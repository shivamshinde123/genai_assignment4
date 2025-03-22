import os  # For file operations
import torch  # PyTorch core library
import torch.nn as nn  # Neural network module
import numpy as np  # For handling image arrays
import matplotlib.pyplot as plt  # For plotting images
import torch.nn.functional as F  # Common functions like loss
from torchvision.transforms import v2  # Additional transformations
from torch.utils.data import Dataset, DataLoader  # For batching data
from torchvision import datasets, transforms  # For image preprocessing

# Custom dataset to load face images and their ages
class CustomDataset(Dataset):
    def __init__(self, image_file, age_file):
        raw = np.load(image_file)  # Load face image data from .npy
        self.ages = np.load(age_file)  # Load corresponding age labels
        self.data = raw / 255.  # Normalize pixel values to [0, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.RandomHorizontalFlip(),  # Randomly flip image
            transforms.v2.RandomPhotometricDistort(),  # Light/color jitter
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.data)  # Total number of samples

    def __getitem__(self, idx):
        image = self.data[idx]  # Get image
        image = self.transform(image)  # Apply transforms
        return image, self.ages[idx]  # Return image and age

# Load the dataset and wrap it in a DataLoader
face_dataset = CustomDataset("faces23k_48x48.npy", "ages23k.npy")
data_loader = DataLoader(face_dataset, batch_size=64, shuffle=True)

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
T = 1000  # Number of diffusion steps
TIME_EMBEDDING_DIM = 64  # Dimension of time embedding

# Generate sinusoidal embeddings for timestep indices
def sinusoidal_embedding(times):
    frequencies = torch.exp(
        torch.linspace(
            np.log(1.0), np.log(1000.), TIME_EMBEDDING_DIM // 2
        )
    ).view(1, -1).to(times.device)  # Frequencies shape: (1, D/2)
    angular_speeds = 2.0 * torch.pi * frequencies  # Angular frequencies
    times = times.view(-1, 1).float()  # Make times shape (B, 1)
    embeddings = torch.cat([
        torch.sin(times.matmul(angular_speeds) / T),  # Sinusoidal part
        torch.cos(times.matmul(angular_speeds) / T)   # Cosine part
    ], dim=1)
    return embeddings  # Final shape: (B, D)

# Block of two conv layers with ReLU
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

# U-Net architecture used for noise prediction
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Downsampling layers
        self.dconv_down1 = double_conv(in_channels + TIME_EMBEDDING_DIM, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        # Upsampling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = double_conv(832, 256)  # 512 + 320 (time embedding)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, out_channels, 1)  # Final 1x1 conv

    def forward(self, x, time_index):
        time_embedding = sinusoidal_embedding(time_index)  # Shape: (B, D)
        # Expand and concatenate time embedding to image input
        x = torch.cat([
            x,
            time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2), x.size(3))
        ], dim=1)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        # Add time embedding again before upsampling
        x = torch.cat([
            x,
            time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2), x.size(3))
        ], dim=1)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        return self.conv_last(x)  # Output predicted noise

# === Noise schedule and helper tensors ===
betas = torch.linspace(1e-4, 0.01, T).to(device)  # Linearly spaced betas
alphas = 1. - betas  # alpha_t = 1 - beta_t
alphas_cumprod = torch.cumprod(alphas, dim=0)  # cumulative product of alphas
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

# === Forward process: q(x_t | x_0) ===
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)  # Sample noise if not provided
    sqrt_alpha_prod = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)  # Reshape for broadcasting
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha_prod * x_start + sqrt_one_minus * noise  # Apply forward formula

# === Train the model to predict noise ===
def train_ddpm(model, dataloader, epochs=30, lr=2e-4):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
    for epoch in range(epochs):
        for x, _ in dataloader:
            x = x.to(device)  # Move to GPU if available
            t = torch.randint(0, T, (x.shape[0],), device=device)  # Sample random timestep for each image
            noise = torch.randn_like(x)  # Sample Gaussian noise
            x_noisy = q_sample(x, t, noise)  # Add noise to image
            noise_pred = model(x_noisy, t)  # Predict the noise
            loss = F.mse_loss(noise_pred, noise)  # MSE between predicted and true noise

            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model weights

        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "ddpm_trained_faces.pth")  # Save trained model
    print("Model saved as ddpm_trained_faces.pth")

# === Reverse sampling from noise to image ===
def p_sample(x, t):
    t_tensor = torch.tensor([t] * x.shape[0], device=device)  # Create batch of current timestep t
    with torch.no_grad():  # Disable gradient calculation
        eps_theta = model(x, t_tensor)  # Predict noise using model
        beta_t = betas[t]  # Get beta for timestep t
        alpha_t = alphas[t]  # Get alpha for timestep t
        alpha_bar_t = alphas_cumprod[t]  # Get cumulative alpha product

        coef1 = 1 / torch.sqrt(alpha_t)  # Precompute scaling factor
        coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)  # Scale for predicted noise
        mean = coef1 * (x - coef2 * eps_theta)  # Compute mean of reverse step

        noise = torch.randn_like(x) if t > 0 else 0  # Add noise only if t > 0
        sigma_t = torch.sqrt(beta_t)  # Standard deviation for noise
        return mean + sigma_t * noise  # Return sampled x_{t-1}

# === Generate a batch of images by reverse denoising ===
def sample_images(n=100):
    x = torch.randn(n, 1, 48, 48).to(device)  # Start with pure Gaussian noise
    for t in reversed(range(T)):  # Reverse denoising loop from T to 0
        x = p_sample(x, t)  # Apply one denoising step
    return x  # Final generated samples

# === Visualize a 10x10 grid of generated images ===
def show_images(x):
    x = x.detach().cpu().numpy()  # Move tensor to CPU and convert to NumPy
    x = np.clip((x + 1) / 2.0, 0, 1)  # Convert from [-1,1] to [0,1] range
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))  # Create 10x10 subplot grid
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i][0], cmap='gray')  # Show image in grayscale
        ax.axis('off')  # Hide axis ticks
    plt.tight_layout()  # Adjust spacing
    os.makedirs('Images', exist_ok=True)
    plt.savefig(os.path.join('Images', "question1_generated_images_custom_trained_model.png"))  # Save the collage to file
    plt.show()  # Display the collage

# === Run training if script is main ===
if __name__ == '__main__':
    model = UNet(1, 1).to(device)  # Create U-Net model
    train_ddpm(model, data_loader)  # Train the model

    # === Load trained model and test sampling ===
    model.load_state_dict(torch.load("ddpm_trained_faces.pth", map_location=device))  # Load saved model
    model.eval()  # Set model to evaluation mode

    print("Sampling from trained DDPM...")  # Log message
    sampled = sample_images(100)  # Generate 100 samples
    show_images(sampled)  # Show and save the output
