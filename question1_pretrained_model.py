import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, image_file, age_file):
        raw = np.load(image_file)  # Load data from .npy file
        self.ages = np.load(age_file)
        self.data = raw / 255.
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert numpy array to PyTorch tensor
            transforms.RandomHorizontalFlip(),
            transforms.v2.RandomPhotometricDistort(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize data
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = self.transform(image)
        return image, self.ages[idx]

face_dataset = CustomDataset("faces23k_48x48.npy", "ages23k.npy")
data_loader = DataLoader(face_dataset, batch_size=64)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
T = 1000
TIME_EMBEDDING_DIM = 64

def sinusoidal_embedding(times):
    embedding_min_frequency = 1.0
    frequencies = torch.exp(
        torch.linspace(
            np.log(1.0),
            np.log(1000.),
            TIME_EMBEDDING_DIM // 2
        )
    ).view(1, -1).to(times.device)
    angular_speeds = 2.0 * torch.pi * frequencies
    times = times.view(-1, 1).float()
    embeddings = torch.cat(
        [torch.sin(times.matmul(angular_speeds) / T), torch.cos(times.matmul(angular_speeds) / T)], dim=1
    )
    return embeddings

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.dconv_down1 = double_conv(in_channels + TIME_EMBEDDING_DIM, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(832, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, time_index):
        time_embedding = sinusoidal_embedding(time_index)
        x = torch.cat([x, time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2), x.size(3))], dim=1)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)
        x = torch.cat([x, time_embedding.unsqueeze(-1).unsqueeze(-1).expand(x.size(0), -1, x.size(2), x.size(3))], dim=1)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out

model = UNet(1, 1).to(device)
model.load_state_dict(torch.load("diff_unet_faces.cpt", map_location=torch.device(device)))

betas = torch.linspace(
    1e-4,
    .01,
    T,
    dtype=torch.float32,
).to(device)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

# === Forward process: Adds noise to clean image ===
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha_prod = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha_prod * x_start + sqrt_one_minus * noise

# === Reverse process: Removes noise step by step (sampling) ===
def p_sample(x, t):
    t_tensor = torch.tensor([t] * x.shape[0], device=device)
    with torch.no_grad():
        eps_theta = model(x, t_tensor)
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]

        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_bar_t)
        mean = coef1 * (x - coef2 * eps_theta)
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        sigma_t = torch.sqrt(beta_t)
        return mean + sigma_t * noise

# === Full sampling process: From noise to generated image ===
def sample_images(n=100):
    x = torch.randn(n, 1, 48, 48).to(device)
    for t in reversed(range(T)):
        x = p_sample(x, t)
    return x

# === Display 10x10 collage ===
def show_images(x):
    x = x.detach().cpu().numpy()
    x = np.clip((x + 1) / 2.0, 0, 1)  # Denormalize
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i][0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    os.makedirs('Images', exist_ok=True)
    plt.savefig(os.path.join("Images", "question1_generated_images_pretrained_trained_model.png"))
    plt.show()

# === 1. Unconditional Generation ===
if __name__ == '__main__':
    print("Generating samples using DDPM...")
    generated = sample_images(100)
    show_images(generated)
