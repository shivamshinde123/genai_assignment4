# === guided_ddpm.py ===

import torch                               # Core PyTorch library
import os                                  # For creating directories and saving files
import torch.nn.functional as F            # Functional API for computing loss functions
import matplotlib.pyplot as plt            # For plotting and visualizing generated images
import torchvision.utils as vutils         # For creating a grid of images for visualization

from age_regressor import AgeRegressorUNetDown        # Import the pretrained age regressor
from question1_pretrained_model import UNet           # Import the pretrained DDPM denoiser (UNet)

# === Set Diffusion Hyperparameters ===

T = 1000  # Total number of diffusion steps

# Define a linear beta schedule: linearly increasing variance over time
betas = torch.linspace(1e-4, 0.02, T)

# Compute alpha_t = 1 - beta_t
alphas = 1.0 - betas

# Compute cumulative product of alphas: α_bar_t = prod_{s=1}^t alpha_s
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Compute α_bar_{t-1} by shifting the cumulative product and prepending a 1.0 for t=0
alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

# Set device to GPU if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move all diffusion-related tensors to the selected device
betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
alphas_cumprod_prev = alphas_cumprod_prev.to(device)

# === Predict x_0 from x_t and epsilon_theta using DDPM equation ===
def predict_x0_from_eps(x_t, eps_theta, t):
    # Compute sqrt(α_bar_t) and reshape to match image tensor dimensions
    sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)

    # Compute sqrt(1 - α_bar_t) and reshape
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(-1, 1, 1, 1)

    # Use the DDPM reverse formula to estimate x_0 from x_t and eps_theta
    return (x_t - sqrt_one_minus_alpha_cumprod_t * eps_theta) / sqrt_alpha_cumprod_t

# === Sample x_{t-1} from posterior given x_t and epsilon_theta ===
def sample_posterior(x_t, eps_theta, t):
    # Get the necessary diffusion coefficients for timestep t
    beta_t = betas[t].view(-1, 1, 1, 1)
    alpha_t = alphas[t].view(-1, 1, 1, 1)
    alpha_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_cumprod_prev_t = alphas_cumprod_prev[t].view(-1, 1, 1, 1)

    # Estimate x_0 from the current noisy image x_t
    x0_pred = predict_x0_from_eps(x_t, eps_theta, t)

    # Compute the posterior mean for p(x_{t-1} | x_t, x_0)
    posterior_mean = (
        beta_t * torch.sqrt(alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * x0_pred +
        (1 - alpha_cumprod_prev_t) * torch.sqrt(alpha_t) / (1 - alpha_cumprod_t) * x_t
    )

    # Compute the variance of the posterior
    posterior_variance = beta_t * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)

    # Sample Gaussian noise (or 0 if t == 0)
    noise = torch.randn_like(x_t) if (t[0] > 0) else torch.zeros_like(x_t)

    # Return the sampled x_{t-1}
    return posterior_mean + torch.sqrt(posterior_variance) * noise

# === Classifier-Guided DDPM Sampling ===
def classifier_guided_ddpm_sampling(
    denoiser,         # The trained DDPM denoising model (UNet)
    regressor,        # The trained age regressor model
    num_steps,        # Total number of diffusion steps (T)
    shape,            # Shape of the sample to generate (e.g., [1, 1, 48, 48])
    y_target,         # Target regression value to guide toward
    guidance_scale=1.0,  # Strength of the guidance term
    device="cuda"        # Device to run computation on
):
    # Start from pure Gaussian noise
    x = torch.randn(shape).to(device)

    # Loop over timesteps in reverse order (T-1 to 0)
    for t in reversed(range(num_steps)):
        # Create tensor filled with timestep value t
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Enable gradient tracking for x
        x.requires_grad_(True)

        # Predict noise ε_θ(x_t, t)
        eps_theta = denoiser(x, t_tensor)

        # Estimate clean image x_0 from x_t and ε_θ
        x_denoised = predict_x0_from_eps(x, eps_theta, t_tensor)

        # Predict the regressed value (e.g., age) from x_0
        pred = regressor(x_denoised, t_tensor.float())

        # Construct target tensor of same shape as prediction
        target = torch.full_like(pred, y_target).to(device)

        # Compute regression loss (mean squared error)
        loss = F.mse_loss(pred, target)

        # Compute gradient of loss w.r.t. x
        grad = torch.autograd.grad(loss, x)[0]

        # Apply gradient-based guidance to sample x
        x = x.detach() - guidance_scale * grad

        # Recompute ε_θ using updated x, but without tracking gradients
        with torch.no_grad():
            eps_theta = denoiser(x, t_tensor)
            x = sample_posterior(x, eps_theta, t_tensor)  # Get x_{t-1}

    # Return final generated image (x_0)
    return x

# === Plot and Save Grid of Guided Samples ===
def plot_guided_samples(denoiser, regressor, num_steps, target_values, samples_per_target, shape, guidance_scale=1.0, device="cuda"):
    denoiser.eval()      # Set denoiser to evaluation mode
    regressor.eval()     # Set regressor to evaluation mode

    all_rows = []        # Store generated rows of images

    # Loop through each target value (e.g., different ages)
    for y_target in target_values:
        row = []         # List to hold samples for this target

        # Generate multiple samples for each target
        for _ in range(samples_per_target):
            sample = classifier_guided_ddpm_sampling(
                denoiser, regressor, num_steps, shape, y_target, guidance_scale, device
            )
            sample = sample.clamp(0, 1).detach().cpu()  # Clamp and move to CPU for display
            row.append(sample)  # Add to the row

        row = torch.cat(row, dim=0)  # Concatenate all samples in row
        all_rows.append(row)         # Add row to overall grid

    # Concatenate all rows into one big tensor
    grid = torch.cat(all_rows, dim=0)  # Shape: [B_total, C, H, W]

    # Create a grid of images for visualization
    grid_img = vutils.make_grid(grid, nrow=samples_per_target, padding=2, normalize=True)

    # Plot the image grid
    plt.figure(figsize=(samples_per_target * 2, len(target_values) * 2))
    plt.imshow(grid_img.permute(1, 2, 0).numpy())  # Convert to HWC format for plotting
    plt.axis("off")
    plt.title("Classifier-Guided DDPM Samples")

    # Create folder to save images
    os.makedirs("Images", exist_ok=True)

    # Save the image grid to file
    plt.savefig(os.path.join("Images", "classifier-guided-DDPM.png"))
    plt.show()  # Show the plot

# === Main Entry Point ===
if __name__ == "__main__":
    # Load pretrained DDPM denoiser model
    denoiser = UNet(1, 1).to(device)
    denoiser.load_state_dict(torch.load("diff_unet_faces.cpt", map_location=device))

    # Load pretrained regression model
    regressor = AgeRegressorUNetDown().to(device)
    regressor.load_state_dict(torch.load("age_regressor.pth", map_location=device))

    # Run guided sampling and generate visualization
    plot_guided_samples(
        denoiser=denoiser,                  # Pretrained DDPM model
        regressor=regressor,                # Pretrained regression model
        num_steps=1000,                     # Number of diffusion steps
        target_values=[0.0, 0.3, 0.5, 0.7, 1.0],  # Regression targets to guide toward
        samples_per_target=5,               # Number of samples per target
        shape=[1, 1, 48, 48],               # Shape of generated image
        guidance_scale=1.5,                 # Strength of classifier guidance
        device=device                       # Device to run the model on
    )
