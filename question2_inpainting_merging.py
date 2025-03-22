import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from question1_custom_model import UNet

# -----------------------
# 1. Setup and Parameters
# -----------------------

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Diffusion schedule parameters
T = 1000  # total number of diffusion steps
betas = torch.linspace(1e-4, 0.02, T).to(device)  # linear schedule
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# -----------------------
# 2. Load Dataset
# -----------------------

# Load and normalize faces dataset (48x48 grayscale images)
faces_np = np.load("faces23k_48x48.npy") / 255.0  # scale to [0,1]
faces = torch.tensor(faces_np, dtype=torch.float32).unsqueeze(1).to(device)  # shape (N, 1, 48, 48)

# -----------------------
# 3. Load Pretrained DDPM Model
# -----------------------

# Load the pretrained noise prediction model
model = UNet(1, 1).to(device)
model.load_state_dict(torch.load("diff_unet_faces.cpt", map_location=torch.device(device)))
model.eval()  # set to evaluation mode

# -----------------------
# 4. Forward Diffusion Function
# -----------------------

def add_noise(x_0, t_step):
    """
    Applies the forward diffusion process at timestep t_step to input x_0.
    """
    t = torch.tensor(t_step, device=x_0.device)
    alpha_bar_t = alphas_cumprod[t]

    # Expand scalar terms to match image shape
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t).view(-1, 1, 1, 1)
    sqrt_one_minus = torch.sqrt(1 - alpha_bar_t).view(-1, 1, 1, 1)

    # Sample Gaussian noise and combine with original image
    noise = torch.randn_like(x_0)
    z_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise
    return z_t

# -----------------------
# 5. Reverse Sampling Step (Your Structure)
# -----------------------

def reverse_step_for_merge(z_t, timestep, diffusion_model):
    """
    Performs one reverse DDPM step: z_t → z_{t-1}
    """
    t_batch = torch.full((z_t.size(0),), timestep, device=z_t.device, dtype=torch.long)

    with torch.no_grad():
        predicted_noise = diffusion_model(z_t, t_batch)

        beta_t = betas[timestep]
        alpha_t = alphas[timestep]
        alpha_bar_t = alphas_cumprod[timestep]

        scale_x = 1 / torch.sqrt(alpha_t)
        scale_eps = beta_t / torch.sqrt(1 - alpha_bar_t)
        mean_est = scale_x * (z_t - scale_eps * predicted_noise)

        noise = torch.randn_like(z_t) if timestep > 0 else 0.0
        return mean_est + torch.sqrt(beta_t) * noise

# -----------------------
# 6. Full Reverse Sampling Process
# -----------------------

def denoise_from_latent(z_start, t_start, model):
    """
    Performs full reverse sampling from z_start at timestep t_start to final image x_0.
    """
    z = z_start.clone()
    for t in reversed(range(t_start)):
        z = reverse_step_for_merge(z, t, model)
    return z

# -----------------------
# 7. Inpainting/Merging Loop
# -----------------------

# Define 4 different timesteps to explore
timesteps = [50, 100, 200, 400]

# Select 5 random pairs of images
num_pairs = 5
rand_indices = torch.randperm(len(faces))[:2 * num_pairs]
pairs = [(faces[i], faces[i+1]) for i in range(0, len(rand_indices), 2)]

# Container for all collage rows
collage_rows = []

for t in timesteps:
    row_images = []

    for x1, x2 in pairs:
        x1 = x1.unsqueeze(0)  # shape (1, 1, 48, 48)
        x2 = x2.unsqueeze(0)

        # Forward diffusion to timestep t
        z1 = add_noise(x1, t)
        z2 = add_noise(x2, t)

        # Merge latents: left half from z1, right half from z2
        z_merge = z1.clone()
        z_merge[:, :, :, :24] = z1[:, :, :, :24]
        z_merge[:, :, :, 24:] = z2[:, :, :, 24:]

        # Denoise merged latent to get final image
        x_merged = denoise_from_latent(z_merge, t, model)

        # Save original and output images for plotting
        row_images.extend([
            x1.squeeze().cpu().numpy(),
            x2.squeeze().cpu().numpy(),
            x_merged.squeeze().cpu().numpy()
        ])

    collage_rows.append(row_images)

# -----------------------
# 8. Plotting the 4x15 Collage
# -----------------------

fig, axes = plt.subplots(nrows=4, ncols=15, figsize=(15, 8))

for i in range(4):  # each row corresponds to one timestep
    for j in range(15):  # 5 triplets per row: (x1, x2, merged)
        ax = axes[i, j]
        ax.imshow(collage_rows[i][j], cmap='gray')
        ax.axis('off')

plt.tight_layout()
os.makedirs('Images', exist_ok=True)
plt.savefig(os.path.join("Images", "question2_inpainting_merging.png"))
plt.show()
