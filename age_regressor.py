# === age_regressor.py ===

import torch.nn as nn                  # For building neural networks
import numpy as np                     # For loading image and age arrays
from torch.utils.data import Dataset, DataLoader  # For managing training data
import torch                           # For tensors and neural network operations
import math                            # For math operations (used in time embedding)

# === Sinusoidal Time Embedding Module ===
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()             # Initialize base nn.Module
        self.dim = dim                 # Save the embedding dimension

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2       # Use half for sin, half for cos
        emb = math.log(10000) / (half_dim - 1)  # Scale frequencies
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # Shape: [half_dim]
        emb = t[:, None] * emb[None, :]  # Outer product → Shape: [B, half_dim]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # Final shape: [B, dim]

# === Downsampling Block for U-Net ===
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_ch)  # Projects time embedding to feature space
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)  # Downsample by 2

    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))  # First conv layer + ReLU
        t_proj = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        h = self.relu(self.conv2(h + t_proj))  # Inject time embedding, then conv + ReLU
        return self.pool(h)  # Downsample

# === Full U-Net Style Regressor ===
class AgeRegressorUNetDown(nn.Module):
    def __init__(self, time_emb_dim=128):
        super().__init__()
        self.time_mlp = SinusoidalTimeEmbedding(time_emb_dim)
        self.down1 = DownBlock(1, 32, time_emb_dim)
        self.down2 = DownBlock(32, 64, time_emb_dim)
        self.down3 = DownBlock(64, 128, time_emb_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(128, 1)  # Final regression head

    def forward(self, z_t, t):
        t_emb = self.time_mlp(t)         # Embed the timestep
        h = self.down1(z_t, t_emb)
        h = self.down2(h, t_emb)
        h = self.down3(h, t_emb)
        h = self.avgpool(h).squeeze(-1).squeeze(-1)  # Flatten to [B, 128]
        return self.fc(h).squeeze(1)     # Output shape: [B]

# === Dataset Class with Forward Diffusion Noise ===
class DiffusedAgeDataset(Dataset):
    def __init__(self, faces, ages, beta_schedule, T):
        self.faces = faces
        self.ages = ages
        self.T = T
        self.beta_schedule = beta_schedule
        self.alpha_bars = torch.cumprod(1 - beta_schedule, dim=0)

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        x = self.faces[idx]  # Original image
        y = self.ages[idx]   # True age
        t = torch.randint(1, self.T, (1,)).item()  # Random timestep
        alpha_bar_t = self.alpha_bars[t]
        noise = torch.randn_like(x)
        z_t = alpha_bar_t.sqrt() * x + (1 - alpha_bar_t).sqrt() * noise  # Forward diffusion
        return z_t, torch.tensor(t).float(), y

# === Training Loop ===
def train_regressor(model, dataloader, epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for z_t, t, y in dataloader:
            z_t, t, y = z_t.to(device), t.to(device), y.to(device)
            pred = model(z_t, t)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "age_regressor.pth")  # Save model

# === Entry Point for Training ===
if __name__ == "__main__":
    faces = np.load("faces23k_48x48.npy")       # Shape: (N, 48, 48)
    ages = np.load("ages23k.npy")               # Shape: (N,)

    mask = (ages >= 0) & (ages <= 100)
    faces = faces[mask] / 255.0
    ages = ages[mask].astype(np.float32)

    faces = torch.tensor(faces, dtype=torch.float32).unsqueeze(1)  # [N, 1, 48, 48]
    ages = torch.tensor(ages, dtype=torch.float32)

    T = 1600
    beta_schedule = torch.linspace(1e-4, 0.02, T)
    dataset = DiffusedAgeDataset(faces, ages, beta_schedule, T)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = AgeRegressorUNetDown()
    train_regressor(model, loader, epochs=10)
