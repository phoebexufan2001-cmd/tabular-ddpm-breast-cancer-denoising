"""Core tabular DDPM components used in the denoising experiment."""

import math

import torch
from torch import nn


def make_beta_schedule(steps, beta_start=1e-4, beta_end=2e-2):
    """Create a linear diffusion variance schedule."""
    return torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)


def sinusoidal_timestep_embedding(timesteps, dimension):
    """Encode diffusion timesteps for the noise-prediction network."""
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    arguments = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=1)

    if dimension % 2 == 1:
        padding = torch.zeros((embedding.size(0), 1), device=timesteps.device)
        embedding = torch.cat([embedding, padding], dim=1)

    return embedding


class NoisePredictionMLP(nn.Module):
    """MLP that predicts Gaussian noise from a noisy tabular sample."""

    def __init__(self, feature_count, time_dimension=128, hidden=256, depth=3):
        super().__init__()
        self.time_dimension = time_dimension

        layers = []
        input_dimension = feature_count + time_dimension
        for _ in range(depth):
            layers.extend([nn.Linear(input_dimension, hidden), nn.ReLU()])
            input_dimension = hidden
        layers.append(nn.Linear(hidden, feature_count))
        self.network = nn.Sequential(*layers)

    def forward(self, features, timesteps):
        time_embedding = sinusoidal_timestep_embedding(
            timesteps, self.time_dimension
        )
        return self.network(torch.cat([features, time_embedding], dim=1))


class TabularDDPM:
    """Forward diffusion and reverse denoising operations for tabular data."""

    def __init__(self, steps=500, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.steps = steps
        self.device = device
        self.betas = make_beta_schedule(
            steps, beta_start, beta_end
        ).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        alpha_bar_previous = torch.cat(
            [torch.tensor([1.0], device=device), self.alpha_bar[:-1]]
        )
        posterior_variance = (
            self.betas
            * (1.0 - alpha_bar_previous)
            / (1.0 - self.alpha_bar)
        )
        self.posterior_variance = torch.clamp(
            posterior_variance, min=1e-20
        )

    def add_noise(self, clean_features, timesteps, noise=None):
        """Sample noisy features from the forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(clean_features)
        alpha_bar = self.alpha_bar[timesteps].unsqueeze(1)
        noisy_features = (
            torch.sqrt(alpha_bar) * clean_features
            + torch.sqrt(1.0 - alpha_bar) * noise
        )
        return noisy_features, noise

    @torch.no_grad()
    def denoise(self, model, noisy_features, start_step):
        """Run deterministic reverse diffusion from a selected start step."""
        features = noisy_features
        for step in range(start_step, -1, -1):
            timesteps = torch.full(
                (features.shape[0],),
                step,
                device=features.device,
                dtype=torch.long,
            )
            predicted_noise = model(features, timesteps)
            beta = self.betas[timesteps].unsqueeze(1)
            alpha = self.alphas[timesteps].unsqueeze(1)
            alpha_bar = self.alpha_bar[timesteps].unsqueeze(1)
            features = (1.0 / torch.sqrt(alpha)) * (
                features
                - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise
            )
        return features

