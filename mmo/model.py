"""Bayesian MMM fit via variational inference (mean-field Gaussian) in PyTorch."""
import torch
import torch.nn as nn
from torch.distributions import Normal, HalfNormal, kl_divergence

from .transforms import adstock, hill


class BayesianMMM(nn.Module):
    """y_t = base + sum_c beta_c * hill(adstock(x_tc)) + controls @ gamma + noise."""

    def __init__(self, n_channels: int, n_controls: int = 0):
        super().__init__()
        self.C, self.K = n_channels, n_controls
        # variational params (unconstrained)
        self.beta_mu = nn.Parameter(torch.zeros(n_channels))
        self.beta_logsd = nn.Parameter(torch.full((n_channels,), -2.0))
        self.gamma = nn.Parameter(torch.zeros(n_controls))
        self.base = nn.Parameter(torch.zeros(1))
        self.decay_logit = nn.Parameter(torch.zeros(n_channels))
        self.half_sat_log = nn.Parameter(torch.zeros(n_channels))
        self.slope_log = nn.Parameter(torch.zeros(n_channels))
        self.noise_log = nn.Parameter(torch.zeros(1))

    def channel_effects(self, x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        decay = torch.sigmoid(self.decay_logit)
        sat = hill(adstock(x, decay), self.half_sat_log.exp(), self.slope_log.exp())
        return sat * beta  # (T, C)

    def forward(self, x, controls=None, beta=None):
        beta = self.beta_mu.exp() if beta is None else beta
        y = self.base + self.channel_effects(x, beta).sum(1)
        if controls is not None and self.K:
            y = y + controls @ self.gamma
        return y

    def elbo(self, x, y, controls=None, n_samples: int = 4):
        q = Normal(self.beta_mu, self.beta_logsd.exp())
        prior = Normal(torch.zeros(self.C), torch.ones(self.C))  # prior on log-beta
        ll = 0.0
        for _ in range(n_samples):
            beta = q.rsample().exp()  # positive channel effects
            pred = self.forward(x, controls, beta)
            ll = ll + Normal(pred, self.noise_log.exp()).log_prob(y).sum()
        kl = kl_divergence(q, prior).sum()
        kl = kl + (-HalfNormal(1.0).log_prob(self.noise_log.exp())).sum()
        return ll / n_samples - kl

    def fit(self, x, y, controls=None, epochs: int = 2000, lr: float = 0.02, verbose: bool = True):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for i in range(epochs):
            opt.zero_grad()
            loss = -self.elbo(x, y, controls)
            loss.backward()
            opt.step()
            if verbose and i % 500 == 0:
                print(f"epoch {i:5d}  -elbo {loss.item():.2f}")
        return self

    @torch.no_grad()
    def contributions(self, x, n_samples: int = 200):
        """Posterior samples of per-channel total contribution: (S, C)."""
        q = Normal(self.beta_mu, self.beta_logsd.exp())
        return torch.stack([self.channel_effects(x, q.sample().exp()).sum(0) for _ in range(n_samples)])
