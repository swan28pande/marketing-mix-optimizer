"""Media transforms: geometric adstock and Hill saturation (PyTorch)."""
import torch


def adstock(x: torch.Tensor, decay: torch.Tensor, max_lag: int = 8) -> torch.Tensor:
    """Geometric adstock along time axis. x: (T, C), decay: (C,) in (0, 1)."""
    T = x.shape[0]
    weights = decay.unsqueeze(0) ** torch.arange(max_lag, dtype=x.dtype).unsqueeze(1)  # (L, C)
    weights = weights / weights.sum(0, keepdim=True)
    out = torch.zeros_like(x)
    for lag in range(max_lag):
        out[lag:] += weights[lag] * x[: T - lag]
    return out


def hill(x: torch.Tensor, half_sat: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    """Hill saturation curve, maps spend to (0, 1)."""
    return x.pow(slope) / (x.pow(slope) + half_sat.pow(slope))
