"""End-to-end demo on synthetic data: fit, decompose ROI, optimize budget."""
import torch
from sklearn.preprocessing import StandardScaler

from mmo.data import synthetic
from mmo.model import BayesianMMM
from mmo.optimize import optimize_budget

CHANNELS = ["tv", "search", "social", "display"]

df = synthetic(channels=CHANNELS)
scale = df[CHANNELS].values.mean()
x = torch.tensor(df[CHANNELS].values / scale, dtype=torch.float32)
controls = torch.tensor(StandardScaler().fit_transform(df[["seasonality"]]), dtype=torch.float32)
y_mean, y_std = df["revenue"].mean(), df["revenue"].std()
y = torch.tensor((df["revenue"].values - y_mean) / y_std, dtype=torch.float32)

model = BayesianMMM(n_channels=len(CHANNELS), n_controls=1).fit(x, y, controls)

contrib = model.contributions(x) * y_std  # revenue units
spend = df[CHANNELS].values.sum(0)
print("\nChannel ROI (posterior mean [5%, 95%]):")
for i, c in enumerate(CHANNELS):
    roi = contrib[:, i] / spend[i]
    lo, hi = roi.quantile(0.05).item(), roi.quantile(0.95).item()
    print(f"  {c:8s} {roi.mean().item():.3f}  [{lo:.3f}, {hi:.3f}]")

budget = x.mean(0).sum().item()
result = optimize_budget(model, x, total_budget=budget, n_trials=200)
print("\nOptimal weekly allocation (same total budget):")
for c, a in zip(CHANNELS, result["allocation"]):
    print(f"  {c:8s} {a * scale:8.1f}")
