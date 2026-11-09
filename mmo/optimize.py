"""Budget allocation with Optuna; hyperparameter search for the model."""
import optuna
import torch
from sklearn.model_selection import TimeSeriesSplit

from .model import BayesianMMM


def optimize_budget(model: BayesianMMM, x_hist: torch.Tensor, total_budget: float,
                    bounds=(0.5, 1.5), n_trials: int = 300) -> dict:
    """Find per-channel weekly spend maximizing predicted response under a fixed total budget.

    bounds: allowed multiplier range relative to historical mean spend per channel.
    """
    C = x_hist.shape[1]
    hist_mean = x_hist.mean(0)

    def objective(trial):
        w = torch.tensor([trial.suggest_float(f"w{c}", *bounds) for c in range(C)])
        alloc = hist_mean * w
        alloc = alloc / alloc.sum() * total_budget
        plan = alloc.repeat(x_hist.shape[0], 1)
        with torch.no_grad():
            return model(plan).mean().item()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    w = torch.tensor([study.best_params[f"w{c}"] for c in range(C)])
    alloc = hist_mean * w
    alloc = alloc / alloc.sum() * total_budget
    return {"allocation": alloc.tolist(), "predicted_response": study.best_value}


def tune_hyperparams(x: torch.Tensor, y: torch.Tensor, n_trials: int = 10) -> dict:
    """Tune learning rate / epochs via time-series cross-validation."""
    cv = TimeSeriesSplit(n_splits=3)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
        epochs = trial.suggest_int("epochs", 300, 1500, step=300)
        errs = []
        for tr, te in cv.split(x):
            m = BayesianMMM(x.shape[1]).fit(x[: tr[-1] + 1], y[: tr[-1] + 1], epochs=epochs, lr=lr, verbose=False)
            with torch.no_grad():
                pred = m(x[: te[-1] + 1])[te]
            errs.append(((pred - y[te]) ** 2).mean().item())
        return sum(errs) / len(errs)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
