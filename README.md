# Marketing Mix Optimizer

A minimal Bayesian media mix modeling (MMM) and budget optimization system. It splits revenue into per-channel contributions, estimates ROI with uncertainty, and reallocates spend to maximize predicted response.

## Components

- `mmo/transforms.py` — geometric adstock and Hill saturation (PyTorch)
- `mmo/model.py` — Bayesian MMM fit with variational inference; posterior channel contributions
- `mmo/data.py` — synthetic data generator and Spark-based weekly aggregation of raw spend logs
- `mmo/optimize.py` — Optuna budget allocation under a fixed total budget, plus hyperparameter tuning with scikit-learn time-series CV

## Quickstart

```bash
pip install -r requirements.txt
python -m scripts.run_demo
```

The demo fits the model on synthetic weekly data, prints per-channel ROI with 90% credible intervals, and proposes an optimal allocation.

## Stack

Python, PyTorch, scikit-learn, Spark (PySpark), Optuna.
