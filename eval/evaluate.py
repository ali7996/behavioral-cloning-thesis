"""Evaluate a trained model on the held-out validation set with honest regression metrics.

Reports MSE, MAE, RMSE, and R^2 (coefficient of determination).
NOTE: We deliberately do NOT report "accuracy" — accuracy is for classification.
The thesis used 1/(1+MSE) as a pseudo-accuracy; we drop it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model

from bc.augment import batch_generator
from bc.data import balance_steering, load_img_steering, load_log, train_val_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "track"


def collect_predictions(model, image_paths, steerings, batch_size: int = 100):
    """Materialize one pass of the validation set through the generator and predict."""
    n = len(image_paths)
    steps = max(1, n // batch_size)
    y_true: list[float] = []
    y_pred: list[float] = []
    gen = batch_generator(image_paths, steerings, batch_size, training=False, seed=42)
    for _ in range(steps):
        X, y = next(gen)
        preds = model.predict(X, verbose=0).ravel()
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "n_samples": int(len(y_true))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    df = balance_steering(load_log(DATA_DIR))
    drive = load_img_steering(DATA_DIR / "IMG", df)
    _, val = train_val_split(drive)

    model = load_model(args.model_path)
    y_true, y_pred = collect_predictions(model, val.image_paths, val.steerings, args.batch_size)
    m = metrics(y_true, y_pred)
    m["model"] = args.model_path.stem

    print(json.dumps(m, indent=2))
    out_path = args.model_path.with_suffix(".eval.json")
    out_path.write_text(json.dumps(m, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
