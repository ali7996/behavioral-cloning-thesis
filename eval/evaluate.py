"""Evaluate a trained model on the validation set with honest regression metrics.

Reports MSE, MAE, RMSE, and R^2 (coefficient of determination).
NOTE: We deliberately do NOT report "accuracy" — accuracy is for classification.
The thesis used 1/(1+MSE) as a pseudo-accuracy; we drop it.

Two evaluation modes:

* **random split** (default, frame models only) — the original 80/20 random
  frame split. This produced the headline CNN number (MAE 0.168).
* **contiguous split** (`--contiguous`, and always for temporal models) — the
  timeline-ordered split from `bc.sequence`. Temporal models *require* it.
  Passing `--contiguous` to the CNN re-scores it on the last frame of every
  temporal val window, giving a like-for-like comparison against the LSTMs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model

from bc.augment import batch_generator, preprocess
from bc.data import balance_steering, load_img_steering, load_log, train_val_split
from bc.sequence import build_sequences

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "track"


def collect_predictions(model, image_paths, steerings, batch_size: int = 100):
    """One pass of the random-split val frames through the generator (CNN headline)."""
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


def predict_frames(model, image_paths, batch_size: int = 64) -> np.ndarray:
    """Deterministic full pass over single frames — preprocess once, no augmentation."""
    preds: list[float] = []
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        X = np.stack([preprocess(mpimg.imread(p)) for p in chunk])
        preds.extend(model.predict(X, verbose=0).ravel().tolist())
    return np.asarray(preds)


def predict_sequences(model, windows, batch_size: int = 16) -> np.ndarray:
    """Deterministic full pass over temporal windows — no augmentation."""
    preds: list[float] = []
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        X = np.stack(
            [np.stack([preprocess(mpimg.imread(p)) for p in w]) for w in chunk]
        )
        preds.extend(model.predict(X, verbose=0).ravel().tolist())
    return np.asarray(preds)


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--contiguous", action="store_true",
                        help="evaluate on the contiguous timeline split (for LSTM comparison)")
    args = parser.parse_args()

    model = load_model(args.model_path)
    is_sequence = len(model.input_shape) == 5

    if is_sequence or args.contiguous:
        _, val = build_sequences(DATA_DIR, balance=False)
        y_true = val.steerings
        if is_sequence:
            y_pred = predict_sequences(model, val.windows, args.batch_size)
        else:
            last_frames = np.asarray([w[-1] for w in val.windows])
            y_pred = predict_frames(model, last_frames, args.batch_size)
        split = "contiguous"
    else:
        df = balance_steering(load_log(DATA_DIR))
        drive = load_img_steering(DATA_DIR / "IMG", df)
        _, val = train_val_split(drive)
        y_true, y_pred = collect_predictions(model, val.image_paths, val.steerings,
                                             args.batch_size)
        split = "random"

    m = metrics(y_true, y_pred)
    m["model"] = getattr(model, "name", args.model_path.stem)
    m["split"] = split

    print(json.dumps(m, indent=2))
    out_path = args.model_path.with_suffix(f".eval_{split}.json")
    out_path.write_text(json.dumps(m, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
