"""Train a temporal model (lstm_steering or cnn_lstm) on real 10-frame windows.

Unlike the CNN, the temporal models need contiguous sequences, so this script
uses `bc.sequence` (window-first, contiguous train/val split) instead of the
frame-level pipeline in `bc.data`.

    python -m scripts.train_lstm --model lstm
    python -m scripts.train_lstm --model cnn_lstm

EarlyStopping is on by default: ~3k balanced windows against a 1M–20M parameter
recurrent net overfits quickly, so we stop when validation loss stops improving.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from bc.augment import sequence_batch_generator
from bc.models import REGISTRY
from bc.sequence import SEQ_LEN, build_sequences

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "track"
MODELS_DIR = ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "cnn_lstm"], default="cnn_lstm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps-per-epoch", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4, help="EarlyStopping patience")
    parser.add_argument("--no-balance", action="store_true", help="keep natural distribution")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    print("Building temporal windows from driving_log.csv …")
    train, val = build_sequences(DATA_DIR, seq_len=SEQ_LEN, balance=not args.no_balance)
    print(f"Train windows: {len(train):,}  Val windows: {len(val):,}")

    model = REGISTRY[args.model](lr=args.lr)
    model.summary()

    val_steps = max(1, len(val) // args.batch_size)
    train_gen = sequence_batch_generator(
        train.windows, train.steerings, args.batch_size, training=True, seed=1
    )
    val_gen = sequence_batch_generator(
        val.windows, val.steerings, args.batch_size, training=False, seed=2
    )

    weights_path = MODELS_DIR / f"{model.name}.keras"
    callbacks = [
        ModelCheckpoint(weights_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=args.patience,
                      restore_best_weights=True, verbose=1),
    ]

    t0 = perf_counter()
    history = model.fit(
        train_gen,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = perf_counter() - t0

    history_path = MODELS_DIR / f"{model.name}_history.json"
    history_path.write_text(json.dumps({
        "model": model.name,
        "elapsed_seconds": round(elapsed, 1),
        "args": vars(args),
        "train_windows": len(train),
        "val_windows": len(val),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }, indent=2))
    print(f"\nTraining done in {elapsed/60:.1f} min")
    print(f"Best weights: {weights_path}")
    print(f"History:      {history_path}")


if __name__ == "__main__":
    main()
