"""Train the NVIDIA CNN on the real Udacity track data and persist weights+history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from tensorflow.keras.callbacks import ModelCheckpoint

from bc.augment import batch_generator
from bc.data import balance_steering, load_img_steering, load_log, train_val_split
from bc.models import nvidia_cnn

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "track"
MODELS_DIR = ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading driving_log.csv …")
    df = balance_steering(load_log(DATA_DIR))
    drive = load_img_steering(DATA_DIR / "IMG", df)
    train, val = train_val_split(drive)
    print(f"Train pairs: {len(train.image_paths):,}  Val pairs: {len(val.image_paths):,}")

    model = nvidia_cnn(lr=args.lr)
    model.summary()

    val_steps = max(1, len(val.image_paths) // args.batch_size)

    train_gen = batch_generator(train.image_paths, train.steerings, args.batch_size,
                                training=True, seed=1)
    val_gen = batch_generator(val.image_paths, val.steerings, args.batch_size,
                              training=False, seed=2)

    weights_path = MODELS_DIR / "nvidia_cnn.keras"
    ckpt = ModelCheckpoint(weights_path, monitor="val_loss", save_best_only=True, verbose=1)

    t0 = perf_counter()
    history = model.fit(
        train_gen,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=[ckpt],
        verbose=1,
    )
    elapsed = perf_counter() - t0

    history_path = MODELS_DIR / "nvidia_cnn_history.json"
    history_path.write_text(json.dumps({
        "model": "nvidia_cnn",
        "elapsed_seconds": round(elapsed, 1),
        "args": vars(args),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }, indent=2))
    print(f"\nTraining done in {elapsed/60:.1f} min")
    print(f"Best weights: {weights_path}")
    print(f"History:      {history_path}")


if __name__ == "__main__":
    main()
