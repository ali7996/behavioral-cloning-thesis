"""Load and balance the Udacity simulator driving dataset."""

from __future__ import annotations

import ntpath
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

DEFAULT_COLUMNS = ("center", "left", "right", "steering", "throttle", "reverse", "speed")
STEERING_CORRECTION = 0.15  # offset applied to side-camera images
DEFAULT_NUM_BINS = 25
DEFAULT_SAMPLES_PER_BIN = 400


@dataclass
class DriveData:
    image_paths: np.ndarray
    steerings: np.ndarray


def _path_leaf(path: str) -> str:
    """Strip directory from a (possibly Windows-style) path."""
    _, tail = ntpath.split(path)
    return tail


def load_log(data_dir: Path, columns: tuple[str, ...] = DEFAULT_COLUMNS) -> pd.DataFrame:
    """Read driving_log.csv and reduce camera paths to bare filenames."""
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "driving_log.csv", names=list(columns))
    for col in ("center", "left", "right"):
        df[col] = df[col].apply(_path_leaf)
    return df


def balance_steering(
    df: pd.DataFrame,
    num_bins: int = DEFAULT_NUM_BINS,
    samples_per_bin: int = DEFAULT_SAMPLES_PER_BIN,
    seed: int = 42,
) -> pd.DataFrame:
    """Cap each steering-angle bin to reduce the heavy zero-steering bias."""
    rng = np.random.default_rng(seed)
    _, bins = np.histogram(df["steering"], num_bins)
    drop_idx: list[int] = []
    for j in range(num_bins):
        in_bin = df.index[(df["steering"] >= bins[j]) & (df["steering"] <= bins[j + 1])].tolist()
        rng.shuffle(in_bin)
        drop_idx.extend(in_bin[samples_per_bin:])
    return df.drop(index=drop_idx).reset_index(drop=True)


def load_img_steering(img_dir: Path, df: pd.DataFrame) -> DriveData:
    """Stack center+left+right camera frames with steering correction."""
    img_dir = Path(img_dir)
    paths: list[str] = []
    steerings: list[float] = []
    for _, row in df.iterrows():
        steer = float(row["steering"])
        paths.append(str(img_dir / row["center"].strip()))
        steerings.append(steer)
        paths.append(str(img_dir / row["left"].strip()))
        steerings.append(steer + STEERING_CORRECTION)
        paths.append(str(img_dir / row["right"].strip()))
        steerings.append(steer - STEERING_CORRECTION)
    return DriveData(np.asarray(paths), np.asarray(steerings, dtype=np.float32))


def train_val_split(
    drive: DriveData, test_size: float = 0.2, seed: int = 6
) -> tuple[DriveData, DriveData]:
    x_train, x_val, y_train, y_val = train_test_split(
        drive.image_paths, drive.steerings, test_size=test_size, random_state=seed
    )
    return DriveData(x_train, y_train), DriveData(x_val, y_val)
