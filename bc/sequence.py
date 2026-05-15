"""Build contiguous temporal sequences for the LSTM models.

The Udacity track dataset is a single uninterrupted driving run (~14 Hz, with
no inter-frame gap larger than ~0.48 s), so sliding a fixed-length window over
the timestamp-sorted frames yields valid temporal samples.

Two things differ from the CNN pipeline in ``bc.data`` and matter for correctness:

1. **Window first, balance second.** ``balance_steering`` drops individual frames,
   which would punch holes in the timeline. Here we window the full contiguous
   run first, then (optionally) cap windows per steering bin — dropping a whole
   window is fine because windows are independent training samples.
2. **Contiguous split, never shuffled.** Adjacent windows share ``SEQ_LEN - 1``
   frames. A random train/val split would therefore leak almost-identical
   windows across the boundary. We instead split the timeline into a contiguous
   train block and a contiguous val block and window each independently, so no
   window ever straddles the split.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bc.data import STEERING_CORRECTION, load_log

SEQ_LEN = 10  # frames per temporal window — matches the thesis LSTM input
MAX_GAP_S = 0.5  # inter-frame gap above this starts a new contiguous run
DEFAULT_NUM_BINS = 25
DEFAULT_SAMPLES_PER_BIN = 200  # per-bin window cap (lower than the CNN's 400:
#                                windows overlap heavily, so fewer are needed)

# Matches the timestamp tail of a frame filename, e.g.
# "center_2018_07_16_17_11_43_382.jpg" -> 2018-07-16 17:11:43.382
_TS_RE = re.compile(r"_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{3})\.jpg$")


@dataclass
class SequenceData:
    """A set of temporal windows.

    ``windows`` has shape ``(n, SEQ_LEN)``; each row is an ordered list of image
    paths. ``steerings`` has shape ``(n,)`` and holds the target for each window
    — the steering angle of the window's final (most recent) frame.
    """

    windows: np.ndarray
    steerings: np.ndarray

    def __len__(self) -> int:
        return len(self.steerings)


def parse_timestamp(filename: str) -> datetime.datetime:
    """Extract the capture time encoded in a frame filename."""
    match = _TS_RE.search(filename)
    if match is None:
        raise ValueError(f"no timestamp in filename: {filename!r}")
    year, month, day, hour, minute, sec, millis = map(int, match.groups())
    return datetime.datetime(year, month, day, hour, minute, sec, millis * 1000)


def contiguous_runs(
    timestamps: list[datetime.datetime], max_gap_s: float = MAX_GAP_S
) -> list[tuple[int, int]]:
    """Split a sorted timestamp list into ``[start, end)`` index ranges.

    A gap larger than ``max_gap_s`` between consecutive frames ends a run — the
    car was reset or recording paused, so frames either side are not temporally
    adjacent.
    """
    runs: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(timestamps)):
        gap = (timestamps[i] - timestamps[i - 1]).total_seconds()
        if gap > max_gap_s:
            runs.append((start, i))
            start = i
    runs.append((start, len(timestamps)))
    return runs


def _window_camera(
    df, img_dir: Path, camera: str, correction: float, seq_len: int
) -> SequenceData:
    """Slide a ``seq_len`` window over one camera's frames within each run."""
    files = [str(name).strip() for name in df[camera].tolist()]
    timestamps = [parse_timestamp(name) for name in files]
    steerings = df["steering"].to_numpy(dtype=np.float32) + correction

    windows: list[list[str]] = []
    targets: list[float] = []
    for start, end in contiguous_runs(timestamps, MAX_GAP_S):
        for i in range(start, end - seq_len + 1):
            windows.append([str(img_dir / files[j]) for j in range(i, i + seq_len)])
            targets.append(float(steerings[i + seq_len - 1]))
    return SequenceData(
        np.asarray(windows, dtype=object), np.asarray(targets, dtype=np.float32)
    )


def windows_all_cameras(df, img_dir: Path, seq_len: int = SEQ_LEN) -> SequenceData:
    """Build windows from all three cameras and concatenate them.

    Each camera is windowed independently — a window is always single-camera, so
    it stays a coherent temporal sequence. Side cameras get the same ``±0.15``
    steering correction the CNN pipeline uses.
    """
    parts = [
        _window_camera(df, img_dir, "center", 0.0, seq_len),
        _window_camera(df, img_dir, "left", +STEERING_CORRECTION, seq_len),
        _window_camera(df, img_dir, "right", -STEERING_CORRECTION, seq_len),
    ]
    return SequenceData(
        np.concatenate([p.windows for p in parts]),
        np.concatenate([p.steerings for p in parts]),
    )


def balance_windows(
    data: SequenceData,
    num_bins: int = DEFAULT_NUM_BINS,
    samples_per_bin: int = DEFAULT_SAMPLES_PER_BIN,
    seed: int = 42,
) -> SequenceData:
    """Cap windows per steering-target bin to curb the zero-steering bias."""
    rng = np.random.default_rng(seed)
    _, bins = np.histogram(data.steerings, num_bins)
    keep: list[int] = []
    for j in range(num_bins):
        in_bin = np.where(
            (data.steerings >= bins[j]) & (data.steerings <= bins[j + 1])
        )[0]
        rng.shuffle(in_bin)
        keep.extend(in_bin[:samples_per_bin].tolist())
    keep_arr = np.asarray(sorted(keep))
    return SequenceData(data.windows[keep_arr], data.steerings[keep_arr])


def build_sequences(
    data_dir: Path,
    seq_len: int = SEQ_LEN,
    val_frac: float = 0.2,
    balance: bool = True,
    seed: int = 42,
) -> tuple[SequenceData, SequenceData]:
    """Load the dataset and return ``(train, val)`` temporal windows.

    The frame timeline is sorted by capture time and split into a contiguous
    train block (first ``1 - val_frac``) and val block (last ``val_frac``).
    Windows are built within each block, so none crosses the split. Only the
    training set is balanced — the validation set keeps its natural steering
    distribution.
    """
    data_dir = Path(data_dir)
    df = load_log(data_dir)
    # Sort by the center camera's capture time so the timeline is monotonic.
    order = np.argsort([parse_timestamp(str(n).strip()) for n in df["center"]])
    df = df.iloc[order].reset_index(drop=True)

    split = int(len(df) * (1.0 - val_frac))
    img_dir = data_dir / "IMG"
    train = windows_all_cameras(df.iloc[:split].reset_index(drop=True), img_dir, seq_len)
    val = windows_all_cameras(df.iloc[split:].reset_index(drop=True), img_dir, seq_len)

    if balance:
        train = balance_windows(train, seed=seed)
    return train, val
