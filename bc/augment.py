"""Image augmentation, preprocessing, and the training batch generator.

Augmentation pipeline matches the thesis (Section 4.1.10):
zoom, pan, brightness, flip — each applied independently with p=0.5.
Preprocessing matches the NVIDIA paper: crop sky/hood, RGB->YUV, blur, resize to 200x66, normalize.
"""

from __future__ import annotations

from typing import Iterator

import cv2
import matplotlib.image as mpimg
import numpy as np
from imgaug import augmenters as iaa

# Singleton augmenters (avoid rebuilding per call)
_ZOOM = iaa.Affine(scale=(1.0, 1.3))
_PAN = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
_BRIGHTNESS = iaa.Multiply((0.2, 1.2))


def zoom(image: np.ndarray) -> np.ndarray:
    return _ZOOM.augment_image(image)


def pan(image: np.ndarray) -> np.ndarray:
    return _PAN.augment_image(image)


def random_brightness(image: np.ndarray) -> np.ndarray:
    return _BRIGHTNESS.augment_image(image)


def horizontal_flip(image: np.ndarray, steering: float) -> tuple[np.ndarray, float]:
    return cv2.flip(image, 1), -steering


def random_augment(image_path: str, steering: float, rng: np.random.Generator | None = None
                   ) -> tuple[np.ndarray, float]:
    """Load image then apply each augmentation with p=0.5."""
    rng = rng or np.random.default_rng()
    image = mpimg.imread(image_path)
    if rng.random() < 0.5:
        image = pan(image)
    if rng.random() < 0.5:
        image = zoom(image)
    if rng.random() < 0.5:
        image = random_brightness(image)
    if rng.random() < 0.5:
        image, steering = horizontal_flip(image, steering)
    return image, steering


def preprocess(img: np.ndarray) -> np.ndarray:
    """Crop, convert to YUV, blur, resize to 200x66, scale to [0, 1]."""
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    return (img / 255.0).astype(np.float32)


def batch_generator(
    image_paths: np.ndarray,
    steerings: np.ndarray,
    batch_size: int,
    training: bool,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield infinite (X, y) batches for model.fit().

    Training: augment then preprocess. Validation: load + preprocess only.
    """
    rng = np.random.default_rng(seed)
    n = len(image_paths)
    while True:
        idxs = rng.integers(0, n, size=batch_size)
        batch_imgs = np.empty((batch_size, 66, 200, 3), dtype=np.float32)
        batch_steerings = np.empty(batch_size, dtype=np.float32)
        for i, idx in enumerate(idxs):
            if training:
                img, steering = random_augment(image_paths[idx], float(steerings[idx]), rng)
            else:
                img = mpimg.imread(image_paths[idx])
                steering = float(steerings[idx])
            batch_imgs[i] = preprocess(img)
            batch_steerings[i] = steering
        yield batch_imgs, batch_steerings


def augment_window(
    images: list[np.ndarray], steering: float, rng: np.random.Generator
) -> tuple[list[np.ndarray], float]:
    """Augment a temporal window — the SAME transform on every frame.

    A window is a contiguous run of frames, so the augmentation must be locked
    across it: zooming frame 3 but not frame 4, or flipping only half a window,
    would destroy the temporal signal the LSTM is meant to learn. We draw the
    augmentation decisions once and apply them to all frames. `to_deterministic`
    freezes each imgaug augmenter's random parameters so it repeats identically.
    """
    ops = []
    if rng.random() < 0.5:
        ops.append(_PAN.to_deterministic())
    if rng.random() < 0.5:
        ops.append(_ZOOM.to_deterministic())
    if rng.random() < 0.5:
        ops.append(_BRIGHTNESS.to_deterministic())
    do_flip = rng.random() < 0.5

    out: list[np.ndarray] = []
    for img in images:
        for op in ops:
            img = op.augment_image(img)
        if do_flip:
            img = cv2.flip(img, 1)
        out.append(img)
    if do_flip:
        steering = -steering
    return out, steering


def sequence_batch_generator(
    windows: np.ndarray,
    steerings: np.ndarray,
    batch_size: int,
    training: bool,
    seq_len: int = 10,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield infinite (X, y) batches of temporal windows for the LSTM models.

    X has shape `(batch_size, seq_len, 66, 200, 3)`. Each row of `windows` is a
    list of `seq_len` image paths (built by `bc.sequence`). Training augments
    each window as a unit; validation only loads and preprocesses.
    """
    rng = np.random.default_rng(seed)
    n = len(windows)
    while True:
        idxs = rng.integers(0, n, size=batch_size)
        batch = np.empty((batch_size, seq_len, 66, 200, 3), dtype=np.float32)
        batch_steerings = np.empty(batch_size, dtype=np.float32)
        for i, idx in enumerate(idxs):
            frames = [mpimg.imread(p) for p in windows[idx]]
            steering = float(steerings[idx])
            if training:
                frames, steering = augment_window(frames, steering, rng)
            for t, frame in enumerate(frames):
                batch[i, t] = preprocess(frame)
            batch_steerings[i] = steering
        yield batch, batch_steerings
