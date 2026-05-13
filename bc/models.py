"""Model architectures from the thesis (Section 4.2).

All three match the thesis tables 1–3 in layer order, filter counts, and parameter totals,
but use the modern Keras API (no `lr=`, no `fit_generator`, etc.).
"""

from __future__ import annotations

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    LSTM,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    Reshape,
)
from tensorflow.keras.optimizers import Adam

INPUT_SHAPE = (66, 200, 3)
DEFAULT_LR = 1e-3


def nvidia_cnn(lr: float = DEFAULT_LR) -> Sequential:
    """CNN based on the NVIDIA self-driving architecture (thesis Table 1, ~252k params)."""
    model = Sequential(name="nvidia_cnn")
    model.add(Input(shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Conv2D(64, (3, 3), activation="elu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer=Adam(learning_rate=lr), metrics=["mae"])
    return model


def lstm_steering(lr: float = DEFAULT_LR) -> Model:
    """LSTM regressor over flattened image sequences (thesis Table 2).

    NOTE on data shape: the thesis feeds shape (batch, 10, 66, 200, 3). Each "sequence" is
    10 frames. With the current single-frame batch_generator, you have to assemble sequences
    yourself (see scripts/build_sequences.py). The 10-frame window must come from contiguous
    driving timesteps, NOT random frames, or the LSTM has nothing temporal to learn.
    """
    inp = Input(shape=(10, *INPUT_SHAPE))
    x = Reshape((10, INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]))(inp)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(1)(x)
    model = Model(inp, out, name="lstm_steering")
    model.compile(loss="mse", optimizer=Adam(learning_rate=lr), metrics=["mae"])
    return model


def cnn_lstm(lr: float = DEFAULT_LR) -> Model:
    """2D CNN feature extractor + LSTM head (thesis Table 3)."""
    inp = Input(shape=INPUT_SHAPE)
    x = Conv2D(32, (3, 3), activation="elu", padding="same")(inp)
    x = Conv2D(32, (3, 3), activation="elu", padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    # Reshape to a length-1 "sequence" so the LSTM layers run on the per-frame feature vector.
    # This matches the thesis but is effectively a fancy dense layer — kept for parity.
    feat_dim = (INPUT_SHAPE[0] // 2) * (INPUT_SHAPE[1] // 2) * 32
    x = Reshape((1, feat_dim))(x)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    x = Dense(256, activation="elu")(x)
    x = Dropout(0.5)(x)
    out = Dense(1)(x)
    model = Model(inp, out, name="cnn_lstm")
    model.compile(loss="mse", optimizer=Adam(learning_rate=lr), metrics=["mae"])
    return model


REGISTRY = {
    "nvidia_cnn": nvidia_cnn,
    "lstm": lstm_steering,
    "cnn_lstm": cnn_lstm,
}
