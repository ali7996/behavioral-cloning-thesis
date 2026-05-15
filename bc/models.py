"""Model architectures.

`nvidia_cnn` reproduces the thesis Table 1 exactly (layer order, filter counts,
parameter total) using the modern Keras API.

The two temporal models are revisited rather than reproduced verbatim. The
thesis trained its LSTM and 2D-CNN+LSTM on `np.random.random()` placeholders,
and its Table 3 design fed a length-1 "sequence" — i.e. it was not temporal at
all. Here both consume genuine 10-frame windows (see `bc.sequence`):

* `lstm_steering` keeps the thesis Table 2 design (flatten each frame, LSTM
  over the 10 timesteps).
* `cnn_lstm` is corrected into a proper temporal model: a per-frame CNN encoder
  (`TimeDistributed`) followed by an LSTM over the 10 encoded frames.
"""

from __future__ import annotations

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (
    LSTM,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Reshape,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam

INPUT_SHAPE = (66, 200, 3)
SEQ_LEN = 10  # frames per temporal window — matches bc.sequence.SEQ_LEN
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

    Input shape `(batch, SEQ_LEN, 66, 200, 3)`: each sample is `SEQ_LEN`
    consecutive frames. Every frame is flattened to a 39,600-vector and the LSTM
    runs over the timesteps. The flatten makes this layer huge (~20M params) —
    feeding raw pixels to an LSTM is not how you'd build this today, but it is
    the thesis design, kept here so the temporal experiment is a like-for-like
    correction of the thesis. Build the windows with `bc.sequence`.
    """
    inp = Input(shape=(SEQ_LEN, *INPUT_SHAPE))
    x = Reshape((SEQ_LEN, INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]))(inp)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(128)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(1)(x)
    model = Model(inp, out, name="lstm_steering")
    model.compile(loss="mse", optimizer=Adam(learning_rate=lr), metrics=["mae"])
    return model


def cnn_lstm(lr: float = DEFAULT_LR) -> Model:
    """Per-frame CNN encoder + LSTM over time — a corrected temporal CNN-LSTM.

    Input shape `(batch, SEQ_LEN, 66, 200, 3)`. A `TimeDistributed` strided-conv
    stack (NVIDIA-style) encodes each of the `SEQ_LEN` frames into a feature
    vector, giving a `(SEQ_LEN, features)` sequence; an LSTM then models the
    motion across the 10 frames. This is what a CNN-LSTM is meant to be — the
    thesis Table 3 design reshaped a single frame to a length-1 sequence and so
    learned nothing temporal. ~1.1M params.
    """
    inp = Input(shape=(SEQ_LEN, *INPUT_SHAPE))
    x = TimeDistributed(Conv2D(24, (5, 5), strides=(2, 2), activation="elu"))(inp)
    x = TimeDistributed(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))(x)
    x = TimeDistributed(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation="elu"))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(64)(x)
    x = Dense(100, activation="elu")(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation="elu")(x)
    out = Dense(1)(x)
    model = Model(inp, out, name="cnn_lstm")
    model.compile(loss="mse", optimizer=Adam(learning_rate=lr), metrics=["mae"])
    return model


REGISTRY = {
    "nvidia_cnn": nvidia_cnn,
    "lstm": lstm_steering,
    "cnn_lstm": cnn_lstm,
}
