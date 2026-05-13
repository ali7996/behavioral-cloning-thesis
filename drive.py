"""Drive the Udacity self-driving simulator in autonomous mode.

Run:    python drive.py models/nvidia_cnn.keras
Then:   open the Udacity self-driving-car simulator and click Autonomous Mode.

The simulator streams telemetry (camera image + speed) over Socket.IO; we predict the
steering angle from the preprocessed center camera frame and stream it back, along with
a throttle command that slows the car down as it approaches a target speed.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO

import eventlet
import numpy as np
import socketio
from flask import Flask
from PIL import Image
from tensorflow.keras.models import load_model

from bc.augment import preprocess

TARGET_SPEED = 10
sio = socketio.Server()
app = Flask(__name__)
model = None  # populated in main


@sio.on("telemetry")
def telemetry(sid, data):
    if not data:
        sio.emit("manual", data={}, skip_sid=True)
        return
    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    frame = np.asarray(image)
    frame = preprocess(frame)
    steering = float(model.predict(np.expand_dims(frame, 0), verbose=0)[0, 0])
    throttle = 1.0 - (speed / TARGET_SPEED)
    print(f"steering={steering:+.3f}  throttle={throttle:+.3f}  speed={speed:.2f}")
    sio.emit("steer", data={"steering_angle": str(steering), "throttle": str(throttle)})


@sio.on("connect")
def connect(sid, environ):
    print(f"connected: {sid}")
    sio.emit("steer", data={"steering_angle": "0", "throttle": "0"})


def main() -> None:
    global model
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to .keras or .h5 weights")
    parser.add_argument("--port", type=int, default=4567)
    args = parser.parse_args()

    print(f"Loading {args.model_path} …")
    model = load_model(args.model_path)
    wrapped = socketio.WSGIApp(sio, app)
    print(f"Listening on :{args.port} for the Udacity simulator …")
    eventlet.wsgi.server(eventlet.listen(("", args.port)), wrapped)


if __name__ == "__main__":
    main()
