# Behavioral Cloning for Self-Driving Cars

End-to-end neural network that maps a forward-facing camera frame to a steering angle, trained on the Udacity simulator dataset. This repository implements the **NVIDIA self-driving architecture** described in the original 2016 paper *End to End Learning for Self-Driving Cars* (Bojarski et al.) and corresponds to the practical part of my Master's thesis *"Artificial Neural Networks Implementation in Self-Driving Car Behavioral Cloning"* (2023).

```
Camera frame ──► crop ──► YUV ──► blur ──► resize 200×66 ──► /255 ──► CNN ──► steering angle
```

## Results

Evaluated on 800 held-out validation frames after 10 training epochs on a single CPU (≈5 min on an M-series Mac, no GPU needed).

| Metric | Value | Notes |
|---|---|---|
| MAE | **0.168** | Mean absolute steering error |
| RMSE | **0.233** | Root mean squared error |
| MSE | **0.054** | Mean squared error |
| R² | **0.52** | Fraction of steering variance explained |

Honest framing: R² of 0.52 means the model captures roughly half the variation in human steering behaviour from a single forward image — strong enough for stable lane following on the training circuit, but not for novel terrain. See the thesis (Chapter 6) for a fuller discussion.

> **Note on the thesis vs. this repo.** The thesis tables report three model variants (CNN, LSTM, 2D CNN+LSTM). On revisiting the original code I discovered the LSTM variants were inadvertently trained on `np.random.random()` placeholders rather than the driving data, so their numbers were not measurements of real learning. This repository ships only the NVIDIA CNN model, retrained on real data with the original augmentation and preprocessing pipeline intact. A follow-up commit will add the LSTM variants with proper temporal sequence construction (10-frame consecutive windows from contiguous timestamps), which is the experiment the LSTM architecture actually requires.

## Architecture

The CNN follows Bojarski et al. (2016) exactly. Trainable parameters: **252,219**.

| Layer | Output | Params |
|---|---|---|
| Conv2D 24 · 5×5 · stride 2 · ELU | 31 × 98 × 24 | 1,824 |
| Conv2D 36 · 5×5 · stride 2 · ELU | 14 × 47 × 36 | 21,636 |
| Conv2D 48 · 5×5 · stride 2 · ELU | 5 × 22 × 48 | 43,248 |
| Conv2D 64 · 3×3 · ELU | 3 × 20 × 64 | 27,712 |
| Conv2D 64 · 3×3 · ELU | 1 × 18 × 64 | 36,928 |
| Dropout 0.5 | — | 0 |
| Flatten | 1152 | 0 |
| Dense 100 · ELU + Dropout 0.5 | 100 | 115,300 |
| Dense 50 · ELU + Dropout 0.5 | 50 | 5,050 |
| Dense 10 · ELU + Dropout 0.5 | 10 | 510 |
| Dense 1 (steering) | 1 | 11 |

Loss: MSE. Optimizer: Adam (lr=1e-3).

## Data pipeline

Dataset: [`rslim087a/track`](https://github.com/rslim087a/track) — 4,053 frames captured in the Udacity Self-Driving Car Simulator (track 1, three laps in each direction). Each frame has center / left / right camera images plus telemetry.

1. **Load** `driving_log.csv`, strip Windows-style paths to bare filenames.
2. **Balance** steering distribution — cap each of 25 bins at 400 samples to suppress zero-steering bias (4,053 → 1,463 rows).
3. **Triple via side cameras** — left/right frames get a ±0.15 steering correction (4,389 (image, steering) pairs).
4. **Train/val split** — 80/20 → 3,511 train, 878 val.
5. **Augmentation** (training only, each applied independently with p=0.5):
   zoom (1.0–1.3×), pan (±10%), brightness (0.2–1.2×), horizontal flip (with sign-flipped steering).
6. **Preprocess** (always): crop `[60:135]` (drop sky + hood) → RGB→YUV → 3×3 Gaussian blur → resize to 200×66 → divide by 255.

## Quickstart

```bash
git clone https://github.com/ali7996/behavioral-cloning-thesis.git
cd behavioral-cloning-thesis
git clone --depth 1 https://github.com/rslim087a/track.git
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train (5 min on CPU)
python -m scripts.train_cnn --epochs 10

# Evaluate
python -m eval.evaluate models/nvidia_cnn.keras
```

## Driving in the Udacity simulator

```bash
python drive.py models/nvidia_cnn.keras
```

Open the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim), select **Autonomous Mode**, and the car will start driving. The script streams steering commands over Socket.IO on port 4567 and uses a basic proportional throttle controller targeting 10 mph.

## Repository structure

```
behavioral-cloning-thesis/
├── bc/                      # Library code
│   ├── data.py              # CSV loading + steering balancing + train/val split
│   ├── augment.py           # Zoom/pan/brightness/flip + YUV preprocess + batch generator
│   └── models.py            # nvidia_cnn, lstm_steering, cnn_lstm (all 3 thesis architectures)
├── scripts/
│   └── train_cnn.py         # Train + checkpoint best weights
├── eval/
│   └── evaluate.py          # MSE / MAE / RMSE / R² on held-out validation set
├── notebooks/
│   ├── thesis_original.ipynb  # Original Colab notebook (preserved for reference)
│   └── thesis.pdf           # The thesis itself
├── models/                  # Saved weights + training history (gitignored)
├── drive.py                 # Autonomous-mode server for the Udacity simulator
└── track/                   # Dataset (gitignored — clone separately)
```

## References

- Bojarski, M. et al. (2016). *End to End Learning for Self-Driving Cars*. arXiv:1604.07316.
- Codevilla, F. et al. (2018). *End-to-end Driving via Conditional Imitation Learning*. ICRA.
- Codevilla, F. et al. (2019). *Exploring the Limitations of Behavior Cloning for Autonomous Driving*. ICCV.
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim

## Author

**Aly Elgemei** — AI & Automation Developer · MSc Data Science & AI · PhD Applicant
[GitHub: @ali7996](https://github.com/ali7996) · Siegen, Germany
