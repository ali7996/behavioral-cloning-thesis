# Behavioral Cloning for Self-Driving Cars

End-to-end neural network that maps a forward-facing camera frame to a steering angle, trained on the Udacity simulator dataset. This repository implements the **NVIDIA self-driving architecture** described in the original 2016 paper *End to End Learning for Self-Driving Cars* (Bojarski et al.) and corresponds to the practical part of my Master's thesis *"Artificial Neural Networks Implementation in Self-Driving Car Behavioral Cloning"* (2023). It also revisits the thesis's two temporal (LSTM) models — see [Temporal models](#temporal-models--a-leak-free-comparison).

```
Camera frame ──► crop ──► YUV ──► blur ──► resize 200×66 ──► /255 ──► CNN ──► steering angle
```

> 📄 **Technical report.** The methodology — the corrected temporal models, the leak-free contiguous-split evaluation, and the negative result on recurrent architectures — is written up as a short paper in [`paper/`](paper/) (LaTeX; compile on Overleaf, see [`paper/README.md`](paper/README.md)).

## Results

### NVIDIA CNN vs. the thesis

Same protocol as the thesis — random 80/20 frame split, balanced steering bins, 10 training epochs on a single CPU (≈5 min on an M-series Mac, no GPU needed).

| | MAE | RMSE | R² |
|---|---|---|---|
| **This repo** (retrained on real data) | **0.168** | **0.233** | **0.52** |
| Thesis (2023) | 0.300 | 0.408 | — |

Retrained on the real Udacity track data with the original augmentation and preprocessing pipeline intact, the CNN roughly halves the thesis's steering error. R² of 0.52 means the model captures about half the variation in human steering from a single forward image — enough for stable lane following on the training circuit, not for novel terrain.

> **Note on the thesis vs. this repo.** The thesis tables report three model variants (CNN, LSTM, 2D CNN+LSTM). On revisiting the original code I found the two LSTM variants were inadvertently trained on `np.random.random()` placeholders rather than the driving data — so their reported numbers measured nothing. The 2D-CNN+LSTM was also not actually temporal: it reshaped a single frame into a length-1 "sequence". This repository retrains the CNN honestly, and re-runs the temporal experiment properly. See below.

### Temporal models — a leak-free comparison

The thesis's LSTM models were meant to use *temporal context* — several consecutive frames — rather than a single image. Here that experiment is done properly: real 10-frame windows of consecutive driving, built from the timestamps encoded in each frame's filename (`bc/sequence.py`).

Two methodology points matter, and both differ from the CNN pipeline:

- **Window first, balance second.** Steering balancing drops individual frames; doing it before windowing would punch holes in the timeline. We window the contiguous run first, then cap windows per steering bin.
- **Contiguous, leak-free split.** The dataset is one uninterrupted run at ~14 Hz — consecutive frames are ~0.07 s apart and nearly identical. A *random* frame split therefore lets a model see near-copies of its validation frames during training. The temporal models require a timeline-ordered split (first 80 % of the run → train, last 20 % → val, no window crossing the boundary). For a fair comparison the CNN is **re-scored on that same split**.

All three models below are evaluated on the **identical contiguous validation set** (2,406 windows). The temporal models train on ~2,300 balanced 10-frame windows.

| Model | Params | MAE | RMSE | R² |
|---|---|---|---|---|
| `nvidia_cnn` (single frame) | 252 k | **0.106** | **0.138** | **+0.31** |
| `lstm_steering` (10-frame, flatten→LSTM) | 20.5 M | 0.128 | 0.169 | −0.03 |
| `cnn_lstm` (10-frame, TimeDistributed CNN→LSTM) | 1.1 M | 0.138 | 0.170 | −0.05 |

**Finding — a negative result, reported as-is.** Neither temporal model beats the single-frame CNN; in fact neither beats simply predicting the mean steering angle (R² ≤ 0). Their training loss never dropped below ≈0.18 MSE either — they *underfit*. On this dataset the steering angle is essentially determined by the current frame's appearance, and ~2,300 windows is far too little for a recurrent network to extract useful temporal structure. This is also why the thesis's original LSTM numbers — reported as *better* than the CNN — cannot have been real measurements: trained on noise, they measured nothing.

> The CNN's R² on this contiguous split (0.31) is lower than its random-split R² (0.52) above. That gap is the leakage: the random split's near-duplicate adjacent frames inflate the score. The contiguous 0.31 is the honest number; the two CNN rows use different validation sets, so their MAEs are not directly comparable — compare models *within* a split only.

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

The two temporal models (`bc/models.py`) take a `(10, 66, 200, 3)` window. `lstm_steering` is the thesis Table 2 design (flatten each frame, LSTM over the 10 timesteps). `cnn_lstm` is a *corrected* CNN-LSTM: a `TimeDistributed` strided-conv encoder runs on each frame, then an LSTM models the 10-frame dynamics — unlike the thesis Table 3 design, it is genuinely temporal.

## Data pipeline

Dataset: [`rslim087a/track`](https://github.com/rslim087a/track) — 4,053 frames captured in the Udacity Self-Driving Car Simulator (track 1, three laps in each direction). Each frame has center / left / right camera images plus telemetry.

1. **Load** `driving_log.csv`, strip Windows-style paths to bare filenames.
2. **Balance** steering distribution — cap each of 25 bins at 400 samples to suppress zero-steering bias (4,053 → 1,463 rows).
3. **Triple via side cameras** — left/right frames get a ±0.15 steering correction (4,389 (image, steering) pairs).
4. **Train/val split** — 80/20 → 3,511 train, 878 val.
5. **Augmentation** (training only, each applied independently with p=0.5):
   zoom (1.0–1.3×), pan (±10%), brightness (0.2–1.2×), horizontal flip (with sign-flipped steering).
6. **Preprocess** (always): crop `[60:135]` (drop sky + hood) → RGB→YUV → 3×3 Gaussian blur → resize to 200×66 → divide by 255.

The temporal pipeline (`bc/sequence.py`) reuses the same preprocessing and augmentation, but builds 10-frame windows over the timestamp-sorted run, splits the timeline contiguously, and locks each augmentation across all 10 frames of a window (`augment_window`).

## Quickstart

```bash
git clone https://github.com/ali7996/behavioral-cloning-thesis.git
cd behavioral-cloning-thesis
git clone --depth 1 https://github.com/rslim087a/track.git
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train + evaluate the NVIDIA CNN (≈5 min on CPU)
python -m scripts.train_cnn --epochs 10
python -m eval.evaluate models/nvidia_cnn.keras

# Train the temporal models on real 10-frame windows
python -m scripts.train_lstm --model cnn_lstm
python -m scripts.train_lstm --model lstm

# Evaluate every model on the same leak-free contiguous split
python -m eval.evaluate models/cnn_lstm.keras
python -m eval.evaluate models/lstm_steering.keras
python -m eval.evaluate models/nvidia_cnn.keras --contiguous
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
│   ├── augment.py           # Zoom/pan/brightness/flip + YUV preprocess + batch generators
│   ├── sequence.py          # Temporal 10-frame windows + contiguous leak-free split
│   └── models.py            # nvidia_cnn, lstm_steering, cnn_lstm
├── scripts/
│   ├── train_cnn.py         # Train the NVIDIA CNN + checkpoint best weights
│   └── train_lstm.py        # Train lstm_steering / cnn_lstm on real sequences
├── eval/
│   └── evaluate.py          # MSE / MAE / RMSE / R² — random or contiguous split
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
