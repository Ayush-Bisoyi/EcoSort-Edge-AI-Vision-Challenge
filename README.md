# EcoSort Edge-AI Challenge: Submission

This repository contains my submission for the EcoSort recyclable-waste classification challenge: a compact CNN that classifies 96×96 RGB images into six classes (`cardboard, glass, metal, paper, plastic, trash`) and is exported as a single ONNX file within the edge-hardware limits.

## Repository contents

| File | Purpose |
|---|---|
| `model.onnx` | **My submission model** (the file uploaded to the portal). |
| `train_ecosort.py` | Training and ONNX export code that produces `model.onnx`. |
| `verify_submission.py` | The starter kit's pre-submission checker, included unmodified. |
| `predict.py` | Optional demo that classifies a single image with `model.onnx`. |
| `requirements.txt` | Dependencies for training, export and verification. |
| `requirements_inference.txt` | Minimal dependencies for `predict.py` only. |

The dataset is not included in this repository.

## Constraint checklist

I checked the final `model.onnx` with the starter kit's `verify_submission.py` against the public validation set:

| Requirement | Limit | My result |
|---|---:|---:|
| Valid ONNX file | `.onnx` | pass (opset 18, single self-contained file) |
| Parameters (ONNX initializers) | ≤ 350,000 | **31,782** |
| File size | ≤ 1.50 MB | **140,357 bytes (0.13 MB)** |
| Input signature | `(batch, 3, 96, 96)` float32 | pass |
| Output signature | `(batch, 6)` float32 | pass (raw logits, no softmax) |
| CPU latency (kit target ≤ 35 ms) | not a hard limit | about 0.2 ms/image on my machine |
| Accuracy | ≥ 75% on the hidden test set | **100.0% (240/240) on the public validation set** |

The hidden-test accuracy is determined by the evaluators, so I cannot measure it myself. Two caveats about my 100% validation score:

- The exported model is the **final training epoch**. I did not pick a checkpoint using the validation set, so the score is not inflated by epoch selection.
- The public images are synthetic and clean (flat icons on plain backgrounds), so a near-perfect public score is expected and does not by itself show how the model handles harder images. To address this, I tested robustness (below).

## Robustness check

Because the public images are clean, I added noise, blur, wide brightness/contrast changes and occasional grayscale to the training augmentation. I then evaluated `model.onnx` on perturbed copies of the 240 public validation images (perturbations were applied only at test time):

| Perturbation | Accuracy |
|---|---:|
| None | 100.0% |
| Gaussian noise, sd = 15/255 | 100.0% |
| Gaussian noise, sd = 30/255 | 99.6% |
| Blur (radius 1.5) | 100.0% |
| JPEG quality 20 | 100.0% |
| Brightness ×0.6 | 100.0% |
| Brightness ×1.4 | 92.1% |
| Contrast ×0.6 | 100.0% |
| Rotation 25° | 100.0% |
| Grayscale | 99.6% |
| Downscale to 64 px and back | 100.0% |

Brightened images (×1.4) are the weakest case I found.

## Model

The network is a MobileNet-style stack of depthwise-separable blocks:

- Stem: 3×3 convolution with stride 2 (96 → 48).
- Six depthwise-separable blocks (3×3 depthwise followed by 1×1 pointwise) with channels 32 → 48 → 48 → 64 → 64 → 96 → 96. Stride-2 blocks reduce the resolution to 6×6.
- Identity skip connections wherever a block's input and output shapes match.
- BatchNorm and ReLU6 throughout, global average pooling, dropout (0.3), and a linear layer that outputs 6 logits.

I chose depthwise-separable convolutions to stay far below the parameter limit (31,782 of 350,000) while keeping inference very cheap.

**Preprocessing expected by the model** (the same as the evaluator): convert to RGB, resize to 96×96, scale to 0–1, normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`, NCHW layout. Normalization is done outside the network, so the ONNX model takes already-normalized input.

## Training recipe

- **Augmentation:** horizontal/vertical flips, random affine (rotation, translation, scale), per-sample brightness and contrast jitter, Gaussian noise, blur, and occasional grayscale.
- **Regularization:** MixUp (alpha 0.2), label smoothing (0.1), dropout (0.3), weight decay (0.05).
- **Optimization:** AdamW with a OneCycleLR schedule, 40 epochs, batch size 48, seed 11.
- **Export:** opset 18, dynamic batch axis, input name `input`, output name `output`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the dataset so the layout is:

```
data/public/
├── train/{cardboard,glass,metal,paper,plastic,trash}/*.png
└── val/{cardboard,glass,metal,paper,plastic,trash}/*.png
```

## Reproduce training and verification

```bash
python train_ecosort.py --data data/public --epochs 40 --out model.onnx
python verify_submission.py --model model.onnx --val_dir data/public/val
```

With a fixed seed (`--seed`, default 11), two runs of this command on CPU on my machine produced byte-identical `model.onnx` files. On different hardware (CUDA or Apple MPS) or different library versions the weights may differ slightly, so the behavior should match but the file bytes may not. The script uses CUDA if available, then Apple MPS, then CPU. A full run takes a few minutes on CPU.

The script also has a `--full-data` option that trains on train+val. I did not use it for the submitted model, and its validation score would be meaningless because the model would have seen those images.

## Run a single prediction

```bash
pip install -r requirements_inference.txt
python predict.py --image path/to/image.png                      # uses ./model.onnx by default
python predict.py --image path/to/image.png --model path/to/model.onnx
```

The script prints the predicted class and the class probabilities.
