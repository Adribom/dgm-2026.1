# FakeImageDetection Classifier for FakeClue Dataset

Classifies FakeClue dataset images as fake or real using pre-trained models
from [Doloriel & Cheung, "Frequency Masking for Universal DeepFake Detection"
(ICASSP 2024)](https://arxiv.org/abs/2401.06506).

## Requirements

- Python 3.12
- Dependencies: see `requirements.txt`

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic (test split with RN50 model)

```bash
python classify.py \
    --checkpoint ../../../../models/FakeImageDetection/mask_0/rn50ft.pth \
    --model-name rn50 \
    --split test
```

### With frequency-masked model

```bash
python classify.py \
    --checkpoint ../../../../models/FakeImageDetection/mask_15/rn50ft_highspectralmask.pth \
    --model-name rn50 \
    --split test
```

### With CLIP model

```bash
python classify.py \
    --checkpoint ../../../../models/FakeImageDetection/mask_15/clipft_spectralmask.pth \
    --model-name clip \
    --split test
```

### Quick test run

```bash
python classify.py \
    --checkpoint ../../../../models/FakeImageDetection/mask_0/rn50ft.pth \
    --model-name rn50 \
    --split test \
    --limit 100
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | *required* | Path to `.pth` checkpoint file |
| `--model-name` | `rn50` | Model architecture: `rn50`, `rn50_mod`, or `clip` |
| `--split` | `test` | FakeClue split: `train`, `test`, or `both` |
| `--batch-size` | `64` | Images per batch |
| `--num-workers` | `4` | DataLoader worker processes |
| `--data-dir` | `../../../../data/external/FakeClue` | FakeClue dataset root |
| `--output-dir` | `DATA_DIR/data_json/FakeImageDetection` | Output directory for JSON results |
| `--limit` | None | Process only N images (for debugging) |

## Output Format

Results are written to `{split}_frequency_{checkpoint_stem}.json`:

```json
[
  {
    "image": "ff++/fake/Deepfakes/c23/frames/856_881/1066.png",
    "ground_truth": 0,
    "predicted_label": 0,
    "confidence": 0.9873,
    "model": "rn50ft",
    "category": "deepfake",
    "error": null
  }
]
```

- `ground_truth`: label from the FakeClue dataset (0 = fake, 1 = real)
- `predicted_label`: model prediction mapped to FakeClue convention (0 = fake, 1 = real)
- `confidence`: model confidence score in [0, 1]
- `error`: error message if the image could not be processed, otherwise null

A summary with accuracy, confusion matrix, and per-category breakdown is
printed to stdout after processing.

## Label Conventions

The two projects use opposite conventions:

- **FakeClue**: 0 = fake, 1 = real
- **FakeImageDetection**: 0 = real, 1 = fake

This script maps model predictions to the FakeClue convention automatically.

## Models

Pre-trained models from the [FakeImageDetection Google Drive](https://drive.google.com/drive/folders/1ePTY4x2qvD7AVlNJXFLozFbUF6Y0_hET).

Available checkpoints:

**No masking (mask_0/):**
- `rn50ft.pth` — ResNet-50 (`--model-name rn50`)
- `rn50_modft.pth` — Modified ResNet-50 (`--model-name rn50_mod`)

**With masking (mask_15/):**
- `rn50ft_spectralmask.pth` — ResNet-50, all-band spectral mask (`--model-name rn50`)
- `rn50ft_spectralmask(0.5).pth` — ResNet-50, spectral mask ratio 0.5 (`--model-name rn50`)
- `rn50ft_lowspectralmask.pth` — ResNet-50, low-frequency spectral mask (`--model-name rn50`)
- `rn50ft_midspectralmask.pth` — ResNet-50, mid-frequency spectral mask (`--model-name rn50`)
- `rn50ft_highspectralmask.pth` — ResNet-50, high-frequency spectral mask (`--model-name rn50`)
- `rn50ft_pixelmask.pth` — ResNet-50, pixel mask (`--model-name rn50`)
- `rn50ft_patchmask.pth` — ResNet-50, patch mask (`--model-name rn50`)
- `rn50_modft_spectralmask.pth` — Modified ResNet-50, spectral mask (`--model-name rn50_mod`)
- `clipft.pth` — CLIP ViT-L/14, no mask (`--model-name clip`)
- `clipft_spectralmask.pth` — CLIP ViT-L/14, spectral mask (`--model-name clip`)

## Notes

- Images are center-cropped to 224x224. ResNet models use ImageNet normalization;
  CLIP models use CLIP-specific normalization. Both match the original testing pipeline.
- Images smaller than 224x224 will fail to crop. These are logged and skipped.
- Runs on GPU if available, otherwise falls back to CPU.

## References

- Doloriel and Cheung, "Frequency Masking for Universal DeepFake Detection,"
  ICASSP 2024. https://arxiv.org/abs/2401.06506
- Doloriel et al., "Towards Sustainable Universal Deepfake Detection with
  Frequency-Domain Masking," ACM TOMM 2026. https://arxiv.org/abs/2512.08042
- Wen et al., "Spot the Fake: Large Multimodal Model-Based Synthetic Image
  Detection with Artifact Explanation," 2025. https://arxiv.org/abs/2503.14905
