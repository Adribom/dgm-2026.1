import torch
from pathlib import Path

# ecgpcx-ray/ root (3 levels up from models/CycleGAN/config.py)
_DATA_ROOT = Path(__file__).parent.parent.parent / "data" / "processed"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 128

BATCH_SIZE = 4

NUM_WORKERS = 4

EPOCHS = 200

LR = 2e-4

LAMBDA_CYCLE = 10.0

LAMBDA_IDENTITY = 5.0

CHECKPOINT_DIR = "checkpoints"

TRAIN_HEALTHY_DIR   = str(_DATA_ROOT / "train" / "healthy")
TRAIN_PNEUMONIA_DIR = str(_DATA_ROOT / "train" / "pneumonia")
VAL_HEALTHY_DIR     = str(_DATA_ROOT / "val"   / "healthy")
VAL_PNEUMONIA_DIR   = str(_DATA_ROOT / "val"   / "pneumonia")