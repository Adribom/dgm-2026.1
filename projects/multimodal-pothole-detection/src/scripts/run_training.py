"""Run Training Orchestrator.

This CLI script provides the executable entry point to initiate or resume
the Point-E training process relying on hyperparameters or a resolved JSON
configuration file.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch

# Ensure src modules can be resolved dynamically from the script execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.point_e_model import PotholePointE
from src.models.train_engine import PointETrainer
from src.data.pothole_dataset import create_dataloader


DEFAULT_CONFIG = {
    "model": {
        "batch_size": 8,
        "epochs": 50,
        "learning_rate": 1e-5,
        "save_interval": 5,
        "save_dir": "artifacts/checkpoints",
    },
    "seed": None,
    "augmentation": {
        "active_transforms": [],
        "probabilities": {
            "horizontal_flip": 0.5,
            "fake_shadow": 0.3,
            "color_jitter": 0.4,
            "gaussian_blur": 0.2,
            "motion_blur": 0.2,
            "cutout": 0.3,
        },
    },
}


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Merge override values into a nested default dictionary."""
    merged = dict(defaults)
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | None, cli_overrides: dict) -> dict:
    """Load the training configuration and merge CLI overrides.

    Parameters
    ----------
    config_path:
        Optional path to a JSON configuration file.
    cli_overrides:
        Dictionary with non-None CLI arguments that should win over file values.

    Returns
    -------
    dict
        Fully resolved configuration dictionary.
    """
    config = _deep_merge(DEFAULT_CONFIG, {})

    if config_path is not None:
        config_file = Path(config_path)
        with config_file.open("r", encoding="utf-8") as handle:
            config = _deep_merge(config, json.load(handle))

    config = _deep_merge(config, cli_overrides)
    config["model"] = _deep_merge(DEFAULT_CONFIG["model"], config.get("model", {}))
    config["augmentation"] = _deep_merge(DEFAULT_CONFIG["augmentation"], config.get("augmentation", {}))
    config.setdefault("data", {})
    return config


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main CLI entrypoint.

    Parses command-line arguments, prepares dataloaders and model
    artifacts paths, and launches the `PointETrainer` training loop.
    """
    # T018: Create argparse structure configuring parameters
    parser = argparse.ArgumentParser(description="Point-E Fine-Tuning CLI Orchestrator")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON training config file")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing RGB images (.jpg)")
    parser.add_argument("--cloud-dir", type=str, default=None, help="Directory containing target Point Clouds (.npy)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=None, help="Total number of epochs to train")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate for AdamW")
    parser.add_argument("--save-interval", type=int, default=None, help="Epoch interval to save checkpoints")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save checkpoints")
    
    # T019: Inject --resume-from parameter
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a .pt checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # Resolve Paths using pure Pathlib dynamically to root
    root_dir = Path(__file__).resolve().parent.parent.parent
    cli_overrides = {
        "model": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "save_interval": args.save_interval,
            "save_dir": args.save_dir,
        },
        "data": {
            "image_dir": args.image_dir,
            "cloud_dir": args.cloud_dir,
        },
    }
    config = load_config(args.config, cli_overrides)

    if config["data"].get("image_dir") is None or config["data"].get("cloud_dir") is None:
        parser.error("--image-dir and --cloud-dir are required unless provided in --config")

    if config["seed"] is not None:
        set_seed(config["seed"])

    image_dir_value = config["data"]["image_dir"]
    cloud_dir_value = config["data"]["cloud_dir"]
    save_dir_value = config["model"]["save_dir"]

    image_dir = root_dir / image_dir_value if image_dir_value and not Path(image_dir_value).is_absolute() else Path(image_dir_value)
    cloud_dir = root_dir / cloud_dir_value if cloud_dir_value and not Path(cloud_dir_value).is_absolute() else Path(cloud_dir_value)
    save_dir = root_dir / save_dir_value if save_dir_value and not Path(save_dir_value).is_absolute() else Path(save_dir_value)
    save_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = save_dir / f"run_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    resolved_config = {
        **config,
        "data": {
            "image_dir": str(image_dir),
            "cloud_dir": str(cloud_dir),
        },
        "model": {
            **config["model"],
            "save_dir": str(save_dir),
        },
    }
    with resolved_config_path.open("w", encoding="utf-8") as handle:
        json.dump(resolved_config, handle, indent=2)
    
    print(f"==================================================")
    print(f" Starting Point-E Fine-Tuning Orchestrator")
    print(f"==================================================")
    print(f"Images Target: {image_dir}")
    print(f"Clouds Target: {cloud_dir}")
    print(f"Batch Size: {config['model']['batch_size']} | Epochs: {config['model']['epochs']}")
    print(f"==================================================")

    augmentation_record_dir = root_dir / "artifacts" / "augmentation_records"
    augmentation_record_dir.mkdir(parents=True, exist_ok=True)
    augmentation_record_path = augmentation_record_dir / f"aug_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    # T020: Execute full PointETrainer launch integration
    dataloader = create_dataloader(
        image_dir=image_dir,
        cloud_dir=cloud_dir,
        batch_size=config["model"]["batch_size"],
        shuffle=True,
        num_workers=4,
        augmentation_config=config.get("augmentation"),
    )
    
    print("Loading Point-E architecture (Base40M)...")
    point_e_model = PotholePointE(base_model_name="base40M")
    
    trainer = PointETrainer(pothole_point_e=point_e_model, learning_rate=config["model"]["learning_rate"])
    
    # Handle Resumption if specified
    start_epoch = 0
    if args.resume_from:
        resume_path = root_dir / args.resume_from if not Path(args.resume_from).is_absolute() else Path(args.resume_from)
        start_epoch = trainer.load_checkpoint(resume_path)
        
    # Launch Loop
    with augmentation_record_path.open("a", encoding="utf-8") as augmentation_record_file:
        trainer.train_step(
            dataloader=dataloader,
            epochs=config["model"]["epochs"],
            start_epoch=start_epoch,
            save_dir=save_dir,
            save_interval=config["model"]["save_interval"],
            augmentation_record_file=augmentation_record_file,
        )
    
    print("Training session complete!")

if __name__ == "__main__":
    main()

