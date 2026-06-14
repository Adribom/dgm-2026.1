"""Run Training Orchestrator.

This CLI script provides the executable entry point to initiate or resume
the Point-E training process relying on hyperparameters or a resolved JSON
configuration file. Supports a configurable validation loop with best-checkpoint
selection via `checkpoint_best.pt`, augmentation-free val evaluation, and an
optional Chamfer Distance monitoring module.
"""

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import torch

# Ensure src modules can be resolved dynamically from the script execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.point_e_model import PotholePointE
from src.models.train_engine import PointETrainer
from src.data.pothole_dataset import (
    PotholeDataset,
    create_dataloader,
    point_e_collate_fn,
    IMAGE_EXTENSIONS,
)
from torch.utils.data import DataLoader


DEFAULT_CONFIG = {
    "model": {
        "batch_size": 8,
        "epochs": 50,
        "learning_rate": 1e-5,
        "save_interval": 5,
        "save_dir": "checkpoints",
    },
    "lr_scheduler": {
        "type": None,
        "eta_min": 1e-6,
    },
    "training": {
        "max_grad_norm": None,
    },
    "seed": None,
    "augmentation": {
        "active_transforms": [],
        "probabilities": {
            "pure_image": 0.1,
            "horizontal_flip": 0.5,
            "fake_shadow": 0.3,
            "color_jitter": 0.4,
            "gaussian_blur": 0.2,
            "motion_blur": 0.2,
            "cutout": 0.3,
        },
    },
    "validation": {
        "val_sample_ids": [],
        "val_interval": 1,
        "checkpoint_metric": "val_loss",
        "compute_chamfer_distance": False,
        "cd_sampling_steps": 64,
        "val_fixed_timesteps": None,
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
    config["lr_scheduler"] = _deep_merge(DEFAULT_CONFIG["lr_scheduler"], config.get("lr_scheduler", {}))
    config["training"] = _deep_merge(DEFAULT_CONFIG["training"], config.get("training", {}))
    config["augmentation"] = _deep_merge(DEFAULT_CONFIG["augmentation"], config.get("augmentation", {}))
    config["validation"] = _deep_merge(DEFAULT_CONFIG["validation"], config.get("validation", {}))
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


def _resolve_run_artifact_path(run_root: Path, path_value: str | None) -> Path:
    """Resolve a configured artifact path under the current run directory."""
    if path_value is None:
        return run_root

    configured_path = Path(path_value)
    if configured_path.is_absolute():
        return configured_path

    if configured_path.parts[:1] == ("artifacts",):
        configured_path = Path(*configured_path.parts[1:])

    return run_root / configured_path


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

    run_timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_artifacts_dir = root_dir / "artifacts" / f"run_{run_timestamp}"
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)

    image_dir = root_dir / image_dir_value if image_dir_value and not Path(image_dir_value).is_absolute() else Path(image_dir_value)
    cloud_dir = root_dir / cloud_dir_value if cloud_dir_value and not Path(cloud_dir_value).is_absolute() else Path(cloud_dir_value)
    save_dir = _resolve_run_artifact_path(run_artifacts_dir, save_dir_value)
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

    augmentation_record_dir = run_artifacts_dir / "augmentation_records"
    augmentation_record_dir.mkdir(parents=True, exist_ok=True)
    augmentation_record_path = augmentation_record_dir / f"aug_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    # Val records directory
    val_records_dir = run_artifacts_dir / "val_records"
    val_records_dir.mkdir(parents=True, exist_ok=True)
    val_log_path = val_records_dir / f"val_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    # Build train/val split from config
    val_ids = set(config["validation"].get("val_sample_ids", []))
    all_stems = {
        path.stem
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    } if image_dir.exists() else set()
    train_ids_filter = (all_stems - val_ids) if val_ids else None

    # T020: Execute full PointETrainer launch integration
    dataloader = create_dataloader(
        image_dir=image_dir,
        cloud_dir=cloud_dir,
        batch_size=config["model"]["batch_size"],
        shuffle=True,
        num_workers=4,
        augmentation_config=config.get("augmentation"),
        sample_ids_filter=train_ids_filter,
    )

    # Construct val dataloader (augmentation always disabled)
    val_dataloader = None
    if val_ids:
        val_dataset = PotholeDataset(
            image_dir,
            cloud_dir,
            augmentation_config=config.get("augmentation"),
            sample_ids_filter=val_ids,
            disable_augmentation=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["model"]["batch_size"],
            shuffle=False,
            num_workers=4,
            collate_fn=point_e_collate_fn,
        )
        print(f"Val set: {len(val_ids)} samples | Train set: {len(all_stems) - len(val_ids)} samples")
    
    print("Loading Point-E architecture (Base40M)...")
    point_e_model = PotholePointE(base_model_name="base40M")
    
    trainer = PointETrainer(pothole_point_e=point_e_model, learning_rate=config["model"]["learning_rate"])
    
    # Handle Resumption if specified
    start_epoch = 0
    if args.resume_from:
        resume_path = root_dir / args.resume_from if not Path(args.resume_from).is_absolute() else Path(args.resume_from)
        start_epoch = trainer.load_checkpoint(resume_path)
        
    # Launch Loop
    with augmentation_record_path.open("a", encoding="utf-8") as augmentation_record_file, \
         val_log_path.open("w", encoding="utf-8") as val_log_file:
        run_result = trainer.train_step(
            dataloader=dataloader,
            epochs=config["model"]["epochs"],
            start_epoch=start_epoch,
            save_dir=save_dir,
            save_interval=config["model"]["save_interval"],
            augmentation_record_file=augmentation_record_file,
            val_dataloader=val_dataloader,
            val_config=config["validation"],
            val_log_file=val_log_file,
            scheduler_config=config.get("lr_scheduler"),
            max_grad_norm=config.get("training", {}).get("max_grad_norm"),
        )

    metadata_path = run_artifacts_dir / "metadata.json"
    metadata = {
        "run_timestamp": run_timestamp,
        "artifacts_dir": str(run_artifacts_dir),
        "save_dir": str(save_dir),
        "base_model": "base40M",
        "resumed_from": str(resume_path) if args.resume_from else None,
        "start_epoch": start_epoch,
        "final_epoch_requested": config["model"]["epochs"],
        "total_epochs_trained": run_result["total_epochs_trained"],
        "best_epoch": run_result["best_epoch"],
        "best_val_metric": run_result["best_val_metric"],
        "elapsed_seconds": run_result["elapsed_seconds"],
        "elapsed_formatted": str(timedelta(seconds=int(run_result["elapsed_seconds"]))),
        "final_lr": run_result["final_lr"],
        "learning_rate": config["model"]["learning_rate"],
        "lr_scheduler": config.get("lr_scheduler"),
        "batch_size": config["model"]["batch_size"],
        "epochs": config["model"]["epochs"],
        "train_set_size": len(all_stems) - len(val_ids),
        "val_set_size": len(val_ids),
        "val_interval": config["validation"].get("val_interval"),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    
    print("Training session complete!")

if __name__ == "__main__":
    main()

