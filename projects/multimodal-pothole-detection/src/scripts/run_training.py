"""
Run Training Orchestrator.

This CLI script provides the executable entry point to initiate or resume
the Point-E training process relying on hyperparameters.
"""

import argparse
from pathlib import Path
import sys

# Ensure src modules can be resolved dynamically from the script execution
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.point_e_model import PotholePointE
from src.models.train_engine import PointETrainer
from src.data.pothole_dataset import create_dataloader

def main():
    """Main CLI entrypoint.

    Parses command-line arguments, prepares dataloaders and model
    artifacts paths, and launches the `PointETrainer` training loop.
    """
    # T018: Create argparse structure configuring parameters
    parser = argparse.ArgumentParser(description="Point-E Fine-Tuning CLI Orchestrator")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing RGB images (.jpg)")
    parser.add_argument("--cloud-dir", type=str, required=True, help="Directory containing target Point Clouds (.npy)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs to train")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for AdamW")
    parser.add_argument("--save-interval", type=int, default=5, help="Epoch interval to save checkpoints")
    parser.add_argument("--save-dir", type=str, default="artifacts/checkpoints", help="Directory to save checkpoints")
    
    # T019: Inject --resume-from parameter
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a .pt checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # Resolve Paths using pure Pathlib dynamically to root
    root_dir = Path(__file__).resolve().parent.parent.parent
    image_dir = root_dir / args.image_dir if not Path(args.image_dir).is_absolute() else Path(args.image_dir)
    cloud_dir = root_dir / args.cloud_dir if not Path(args.cloud_dir).is_absolute() else Path(args.cloud_dir)
    save_dir = root_dir / args.save_dir if not Path(args.save_dir).is_absolute() else Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"==================================================")
    print(f" Starting Point-E Fine-Tuning Orchestrator")
    print(f"==================================================")
    print(f"Images Target: {image_dir}")
    print(f"Clouds Target: {cloud_dir}")
    print(f"Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print(f"==================================================")
    
    # T020: Execute full PointETrainer launch integration
    dataloader = create_dataloader(
        image_dir=image_dir,
        cloud_dir=cloud_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    print("Loading Point-E architecture (Base40M)...")
    point_e_model = PotholePointE(base_model_name="base40M")
    
    trainer = PointETrainer(pothole_point_e=point_e_model, learning_rate=args.learning_rate)
    
    # Handle Resumption if specified
    start_epoch = 0
    if args.resume_from:
        resume_path = root_dir / args.resume_from if not Path(args.resume_from).is_absolute() else Path(args.resume_from)
        start_epoch = trainer.load_checkpoint(resume_path)
        
    # Launch Loop
    trainer.train_step(
        dataloader=dataloader,
        epochs=args.epochs,
        start_epoch=start_epoch,
        save_dir=save_dir,
        save_interval=args.save_interval
    )
    
    print("Training session complete!")

if __name__ == "__main__":
    main()

