import matplotlib
matplotlib.use('Agg')  # non-interactive backend — avoids Tkinter/threading conflict on Windows

import sys
from pathlib import Path

# Allow imports from the parent CycleGAN directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import mask.config as config
from model_utils.cyclegan import CycleGAN
from training.dataloaders import get_train_dataloader, get_val_dataloader
from training.trainer import CycleGANTrainer
from training.utils import save_checkpoint
from training.validation import validate
from utils.metrics import LossTracker
from utils.plotting import plot_losses
from utils.visualization import save_training_progress


def main():

    model = CycleGAN(image_channels=1, use_mask=True)

    trainer = CycleGANTrainer(
        model=model,
        device=config.DEVICE,
        lambda_cycle=config.LAMBDA_CYCLE,
        lambda_identity=config.LAMBDA_IDENTITY,
        use_mask=True,
        lambda_mask_inside=config.LAMBDA_MASK_INSIDE,
        lambda_mask_outside=config.LAMBDA_MASK_OUTSIDE,
    )

    train_loader = get_train_dataloader(
        healthy_dir=config.TRAIN_HEALTHY_DIR,
        pneumonia_dir=config.TRAIN_PNEUMONIA_DIR,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS,
        healthy_masks_dir=config.TRAIN_HEALTHY_MASKS_DIR,
        pneumonia_masks_dir=config.TRAIN_PNEUMONIA_MASKS_DIR,
    )

    val_loader = get_val_dataloader(
        healthy_dir=config.VAL_HEALTHY_DIR,
        pneumonia_dir=config.VAL_PNEUMONIA_DIR,
        batch_size=config.BATCH_SIZE,
        image_size=config.IMAGE_SIZE,
        num_workers=config.NUM_WORKERS,
        healthy_masks_dir=config.VAL_HEALTHY_MASKS_DIR,
        pneumonia_masks_dir=config.VAL_PNEUMONIA_MASKS_DIR,
    )

    tracker = LossTracker()

    for epoch in range(config.EPOCHS):

        print(f"\nEpoch {epoch + 1}")

        train_output = trainer.train_epoch(train_loader)
        train_losses = train_output["losses"]
        generated_images = train_output["images"]

        val_losses = validate(trainer, val_loader)

        print(f"Train G: {train_losses['G']:.4f}")
        print(f"Val Cycle: {val_losses['cycle']:.4f}")

        tracker.update(
            G_loss=train_losses["G"],
            D_H_loss=train_losses["D_H"],
            D_P_loss=train_losses["D_P"],
            cycle_loss=train_losses["cycle"],
            identity_loss=train_losses["identity"],
            gan_loss=train_losses["gan"],
            val_cycle_loss=val_losses["cycle"],
        )

        plot_losses(
            tracker.get_history(),
            save_dir="mask/outputs/losses",
            filename="loss_curve.png",
        )

        save_training_progress(
            images=generated_images,
            epoch=epoch + 1,
            save_dir="mask/outputs/progress",
            max_images=4,
        )

        save_checkpoint(trainer, epoch, config.CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
