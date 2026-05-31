"""
Point-E Train Engine Module.

This script contains the OOP architecture to orchestrate the Point-E fine-tuning
loop utilizing PyTorch AMP, Checkpointing, and native Diffusion logic.
"""

import logging
from pathlib import Path
import torch
from tqdm import tqdm


class PointETrainer:
    def __init__(self, pothole_point_e, learning_rate: float = 1e-5):
        """
        T010: OOP class init injecting PotholePointE models.
        """
        self.device = pothole_point_e.device
        self.base_model = pothole_point_e.base_model
        self.diffusion = pothole_point_e.base_diffusion
        
        # T013: Setup standard Python logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing PointETrainer on device: {self.device}")
        
        # T011: Optimizer config filtering frozen layers using requires_grad
        self.logger.info("Configuring AdamW optimizer with filtered (unfrozen) parameters.")
        trainable_params = filter(lambda p: p.requires_grad, self.base_model.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        
        # T012: GradScaler / Automatic Mixed Precision wrapper setup
        self.logger.info("Initializing AMP GradScaler for Memory Optimization.")
        # Determine the device type string for AMP ('cuda' or 'cpu')
        device_type = self.device.type if isinstance(self.device, torch.device) else (self.device if isinstance(self.device, str) else 'cuda')
        self.scaler = torch.amp.GradScaler(device_type)

    def train_step(self, dataloader, epochs: int = 1, start_epoch: int = 0, save_dir=None, save_interval: int = 1):
        """
        Executes the core training loop for a given number of epochs.
        """
        self.base_model.train()
        
        for epoch in range(start_epoch, epochs):
            self.logger.info(f"Starting Epoch {epoch + 1}/{epochs}")
            
            # T014: Wrap the loop loader using the tqdm library for real-time postfix observation
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            
            for step, batch in enumerate(pbar):
                images = batch["images"]
                point_clouds = batch["point_cloud_6d"].to(self.device)
                
                batch_size = point_clouds.shape[0]
                
                # Random timesteps for the diffusion process
                t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)
                
                self.optimizer.zero_grad()
                
                device_type = self.device.type if isinstance(self.device, torch.device) else (self.device if isinstance(self.device, str) else 'cuda')
                # T015: Execute diffusion.training_losses dynamically applying model_kwargs
                with torch.amp.autocast(device_type):
                    model_kwargs = {"images": images}
                    loss_dict = self.diffusion.training_losses(
                        model=self.base_model,
                        x_start=point_clouds,
                        t=t,
                        model_kwargs=model_kwargs
                    )
                    loss = loss_dict["loss"].mean()
                    
                # Backpropagation governed by GradScaler (AMP)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update observability metrics
                loss_val = loss.item()
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})
                
                # Log periodically to standard output independently from tqdm
                if step % 50 == 0:
                    self.logger.info(f"Epoch {epoch+1} | Step {step} | Loss: {loss_val:.4f}")

            # Handle checkpoint saving at the end of epoch
            if save_dir and (epoch + 1) % save_interval == 0:
                save_path = Path(save_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(epoch + 1, save_path)

    def save_checkpoint(self, epoch: int, filepath):
        """
        T016: Extract and save the model, optimizer, scaler, and epoch params to a dict.

        Ensures the parent directory exists before attempting to write the
        checkpoint file to avoid a FileNotFoundError when running from a
        different working directory.
        """
        filepath = Path(filepath)
        # Ensure the parent directory exists (handles relative/absolute paths)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving checkpoint to {filepath}...")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.base_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(checkpoint, filepath)
        self.logger.info("Checkpoint saved successfully.")

    def load_checkpoint(self, filepath) -> int:
        """
        T017: Update model attributes safely from the loaded dict path.
        Returns the epoch from which to resume.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")
            
        self.logger.info(f"Loading checkpoint from {filepath}...")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.base_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        start_epoch = checkpoint["epoch"]
        self.logger.info(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}.")
        return start_epoch


