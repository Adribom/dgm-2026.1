"""Point-E Train Engine Module.

This script contains the OOP architecture to orchestrate the Point-E fine-tuning
loop utilizing PyTorch AMP, Checkpointing, and native Diffusion logic.
Now includes a configurable validation loop, best-checkpoint selection via
`checkpoint_best.pt`, and optional Chamfer Distance monitoring.
"""

import json
import logging
from collections import Counter
from datetime import timedelta
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.evaluation.chamfer import compute_mean_chamfer_distance


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

        # Best validation metric seen so far; used for checkpoint_best.pt selection.
        # Restored from checkpoint on resume; defaults to inf for fresh runs.
        self.best_val_metric: float = float("inf")
        self.best_epoch: int = 0
        self._restored_scheduler_state_dict: dict | None = None
        self._max_grad_norm: float | None = None

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Format elapsed time as H:MM:SS for log messages."""
        return str(timedelta(seconds=int(seconds)))

    @staticmethod
    def _json_safe(value):
        """Convert NumPy-heavy replay metadata into JSON-serializable primitives."""
        if isinstance(value, dict):
            return {key: PointETrainer._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [PointETrainer._json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [PointETrainer._json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def train_step(
        self,
        dataloader,
        epochs: int = 1,
        start_epoch: int = 0,
        save_dir=None,
        save_interval: int = 1,
        augmentation_record_file=None,
        val_dataloader=None,
        val_config=None,
        val_log_file=None,
        scheduler_config: dict | None = None,
        max_grad_norm: float | None = None,
    ):
        """
        Executes the core training loop for a given number of epochs.
        """
        self._max_grad_norm = max_grad_norm
        train_start_time = perf_counter()
        scheduler = None
        if scheduler_config and scheduler_config.get("type") == "cosine":
            total_epochs = max(1, epochs - start_epoch)
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=scheduler_config.get("eta_min", 1e-6),
            )
            if self._restored_scheduler_state_dict is not None:
                scheduler.load_state_dict(self._restored_scheduler_state_dict)
                self.logger.info("Restored cosine scheduler state from checkpoint.")

        self.base_model.train()
        
        for epoch in range(start_epoch, epochs):
            elapsed_at_epoch_start = self._format_elapsed(perf_counter() - train_start_time)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"Starting Epoch {epoch + 1}/{epochs} | LR: {current_lr:.6e} | Elapsed: {elapsed_at_epoch_start}")
            augmented_sample_entries: list[dict] = []
            transform_counter: Counter[str] = Counter()
            augmented_sample_count = 0
            
            # T014: Wrap the loop loader using the tqdm library for real-time postfix observation
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            
            for step, batch in enumerate(pbar):
                images = batch["images"]
                point_clouds = batch["point_cloud_6d"].to(self.device)
                sample_ids = batch.get("sample_id", [])
                batch_applied_transforms = batch.get("applied_transforms", [])

                for sample_id, applied_transforms in zip(sample_ids, batch_applied_transforms):
                    if not applied_transforms:
                        continue
                    augmented_sample_count += 1
                    # applied_transforms is now a list of dicts: [{"name": "...", "params": {...}}, ...]
                    transform_names = [t["name"] for t in applied_transforms]
                    transform_counter.update(transform_names)
                    augmented_sample_entries.append(
                        {
                            "epoch": epoch + 1,
                            "sample_id": sample_id,
                            "transforms": applied_transforms,
                        }
                    )
                
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
                if self._max_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), self._max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update observability metrics
                loss_val = loss.item()
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})
                
                # Log periodically to standard output independently from tqdm
                if step % 50 == 0:
                    elapsed_at_step = self._format_elapsed(perf_counter() - train_start_time)
                    self.logger.info(f"Epoch {epoch+1} | Step {step} | Loss: {loss_val:.4f} | LR: {current_lr:.6e} | Elapsed: {elapsed_at_step}")

            if augmentation_record_file is not None and augmented_sample_entries:
                for entry in augmented_sample_entries:
                    augmentation_record_file.write(json.dumps(self._json_safe(entry)) + "\n")
                augmentation_record_file.flush()

            if transform_counter:
                transform_summary = " ".join(f"{name}:{count}" for name, count in sorted(transform_counter.items()))
            else:
                transform_summary = "no augmentations"
            elapsed_after_epoch = self._format_elapsed(perf_counter() - train_start_time)
            self.logger.info(
                f"Epoch {epoch + 1} augmentation summary: {augmented_sample_count} samples | {transform_summary} | LR: {current_lr:.6e} | Elapsed: {elapsed_after_epoch}"
            )

            # Handle checkpoint saving at the end of epoch
            if save_dir and (epoch + 1) % save_interval == 0:
                save_path = Path(save_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
                scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
                self.save_checkpoint(epoch + 1, save_path, scheduler_state_dict=scheduler_state_dict)

            # Run validation pass if configured
            if val_dataloader is not None and val_config is not None:
                val_interval = val_config.get("val_interval", 1)
                if (epoch + 1) % val_interval == 0:
                    result = self._run_val_pass(val_dataloader, val_config)
                    cd_str = f" | CD: {result['chamfer_distance']:.4f}" if result["chamfer_distance"] is not None else " | CD: N/A"
                    best_str = "  →  checkpoint_best.pt saved" if result["new_best"] else ""
                    elapsed_after_val = self._format_elapsed(perf_counter() - train_start_time)
                    self.logger.info(
                        f"Epoch {epoch + 1} | Val Loss: {result['val_loss']:.4f}{cd_str} | New Best: {result['new_best']}{best_str} | LR: {current_lr:.6e} | Elapsed: {elapsed_after_val}"
                    )
                    if result["new_best"] and save_dir:
                        self.best_epoch = epoch + 1
                        scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
                        self.save_checkpoint(epoch + 1, Path(save_dir) / "checkpoint_best.pt", scheduler_state_dict=scheduler_state_dict)
                    if val_log_file is not None:
                        val_log_file.write(json.dumps({
                            "epoch": epoch + 1,
                            "val_loss": result["val_loss"],
                            "chamfer_distance": result["chamfer_distance"],
                            "new_best": result["new_best"],
                        }) + "\n")
                        val_log_file.flush()

            if scheduler is not None:
                scheduler.step()

        total_epochs_trained = max(0, epochs - start_epoch)
        elapsed_seconds = perf_counter() - train_start_time
        return {
            "best_epoch": self.best_epoch,
            "best_val_metric": self.best_val_metric,
            "total_epochs_trained": total_epochs_trained,
            "elapsed_seconds": elapsed_seconds,
            "final_lr": self.optimizer.param_groups[0]["lr"],
        }

    def _run_val_pass(self, val_dataloader, val_config: dict) -> dict:
        """Run a full validation pass and return metrics.

        Parameters
        ----------
        val_dataloader:
            DataLoader for the validation set. Must have augmentation disabled.
        val_config:
            The resolved 'validation' section of the training config.

        Returns
        -------
        dict
            Keys: 'val_loss' (float), 'chamfer_distance' (float|None), 'new_best' (bool).
        """
        checkpoint_metric = val_config.get("checkpoint_metric", "val_loss")
        compute_cd = val_config.get("compute_chamfer_distance", False)
        fixed_timesteps = val_config.get("val_fixed_timesteps") or None

        # Fallback warning: CD metric requested but CD computation disabled
        if checkpoint_metric == "chamfer_distance" and not compute_cd:
            self.logger.warning(
                "checkpoint_metric='chamfer_distance' requires compute_chamfer_distance=true. "
                "Falling back to val_loss."
            )
            checkpoint_metric = "val_loss"

        if fixed_timesteps is not None:
            fixed_timesteps = [int(timestep) for timestep in fixed_timesteps]
            invalid_timesteps = [timestep for timestep in fixed_timesteps if timestep < 0 or timestep >= self.diffusion.num_timesteps]
            if invalid_timesteps:
                raise ValueError(
                    "val_fixed_timesteps must contain values in the range "
                    f"[0, {self.diffusion.num_timesteps - 1}]. Invalid values: {invalid_timesteps}"
                )

        self.base_model.eval()
        total_loss = 0.0
        total_samples = 0
        device_type = self.device.type if isinstance(self.device, torch.device) else "cuda"

        with torch.no_grad():
            if fixed_timesteps is None:
                for batch in val_dataloader:
                    images = batch["images"]
                    point_clouds = batch["point_cloud_6d"].to(self.device)
                    batch_size = point_clouds.shape[0]
                    t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device)

                    with torch.amp.autocast(device_type):
                        loss_dict = self.diffusion.training_losses(
                            model=self.base_model,
                            x_start=point_clouds,
                            t=t,
                            model_kwargs={"images": images},
                        )
                        loss = loss_dict["loss"].mean()

                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
            else:
                for timestep in fixed_timesteps:
                    for batch in val_dataloader:
                        images = batch["images"]
                        point_clouds = batch["point_cloud_6d"].to(self.device)
                        batch_size = point_clouds.shape[0]
                        t = torch.full((batch_size,), timestep, dtype=torch.long, device=self.device)

                        with torch.amp.autocast(device_type):
                            loss_dict = self.diffusion.training_losses(
                                model=self.base_model,
                                x_start=point_clouds,
                                t=t,
                                model_kwargs={"images": images},
                            )
                            loss = loss_dict["loss"].mean()

                        total_loss += loss.item() * batch_size
                        total_samples += batch_size

        val_loss = total_loss / total_samples if total_samples > 0 else float("inf")

        # Optional Chamfer Distance computation
        chamfer_distance = None
        if compute_cd:
            n_steps = val_config.get("cd_sampling_steps", 64)
            chamfer_distance = compute_mean_chamfer_distance(
                self.base_model, self.diffusion, val_dataloader, n_steps, self.device
            )

        # Determine active metric value for best-checkpoint comparison
        active_metric = chamfer_distance if checkpoint_metric == "chamfer_distance" and chamfer_distance is not None else val_loss
        new_best = active_metric < self.best_val_metric
        if new_best:
            self.best_val_metric = active_metric

        self.base_model.train()
        return {"val_loss": val_loss, "chamfer_distance": chamfer_distance, "new_best": new_best}

    def save_checkpoint(self, epoch: int, filepath, scheduler_state_dict: dict | None = None):
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
            "best_val_metric": self.best_val_metric,
            "best_epoch": self.best_epoch,
            "scheduler_state_dict": scheduler_state_dict,
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

        # Restore best val metric for resume-compatible checkpoint selection.
        # Defaults to inf for backwards compatibility with pre-005 checkpoints.
        self.best_val_metric = checkpoint.get("best_val_metric", float("inf"))
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self._restored_scheduler_state_dict = checkpoint.get("scheduler_state_dict")
        self.logger.info(f"Restored best_val_metric: {self.best_val_metric}")
        self.logger.info(f"Restored best_epoch: {self.best_epoch}")

        start_epoch = checkpoint["epoch"]
        self.logger.info(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}.")
        return start_epoch


