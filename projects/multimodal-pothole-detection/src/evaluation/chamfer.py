"""Isolated Chamfer Distance evaluation utility for Point-E validation.

Called only when `compute_chamfer_distance` is enabled in the training config.
Computes mean symmetric Chamfer Distance between Point-E generated point clouds
and ground-truth tensors over a validation DataLoader. No side effects outside
the returned scalar.

CD is computed in the already-normalised [-1, 1] spatial space to avoid
de-normalisation overhead.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def _chamfer_distance_batch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute symmetric Chamfer Distance for a batch of point cloud pairs.

    Parameters
    ----------
    pred:
        Predicted point clouds with shape [B, N, 3].
    gt:
        Ground-truth point clouds with shape [B, M, 3].

    Returns
    -------
    torch.Tensor
        Per-sample CD values with shape [B].
    """
    # pairwise squared distances: [B, N, M]
    dist = torch.cdist(pred, gt, p=2).pow(2)

    # Chamfer: mean of nearest-neighbour distances in both directions
    cd_pred_to_gt = dist.min(dim=2).values.mean(dim=1)   # [B]
    cd_gt_to_pred = dist.min(dim=1).values.mean(dim=1)   # [B]

    return cd_pred_to_gt + cd_gt_to_pred


def compute_mean_chamfer_distance(
    model,
    diffusion,
    dataloader: DataLoader,
    n_steps: int,
    device: torch.device,
) -> float:
    """Compute mean Chamfer Distance between generated and ground-truth point clouds.

    Runs the Point-E DDIM sampler on every val sample, extracts spatial coords,
    and returns the scalar mean CD over the full val set.

    Parameters
    ----------
    model:
        Point-E base model. Will be temporarily set to eval mode and restored
        to train mode after computation.
    diffusion:
        Point-E diffusion object providing the fast sampler API.
    dataloader:
        Validation DataLoader. Must have augmentation disabled.
    n_steps:
        Number of DDIM sampling steps (e.g. 64 for fast approximate evaluation).
    device:
        Torch device for inference.

    Returns
    -------
    float
        Mean symmetric Chamfer Distance (lower is better) in normalised [-1, 1] space.
    """
    model.eval()
    total_cd = 0.0
    total_samples = 0
    device_type = device.type if isinstance(device, torch.device) else "cuda"

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"]
            gt_clouds = batch["point_cloud_6d"].to(device)  # [B, 6, K]
            batch_size = gt_clouds.shape[0]

            # Sample predicted point clouds using fast DDIM
            with torch.amp.autocast(device_type):
                samples = diffusion.ddim_sample_loop(
                    model,
                    shape=(batch_size, gt_clouds.shape[1], gt_clouds.shape[2]),
                    model_kwargs={"images": images},
                    num_inference_steps=n_steps,
                    device=device,
                )

            # Extract spatial coords [B, K, 3] from channel-first [B, 6, K]
            pred_spatial = samples[:, :3, :].permute(0, 2, 1)   # [B, K, 3]
            gt_spatial = gt_clouds[:, :3, :].permute(0, 2, 1)   # [B, K, 3]

            cd_per_sample = _chamfer_distance_batch(pred_spatial, gt_spatial)
            total_cd += cd_per_sample.sum().item()
            total_samples += batch_size

    model.train()
    return total_cd / total_samples if total_samples > 0 else float("inf")


def compute_symmetric_chamfer_distance_points(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute symmetric Chamfer Distance for two point clouds.

    Parameters
    ----------
    pred:
        Predicted point cloud with shape [N, 3].
    gt:
        Ground-truth point cloud with shape [M, 3].

    Returns
    -------
    float
        Symmetric Chamfer Distance using squared Euclidean distances.
    """

    if pred.size == 0 or gt.size == 0:
        return float("inf")

    dist = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1) ** 2
    return float(dist.min(axis=1).mean() + dist.min(axis=0).mean())
