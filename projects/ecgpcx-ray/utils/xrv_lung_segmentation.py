"""TorchXRayVision lung segmentation helpers for chest X-ray preprocessing."""

from __future__ import annotations

import numpy as np
import torchxrayvision as xrv
import torch
import torch.nn.functional as F
from PIL import Image


class TorchXRayVisionLungSegmenter:
    """Apply TorchXRayVision's anatomical PSPNet lung mask to PIL X-rays."""

    def __init__(self, device=None, threshold=0.5, mask_value=0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        self.mask_value = mask_value
        self.model = xrv.baseline_models.chestx_det.PSPNet().to(self.device)
        self.model.eval()
        self.lung_indices = [
            self.model.targets.index("Left Lung"),
            self.model.targets.index("Right Lung"),
        ]

    @staticmethod
    def _to_xrv_tensor(image):
        """Convert a PIL image to the single-channel XRV range expected by PSPNet."""
        gray = image.convert("L")
        arr = np.asarray(gray, dtype=np.float32)
        arr = (arr / 255.0) * 2048.0 - 1024.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    def lung_mask(self, image):
        """Return a binary lung mask as a float tensor with shape (H, W)."""
        original_size = image.size[::-1]
        x = self._to_xrv_tensor(image).to(self.device)

        with torch.no_grad():
            pred = self.model(x)
            pred = torch.sigmoid(pred[:, self.lung_indices]).amax(dim=1, keepdim=True)
            pred = F.interpolate(pred, size=original_size, mode="bilinear", align_corners=False)

        return (pred.squeeze().cpu() >= self.threshold).float()

    def apply(self, image):
        """Return a PIL image with pixels outside both lungs suppressed."""
        gray = image.convert("L")
        arr = np.asarray(gray, dtype=np.float32)
        mask = self.lung_mask(gray).numpy()
        segmented = arr * mask + float(self.mask_value) * (1.0 - mask)
        segmented = np.clip(segmented, 0, 255).astype(np.uint8)
        return Image.fromarray(segmented, mode="L")

    def mask_image(self, image):
        """Return the binary lung mask as a PIL image with values 0 or 255."""
        mask = self.lung_mask(image).numpy()
        mask = (mask * 255).astype(np.uint8)
        return Image.fromarray(mask, mode="L")


def apply_lung_segmentation(
    images,
    device=None,
    threshold=0.5,
    mask_value=0,
    verbose=True,
    segmenter=None,
):
    """Segment a list of PIL images with TorchXRayVision lung masks."""
    if segmenter is None:
        segmenter = TorchXRayVisionLungSegmenter(
            device=device,
            threshold=threshold,
            mask_value=mask_value,
        )
    segmented_images = []

    for idx, image in enumerate(images, start=1):
        segmented_images.append(segmenter.apply(image))
        if verbose and idx % 250 == 0:
            print(f"Segmented {idx}/{len(images)} images.")

    return segmented_images


def create_lung_mask_images(
    images,
    device=None,
    threshold=0.5,
    mask_value=0,
    verbose=True,
    segmenter=None,
):
    """Create binary lung mask PIL images from a list of chest X-rays."""
    if segmenter is None:
        segmenter = TorchXRayVisionLungSegmenter(
            device=device,
            threshold=threshold,
            mask_value=mask_value,
        )
    mask_images = []

    for idx, image in enumerate(images, start=1):
        mask_images.append(segmenter.mask_image(image))
        if verbose and idx % 250 == 0:
            print(f"Created {idx}/{len(images)} lung masks.")

    return mask_images
