import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import matplotlib.cm as mplcm
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
from matplotlib.gridspec import GridSpec
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
INDEX_PATTERN = re.compile(r"img_(\d+)_")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "training-results" / "cvae" / "results"

INPUT_COLOR = "#2c7bb6"
GEN_COLOR = "#d7191c"
HEAT_COLOR = "#4dac26"

ROW_LABELS = [
    ("Healthy\n-> Pneumonia", INPUT_COLOR),
    ("Pneumonia\n-> Healthy", GEN_COLOR),
]
COL_TITLES = ["Input (Real)", "Generated", "Change Heatmap"]
COL_COLORS = [INPUT_COLOR, GEN_COLOR, HEAT_COLOR]


@dataclass
class EvaluationResult:
    num_ssim_pairs: int
    ssim_mean: float
    ssim_std: float
    ssim_min: float
    ssim_max: float
    num_counterfactual_images: int
    num_reference_images: int
    fid: float


class ImageFolderDataset(Dataset):
    """Small image folder dataset returning uint8 RGB tensors for TorchMetrics FID."""

    def __init__(self, image_paths: list[Path], transform: transforms.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


def list_images(folder: Path, include_pairs: bool = False) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")

    paths = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not include_pairs:
        paths = [path for path in paths if "_pair" not in path.stem]
    return sorted(paths)


def extract_index(path: Path) -> str | None:
    match = INDEX_PATTERN.search(path.name)
    if match is None:
        return None
    return match.group(1)


def pair_originals_and_counterfactuals(
    original_dir: Path, counterfactual_dir: Path
) -> list[tuple[Path, Path]]:
    originals = {
        extract_index(path): path
        for path in list_images(original_dir)
        if "_original" in path.stem and extract_index(path) is not None
    }
    counterfactuals = {
        extract_index(path): path
        for path in list_images(counterfactual_dir)
        if "_counterfactual" in path.stem and extract_index(path) is not None
    }

    shared_indices = sorted(set(originals).intersection(counterfactuals))
    return [(originals[index], counterfactuals[index]) for index in shared_indices]


def load_grayscale_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def load_grayscale_array(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32) / 255.0


def label_to_class_index(label) -> int:
    label = torch.as_tensor(label).detach().cpu()
    if label.ndim == 0 or label.numel() == 1:
        return int(label.item())
    return int(torch.argmax(label).item())


def chunked(items: list, size: int):
    """Yield complete non-overlapping chunks of size."""
    for start in range(0, len(items) - size + 1, size):
        yield items[start : start + size]


def chunked_with_reuse(items: list[int], size: int, n_batches: int) -> list[list[int]]:
    """Create fixed-size chunks, reusing from the start only when a class runs out."""
    if not items:
        return []

    chunks = []
    for batch_idx in range(n_batches):
        chunk = []
        for offset in range(size):
            item_idx = batch_idx * size + offset
            chunk.append(items[item_idx] if item_idx < len(items) else items[item_idx % len(items)])
        chunks.append(chunk)
    return chunks


def make_change_heatmap_overlay(
    real: np.ndarray,
    generated: np.ndarray,
    colormap: str = "hot",
    overlay_alpha: float = 0.55,
    blur_sigma: float = 2.0,
) -> np.ndarray:
    """Overlay smoothed |real - generated| on the real grayscale image."""
    if real.shape != generated.shape:
        raise ValueError(
            "Images must have the same shape for heatmap visualization: "
            f"{real.shape} != {generated.shape}"
        )

    diff = np.abs(real - generated)
    if blur_sigma > 0:
        diff_img = Image.fromarray((diff * 255).astype(np.uint8), mode="L")
        diff = (
            np.asarray(
                diff_img.filter(ImageFilter.GaussianBlur(blur_sigma)),
                dtype=np.float32,
            )
            / 255.0
        )

    diff_norm = diff / (diff.max() + 1e-8)
    heatmap_rgb = mplcm.get_cmap(colormap)(diff_norm)[..., :3]
    real_rgb = np.stack([real, real, real], axis=-1)
    return np.clip((1.0 - overlay_alpha) * real_rgb + overlay_alpha * heatmap_rgb, 0, 1)


def style_heatmap_axis(ax, spine_color: str, linewidth: float = 2.0) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        spine.set_linewidth(linewidth)


def image_paths_for_index(
    image_idx: int,
    original_dir: Path,
    counterfactual_dir: Path,
) -> tuple[Path, Path]:
    index = f"{image_idx:06d}"
    original_path = original_dir / f"img_{index}_original.png"
    counterfactual_path = counterfactual_dir / f"img_{index}_counterfactual.png"
    if not original_path.exists():
        raise FileNotFoundError(f"Missing original image: {original_path}")
    if not counterfactual_path.exists():
        raise FileNotFoundError(f"Missing counterfactual image: {counterfactual_path}")
    return original_path, counterfactual_path


def render_cvae_change_heatmap_batch(
    healthy_indices: list[int],
    pneumonia_indices: list[int],
    original_dir: Path,
    counterfactual_dir: Path,
    save_path: Path,
    batch_idx: int,
    n_batches: int,
    colormap: str = "hot",
    overlay_alpha: float = 0.55,
    blur_sigma: float = 2.0,
) -> None:
    rows = [
        (healthy_indices, ROW_LABELS[0]),
        (pneumonia_indices, ROW_LABELS[1]),
    ]
    samples_per_class = len(healthy_indices)
    n_img_cols = 3 * samples_per_class

    fig = plt.figure(figsize=(3.0 * n_img_cols + 0.6, 7.2))
    gs = GridSpec(
        nrows=2,
        ncols=n_img_cols + 1,
        figure=fig,
        width_ratios=[1.0] * n_img_cols + [0.05],
        wspace=0.05,
        hspace=0.18,
        left=0.10,
        right=0.96,
        top=0.86,
        bottom=0.05,
    )

    axes = np.array(
        [[fig.add_subplot(gs[row, col]) for col in range(n_img_cols)] for row in range(2)]
    )
    cbar_ax = fig.add_subplot(gs[:, -1])

    for sample_idx in range(samples_per_class):
        for col_offset, (title, color) in enumerate(zip(COL_TITLES, COL_COLORS)):
            axes[0, 3 * sample_idx + col_offset].set_title(
                title,
                fontsize=9,
                fontweight="bold",
                color=color,
                pad=5,
            )

    for row_idx, (indices, (row_label, row_color)) in enumerate(rows):
        for sample_idx, image_idx in enumerate(indices):
            original_path, counterfactual_path = image_paths_for_index(
                image_idx,
                original_dir,
                counterfactual_dir,
            )
            real = load_grayscale_array(original_path)
            generated = load_grayscale_array(counterfactual_path)
            overlay = make_change_heatmap_overlay(
                real,
                generated,
                colormap=colormap,
                overlay_alpha=overlay_alpha,
                blur_sigma=blur_sigma,
            )

            col = 3 * sample_idx
            axes[row_idx, col].imshow(real, cmap="gray", vmin=0, vmax=1)
            axes[row_idx, col + 1].imshow(generated, cmap="gray", vmin=0, vmax=1)
            axes[row_idx, col + 2].imshow(overlay)

            style_heatmap_axis(axes[row_idx, col], INPUT_COLOR)
            style_heatmap_axis(axes[row_idx, col + 1], GEN_COLOR)
            style_heatmap_axis(axes[row_idx, col + 2], HEAT_COLOR)

            axes[row_idx, col].set_xlabel(f"idx {image_idx:06d}", fontsize=8)

        axes[row_idx, 0].set_ylabel(
            row_label,
            fontsize=11,
            fontweight="bold",
            color=row_color,
            labelpad=10,
            rotation=90,
            va="center",
        )

    scalar_mappable = plt.cm.ScalarMappable(
        cmap=mplcm.get_cmap(colormap),
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, cax=cbar_ax)
    cbar.set_label("Normalized |pixel change|", fontsize=9, labelpad=6)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"CVAE Change Heatmaps [{batch_idx + 1}/{n_batches}]",
        fontsize=13,
        fontweight="bold",
        y=0.95,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_cvae_change_heatmap_sweep(
    test_dataset,
    original_dir: Path | str = DEFAULT_RESULTS_DIR / "original",
    counterfactual_dir: Path | str = DEFAULT_RESULTS_DIR / "counterfactuals",
    output_dir: Path | str = DEFAULT_RESULTS_DIR / "change_heatmaps",
    samples_per_class: int = 2,
    colormap: str = "hot",
    overlay_alpha: float = 0.55,
    blur_sigma: float = 2.0,
    max_batches: int | None = None,
) -> list[Path]:
    """Save CVAE change heatmaps for the full test set.

    Each PNG contains two rows. The first row shows healthy originals and their
    generated pneumonia counterfactuals. The second row shows pneumonia originals
    and their generated healthy counterfactuals. Images are grouped in complete
    batches of `samples_per_class` from each source class.
    """
    original_dir = Path(original_dir)
    counterfactual_dir = Path(counterfactual_dir)
    output_dir = Path(output_dir)

    labels = [label_to_class_index(label) for label in test_dataset.labels]
    healthy_indices = [idx for idx, label in enumerate(labels) if label == 0]
    pneumonia_indices = [idx for idx, label in enumerate(labels) if label == 1]

    n_healthy_batches = int(np.ceil(len(healthy_indices) / samples_per_class))
    n_pneumonia_batches = int(np.ceil(len(pneumonia_indices) / samples_per_class))
    n_batches = max(n_healthy_batches, n_pneumonia_batches)

    if not healthy_indices or not pneumonia_indices:
        raise ValueError(
            "Both classes are required to create CVAE change heatmap batches. "
            f"Found {len(healthy_indices)} healthy and {len(pneumonia_indices)} pneumonia samples."
        )

    healthy_batches = chunked_with_reuse(healthy_indices, samples_per_class, n_batches)
    pneumonia_batches = chunked_with_reuse(pneumonia_indices, samples_per_class, n_batches)

    saved_paths = []
    for batch_idx in tqdm(range(n_batches), desc="Saving CVAE change heatmaps"):
        if max_batches is not None and batch_idx >= max_batches:
            break
        save_path = output_dir / f"cvae_change_heatmap_{batch_idx + 1:03d}.png"
        render_cvae_change_heatmap_batch(
            healthy_batches[batch_idx],
            pneumonia_batches[batch_idx],
            original_dir,
            counterfactual_dir,
            save_path,
            batch_idx,
            n_batches,
            colormap=colormap,
            overlay_alpha=overlay_alpha,
            blur_sigma=blur_sigma,
        )
        saved_paths.append(save_path)

    reused_healthy = max(0, n_batches * samples_per_class - len(healthy_indices))
    reused_pneumonia = max(0, n_batches * samples_per_class - len(pneumonia_indices))
    print(f"Saved {len(saved_paths)} CVAE change heatmap PNGs to: {output_dir}")
    if reused_healthy or reused_pneumonia:
        print(
            "Reused samples to fill fixed-size rows: "
            f"{reused_healthy} healthy, {reused_pneumonia} pneumonia"
        )

    return saved_paths


def compute_paired_ssim(pairs: Iterable[tuple[Path, Path]], device) -> list[dict[str, object]]:
    rows = []
    for original_path, counterfactual_path in tqdm(
        list(pairs), desc="Computing SSIM", leave=False
    ):
        original = load_grayscale_tensor(original_path).to(device)
        counterfactual = load_grayscale_tensor(counterfactual_path).to(device)

        if original.shape != counterfactual.shape:
            raise ValueError(
                "Paired images must have the same shape for SSIM: "
                f"{original_path} has {tuple(original.shape)}, "
                f"{counterfactual_path} has {tuple(counterfactual.shape)}"
            )

        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        rows.append(
            {
                "index": extract_index(original_path),
                "original": str(original_path),
                "counterfactual": str(counterfactual_path),
                "ssim": float(ssim_metric(original, counterfactual).item()),
            }
        )
    return rows


def fid_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.PILToTensor(),
        ]
    )


@torch.no_grad()
def update_fid(
    metric: FrechetInceptionDistance, image_paths: list[Path], device: torch.device, batch_size: int, num_workers: int, real: bool) -> None:
    dataset = ImageFolderDataset(image_paths, fid_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    label = "real reference" if real else "fake counterfactual"
    for batch in tqdm(loader, desc=f"Updating FID ({label})", leave=False):
        metric.update(batch.to(device), real=real)


def compute_fid(
    reference_images: list[Path],
    counterfactual_images: list[Path],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    if len(reference_images) < 2 or len(counterfactual_images) < 2:
        raise ValueError("FID requires at least two images in each image set.")

    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    update_fid(metric, reference_images, device, batch_size, num_workers, real=True)
    update_fid(metric, counterfactual_images, device, batch_size, num_workers, real=False)
    return float(metric.compute().item())


def write_ssim_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["index", "original", "counterfactual", "ssim"]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_json(result: EvaluationResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=2)


def summarize_ssim(rows: list[dict[str, object]]) -> tuple[float, float, float, float]:
    if not rows:
        nan = float("nan")
        return nan, nan, nan, nan

    scores = np.array([row["ssim"] for row in rows], dtype=np.float64)
    return (
        float(scores.mean()),
        float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        float(scores.min()),
        float(scores.max()),
    )

def ssim_metric_calculation(device, output_csv, original_dir, counterfactual_dir):
    pairs = pair_originals_and_counterfactuals(original_dir, counterfactual_dir)
    if not pairs:
        raise ValueError(
            "No matching original/counterfactual pairs were found. Expected names like "
            "img_000000_original.png and img_000000_counterfactual.png."
        )
    ssim_rows = compute_paired_ssim(pairs, device)
    write_ssim_csv(ssim_rows, output_csv)
    ssim_mean, ssim_std, ssim_min, ssim_max = summarize_ssim(ssim_rows)
    return ssim_mean, ssim_std, ssim_min, ssim_max, ssim_rows

def fid_metric_calculation(original_dir, counterfactual_dir, device, batch_size, num_workers):
    counterfactual_images = [
        path for path in list_images(counterfactual_dir) if "_counterfactual" in path.stem
    ]
    reference_images = list_images(original_dir)
    reference_images = [path for path in reference_images if "_original" in path.stem]

    fid = compute_fid(
        reference_images,
        counterfactual_images,
        device,
        batch_size,
        num_workers,
    )
    return fid, len(counterfactual_images), len(reference_images)


