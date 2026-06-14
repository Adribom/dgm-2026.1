import argparse
import sys
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Discover current directory so we can import local modules regardless of where it's run
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
scripts_dir = project_root / "src" / "scripts"

if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from pothole_geometry import load_yolo_mask
from point_e_pipeline_utils import (
    get_square_bbox_from_mask, 
    apply_square_crop,
    apply_square_crop_hybrid,
    compute_leveled_point_cloud, 
    format_point_e_tensor
)

def main():
    parser = argparse.ArgumentParser(description="Generate Point-E ready tensors and images from PothRGDB.")
    parser.add_argument("--dataset_root", type=Path, required=True, 
                        help="Path to the original PothRGDB dataset root folder")
    parser.add_argument("--manifest", type=Path, required=True, 
                        help="Path to the pothrgbd_manifest.csv file")
    parser.add_argument("--output_dir", type=Path, required=True, 
                        help="Directory to save the formatted Point-E data")
    parser.add_argument("--margin", type=int, default=20, 
                        help="Square crop context margin in pixels (default: 20)")
    
    args = parser.parse_args()

    # Hardcoded Intrinsics per Methodology
    FX, FY = 460.0, 460.0
    
    dataset_root = args.dataset_root
    manifest_path = args.manifest
    output_root = args.output_dir
    CONTEXT_MARGIN_PX = args.margin

    # 1. Setup Destination Architecture
    dirs = {
        "images": output_root / "images",
        "tensors": output_root / "tensors",
        "heatmaps": output_root / "heatmaps"
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.json"
    metadata = {}

    # 2. Load Manifest
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    valid_samples = manifest[manifest["status"] == "ok"].reset_index(drop=True)

    print(f"Starting Point-E Dataset Generation for {len(valid_samples)} samples...")
    print(f"Input Dataset: {dataset_root}")
    print(f"Output Target: {output_root}")

    # 3. The Factory Loop
    for idx, row in tqdm(valid_samples.iterrows(), total=len(valid_samples), desc="Processing"):
        sample_id = row['sample_id']
        
        rgb_path = dataset_root / row["image_path"]
        depth_path = dataset_root / row["depth_path"]
        mask_path = dataset_root / row["label_path"]
        
        # Check if critical files exist
        if not rgb_path.exists() or not depth_path.exists() or not mask_path.exists():
            continue
            
        try:
            # --- LOAD DATA ---
            rgb_img = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            depth_img = np.load(depth_path)
            
            # Fix Depth Shape Discrepancy (some depth .npy files have different dimensions than the RGB)
            if depth_img.shape != rgb_img.shape[:2]:
                depth_img = cv2.resize(depth_img, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
            mask_img = load_yolo_mask(mask_path, image_shape=rgb_img.shape[:2])
            
            if mask_img is None or mask_img.sum() == 0:
                continue
                
            cx, cy = rgb_img.shape[1] / 2.0, rgb_img.shape[0] / 2.0

            # --- 2D PHASE ---
            square_bbox = get_square_bbox_from_mask(mask_img, margin_px=CONTEXT_MARGIN_PX)
            rgb_crop = apply_square_crop_hybrid(rgb_img, square_bbox)

            # --- 3D PHASE ---
            points_3d, colors_3d = compute_leveled_point_cloud(
                rgb_img=rgb_img,
                depth_img=depth_img,
                mask_img=mask_img,
                square_bbox=square_bbox,
                fx=FX, fy=FY, cx=cx, cy=cy
            )
            
            # Abort if RANSAC failed to find enough points
            if len(points_3d) < 100:
                continue
                
            tensor_6d, scale_factor = format_point_e_tensor(points_3d, colors_3d, num_points=1024)
            
            # --- SAVING ARTIFACTS ---
            # 1. 2D Image for CLIP
            out_rgb_path = dirs["images"] / f"{sample_id}.png"
            cv2.imwrite(str(out_rgb_path), cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))
            
            # 2. 3D Tensor for Point-E
            out_tensor_path = dirs["tensors"] / f"{sample_id}.npy"
            np.save(out_tensor_path, tensor_6d)
            
            # 3. Metadata
            metadata[sample_id] = float(scale_factor)
            
            # 4. Point 3 (Heatmap Generation)
            spatial = tensor_6d[:, :3]
            
            fig, ax = plt.subplots(figsize=(4, 4))
            # Scatter plot of X and Y, colored by depth Z. Invert Y for top-down view.
            sc = ax.scatter(spatial[:, 0], -spatial[:, 1], c=spatial[:, 2], 
                            cmap='jet', s=5, alpha=0.8, vmin=-0.1, vmax=0.3)
            ax.axis('off')
            ax.set_title(f"Z-Depth Heatmap\n{sample_id}", fontsize=10)
            
            out_heatmap_path = dirs["heatmaps"] / f"{sample_id}_heatmap.png"
            plt.savefig(out_heatmap_path, bbox_inches='tight', dpi=100)
            plt.close(fig) # Prevent memory leak!
            
        except Exception as e:
            # Non-fatal crash for corrupt individual samples
            print(f"\nFailed on {sample_id}: {e}")
            continue

    # 4. Save Global Metadata
    with open(metadata_path, 'w') as f:
        json.dump({"scales": metadata}, f, indent=4)

    print(f"\n✅ Finished! Successfully generated {len(metadata)} valid Point-E samples.")
    print(f"Data saved at: {output_root}")
    print("-> To perform Point 3 cleanup, browse the 'heatmaps' folder in Windows Explorer and delete bad images.")

if __name__ == "__main__":
    main()
