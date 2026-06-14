"""
3D Point Cloud Evaluation Module.

This script evaluates generated .npy point clouds, calculating the anomaly depth
using the P05 (5th percentile) metric on the Z-axis, and classifies the severity
of the pothole into engineering categories.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Define absolute paths dynamically relative to this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Global Configuration Constants ---
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "interim" / "point_e_baseline_predictions"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports" / "point_e_baseline_results.csv"

# Rui Fan MVP severity thresholds (in centimeters)
MVP_THRESHOLD_LOW_CM = 7.0
MVP_THRESHOLD_MEDIUM_CM = 10.0

def calculate_p05_depth(coords: np.ndarray) -> float:
    """
    Calculates the 5th percentile of the negative Z-axis points (depth).
    Returns the absolute depth value as a positive float in meters.
    """
    # Z-axis is typically index 2 in Point-E output
    z_coords = coords[:, 2]
    
    # Isolate negative Z points (points considered below the asphalt surface level)
    negative_z = z_coords[z_coords < 0.0]
    
    if len(negative_z) == 0:
        return 0.0
        
    # P05 is the 5th percentile from the bottom (representing the deepest reliable points).
    # This resists extreme outlier noise points naturally produced by diffusion models.
    p05 = np.percentile(negative_z, 5)
    
    # Return as an absolute positive depth
    return abs(p05)

def classify_severity(depth: float) -> str:
    """
    Classify a depth value in meters into the legacy severity buckets.

    This is kept as a thin compatibility wrapper so existing notebooks and
    scripts can transition to the Rui Fan MVP centimeter-based classifier
    without breaking immediately.

    Args:
        depth: Absolute depth value in meters.

    Returns:
        str: One of ``low``, ``medium``, or ``high``.
    """
    return classify_severity_cm(depth * 100.0)


def classify_severity_cm(depth_cm: float) -> str:
    """
    Classify a depth value in centimeters into the Rui Fan MVP severity bins.

    Args:
        depth_cm: Absolute depth value in centimeters.

    Returns:
        str: One of ``low``, ``medium``, or ``high``.
    """
    if depth_cm < MVP_THRESHOLD_LOW_CM:
        return "low"
    if depth_cm < MVP_THRESHOLD_MEDIUM_CM:
        return "medium"
    return "high"


def calculate_effective_depth_cm(coords: np.ndarray) -> float:
    """
    Calculate the outlier-resistant effective depth in centimeters.

    Args:
        coords: Point cloud coordinates in meters, with the pothole depth encoded
            as negative Z values.

    Returns:
        float: P05 depth converted to centimeters.
    """
    return calculate_p05_depth(coords) * 100.0

def evaluate_batch(args):
    """Execution block for evaluating a directory of .npy point clouds."""
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    
    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print("--- 3D Point Cloud Evaluation ---")
    
    if not input_dir.exists():
        print(f"Error: Input directory not found at {input_dir}")
        return
        
    npy_files = list(input_dir.glob("*.npy"))
    if not npy_files:
        print(f"Warning: No valid .npy point cloud files found in {input_dir}")
        return

    print(f"Found {len(npy_files)} point cloud files. Commencing evaluation...")
    
    results = []
    for npy_path in tqdm(npy_files, desc="Evaluating point clouds"):
        try:
            coords = np.load(npy_path)
            
            # 1. Analyze depth metric
            depth_p05 = calculate_p05_depth(coords)
            # 2. Categorize
            severity = classify_severity(depth_p05)
            
            results.append({
                "image_name": npy_path.stem,
                "p05_depth_meters": round(depth_p05, 4),
                "severity_class": severity
            })
            
        except Exception as e:
            print(f"Error evaluating {npy_path.name}: {e}")
            
    # Load existing CSV if it exists to append/update (checkpointing logic)
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        new_df = pd.DataFrame(results)
        
        # Merge, overwriting old entries with new ones if they share the same image_name
        # Keep old ones that weren't evaluated in this run
        merged_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['image_name'], keep='last')
        final_df = merged_df
    else:
        final_df = pd.DataFrame(results)

    # Export to DataFrame and CSV
    if not final_df.empty:
        final_df.to_csv(output_csv, index=False)
        print(f"\nEvaluation complete! Report exported to: {output_csv}")
        
        # Display summary distribution in the console
        print("\nSeverity Distribution:")
        print(final_df['severity_class'].value_counts().to_string())

def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D point clouds and categorize severity.")
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR), help="Directory containing input .npy files")
    parser.add_argument("--output_csv", type=str, default=str(DEFAULT_OUTPUT_CSV), help="Output path for the results CSV")
    args = parser.parse_args()
    
    evaluate_batch(args)

if __name__ == "__main__":
    main()