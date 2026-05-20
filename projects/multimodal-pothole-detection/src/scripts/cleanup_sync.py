import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Synchronize dataset based on approved heatmaps.")
    parser.add_argument("--dataset_dir", type=Path, required=True, 
                        help="Path to the generated Point-E dataset directory (e.g. data/processed/point_e_ready)")
    parser.add_argument("--dry_run", action="store_true", 
                        help="Print what would be deleted without actually deleting")
    
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    
    dirs = {
        "images": dataset_dir / "images",
        "tensors": dataset_dir / "tensors",
        "heatmaps": dataset_dir / "heatmaps"
    }
    metadata_path = dataset_dir / "metadata.json"
    
    for name, d in dirs.items():
        if not d.exists():
            print(f"Error: Directory {d} does not exist.")
            return
            
    if not metadata_path.exists():
        print(f"Error: Metadata file {metadata_path} does not exist.")
        return

    # 1. Identify approved sample IDs from heatmaps folder
    approved_samples = set()
    for hm_path in dirs["heatmaps"].glob("*_heatmap.png"):
        sample_id = hm_path.name.replace("_heatmap.png", "")
        approved_samples.add(sample_id)
        
    print(f"Found {len(approved_samples)} approved heatmaps.")

    # 2. Find orphans in images and tensors
    files_deleted = 0
    
    for img_path in dirs["images"].glob("*.png"):
        sample_id = img_path.stem
        if sample_id not in approved_samples:
            if args.dry_run:
                print(f"  [Dry Run] Would delete image: {img_path.name}")
            else:
                img_path.unlink()
                print(f"  [Deleted] {img_path.name}")
            files_deleted += 1
            
    for tensor_path in dirs["tensors"].glob("*.npy"):
        sample_id = tensor_path.stem
        if sample_id not in approved_samples:
            if args.dry_run:
                print(f"  [Dry Run] Would delete tensor: {tensor_path.name}")
            else:
                tensor_path.unlink()
                print(f"  [Deleted] {tensor_path.name}")
            files_deleted += 1

    # 3. Clean up metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    original_metadata_len = len(metadata.get("scales", {}))
    new_scales = {k: v for k, v in metadata.get("scales", {}).items() if k in approved_samples}
    metadata["scales"] = new_scales
    metadata_removed = original_metadata_len - len(new_scales)
    
    if not args.dry_run:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
    # 4. Summary
    print("\n" + "="*30)
    print("       CLEANUP SUMMARY")
    print("="*30)
    if args.dry_run:
        print("[DRY RUN MODE] No files were actually deleted.")
    print(f"Approved Samples Retained: {len(approved_samples)}")
    print(f"Orphan Files Cleaned:      {files_deleted}")
    print(f"Metadata Entries Removed:  {metadata_removed}")
    print("="*30)

if __name__ == "__main__":
    main()
