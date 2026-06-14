"""
Gera as mascaras binárias de pulmao para todas as imagens em data/processed/
e as salva em pastas *_masks/ com o mesmo nome de arquivo.

Estrutura gerada:
    data/processed/
        train/healthy_masks/    train/pneumonia_masks/
        val/healthy_masks/      val/pneumonia_masks/
        test/healthy_masks/     test/pneumonia_masks/

Execute a partir de projects/ecgpcx-ray/:
    python prepare_masks.py [--device cuda]
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.xrv_lung_segmentation import TorchXRayVisionLungSegmenter

PROCESSED_DIR = Path("data") / "processed"

SPLITS_DOMAINS = [
    ("train", "healthy"),
    ("train", "pneumonia"),
    ("val",   "healthy"),
    ("val",   "pneumonia"),
    ("test",  "healthy"),
    ("test",  "pneumonia"),
]

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo para inferencia (default: cuda se disponivel, senao cpu).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Dispositivo: {args.device}")
    print("Carregando modelo PSPNet (TorchXRayVision)...")
    segmenter = TorchXRayVisionLungSegmenter(device=args.device)
    print("Modelo carregado.\n")

    total_saved = 0

    for split, domain in SPLITS_DOMAINS:
        src_dir  = PROCESSED_DIR / split / domain
        mask_dir = PROCESSED_DIR / split / f"{domain}_masks"

        if not src_dir.exists():
            print(f"[AVISO] Pasta nao encontrada, pulando: {src_dir}")
            continue

        images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in EXTS)
        if not images:
            print(f"[AVISO] Nenhuma imagem em {src_dir}, pulando.")
            continue

        mask_dir.mkdir(parents=True, exist_ok=True)

        print(f"{split}/{domain}  ({len(images)} imagens)  ->  {mask_dir}")

        for img_path in tqdm(images, desc=f"{split}/{domain}", unit="img"):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                continue  # ja gerada, pular

            img  = Image.open(img_path).convert("L")
            mask = segmenter.mask_image(img)   # PIL 'L', valores 0 ou 255
            mask.save(mask_path)
            total_saved += 1

    print(f"\nConcluido. {total_saved} mascaras geradas.")


if __name__ == "__main__":
    main()
