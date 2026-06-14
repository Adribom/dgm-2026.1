"""
Organiza as imagens em data/processed/ seguindo a mesma lógica do notebook
data-folders-test.ipynb, mas usando os CSVs existentes para definir quais
imagens pertencem a cada split (train / val / test).

Regras:
  - A CLASSE de cada imagem é determinada pela pasta de origem:
      healthy_images/  → healthy
      pnemonia_images/ → pneumonia
    (a coluna Label do CSV não é usada para isso)
  - O SPLIT de cada imagem é determinado pelo CSV em que ela aparece.
  - Balanceamento (por split):
      · Pacientes healthy com > 2 imagens no split são excluídos.
      · Pacientes healthy selecionados = min(disponíveis, 2 × pacientes pneumonia).

Execute a partir de projects/ecgpcx-ray/:
    python prepare_processed_data.py
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

SEED = 42

DATA_DIR      = Path("data")
HEALTHY_SRC   = DATA_DIR / "healthy_images"
PNEUMONIA_SRC = DATA_DIR / "pnemonia_images"
OUTPUT_DIR    = DATA_DIR / "processed"

SPLITS = {
    "train": DATA_DIR / "train_split.csv",
    "val":   DATA_DIR / "val_split.csv",
    "test":  DATA_DIR / "test_split.csv",
}

# =========================================================
# BUILD LOOKUP SETS (evita stat por arquivo)
# =========================================================

healthy_files   = {f.name for f in HEALTHY_SRC.glob("*.png")}
pneumonia_files = {f.name for f in PNEUMONIA_SRC.glob("*.png")}

print(f"healthy_images/   : {len(healthy_files)} arquivos encontrados")
print(f"pnemonia_images/  : {len(pneumonia_files)} arquivos encontrados")

# =========================================================
# PROCESSAR CADA SPLIT
# =========================================================

for split_name, csv_path in SPLITS.items():

    df = pd.read_csv(csv_path)

    # ── Determinar classe pela pasta de origem ─────────────
    def get_domain(name):
        if name in pneumonia_files:
            return "pneumonia"
        if name in healthy_files:
            return "healthy"
        return None

    df["domain"] = df["Image Index"].apply(get_domain)

    not_found = df["domain"].isna().sum()
    if not_found:
        print(f"\n[AVISO] {not_found} imagem(ns) não encontrada(s) em {split_name} — ignoradas.")

    df = df[df["domain"].notna()].copy()

    healthy_df   = df[df["domain"] == "healthy"].copy()
    pneumonia_df = df[df["domain"] == "pneumonia"].copy()

    # ── Filtro: pacientes healthy com ≤ 2 imagens no split ─
    counts = healthy_df.groupby("Patient ID").size()
    valid  = counts[counts <= 2].index
    healthy_df = healthy_df[healthy_df["Patient ID"].isin(valid)]

    # ── Balanceamento: healthy = 2 × pacientes pneumonia ───
    pne_patients = pneumonia_df["Patient ID"].unique()
    n_pne        = len(pne_patients)

    h_patients = healthy_df["Patient ID"].unique()
    n_select   = min(len(h_patients), n_pne * 2)

    rng = np.random.default_rng(SEED)
    selected_h = rng.choice(h_patients, size=n_select, replace=False)
    healthy_df = healthy_df[healthy_df["Patient ID"].isin(selected_h)]

    # ── Relatório ───────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"SPLIT: {split_name.upper()}")
    print(f"{'=' * 50}")
    print(f"  Pneumonia : {len(pneumonia_df):4d} imagens, {n_pne:4d} pacientes")
    print(f"  Healthy   : {len(healthy_df):4d} imagens, {len(selected_h):4d} pacientes")

    # ── Copiar arquivos ─────────────────────────────────────
    for domain, domain_df, src_dir in [
        ("healthy",   healthy_df,   HEALTHY_SRC),
        ("pneumonia", pneumonia_df, PNEUMONIA_SRC),
    ]:
        dest_dir = OUTPUT_DIR / split_name / domain
        dest_dir.mkdir(parents=True, exist_ok=True)

        copied  = 0
        missing = 0

        for img_name in domain_df["Image Index"]:
            src = src_dir / img_name
            dst = dest_dir / img_name
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
            else:
                print(f"    [AVISO] não encontrado: {src}")
                missing += 1

        print(f"  > {domain:9s}: {copied} copiadas"
              + (f"  ({missing} nao encontradas)" if missing else ""))

# =========================================================
# RESUMO FINAL
# =========================================================

print(f"\n{'=' * 50}")
print("RESUMO FINAL")
print(f"{'=' * 50}")

for split_name in SPLITS:
    for domain in ("healthy", "pneumonia"):
        d = OUTPUT_DIR / split_name / domain
        n = len(list(d.glob("*.png"))) if d.exists() else 0
        print(f"  {split_name:5s}/{domain:9s}  {n:4d} imagens")

print("\nConcluído.")
