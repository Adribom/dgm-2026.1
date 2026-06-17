"""
I/O utilities shared by all experiments.

Centralizes:
  - Loading YAML configs with sensible defaults.
  - Saving structured outputs (results + metadata) in a uniform layout
    so that downstream notebooks can find them without surprises.

Conventions:
  - Configs are YAML files under `configs/`.
  - Results live under `results/<experiment>/run_<timestamp>/`.
  - Every result directory contains a `metadata.json` with the exact
    config used, plus library versions and a UTC timestamp. This is
    what makes a result reproducible: without it, a number is just a
    number.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict. Raises if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _git_commit() -> str | None:
    """Return the current git HEAD short hash, or None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _library_versions() -> dict[str, str]:
    """Capture versions of libraries that affect reproducibility."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for lib in ("numpy", "torch", "transformers", "diffusers", "datasets"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not-installed"
    return versions


def make_run_dir(
    experiment_name: str,
    results_root: str | Path = "results",
    timestamp: str | None = None,
) -> Path:
    """
    Create a fresh, timestamped output directory for an experiment run.

    Layout: <results_root>/<experiment_name>/run_<UTC timestamp>/

    Args:
        experiment_name: identifier like "exp3_eval_generation".
        results_root: parent directory for all results.
        timestamp: optional pre-computed timestamp string; if None, use now (UTC).

    Returns:
        Path to the newly created run directory.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = Path(results_root) / experiment_name / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(
    run_dir: Path,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """
    Persist the configuration plus environment fingerprint for this run.

    The resulting `metadata.json` answers, months later, the question
    "what generated this number?" without ambiguity:
      - config used (exact YAML, expanded)
      - library versions
      - git commit (if available)
      - UTC timestamp

    Args:
        run_dir: directory created by `make_run_dir`.
        config: the loaded config dict.
        extra: optional extra fields to embed (e.g., device info).

    Returns:
        Path to the written metadata.json.
    """
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "libraries": _library_versions(),
        "platform": platform.platform(),
        "config": config,
    }
    if extra:
        metadata.update(extra)

    path = run_dir / "metadata.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return path


def hash_file(path: str | Path, algo: str = "sha256") -> str:
    """
    Compute a hex digest of a file's contents. Useful for image checksums
    in generation metadata, so that "same seed + same model" produces a
    file you can verify byte-for-byte across machines.
    """
    h = hashlib.new(algo)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
