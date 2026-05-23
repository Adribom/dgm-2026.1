"""
Reproducibility helpers.

Calling `set_all_seeds(n)` at the start of any experiment makes runs
deterministic across Python's `random`, NumPy, and PyTorch (including
CUDA). This is essential because experimental claims must be replicable:
a result that disappears when you re-run it is not a result.

Usage:
    from binding.seeds import set_all_seeds
    set_all_seeds(42)
"""

from __future__ import annotations

import os
import random


def set_all_seeds(seed: int = 42) -> None:
    """
    Seed every common randomness source.

    Sets seeds for:
      - Python's `random` module
      - NumPy
      - PyTorch CPU and CUDA (if available)
      - The PYTHONHASHSEED environment variable (affects hash-based ops)

    Also configures PyTorch's cuDNN to deterministic mode. This costs
    a bit of throughput but is the right trade-off for research: speed
    is recoverable, lost determinism is not.

    Args:
        seed: integer seed. Default 42 (the conventional choice that
            keeps results comparable across runs).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy: imported lazily so seeds.py stays import-light when only
    # Python's random is needed.
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch: lazy import too, since seeds.py may be used in contexts
    # without torch (e.g., the LAION-only experiment).
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms — trades throughput for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_generator(seed: int) -> "torch.Generator":
    """
    Create a torch.Generator seeded with `seed`.

    Useful for per-call determinism in diffusion pipelines, where you
    want each generated image to use a specific seed independent of
    the global random state.

    Args:
        seed: integer seed for this specific generator.

    Returns:
        A torch.Generator on CPU (move to CUDA if needed by the caller).
    """
    import torch
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g
