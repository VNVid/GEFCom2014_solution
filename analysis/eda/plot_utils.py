"""Shared Matplotlib setup used by all EDA figure modules."""

from __future__ import annotations

import os
from pathlib import Path

# Keep Matplotlib/fontconfig caches inside the ignored, writable project cache.
_CACHE_ROOT = Path(os.environ.get("GEFCOM_CACHE_DIR", ".cache")).resolve()
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, output: Path, dpi: int) -> None:
    fig.savefig(output, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
