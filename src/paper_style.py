"""Small local fallback for the Stage-1 sampling script.

The upstream script imports ``paper_style`` from a user-local Claude skills
directory. That directory is not present on this machine, so this module keeps
the sample script runnable without changing its public behavior.
"""

from __future__ import annotations


def setup_paper_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.8,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
        }
    )
