"""
Final step: assemble the four-panel statistics meme and save as PNG.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white",
) -> None:
    """
    Create a 1×4 figure: Reality | Your Model | Selection Bias | Estimate.

    Parameters
    ----------
    original_img, stipple_img, block_letter_img, masked_stipple_img : np.ndarray
        2D grayscale arrays in [0, 1], all the same shape.
    output_path : str
        Path to write the PNG (e.g. ``statistics_meme.png``).
    dpi : int
        Resolution for ``savefig``.
    background_color : str
        Matplotlib-compatible figure face color.
    """
    panels = [
        ("Reality", original_img),
        ("Your Model", stipple_img),
        ("Selection Bias", block_letter_img),
        ("Estimate", masked_stipple_img),
    ]
    n = len(panels)
    fig_w = 3.4 * n
    fig_h = 4.2
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), facecolor=background_color)
    if n == 1:
        axes = np.array([axes])
    for ax, (title, arr) in zip(axes, panels):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=13, fontweight="bold", color="#1a1a1a", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor("#444444")
            s.set_linewidth(1.0)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.05, wspace=0.15)
    fig.savefig(
        output_path,
        dpi=dpi,
        facecolor=background_color,
        edgecolor="none",
        bbox_inches="tight",
    )
    plt.close(fig)
