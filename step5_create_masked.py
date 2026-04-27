"""
Step 5: Apply the block-letter mask to the stippled image (biased estimate).
"""

from __future__ import annotations

import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Where the mask is dark (selection region), remove stipples (set to 1.0).

    Parameters
    ----------
    stipple_img : np.ndarray
        Stipple pattern, same shape as ``mask_img``, values in [0, 1].
    mask_img : np.ndarray
        Letter mask: 0 = dark (remove data), 1 = light (keep).
    threshold : float
        Pixels with mask value strictly below this are treated as mask-on.

    Returns
    -------
    np.ndarray
        Copy of stipple image with masked regions forced to white (1.0).
    """
    if stipple_img.shape != mask_img.shape:
        raise ValueError(
            f"Shape mismatch: stipple {stipple_img.shape} vs mask {mask_img.shape}"
        )
    out = stipple_img.copy()
    remove = mask_img < threshold
    out[remove] = 1.0
    return out
