"""
Step 4: Render a block letter (e.g. "S") as a grayscale mask for the meme.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    """Return a bold/truetype font at ``size``, or default bitmap font."""
    paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9,
) -> np.ndarray:
    """
    Draw a single centered letter: black (0.0) on white (1.0).

    Parameters
    ----------
    height, width : int
        Output array shape (height, width).
    letter : str
        Character to draw (first character is used).
    font_size_ratio : float
        Font size as a fraction of ``min(height, width)``.

    Returns
    -------
    np.ndarray
        2D float array in [0, 1].
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    ch = (letter or "S")[:1] or "S"

    img = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(img)
    font_size = max(12, int(min(height, width) * font_size_ratio))
    font = _load_font(font_size)

    bbox = draw.textbbox((0, 0), ch, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) / 2 - bbox[0]
    y = (height - text_h) / 2 - bbox[1]
    draw.text((x, y), ch, fill=0, font=font)

    return np.asarray(img, dtype=np.float32) / 255.0
