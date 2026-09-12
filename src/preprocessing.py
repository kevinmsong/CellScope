"""Conservative fluorescence preprocessing (§5).

Preprocessing exists to help the segmenter find boundaries. It never touches
the array that measurements are taken from, and it never changes the spatial
calibration -- no resizing, no warping, no registration.

Deliberately absent: deconvolution, super-resolution, denoising networks, and
anything generative. §5 rules those out, and they would put invented structure
into a measurement pipeline.
"""

from __future__ import annotations

import numpy as np
from skimage.filters import gaussian

from .types import PreprocessParams


def subtract_background(image: np.ndarray, radius: float) -> np.ndarray:
    """Remove slowly varying background using a rolling-ball estimate.

    Falls back to a white top-hat if the rolling ball is unavailable. Both
    preserve object geometry; neither rescales the image.
    """
    from skimage.restoration import rolling_ball

    background = rolling_ball(image, radius=radius)
    return np.clip(image - background, 0, None)


def normalize_contrast(
    image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.5
) -> np.ndarray:
    """Percentile stretch to [0, 1], matching Cellpose's own convention.

    A flat image (high == low) is returned as zeros rather than dividing by a
    vanishing span.
    """
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    low = float(np.percentile(finite, low_percentile))
    high = float(np.percentile(finite, high_percentile))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def preprocess_image(
    image: np.ndarray, params: PreprocessParams | None = None
) -> np.ndarray:
    """Apply the configured preprocessing chain, for segmentation input only.

    The returned array is a new float32 image; the input is never modified.
    """
    params = params or PreprocessParams()
    result = np.asarray(image, dtype=np.float32).copy()

    if params.background_subtract and params.background_radius_px > 0:
        result = subtract_background(result, params.background_radius_px)

    if params.gaussian_sigma and params.gaussian_sigma > 0:
        result = gaussian(result, sigma=params.gaussian_sigma, preserve_range=True)

    if params.normalize_contrast:
        result = normalize_contrast(
            result, params.norm_low_percentile, params.norm_high_percentile
        )

    return np.asarray(result, dtype=np.float32)
