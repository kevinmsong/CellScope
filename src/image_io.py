"""Image loading, channel selection, and display downscaling (§3).

Two rules from the specification are enforced here:

* The array used for measurement is never resized and never overwritten. A
  downscaled copy exists only for browser display.
* The segmentation channel is never chosen automatically. Asking for a colour
  channel of a single-channel image raises rather than silently substituting
  something else.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
from PIL import Image

from .types import Channel, ImageRecord

#: ITU-R BT.601 luma coefficients, the standard RGB→grey weighting.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)

_CHANNEL_INDEX: dict[str, int] = {"red": 0, "green": 1, "blue": 2}

_TIFF_SUFFIXES = {".tif", ".tiff"}


def sha256_file(path: str | os.PathLike) -> str:
    """SHA-256 of the file's bytes, for provenance (§3, §18)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_pixels(path: str | os.PathLike) -> np.ndarray:
    suffix = os.path.splitext(str(path))[1].lower()
    if suffix in _TIFF_SUFFIXES:
        import tifffile

        arr = tifffile.imread(str(path))
    else:
        with Image.open(path) as im:
            # Drop alpha: it carries no fluorescence signal and would be
            # mistaken for a fourth channel.
            if im.mode in ("RGBA", "LA", "PA"):
                im = im.convert("RGB" if im.mode != "LA" else "L")
            arr = np.asarray(im)
    return np.asarray(arr)


def _normalize_shape(arr: np.ndarray) -> np.ndarray:
    """Reduce to 2D (H, W) or 3D (H, W, C) with C <= 3."""
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # A channel-first TIFF (C, H, W) with a small leading axis.
        if arr.shape[0] <= 4 and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        return arr
    raise ValueError(
        f"Unsupported image shape {arr.shape}. CellScope's MVP handles 2D "
        "single-plane images (greyscale or RGB); z-stacks and time series are "
        "out of scope."
    )


def load_image(path: str | os.PathLike) -> ImageRecord:
    """Load an image and record everything §3 requires for provenance."""
    pixels = _normalize_shape(_read_pixels(path))
    pixels = np.ascontiguousarray(pixels)
    # Freeze it: this is the array every measurement traces back to.
    pixels.flags.writeable = False

    height, width = pixels.shape[:2]
    n_channels = 1 if pixels.ndim == 2 else int(pixels.shape[2])

    return ImageRecord(
        filename=os.path.basename(str(path)),
        sha256=sha256_file(path),
        height=int(height),
        width=int(width),
        dtype=str(pixels.dtype),
        n_channels=n_channels,
        pixels=pixels,
    )


def available_channels(record: ImageRecord) -> list[Channel]:
    """Channel choices valid for this image — RGB options only for RGB images."""
    if record.n_channels >= 3:
        return ["grayscale", "red", "green", "blue"]
    return ["grayscale"]


def extract_channel(record: ImageRecord, channel: Channel) -> np.ndarray:
    """Return the chosen channel as a 2D float32 array of raw intensities.

    No rescaling is applied — this is the unmodified signal, which
    preprocessing may then adjust for segmentation purposes only.
    """
    pixels = np.asarray(record.pixels)

    if pixels.ndim == 2:
        if channel != "grayscale":
            raise ValueError(
                f"Cannot extract the {channel!r} channel from a single-channel "
                f"image ({record.filename}). Choose 'grayscale'."
            )
        return pixels.astype(np.float32, copy=True)

    if channel == "grayscale":
        return (pixels[..., :3].astype(np.float64) @ _LUMA).astype(np.float32)

    try:
        index = _CHANNEL_INDEX[channel]
    except KeyError:
        raise ValueError(f"Unknown channel {channel!r}.") from None
    if index >= pixels.shape[-1]:
        raise ValueError(
            f"Image {record.filename} has {pixels.shape[-1]} channel(s); "
            f"the {channel!r} channel does not exist."
        )
    return pixels[..., index].astype(np.float32, copy=True)


def to_display_rgb(record: ImageRecord) -> np.ndarray:
    """The original image as uint8 RGB, at full resolution."""
    pixels = np.asarray(record.pixels)
    if pixels.dtype == np.uint8:
        base = pixels
    else:
        # 16-bit or float source: stretch to 8 bit for viewing only.
        finite = pixels[np.isfinite(pixels)] if pixels.dtype.kind == "f" else pixels
        lo = float(np.min(finite)) if finite.size else 0.0
        hi = float(np.max(finite)) if finite.size else 1.0
        span = hi - lo if hi > lo else 1.0
        base = np.clip((pixels.astype(np.float64) - lo) / span * 255.0, 0, 255).astype(np.uint8)
    if base.ndim == 2:
        return np.repeat(base[:, :, None], 3, axis=2)
    if base.shape[-1] == 1:
        return np.repeat(base, 3, axis=2)
    return np.ascontiguousarray(base[..., :3])


def make_display(record: ImageRecord, max_dim: int = 1024) -> tuple[np.ndarray, float]:
    """Build a browser-sized RGB copy.

    Returns ``(rgb_uint8, display_scale)`` where ``display_scale`` converts
    display coordinates back to original pixel coordinates:
    ``original = display * display_scale``.

    This factor is the only bridge between what the user clicks and the pixel
    grid that calibration is computed on, so it is carried in session state and
    covered by :func:`src.calibration.display_to_original` tests.
    """
    rgb = to_display_rgb(record)
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return rgb, 1.0

    scale = longest / float(max_dim)
    new_w = max(1, int(round(width / scale)))
    new_h = max(1, int(round(height / scale)))
    resized = np.asarray(
        Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
    )
    return resized, scale


def describe_channel_signal(record: ImageRecord) -> dict[str, Any]:
    """Per-channel mean/max, shown in the UI to help pick the right channel.

    Single-fluorophore JPEGs are strongly dominated by one channel, but JPEG
    chroma subsampling bleeds signal into the others, so 'is this channel
    non-empty?' is not a safe test. Showing the actual numbers lets the
    researcher make the call.
    """
    pixels = np.asarray(record.pixels)
    if pixels.ndim == 2:
        return {"grayscale": {"mean": float(pixels.mean()), "max": float(pixels.max())}}
    names = ["red", "green", "blue"]
    return {
        names[i]: {
            "mean": float(pixels[..., i].mean()),
            "max": float(pixels[..., i].max()),
        }
        for i in range(min(3, pixels.shape[-1]))
    }
