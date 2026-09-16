"""Cell segmentation (§6) and conservative touching-cell separation (§7).

Cellpose is the primary engine. It is imported lazily, inside the call, so the
rest of the package -- and the whole test suite -- runs without it installed and
without triggering a model download.

A classical threshold + watershed fallback exists for that case. It is always
reported honestly in ``engine_info``: the app never claims Cellpose ran when it
did not.

Cellpose 4.x (Cellpose-SAM) is a breaking change from 3.x -- ``models.Cellpose``
and ``SizeModel`` were removed, leaving only ``CellposeModel``, and ``eval``
wants a 3-channel image. The adapter detects which generation is installed. In
both cases the 3-channel input is built explicitly from the channel the user
selected, so the channel recorded in metadata is provably the one segmented.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import measure as skmeasure
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_dilation, remove_small_objects
from skimage.segmentation import watershed

from .morphometry import touches_border
from .types import ObjectRecord, SegmentationParams, SplitParams

#: Loading Cellpose weights is expensive, so models are cached per process.
#: This holds no per-session data and is safe to share between users.
_MODEL_CACHE: dict[tuple[str, bool], Any] = {}

#: Serialises model construction. See :func:`_get_cellpose_model` for why.
_MODEL_LOCK = threading.Lock()
# Gradio workflows share the same GPU. Concurrent inference multiplies VRAM peaks.
_INFERENCE_LOCK = threading.Lock()


#: 8-connected neighbourhood, used to find the interface between fragments.
_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


class CellposeUnavailable(RuntimeError):
    """Raised when Cellpose was requested but cannot be imported or run."""


def cellpose_available() -> bool:
    try:
        import cellpose  # noqa: F401
    except Exception:
        return False
    return True


def cellpose_version() -> str | None:
    try:
        import cellpose

        version = getattr(cellpose, "version", None)
        return str(version) if version else None
    except Exception:
        return None


def _resolve_device(use_gpu: bool) -> bool:
    # ``cellscope --cpu`` sets this so a machine with a GPU can still run on the CPU.
    if not use_gpu or os.environ.get("CELLSCOPE_FORCE_CPU") == "1":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_cellpose_model(model_name: str, use_gpu: bool):
    """Return the cached Cellpose model, loading it at most once.

    The lock is load-bearing, not defensive. The app preloads the weights in a
    background thread while the user is still uploading; without it, pressing
    **Run** before that finishes lets both threads miss the cache and each build
    a full model. Two 1.2 GB ViT models loading at once on a 4 GB card contend
    badly -- measured at 49 s to get a model, against ~6 s for a single load.
    """
    key = (model_name, use_gpu)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    with _MODEL_LOCK:
        # Re-check inside the lock: another thread may have finished while this
        # one waited, and that is the whole point of the lock.
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        from cellpose import models as cp_models

        if not hasattr(cp_models, "CellposeModel"):
            raise CellposeUnavailable(
                "Installed Cellpose exposes no CellposeModel class."
            )

        try:
            try:
                model = cp_models.CellposeModel(gpu=use_gpu, pretrained_model=model_name)
            except TypeError:
                # Cellpose 3.x keeps the class name but selects weights via model_type.
                model = cp_models.CellposeModel(gpu=use_gpu, model_type=model_name)
        except Exception as error:
            raise CellposeUnavailable(
                "Cellpose model {!r} could not be loaded: {}: {}".format(
                    model_name, type(error).__name__, error)
            ) from error

        _MODEL_CACHE[key] = model
        return model


def model_is_loaded(model_name: str = "cpsam", use_gpu: bool = True) -> bool:
    """Whether the weights are already in memory, for messaging only."""
    return (model_name, _resolve_device(use_gpu)) in _MODEL_CACHE


def _as_three_channel(image: np.ndarray, nuclear: np.ndarray | None = None) -> np.ndarray:
    """Build the 3-channel array Cellpose 4 expects.

    Building this explicitly is what keeps the recorded segmentation channel
    honest: Cellpose is never given the chance to pick a channel for us.

    With no nuclear image the chosen channel is simply replicated. With one, it
    goes into the second and third planes, which is what lets the model use
    nuclei to tell touching cells apart -- measured on the reference field, this
    is the difference between 9 detected objects and 17.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Expected a 2D single-channel image, got shape {}.".format(image.shape))
    if nuclear is None:
        return np.repeat(image[:, :, None], 3, axis=2)

    nuclear = np.asarray(nuclear, dtype=np.float32)
    if nuclear.shape != image.shape:
        raise ValueError(
            "Nuclear channel shape {} does not match the segmentation channel {}.".format(
                nuclear.shape, image.shape
            )
        )
    return np.stack([image, nuclear, nuclear], axis=-1)



def _tile_batch_for_memory(free_bytes: int, total_bytes: int) -> int:
    """Conservative tile batches to avoid Windows shared-memory spill on small GPUs."""
    gib = 1024**3
    if total_bytes <= 4.5 * gib or free_bytes < 2 * gib:
        return 1
    if total_bytes <= 8 * gib or free_bytes < 4 * gib:
        return 2
    if free_bytes < 7 * gib:
        return 4
    return 8


def _inference_tile_batch(use_gpu: bool) -> int:
    if not use_gpu:
        return 1
    try:
        import torch
        return _tile_batch_for_memory(*torch.cuda.mem_get_info())
    except Exception:
        # Missing memory telemetry should not select the highest-memory mode.
        return 1


def segment_with_cellpose(image, params, nuclear=None):
    """Run Cellpose and return ``(labels, engine_info)``."""
    if not cellpose_available():
        raise CellposeUnavailable(
            "Cellpose is not installed in this environment. Install the pinned "
            "version from requirements.txt, or use the threshold fallback."
        )

    use_gpu = _resolve_device(params.use_gpu)
    model = _get_cellpose_model(params.model, use_gpu)
    stack = _as_three_channel(image, nuclear)
    full_shape = stack.shape[:2]

    scale = _valid_scale(params.analysis_scale)
    if scale < 1.0:
        stack = _rescale_stack(stack, scale)

    kwargs = {
        "flow_threshold": params.flow_threshold,
        "cellprob_threshold": params.cellprob_threshold,
        # Cellpose sees the downscaled grid, so the area floor must be expressed
        # in that grid's pixels or it would reject objects it should keep.
        "min_size": max(1, int(params.min_object_area * scale * scale)),
        "channel_axis": 2,
    }
    if params.diameter and params.diameter > 0:
        kwargs["diameter"] = float(params.diameter) * scale

    # Tile batch size changes memory concurrency, not image resolution or weights.
    # The previous implicit default (8) spilled ~1.35 GiB into shared system RAM
    # on the 4 GiB T1200 under desktop load, causing minute-long inference.
    # Batched bfloat16 kernels can round differently, so the batch actually used
    # is recorded with the result.
    recovery: list[str] = []
    with _INFERENCE_LOCK:
        tile_batch = _inference_tile_batch(use_gpu)
        try:
            result = model.eval(stack, batch_size=tile_batch, **kwargs)
        except Exception as error:
            if not (use_gpu and _is_out_of_memory(error)):
                raise
            result, tile_batch, use_gpu = _recover_from_oom(
                model, stack, kwargs, tile_batch, params, recovery
            )
    masks = np.asarray(result[0]).astype(np.int32)

    if scale < 1.0:
        masks = _upscale_labels(masks, full_shape)

    return _relabel_sequential(masks), {
        "engine": "cellpose",
        "cellpose_version": cellpose_version(),
        "model": params.model,
        "device": "cuda" if use_gpu else "cpu",
        "tile_batch_size": tile_batch,
        "nuclear_channel": params.nuclear_channel if nuclear is not None else None,
        "analysis_scale": scale,
        "parameters": params.describe(),
        **({"oom_recovery": recovery} if recovery else {}),
    }


def _is_out_of_memory(error: BaseException) -> bool:
    try:
        import torch

        if isinstance(error, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass                        # fall back to the message, as older CUDA errors need
    return "out of memory" in str(error).lower()


def _free_cuda_memory() -> None:
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:     # freeing memory is best effort
        print("CellScope: could not free CUDA memory: {}".format(error))


def _recover_from_oom(model, stack, kwargs, tile_batch, params, recovery):
    """Retry an out-of-memory inference: smaller tile batch, then the CPU.

    Each step is recorded, because a CPU result is not bit-identical to a GPU
    one and the export must say which ran.
    """
    _free_cuda_memory()
    if tile_batch > 1:
        recovery.append("CUDA out of memory at tile batch {}; retried with 1".format(tile_batch))
        try:
            return model.eval(stack, batch_size=1, **kwargs), 1, True
        except Exception as error:
            if not _is_out_of_memory(error):
                raise
            _free_cuda_memory()
    recovery.append("CUDA out of memory; this image was segmented on the CPU")
    cpu_model = _get_cellpose_model(params.model, False)
    return cpu_model.eval(stack, batch_size=1, **kwargs), 1, False


def _valid_scale(scale) -> float:
    """Clamp the analysis scale to something Cellpose can actually run."""
    try:
        value = float(scale)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(value) or value <= 0:
        return 1.0
    return min(value, 1.0)


def _rescale_stack(stack: np.ndarray, scale: float) -> np.ndarray:
    """Downscale the 3-channel inference input. Intensities are interpolated."""
    from skimage.transform import resize

    height, width = stack.shape[:2]
    target = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    return resize(
        stack, (*target, stack.shape[2]), order=1, preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def _upscale_labels(labels: np.ndarray, shape) -> np.ndarray:
    """Return labels to the original grid.

    Nearest-neighbour only, and never anti-aliased: interpolating between
    integer IDs would blend label 3 and label 5 into a label 4 that names no
    object anyone segmented.
    """
    from skimage.transform import resize

    return resize(
        np.asarray(labels), tuple(shape), order=0, preserve_range=True,
        anti_aliasing=False,
    ).astype(np.int32)


def _relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Renumber labels to 1..N in ascending order, so IDs are reproducible."""
    labels = np.asarray(labels)
    values = [int(v) for v in np.unique(labels) if int(v) != 0]
    if not values:
        return np.zeros(labels.shape, dtype=np.int32)
    lookup = np.zeros(int(max(values)) + 1, dtype=np.int32)
    for new_id, old in enumerate(sorted(values), start=1):
        lookup[old] = new_id
    return lookup[labels.astype(np.int64)].astype(np.int32)


def _drop_small_labels(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Remove objects below ``min_area`` pixels, in one lookup-table pass."""
    labels = np.asarray(labels)
    counts = np.bincount(labels.ravel())
    keep = np.arange(len(counts), dtype=labels.dtype)
    keep[counts < min_area] = 0
    return keep[labels]


def segment_with_threshold_watershed(image, params):
    """Otsu threshold followed by a distance-transform watershed.

    A deliberately simple, dependency-light engine. It keeps the pipeline and
    its tests runnable without Cellpose, and gives a usable result on the
    well-separated bright objects of a sparse fluorescence field. It is not a
    substitute for Cellpose on crowded images, and ``engine_info`` says so.
    """
    image = np.asarray(image, dtype=np.float32)
    base_info = {"engine": "threshold_watershed", "parameters": params.describe()}

    finite = image[np.isfinite(image)]
    if finite.size == 0 or float(finite.max()) == float(finite.min()):
        return np.zeros(image.shape, dtype=np.int32), dict(
            base_info, note="image had no intensity variation; nothing segmented"
        )

    threshold = float(threshold_otsu(image))
    foreground = ndi.binary_fill_holes(image > threshold)
    min_area = max(1, int(params.min_object_area))
    foreground = remove_small_objects(foreground, min_size=min_area)

    if not foreground.any():
        return np.zeros(image.shape, dtype=np.int32), dict(
            base_info, otsu_threshold=threshold, note="no foreground survived thresholding"
        )

    distance = ndi.distance_transform_edt(foreground)
    if params.diameter and params.diameter > 0:
        min_distance = max(1, int(params.diameter / 3.0))
    else:
        min_distance = max(3, int(math.sqrt(min_area / math.pi)))

    peaks = peak_local_max(
        distance, min_distance=min_distance, labels=foreground, exclude_border=False
    )
    markers = np.zeros(foreground.shape, dtype=np.int32)
    for index, (row, col) in enumerate(peaks, start=1):
        markers[row, col] = index

    if markers.max() == 0:
        labels = skmeasure.label(foreground, connectivity=2).astype(np.int32)
    else:
        labels = watershed(-distance, markers, mask=foreground).astype(np.int32)

    labels = _drop_small_labels(labels, min_area)
    return _relabel_sequential(labels), dict(
        base_info, otsu_threshold=threshold, peak_min_distance_px=min_distance
    )


def segment_cells(image, params=None, engine: str = "auto", nuclear=None):
    """Segment an image into labelled instances (§6).

    ``engine`` is ``"cellpose"``, ``"threshold_watershed"``, or ``"auto"``
    (Cellpose when importable, otherwise the fallback). The engine that
    actually ran is always named in the returned ``engine_info`` and written to
    ``metadata.json``, so a result can never be mistaken for a Cellpose result
    it is not.
    """
    params = params or SegmentationParams()

    if engine == "cellpose":
        return segment_with_cellpose(image, params, nuclear)
    if engine == "threshold_watershed":
        return segment_with_threshold_watershed(image, params)
    if engine != "auto":
        raise ValueError("Unknown segmentation engine {!r}.".format(engine))

    # "auto" falls back only when Cellpose cannot be used at all. A Cellpose
    # failure on one image is reported as that image's error rather than
    # silently replaced by a different algorithm.
    if cellpose_available():
        try:
            return segment_with_cellpose(image, params, nuclear)
        except CellposeUnavailable as error:
            labels, info = segment_with_threshold_watershed(image, params)
            info["cellpose_error"] = "{}: {}".format(type(error).__name__, error)
            info["note"] = "Cellpose could not be loaded; fell back to threshold+watershed"
            return labels, info
    return segment_with_threshold_watershed(image, params)


# --------------------------------------------------------------------------- #
# Conservative touching-cell separation (§7)
# --------------------------------------------------------------------------- #


def _is_suspicious(prop, median_area: float, n_objects: int, params: SplitParams):
    """Does this object look like it might be more than one cell?

    Returns ``(suspicious, grossly_oversized, reason)``.

    ``suspicious`` is a review hint, nothing more. Large and concave describes
    plenty of perfectly ordinary cells -- a spread cardiomyocyte is both -- so
    on its own it is not grounds to refuse per-cell measurements. Objects
    flagged this way are still reported as cells, carrying the flag so a
    researcher can look at them.

    ``grossly_oversized`` is the stronger claim: an object several times the
    median area is not plausibly one cell, so it is withheld as an unresolved
    group even when the splitter found no seam to cut.

    Both area tests are skipped when there are too few objects for a median to
    carry information; with one object the comparison is against itself and can
    never fire.
    """
    area_testable = n_objects >= params.min_objects_for_area_test and median_area > 0

    if area_testable and prop.area > params.oversize_factor * median_area:
        return True, True, (
            "area {:.0f} px is {:.1f}x the median object area ({:.0f} px), "
            "implausible for a single cell".format(
                prop.area, prop.area / median_area, median_area
            )
        )
    if area_testable and prop.area > params.area_factor * median_area:
        return True, False, "area {:.0f} px exceeds {:.2g}x the median object area ({:.0f} px)".format(
            prop.area, params.area_factor, median_area
        )
    if prop.solidity < params.solidity_max:
        return True, False, "solidity {:.3f} is below {:.2f}".format(
            prop.solidity, params.solidity_max
        )
    return False, False, None


def _attempt_split(prop, params: SplitParams, min_fragment_area: int):
    """Try to separate one flagged object.

    Returns ``(local_labels, None, False)`` when every acceptance gate passes,
    or ``(None, reason, ambiguous)`` when one fails.

    ``ambiguous`` distinguishes the two very different ways a split can fail.
    A single maximum means there was never any evidence of a merge, and the
    object is simply one cell. But candidate fragments that were found and then
    rejected -- on a shallow neck, a sliver, a misshapen piece -- mean we saw
    signs of two cells and could not separate them confidently. That is §7's
    "separation confidence is poor", and it makes the object an
    ``unresolved_cluster`` rather than a cell with invented boundaries (§12).
    """
    mask = prop.image
    distance = ndi.distance_transform_edt(mask)
    if params.smoothing_sigma > 0:
        distance = gaussian(distance, sigma=params.smoothing_sigma, preserve_range=True)

    equivalent_radius = math.sqrt(prop.area / math.pi)
    min_peak_distance = max(2, int(params.peak_min_distance_frac * equivalent_radius))
    peaks = peak_local_max(
        distance, min_distance=min_peak_distance, labels=mask, exclude_border=False
    )
    if len(peaks) < 2:
        return None, "only one distance-transform maximum; no evidence of a merge", False

    markers = np.zeros(mask.shape, dtype=np.int32)
    for index, (row, col) in enumerate(peaks, start=1):
        markers[row, col] = index
    fragments = watershed(-distance, markers, mask=mask).astype(np.int32)

    fragment_props = skmeasure.regionprops(fragments)
    if len(fragment_props) < 2:
        return None, "watershed produced a single fragment", False

    # Gate: every fragment must be a plausible cell in its own right.
    for fragment in fragment_props:
        if fragment.area < min_fragment_area:
            return None, "fragment of {:.0f} px is below the {:.0f} px minimum".format(
                fragment.area, min_fragment_area
            ), True
        if fragment.area < params.min_fragment_area_frac * prop.area:
            return None, "sliver fragment: {:.1%} of the parent, below {:.0%}".format(
                fragment.area / prop.area, params.min_fragment_area_frac
            ), True
        if fragment.solidity < params.min_fragment_solidity:
            return None, "fragment solidity {:.3f} is below {:.2f}".format(
                fragment.solidity, params.min_fragment_solidity
            ), True

    # Gate: the neck between each touching pair must be a real constriction and
    # not a ripple in the distance transform.
    peak_height = {
        int(f.label): float(distance[fragments == f.label].max()) for f in fragment_props
    }
    fragment_labels = sorted(peak_height)
    for position, first in enumerate(fragment_labels):
        reach = binary_dilation(fragments == first, _EIGHT_CONNECTED)
        for second in fragment_labels[position + 1 :]:
            interface = reach & (fragments == second)
            if not interface.any():
                continue
            saddle = float(distance[interface].max())
            shallower_peak = min(peak_height[first], peak_height[second])
            if shallower_peak <= 0:
                return None, "degenerate distance transform at the interface", True
            depth = (shallower_peak - saddle) / shallower_peak
            if depth < params.min_saddle_depth_frac:
                return None, (
                    "neck between fragments is too shallow "
                    "(relative depth {:.2f} < {:.2f})".format(depth, params.min_saddle_depth_frac)
                ), True

    return fragments, None, False


def split_touching_cells(labels, params=None, min_object_area: int = 50):
    """Attempt conservative separation of merged objects (§7).

    A split is attempted on every object; the gates in :func:`_attempt_split`
    are what keep it conservative. The very first gate -- requiring two
    distance-transform maxima -- already rejects genuine single cells, so
    nothing is gained by pre-filtering which objects to try.

    Returns ``(labels, objects)``. Three outcomes per input object:

    * **split accepted** -- replaced by fragments, each ``'resolved'`` with
      ``split_from`` pointing at the parent's raw label.
    * **split rejected, object looks ordinary** -- kept whole and ``'resolved'``.
      A single cell has no neck to cut, and that is not a failure.
    * **split rejected, object looks suspicious** -- kept whole as
      ``'unresolved_cluster'``, with the failing gate recorded in
      ``unresolved_reason``. §12 requires this: a group we cannot separate gets
      group-level measurements only, never fabricated per-cell boundaries.

    With ``params.enabled=False`` nothing is split and everything stays
    ``resolved``, but suspicious objects are still marked
    ``flagged_suspicious`` so the diagnostic survives the choice not to act.

    ID scheme: unsplit objects keep their original label; fragments take fresh
    IDs above the highest input label, ordered by parent then centroid, so a
    rerun with identical parameters reproduces identical IDs. A split parent's
    label disappears from the output entirely, which keeps every ID
    unambiguous -- it is either the whole object or one fragment, never both.
    """
    params = params or SplitParams()
    labels = np.asarray(labels).astype(np.int32)

    props = {int(p.label): p for p in skmeasure.regionprops(labels)}
    if not props:
        return np.zeros(labels.shape, dtype=np.int32), []

    median_area = float(np.median([p.area for p in props.values()]))
    fragment_floor = max(int(min_object_area), int(params.min_fragment_area))

    output = np.zeros(labels.shape, dtype=np.int32)
    records: list[ObjectRecord] = []
    next_id = max(props) + 1

    n_objects = len(props)
    for label in sorted(props):
        prop = props[label]
        window = (slice(prop.bbox[0], prop.bbox[2]), slice(prop.bbox[1], prop.bbox[3]))
        flagged, oversized, flag_reason = _is_suspicious(
            prop, median_area, n_objects, params
        )

        if not params.enabled:
            output[window][prop.image] = label
            records.append(
                ObjectRecord(
                    object_id=label,
                    raw_label=label,
                    status="resolved",
                    flagged_suspicious=flagged,
                )
            )
            continue

        fragments, reject_reason, ambiguous = _attempt_split(prop, params, fragment_floor)

        if fragments is None:
            # A single cell with no neck to cut stays a cell -- being merely
            # large or concave describes plenty of real cells. Two things do
            # withhold per-cell values (§12): candidate fragments we could not
            # confidently separate, which is §7's "separation confidence is
            # poor"; and an object far too large to be one cell, where the
            # absence of a seam is not enough to call it single.
            unresolved = ambiguous or oversized
            output[window][prop.image] = label
            records.append(
                ObjectRecord(
                    object_id=label,
                    raw_label=label,
                    status="unresolved_cluster" if unresolved else "resolved",
                    flagged_suspicious=flagged,
                    unresolved_reason=(
                        "{}; split rejected: {}".format(
                            flag_reason or "candidate fragments found", reject_reason
                        )
                        if unresolved
                        else None
                    ),
                )
            )
            continue

        ordered = sorted(
            skmeasure.regionprops(fragments), key=lambda f: (f.centroid[0], f.centroid[1])
        )
        for fragment in ordered:
            output[window][fragments == fragment.label] = next_id
            records.append(
                ObjectRecord(
                    object_id=next_id,
                    raw_label=label,
                    status="resolved",
                    split_from=label,
                    flagged_suspicious=True,
                )
            )
            next_id += 1

    records = _annotate_border_status(output, records)
    return output, records


def _annotate_border_status(labels, records):
    """Set ``touches_border`` from the final label image, so the flag always
    matches the mask that was actually measured."""
    bboxes = {int(p.label): p.bbox for p in skmeasure.regionprops(labels)}
    updated = []
    for record in records:
        bbox = bboxes.get(record.object_id)
        on_border = touches_border(bbox, labels.shape) if bbox else False
        updated.append(
            ObjectRecord(
                object_id=record.object_id,
                raw_label=record.raw_label,
                status=record.status,
                split_from=record.split_from,
                touches_border=on_border,
                flagged_suspicious=record.flagged_suspicious,
                unresolved_reason=record.unresolved_reason,
            )
        )
    return updated
