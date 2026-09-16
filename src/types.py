"""Core data model.

The guiding rule of CellScope — *never report more biological resolution than the
segmentation supports* — is enforced here structurally rather than by convention.

Two invariants carry most of that weight:

1. ``ObjectRecord.status`` separates objects that earned single-cell morphology
   (``resolved``) from objects that did not (``unresolved_cluster``). Only
   ``resolved`` objects ever reach ``cell_measurements.csv``.
2. ``ClusterRecord.cell_count`` is ``None`` (exported as NA) whenever a cluster
   contains any unresolved object. We do not know how many cells are inside a
   blob we could not split, so we do not report a number.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

Channel = Literal["grayscale", "red", "green", "blue"]
CalibrationMethod = Literal[
    "uncalibrated",
    "scale_bar",
    "reference_scale_bar",
    "manual_resolution",
]
ObjectStatus = Literal["resolved", "unresolved_cluster"]
ClusterStatus = Literal[
    "isolated_cell",
    "resolved_cluster",
    "partially_resolved_cluster",
    "unresolved_cluster",
]

#: Perimeter is the one metric that cannot be rescaled by a scalar under
#: anisotropic pixels. Which estimator was used is recorded in metadata.
PerimeterEstimator = Literal["crofton_scaled", "contour_scaled"]


def utc_now() -> str:
    """ISO-8601 UTC timestamp, used for every log entry and for metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class ImageRecord:
    """An immutable record of the source image.

    ``pixels`` is the original, unmodified array that every measurement is
    ultimately derived from. It is held read-only (``flags.writeable = False``)
    so that a bug downstream cannot silently alter the measurement source.
    """

    filename: str
    sha256: str
    height: int
    width: int
    dtype: str
    n_channels: int
    pixels: np.ndarray

    @property
    def is_rgb(self) -> bool:
        return self.n_channels >= 3

    def describe(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "height": self.height,
            "width": self.width,
            "dtype": self.dtype,
            "n_channels": self.n_channels,
        }


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Calibration:
    """Pixel-to-physical mapping.

    In uncalibrated mode both scales are ``None``. Callers must then emit
    pixel-space columns only — never a zero, never a NaN standing in for a
    micron value, because either could be mistaken for a measurement.
    """

    method: CalibrationMethod = "uncalibrated"
    um_per_px_x: float | None = None
    um_per_px_y: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def is_calibrated(self) -> bool:
        return self.um_per_px_x is not None and self.um_per_px_y is not None

    @property
    def is_isotropic(self) -> bool:
        """True when pixels are square (or when uncalibrated, where px are square
        by definition)."""
        if not self.is_calibrated:
            return True
        return abs(float(self.um_per_px_x) - float(self.um_per_px_y)) < 1e-12

    @property
    def scales(self) -> tuple[float, float]:
        """(sx, sy) in µm/px, or (1.0, 1.0) when uncalibrated (i.e. pixel units)."""
        if not self.is_calibrated:
            return (1.0, 1.0)
        return (float(self.um_per_px_x), float(self.um_per_px_y))

    def describe(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "um_per_px_x": self.um_per_px_x,
            "um_per_px_y": self.um_per_px_y,
            "is_isotropic": self.is_isotropic,
            "params": dict(self.params),
        }

    def summary(self) -> str:
        """A short human-readable description, for the status strip.

        The warning about uncalibrated results does not live here -- it is
        carried by the absence of micron columns and by the note on the
        calibration panel, so this can stay quiet and legible.
        """
        if not self.is_calibrated:
            return "Not calibrated · px units"
        sx, sy = self.scales
        method = _CALIBRATION_LABELS.get(self.method, self.method)
        if self.is_isotropic:
            return f"{sx:.6g} µm/px · {method}"
        return f"{sx:.6g} × {sy:.6g} µm/px · {method}"


#: How each calibration method reads in the interface.
_CALIBRATION_LABELS = {
    "scale_bar": "scale bar",
    "reference_scale_bar": "reference frame",
    "manual_resolution": "entered",
}

UNCALIBRATED = Calibration()


# --------------------------------------------------------------------------- #
# Parameters (all recorded verbatim in metadata.json)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PreprocessParams:
    """Conservative fluorescence preprocessing.

    Output feeds segmentation only; measurements always read the original image.
    """

    background_subtract: bool = False
    background_radius_px: float = 50.0
    gaussian_sigma: float = 1.0
    normalize_contrast: bool = True
    norm_low_percentile: float = 1.0
    norm_high_percentile: float = 99.5

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationParams:
    """Cellpose controls (§6). ``model`` is passed to CellposeModel."""

    model: str = "cpsam"
    #: Optional second channel carrying nuclei. Cellpose 4 takes a 3-channel
    #: image, and giving it the nuclear signal alongside the cytoplasmic one
    #: markedly improves separation of touching cells: on the reference
    #: cardiomyocyte field it took detection from 9 objects to 17, against 16
    #: nuclei counted independently in the DAPI channel.
    nuclear_channel: str | None = None
    diameter: float | None = None
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    min_object_area: int = 50
    use_gpu: bool = True
    #: Factor the image is downscaled by *for inference only*. Masks always come
    #: back at original resolution, so measurement and calibration are untouched.
    #: The default of 1.0 changes nothing; below 1 it trades accuracy for speed
    #: and is recorded in metadata so a downscaled result is never mistaken for
    #: a full-resolution one. Measured on the reference field: 0.75 gives 1.36x
    #: at F1 0.971, 0.50 gives 3.1x at F1 0.909.
    analysis_scale: float = 1.0

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitParams:
    """Conservative touching-cell separation (§7).

    A split is attempted on every object and accepted only if every gate
    passes. The first gate -- requiring two distance-transform maxima -- already
    rejects genuine single cells outright (measured: disks and both 2.5:1 and
    4:1 ellipses each produce exactly one maximum), so the gates, not a
    pre-filter, are what keep the splitting conservative.

    The suspicion thresholds decide what a *failed* split means. A suspicious
    object that cannot be separated becomes an ``unresolved_cluster`` rather
    than receiving invented boundaries; an ordinary-looking object that simply
    has no neck to cut stays a resolved single cell.
    """

    enabled: bool = True

    # --- suspicion: does a *failure* to split mean the object is unresolved? ---
    #: Measured on synthetic doublets of two r=40 disks: a blatant merge has
    #: solidity 0.921-0.948, while single disks and 2.5:1 / 4:1 ellipses all sit
    #: at 0.98-0.99. 0.95 separates them with margin.
    solidity_max: float = 0.95
    area_factor: float = 1.8              # area > factor x median area
    #: A separate, much higher bar. Being merely large or concave is not
    #: evidence of a merge -- real cardiomyocytes are both. But an object this
    #: many times the median area is implausible as one cell, so it is held as
    #: an unresolved group even when no neck was found to cut. Below this, an
    #: object the splitter found no seam in is reported as the single cell it
    #: appears to be, carrying ``flagged_suspicious`` for review.
    oversize_factor: float = 3.0
    #: The median-area test is meaningless with only one or two objects, where
    #: an object is compared against itself. Below this count it is skipped.
    min_objects_for_area_test: int = 3

    # --- acceptance gates ---
    smoothing_sigma: float = 1.0
    peak_min_distance_frac: float = 0.4   # x equivalent radius of the parent
    min_fragment_area: int = 50
    min_fragment_area_frac: float = 0.15  # x parent area; kills slivers
    #: Relative depth of the neck between two fragments. Measured on doublets of
    #: two r=40 disks: overlap 10 px -> 0.50, 20 px -> 0.32, 35 px -> 0.15,
    #: 50 px -> 0.05. A visually obvious doublet clears 0.3; a nearly fused blob
    #: sits below 0.15. 0.25 splits the obvious cases and refuses the ambiguous.
    min_saddle_depth_frac: float = 0.25
    min_fragment_solidity: float = 0.80

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClusterParams:
    """Contact-graph construction (§8). Membership is mask-based, never centroid-based."""

    contact_distance_px: int = 2

    def describe(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Objects and clusters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ObjectRecord:
    """One segmented object.

    ``included`` is deliberately absent: QC verdicts live in :class:`QCState` so
    there is exactly one source of truth and the raw segmentation is never
    edited in place.
    """

    object_id: int
    raw_label: int | None
    status: ObjectStatus
    split_from: int | None = None
    touches_border: bool = False
    #: True when the object looked like a possible merge (oversized, or low
    #: solidity). Recorded even when splitting is disabled, so the diagnostic
    #: survives the user's choice not to act on it.
    flagged_suspicious: bool = False
    #: Why an object ended up ``unresolved_cluster`` — surfaced in the UI/export
    #: so the researcher can see which gate rejected the split.
    unresolved_reason: str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.status == "resolved"

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClusterRecord:
    """One connected component of the cell contact graph."""

    cluster_id: int
    member_ids: tuple[int, ...]
    resolved_ids: tuple[int, ...]
    unresolved_ids: tuple[int, ...]

    @property
    def n_resolved_cells(self) -> int:
        return len(self.resolved_ids)

    @property
    def n_unresolved_objects(self) -> int:
        return len(self.unresolved_ids)

    @property
    def status(self) -> ClusterStatus:
        n_res, n_unres = self.n_resolved_cells, self.n_unresolved_objects
        if n_unres and not n_res:
            return "unresolved_cluster"
        if n_unres:
            return "partially_resolved_cluster"
        return "isolated_cell" if n_res == 1 else "resolved_cluster"

    @property
    def cell_count(self) -> int | None:
        """Number of cells, or ``None`` (exported as NA) when unknowable.

        §12: if any member could not be resolved into cells, the cell count of
        this cluster is genuinely unknown and must be reported as such.
        """
        if self.n_unresolved_objects:
            return None
        return self.n_resolved_cells

    @property
    def cluster_size(self) -> int:
        """Total member objects, resolved or not."""
        return len(self.member_ids)


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QCAction:
    """One entry of the QC log (§13)."""

    object_id: int | None      # None for bulk actions that matched nothing specific
    action: str                # exclude | restore
    reason: str
    scope: str = "object"      # object | cluster | filter
    timestamp: str = field(default_factory=utc_now)

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QCState:
    """Mutable QC verdicts. The raw label image is never modified."""

    excluded_ids: set[int] = field(default_factory=set)
    log: list[QCAction] = field(default_factory=list)
    #: Bumped by every mutation. Anything derived from this state keys its cache
    #: on the version, so a stale result is not reachable -- which matters more
    #: here than the speed the cache buys.
    version: int = 0

    def exclude(self, object_id: int, reason: str, scope: str = "object") -> bool:
        """Exclude one object. Returns True if this changed anything."""
        if object_id in self.excluded_ids:
            return False
        self.excluded_ids.add(int(object_id))
        self.log.append(QCAction(int(object_id), "exclude", reason, scope))
        self.version += 1
        return True

    def restore(self, object_id: int, reason: str = "manual restore", scope: str = "object") -> bool:
        """Restore a previously excluded object. Returns True if this changed anything."""
        if object_id not in self.excluded_ids:
            return False
        self.excluded_ids.discard(int(object_id))
        self.log.append(QCAction(int(object_id), "restore", reason, scope))
        self.version += 1
        return True

    def is_included(self, object_id: int) -> bool:
        return int(object_id) not in self.excluded_ids

    def reset(self) -> None:
        self.excluded_ids.clear()
        self.log.clear()
        self.version += 1


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #


@dataclass
class AnalysisSession:
    """Everything belonging to one image in one browser session.

    Held in ``gr.State`` so concurrent users never share analysis data.
    """

    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=utc_now)

    image: ImageRecord | None = None
    display_rgb: np.ndarray | None = None
    display_scale: float = 1.0

    channel: Channel = "grayscale"
    calibration: Calibration = UNCALIBRATED

    preprocess_params: PreprocessParams = field(default_factory=PreprocessParams)
    segmentation_params: SegmentationParams = field(default_factory=SegmentationParams)
    split_params: SplitParams = field(default_factory=SplitParams)
    cluster_params: ClusterParams = field(default_factory=ClusterParams)

    #: Cellpose output, before splitting. Never mutated after assignment.
    raw_labels: np.ndarray | None = None
    #: Post-split labels — the ID space that qc_labels.tif and the CSVs use.
    object_labels: np.ndarray | None = None
    objects: list[ObjectRecord] = field(default_factory=list)
    engine_info: dict[str, Any] = field(default_factory=dict)
    #: Bumped by each segmentation run; part of the derived-results cache key.
    segmentation_version: int = 0
    #: The channel the current masks were actually produced from, captured when
    #: segmentation ran. ``channel`` above is only the current *selection*, and
    #: the two differ the moment someone changes the radio without re-running.
    #: Metadata must report this one, or it would claim a provenance that is not
    #: true of the masks it ships with.
    segmented_channel: Channel | None = None

    #: (key, AnalysisResults) for the last computed results. Never read without
    #: an exact key match -- see :func:`src.pipeline.compute_results`.
    results_cache: Any = field(default=None, repr=False, compare=False)
    #: Per-object and per-cluster geometry, reusable across QC edits of the
    #: same mask. Owned by :func:`src.pipeline.compute_results`; never saved.
    geometry_cache: Any = field(default=None, repr=False, compare=False)
    #: Derived display data (overlay layers, tables). Keyed like the results;
    #: never saved.
    view_cache: dict = field(default_factory=dict, repr=False, compare=False)

    qc: QCState = field(default_factory=QCState)

    #: Pending scale-bar clicks, in ORIGINAL image pixel coordinates.
    scale_bar_points: list[tuple[float, float]] = field(default_factory=list)

    #: A separate frame carrying a burned-in scale bar, captured at the same
    #: magnification. Microscope software often exports the bar this way rather
    #: than drawing it on the data frame.
    reference_image: ImageRecord | None = None
    #: What automatic scale-bar detection proposed for that frame. Advisory
    #: only -- the researcher confirms it before it becomes a calibration.
    reference_detection: Any = None

    # -- batch membership ------------------------------------------------ #

    #: Free-text well label, assigned by hand. Pure provenance: it travels into
    #: the exported CSVs and metadata and changes no measurement. Results are
    #: pooled across the whole batch, not grouped by this.
    well: str = ""
    #: Ticked off in the Review tab. A soft gate -- results still compute, but
    #: the count of unreviewed images is shown and recorded.
    reviewed: bool = False
    review_note: str = ""
    condition: str = ""
    replicate: str = ""
    nuclei_labels: Any = None
    nuclei_excluded: set = field(default_factory=set)
    nuclei_settings: dict = field(default_factory=dict)
    nuclei_log: list = field(default_factory=list)
    selected_object: int = 0
    review_alpha: float = 0.28
    review_original: bool = False
    review_zoom: float = 1.0
    review_pan_x: float = 50.0
    review_pan_y: float = 50.0
    review_mode: str = "Toggle inclusion"
    correction_points: list = field(default_factory=list)
    undo_stack: list = field(default_factory=list, repr=False)
    redo_stack: list = field(default_factory=list, repr=False)
    edit_log: list = field(default_factory=list)
    reviewed_signature: str = ""
    #: A nuclear signal held in a *separate file* rather than a channel of this
    #: image. Guides segmentation only; it is never measured.
    nuclear_image: ImageRecord | None = None
    #: Which of ``nuclear_image``'s channels to read.
    nuclear_image_channel: Channel = "grayscale"
    #: Message from the last failed segmentation of this image. One image
    #: failing must not abort a batch, so the error is recorded here and
    #: surfaced in the image table rather than raised.
    error: str = ""

    # -- derived -------------------------------------------------------- #

    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def has_segmentation(self) -> bool:
        return self.object_labels is not None

    def included_objects(self) -> list[ObjectRecord]:
        return [o for o in self.objects if self.qc.is_included(o.object_id)]

    def included_resolved_objects(self) -> list[ObjectRecord]:
        return [o for o in self.included_objects() if o.is_resolved]

    def object_by_id(self, object_id: int) -> ObjectRecord | None:
        for o in self.objects:
            if o.object_id == int(object_id):
                return o
        return None

    @property
    def display_name(self) -> str:
        return self.image.filename if self.image else "(no image)"


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #


@dataclass
class BatchSession:
    """A batch of images analysed together.

    A batch is the images of one experiment, spanning its wells. Results are
    pooled across the whole batch into a single population -- ``well`` on each
    image is a label for traceability, not a grouping that changes any number.

    Shared parameters here are the *defaults* new images inherit. An image may
    depart from them, and :meth:`overrides_for` reports which, so the export can
    say so rather than implying the batch was uniform.
    """

    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    created_at: str = field(default_factory=utc_now)

    images: list[AnalysisSession] = field(default_factory=list)
    active_index: int = 0

    #: Channel choices for the batch. Set once and applied to every image,
    #: including ones added later -- fields in one experiment almost always share
    #: a stain layout, and per-image selection made it far too easy to segment
    #: eleven of twelve images on the wrong channel. An individual image can
    #: still be overridden from the image table.
    channel: Channel = "grayscale"
    nuclear_channel: str = "none"

    calibration: Calibration = UNCALIBRATED
    preprocess_params: PreprocessParams = field(default_factory=PreprocessParams)
    segmentation_params: SegmentationParams = field(default_factory=SegmentationParams)
    split_params: SplitParams = field(default_factory=SplitParams)
    cluster_params: ClusterParams = field(default_factory=ClusterParams)

    #: Batch-level derived tables (pooled cells, summaries), keyed on every
    #: image's results key. Never saved.
    derived_cache: dict = field(default_factory=dict, repr=False, compare=False)

    #: Parameter attributes an image may diverge from the batch on.
    SHARED_PARAMS = (
        "calibration",
        "preprocess_params",
        "segmentation_params",
        "split_params",
        "cluster_params",
    )

    # -- membership ------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.images)

    @property
    def is_empty(self) -> bool:
        return not self.images

    @property
    def active(self) -> AnalysisSession | None:
        if not self.images:
            return None
        index = min(max(self.active_index, 0), len(self.images) - 1)
        return self.images[index]

    def add(self, session: AnalysisSession) -> AnalysisSession:
        """Add an image, seeded with the batch's shared defaults and channels.

        Inheriting the channels is what makes uploading in two goes safe: images
        added later arrive on the same channel as the ones already there, rather
        than silently defaulting to greyscale.
        """
        for name in self.SHARED_PARAMS:
            setattr(session, name, getattr(self, name))
        self.images.append(session)
        return session

    def apply_channels(self, session: AnalysisSession) -> None:
        """Push the batch's channel choices onto one image, where they apply.

        A channel the image does not have is skipped rather than forced, so a
        greyscale frame in an RGB batch keeps its own.
        """
        from .image_io import available_channels

        if session.image is None:
            return
        available = available_channels(session.image)
        if self.channel in available:
            session.channel = self.channel
        nuclear = None if self.nuclear_channel in (None, "", "none") else self.nuclear_channel
        if nuclear is None or nuclear in available:
            session.segmentation_params = replace(
                session.segmentation_params, nuclear_channel=nuclear
            )

    def remove(self, index: int) -> AnalysisSession | None:
        """Drop an image from the batch entirely.

        Removal rather than an exclusion flag: an image that should not count
        is simply not part of the batch, which keeps one source of truth for
        what was analysed.
        """
        if not 0 <= index < len(self.images):
            return None
        removed = self.images.pop(index)
        self.active_index = min(self.active_index, max(0, len(self.images) - 1))
        return removed

    def apply_shared(self, *names: str) -> int:
        """Push the batch defaults onto every image. Returns how many changed.

        Used when calibration or parameters are set at the batch level after
        images are already loaded.
        """
        targets = names or self.SHARED_PARAMS
        changed = 0
        for session in self.images:
            for name in targets:
                if getattr(session, name) != getattr(self, name):
                    setattr(session, name, getattr(self, name))
                    session.results_cache = None
                    changed += 1
        return changed

    def overrides_for(self, session: AnalysisSession) -> list[str]:
        """Which shared parameters this image departs from."""
        return [
            name
            for name in self.SHARED_PARAMS
            if getattr(session, name) != getattr(self, name)
        ]

    # -- review state ---------------------------------------------------- #

    @property
    def segmented(self) -> list[AnalysisSession]:
        return [s for s in self.images if s.has_segmentation]

    @property
    def n_reviewed(self) -> int:
        from .review import validate_review
        for session in self.images:
            validate_review(session)
        return sum(1 for s in self.images if s.reviewed)

    @property
    def n_unreviewed(self) -> int:
        return len(self.images) - self.n_reviewed

    @property
    def unreviewed_names(self) -> list[str]:
        return [s.display_name for s in self.images if not s.reviewed]

    @property
    def failed(self) -> list[AnalysisSession]:
        return [s for s in self.images if s.error]
