"""Reproducible export (§17) and provenance metadata (§18).

Every number in the exported CSVs must be traceable back through the mask that
produced it, the raw segmentation behind that, the source pixels, and the
calibration applied. That chain is what ``metadata.json`` records, and what the
integer masks preserve.

Both label images ship: ``raw_labels.tif`` is the segmenter's own output,
``qc_labels.tif`` is what was actually measured. Keeping both is what makes a
QC decision auditable rather than merely asserted.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
import zipfile
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .morphometry import perimeter_estimator_for
from .qc import qc_log_dataframe
from .segmentation import cellpose_version
from .types import AnalysisSession, utc_now


def _mask_dtype(labels: np.ndarray):
    """Smallest integer type that preserves every ID exactly."""
    maximum = int(np.asarray(labels).max()) if np.asarray(labels).size else 0
    return np.uint16 if maximum <= np.iinfo(np.uint16).max else np.uint32


def write_mask(path: str, labels: np.ndarray) -> None:
    """Write an integer label TIFF, preserving IDs exactly (§17)."""
    import tifffile

    array = np.asarray(labels)
    tifffile.imwrite(path, array.astype(_mask_dtype(array)), compression="zlib")


def contact_graph_dataframe(edges) -> pd.DataFrame:
    """The cell contact graph as an edge list.

    Beyond §17's required files, but cheap and it makes cluster membership
    auditable rather than something the reader must take on trust (§18).
    """
    return pd.DataFrame(list(edges), columns=["object_id_a", "object_id_b"])


MANIFEST_SCHEMA = 1

#: Distributions whose versions can change a result or its presentation.
_DISTRIBUTIONS = (
    "numpy", "scipy", "scikit-image", "pandas", "Pillow", "tifffile", "cellpose",
    "torch", "gradio", "reportlab", "plotly",
)


def software_versions() -> dict[str, Any]:
    """Versions of everything that took part in producing an export.

    Read from package metadata, so nothing heavy (torch, gradio) is imported
    just to report its version. CUDA details are included only when torch is
    already loaded, which is exactly when it ran.
    """
    import importlib.metadata as metadata
    import sys

    versions: dict[str, Any] = {
        "cellscope": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in _DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            versions["cuda"] = torch.version.cuda
            versions["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                versions["gpu"] = torch.cuda.get_device_name(0)
        except Exception as error:
            versions["cuda"] = "unknown ({})".format(type(error).__name__)
    return versions


def _replace_file(source: str, target: str, attempts: int = 6) -> None:
    """Move a finished archive into place, retrying while a sync client holds it."""
    import time

    delay = 0.05
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2


def _zip_atomically(archive_path: str, members: list[tuple[str, str]]) -> None:
    """Write ``(source, name)`` pairs to a zip that appears only when complete."""
    directory = os.path.dirname(archive_path) or "."
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=".cellscope-", suffix=".tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for source, name in members:
                archive.write(source, name)
        _replace_file(temporary, archive_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image_provenance(session: AnalysisSession, results=None) -> dict[str, Any]:
    """Everything needed to say how one image's numbers were produced."""
    from .qc import exclusion_counts
    from .workflow import is_stale

    engine = dict(session.engine_info or {})
    engine.pop("parameters", None)
    return {
        "filename": session.display_name,
        "analysis_id": session.analysis_id,
        "source_image": session.image.describe() if session.image else None,
        "design": {"well": session.well, "condition": session.condition,
                   "biological_replicate": session.replicate},
        "channels": {
            "segmented": session.segmented_channel,
            "selected_now": session.channel,
            "nuclear_guide": session.segmentation_params.nuclear_channel,
            "nuclear_file": session.nuclear_image.describe() if session.nuclear_image else None,
        },
        "calibration": session.calibration.describe(),
        "parameters": {
            "preprocessing": session.preprocess_params.describe(),
            "segmentation": session.segmentation_params.describe(),
            "splitting": session.split_params.describe(),
            "clustering": session.cluster_params.describe(),
            "automatic_qc": session.qc_params.describe(),
        },
        "inference": engine,
        "segmentation_fingerprint": session.segmented_with,
        "settings_changed_since_segmentation": is_stale(session),
        "burned_in_annotation": session.annotation.describe() if session.annotation else None,
        "qc": {
            "excluded_object_ids": sorted(session.qc.excluded_ids),
            "exclusions_by_reason": exclusion_counts(session.objects, session.qc),
            "logged_actions": len(session.qc.log),
        },
        "corrections": {"edits": len(session.edit_log),
                        "operations": sorted({e.get("action", "") for e in session.edit_log})},
        "review": {"reviewed": session.reviewed, "note": session.review_note,
                   "signature": session.reviewed_signature or None},
        "counts": results.counts if results is not None else None,
        "error": session.error or None,
    }


def build_manifest(batch, per_image_dirs: dict[str, str], staging: str) -> dict[str, Any]:
    """The machine-readable description of a batch archive (``manifest.json``)."""
    from .batch import calibration_consistency
    from .pipeline import compute_results
    from .quality import AGGREGATION_RULE

    verdict = calibration_consistency(batch)
    files = []
    for root, _, names in os.walk(staging):
        for name in sorted(names):
            full = os.path.join(root, name)
            relative = os.path.relpath(full, staging).replace(os.sep, "/")
            if relative == "manifest.json":
                continue
            files.append({"path": relative, "bytes": os.path.getsize(full), "sha256": _sha256(full)})
    files.sort(key=lambda f: f["path"])
    by_session = {v: k for k, v in per_image_dirs.items()}
    images = []
    for session in batch.images:
        entry = image_provenance(
            session, compute_results(session) if session.has_segmentation else None
        )
        entry["folder"] = by_session.get(session.analysis_id)
        images.append(entry)
    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "generated_at": utc_now(),
        "software": software_versions(),
        "batch": {
            "batch_id": batch.batch_id,
            "label": batch.label,
            "created_at": batch.created_at,
            "n_images": len(batch.images),
            "n_segmented": len(batch.segmented),
            "n_reviewed": batch.n_reviewed,
            "calibration_confirmed": batch.calibration_confirmed,
            "units": "um" if verdict.get("calibrated") else "px",
            "calibration_consistent": verdict["consistent"],
        },
        "statistics": {
            "replicate_rule": AGGREGATION_RULE,
            "pooled_summary_rule": (
                "batch_summary.csv pools every accepted cell of every image; it is "
                "descriptive and does not treat cells as replicates."
            ),
            "cluster_cell_counts": (
                "A cluster containing any unresolved object has cell_count NA; "
                "no count is inferred."
            ),
            "tests": "none",
        },
        "shared_parameters": {
            "calibration": batch.calibration.describe(),
            "preprocessing": batch.preprocess_params.describe(),
            "segmentation": batch.segmentation_params.describe(),
            "splitting": batch.split_params.describe(),
            "clustering": batch.cluster_params.describe(),
            "automatic_qc": batch.qc_params.describe(),
        },
        "images": images,
        "files": files,
    }


def build_metadata(session: AnalysisSession, results) -> dict[str, Any]:
    """The full provenance record §18 asks for."""
    calibration = session.calibration
    image = session.image

    provenance = [
        {
            "object_id": obj.object_id,
            "raw_label": obj.raw_label,
            "status": obj.status,
            "split_from": obj.split_from,
            "flagged_suspicious": obj.flagged_suspicious,
            "unresolved_reason": obj.unresolved_reason,
            "touches_border": obj.touches_border,
            "touches_annotation": obj.touches_annotation,
            "included": session.qc.is_included(obj.object_id),
        }
        for obj in session.objects
    ]

    excluded_clusters = [
        c.cluster_id
        for c in results.clusters
        if all(not session.qc.is_included(m) for m in c.member_ids)
    ]

    return {
        "condition": session.condition,
        "biological_replicate": session.replicate,
        "manual_corrections": session.edit_log,
        "nuclei_settings": session.nuclei_settings,
        "analysis_id": session.analysis_id,
        "created_at": session.created_at,
        "exported_at": utc_now(),
        "software": {
            "cellscope_version": __version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "cellpose_version": cellpose_version(),
            "versions": software_versions(),
        },
        "source_image": image.describe() if image else None,
        # The channel the masks came from, not whatever is selected now.
        "segmentation_channel": session.segmented_channel,
        "selected_channel_at_export": session.channel,
        "segmentation_engine": session.engine_info,
        "preprocessing": session.preprocess_params.describe(),
        "segmentation_parameters": session.segmentation_params.describe(),
        "splitting_parameters": session.split_params.describe(),
        "clustering_parameters": session.cluster_params.describe(),
        "automatic_qc": session.qc_params.describe(),
        "burned_in_annotation": session.annotation.describe() if session.annotation else None,
        "calibration": calibration.describe(),
        "measurement": {
            "perimeter_estimator": perimeter_estimator_for(calibration),
            "units": "um" if calibration.is_calibrated else "px",
            "pixel_columns_always_present": True,
        },
        # No sub-image ROI in the MVP: the whole frame is analysed. Recorded
        # explicitly so a reader never has to wonder whether one was applied.
        "analysis_roi": None,
        "qc": {
            "excluded_object_ids": sorted(session.qc.excluded_ids),
            "excluded_cluster_ids": excluded_clusters,
            "actions": [action.describe() for action in session.qc.log],
        },
        "object_provenance": provenance,
        "counts": results.counts,
    }


#: The archive layout §17 specifies, plus contact_graph.csv.
ARCHIVE_MEMBERS = (
    "cell_measurements.csv",
    "cluster_measurements.csv",
    "image_summary.csv",
    "qc_log.csv",
    "contact_graph.csv",
    "metadata.json",
    "masks/raw_labels.tif",
    "masks/qc_labels.tif",
    "overlays/segmentation_overlay.png",
    "object_labels.tif", "edit_log.json",
)


def write_analysis_files(
    directory: str,
    session: AnalysisSession,
    results,
    overlay_rgb: np.ndarray | None = None,
) -> None:
    """Write one image's analysis into ``directory``, in the §17 layout.

    Shared by the single-image archive and by each image's folder inside a batch
    archive, so the two cannot drift apart.
    """
    os.makedirs(os.path.join(directory, "masks"), exist_ok=True)
    if session.object_labels is not None:
        write_mask(os.path.join(directory, "object_labels.tif"), session.object_labels)
    with open(os.path.join(directory, "edit_log.json"), "w", encoding="utf-8") as handle:
        json.dump(session.edit_log, handle, indent=2)
    if session.nuclei_labels is not None:
        from .nuclei import write_nuclear_files
        write_nuclear_files(session, directory)

    os.makedirs(os.path.join(directory, "overlays"), exist_ok=True)

    def path(*parts):
        return os.path.join(directory, *parts)

    results.cells.to_csv(path("cell_measurements.csv"), index=False)
    results.cluster_summary.to_csv(path("cluster_measurements.csv"), index=False)
    results.image_summary.to_csv(path("image_summary.csv"), index=False)
    qc_log_dataframe(session.qc).to_csv(path("qc_log.csv"), index=False)
    contact_graph_dataframe(results.contact_edges).to_csv(
        path("contact_graph.csv"), index=False
    )

    with open(path("metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(build_metadata(session, results), handle, indent=2, default=str)

    raw = session.raw_labels if session.raw_labels is not None else results.qc_labels
    write_mask(path("masks", "raw_labels.tif"), raw)
    write_mask(path("masks", "qc_labels.tif"), results.qc_labels)

    if overlay_rgb is not None:
        from PIL import Image

        Image.fromarray(np.asarray(overlay_rgb).astype(np.uint8)).save(
            path("overlays", "segmentation_overlay.png")
        )


def export_analysis(
    session: AnalysisSession,
    results,
    output_dir: str,
    overlay_rgb: np.ndarray | None = None,
    filename: str = "analysis.zip",
) -> str:
    """Write the full analysis archive and return its path (§17).

    Staging happens in a fresh temporary directory that is removed afterwards.
    Reusing one would let a file from an earlier export survive into a later
    archive -- an export without an overlay would silently ship the previous
    run's overlay. In the module responsible for provenance that is the one
    failure mode that must not exist.

    The archive is written under a per-analysis subdirectory so two sessions
    exporting at once cannot overwrite each other's download.
    """
    analysis_dir = os.path.join(output_dir, session.analysis_id)
    os.makedirs(analysis_dir, exist_ok=True)
    archive_path = os.path.join(analysis_dir, filename)
    with tempfile.TemporaryDirectory(prefix="cellscope_stage_") as staging:
        write_analysis_files(staging, session, results, overlay_rgb)
        members = []
        for member in ARCHIVE_MEMBERS + ("nuclei_labels.tif", "nuclei_measurements.csv", "nuclei_metadata.json"):
            source = os.path.join(staging, *member.split("/"))
            if os.path.exists(source):
                members.append((source, member))
        _zip_atomically(archive_path, members)
    return archive_path


# --------------------------------------------------------------------------- #
# Batch archive
# --------------------------------------------------------------------------- #

#: Batch-level members at the archive root. Per-image folders sit under images/.
BATCH_MEMBERS = (
    "batch_summary.csv",
    "per_image_summary.csv",
    "all_cells.csv",
    "all_clusters.csv",
    "qc_log.csv",
    "metadata.json",
    "nuclei_counts.csv", "readiness.csv", "well_summary.csv",
    "replicate_summary.csv", "condition_summary.csv", "report.pdf",
    "design_images.csv", "design_wells.csv", "design_replicates.csv",
    "design_conditions.csv", "manifest.json", "README.txt",
)


def _unique_stem(filename: str, taken: set[str]) -> str:
    """A filesystem-safe folder name, disambiguated if it is already used.

    Two images may legitimately share a filename across wells, and silently
    overwriting one with the other would lose an entire image's masks.
    """
    stem = os.path.splitext(os.path.basename(filename))[0] or "image"
    stem = "".join(c if (c.isalnum() or c in "-_ .") else "_" for c in stem).strip()
    stem = stem or "image"
    candidate, index = stem, 2
    while candidate.lower() in taken:
        candidate = "{}_{}".format(stem, index)
        index += 1
    taken.add(candidate.lower())
    return candidate


ARCHIVE_README = """CellScope batch analysis archive
================================

manifest.json           How every number was produced: software versions, device,
                        parameters, calibration, QC and corrections per image, the
                        statistical rules, and a SHA-256 for every file here.
batch_summary.csv       All accepted cells pooled (descriptive; cells are not replicates).
per_image_summary.csv   One row per image, for spotting an unusual field.
all_cells.csv           Every accepted, resolved cell, tagged with image and well.
all_clusters.csv        Every cluster. cell_count is NA when a cluster holds an
                        object that could not be resolved into cells.
design_*.csv            Image -> well -> biological replicate -> condition summaries
                        of cell area. Condition n is the number of biological replicates.
well/replicate/condition_summary.csv   The same hierarchy in the original layout.
readiness.csv           Checks that were outstanding at export time.
qc_log.csv              Every exclusion and restoration, with its reason.
nuclei_counts.csv       Independent DAPI nuclear counts (not cell counts).
report.pdf              A readable report of all of the above.
images/<name>/          Per image: masks (raw and QC-approved), overlay, measurements,
                        contact graph, edit log and metadata.json.

Measurements are in um/um2 only when the image was calibrated, otherwise px/px2.
Pixel and physical units are never mixed in one column.
"""


def build_batch_metadata(batch, per_image: dict[str, Any]) -> dict[str, Any]:
    """Provenance for the batch as a whole (§18, extended to many images)."""
    from .batch import calibration_consistency
    from .quality import AGGREGATION_RULE

    verdict = calibration_consistency(batch)
    return {
        "batch_id": batch.batch_id,
        "batch_label": batch.label,
        "created_at": batch.created_at,
        "exported_at": utc_now(),
        "software_version": __version__,
        "software": software_versions(),
        "python_version": platform.python_version(),
        "cellpose_version": cellpose_version(),
        "n_images": len(batch.images),
        "n_images_reviewed": batch.n_reviewed,
        "n_images_unreviewed": batch.n_unreviewed,
        "unreviewed_images": batch.unreviewed_names,
        "failed_images": [
            {"image": s.display_name, "error": s.error} for s in batch.failed
        ],
        # How the pooled figure was formed, so a reader need not infer it.
        "aggregation": {
            "rule": "pooled across all cells of all images",
            "weighted_by": "cell count",
            "units": "um" if verdict.get("calibrated") else "px",
            "calibration_consistent": verdict["consistent"],
            "calibration_note": verdict["reason"],
            "distinct_calibrations": verdict.get("distinct_scales", 0),
            "caveat": (
                "Cells are not independent biological replicates. A comparison "
                "between conditions must use the well or sample as the "
                "replicate unit; CellScope computes no such test."
            ),
            "replicate_rule": AGGREGATION_RULE,
        },
        "shared_parameters": {
            "calibration": batch.calibration.describe(),
            "preprocessing": batch.preprocess_params.describe(),
            "segmentation": batch.segmentation_params.describe(),
            "splitting": batch.split_params.describe(),
            "clustering": batch.cluster_params.describe(),
            "automatic_qc": batch.qc_params.describe(),
        },
        "images": per_image,
    }


def export_batch(
    batch,
    output_dir: str,
    overlays: dict[str, np.ndarray] | None = None,
    filename: str = "batch_analysis.zip",
) -> str:
    """Write the batch archive and return its path.

    Batch-level CSVs sit at the root; each image keeps a full single-image
    analysis under ``images/<stem>/`` via :func:`write_analysis_files`, so every
    number stays traceable to the mask it came from.
    """
    from .batch import (
        batch_qc_log,
        batch_summary,
        per_image_summary,
        pooled_cells,
        pooled_clusters,
        results_for,
    )

    overlays = overlays or {}
    batch_dir = os.path.join(output_dir, batch.batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    staging = tempfile.mkdtemp(prefix="cellscope_batch_")
    folders: dict[str, str] = {}

    try:
        batch_summary(batch).to_csv(os.path.join(staging, "batch_summary.csv"), index=False)
        per_image_summary(batch).to_csv(
            os.path.join(staging, "per_image_summary.csv"), index=False
        )
        pooled_cells(batch).to_csv(os.path.join(staging, "all_cells.csv"), index=False)
        pooled_clusters(batch).to_csv(os.path.join(staging, "all_clusters.csv"), index=False)
        batch_qc_log(batch).to_csv(os.path.join(staging, "qc_log.csv"), index=False)

        from .nuclei import batch_nuclear_counts
        from .quality import design_summary, experimental_summary, readiness
        from .report import write_report
        readiness(batch).to_csv(os.path.join(staging, "readiness.csv"), index=False)
        for name, table in zip(("well", "replicate", "condition"), experimental_summary(batch)):
            table.to_csv(os.path.join(staging, name + "_summary.csv"), index=False)
        design = design_summary(batch, "area")
        for level in ("images", "wells", "replicates", "conditions"):
            design[level].to_csv(os.path.join(staging, "design_{}.csv".format(level)), index=False)
        batch_nuclear_counts(batch).to_csv(os.path.join(staging, "nuclei_counts.csv"), index=False)
        write_report(batch, os.path.join(staging, "report.pdf"))

        taken: set[str] = set()
        per_image: dict[str, Any] = {}
        for session, results in results_for(batch):
            stem = _unique_stem(session.display_name, taken)
            folders["images/" + stem] = session.analysis_id
            write_analysis_files(
                os.path.join(staging, "images", stem),
                session,
                results,
                overlays.get(session.analysis_id),
            )
            per_image[stem] = {
                "filename": session.display_name,
                "analysis_id": session.analysis_id,
                "sha256": session.image.sha256 if session.image else None,
                "well": session.well,
                "condition": session.condition,
                "biological_replicate": session.replicate,
                "reviewed": session.reviewed,
                "review_note": session.review_note,
                "segmentation_channel": session.segmented_channel,
                "nuclear_channel": session.segmentation_params.nuclear_channel,
                "nuclear_file": (
                    session.nuclear_image.filename if session.nuclear_image else None
                ),
                "calibration": session.calibration.describe(),
                # Which shared defaults this image departed from, so the archive
                # never implies the batch was uniform when it was not.
                "overrides": batch.overrides_for(session),
                "cells": int(len(results.cells)),
                "error": session.error,
            }

        # DAPI-only images still retain their masks and provenance in a mixed batch.
        from .nuclei import write_nuclear_files
        for session in batch.images:
            if session.nuclei_labels is not None and not session.has_segmentation:
                stem = _unique_stem(session.display_name, taken)
                folders["images/" + stem] = session.analysis_id
                write_nuclear_files(session, os.path.join(staging, "images", stem))
                per_image[stem] = {"filename": session.display_name, "analysis_id": session.analysis_id,
                                   "well": session.well, "condition": session.condition,
                                   "biological_replicate": session.replicate, "cell_segmentation": False}

        with open(os.path.join(staging, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(build_batch_metadata(batch, per_image), handle, indent=2, default=str)

        with open(os.path.join(staging, "README.txt"), "w", encoding="utf-8") as handle:
            handle.write(ARCHIVE_README)
        with open(os.path.join(staging, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(build_manifest(batch, folders, staging), handle, indent=2, default=str)

        archive_path = os.path.join(batch_dir, filename)
        members = []
        for member in BATCH_MEMBERS:
            source = os.path.join(staging, member)
            if os.path.exists(source):
                members.append((source, member))
        images_root = os.path.join(staging, "images")
        for root, dirs, files in os.walk(images_root):
            dirs.sort()
            for name in sorted(files):
                full = os.path.join(root, name)
                members.append((full, os.path.relpath(full, staging).replace(os.sep, "/")))
        _zip_atomically(archive_path, members)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return archive_path
