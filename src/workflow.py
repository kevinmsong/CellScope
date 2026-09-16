"""Where a batch stands in the analysis, and what to do next.

The interface shows this as a stepper -- Import, Calibrate, Segment, Review,
Design, Export -- plus one sentence of guidance. It is computed purely from the
batch, so it can never disagree with what the export would contain, and it is
tested without a browser.

Staleness lives here too. An image is *stale* when the settings that produced
its masks no longer match the settings it would be segmented with now. Stale
images are flagged, never silently re-run: re-segmenting discards manual review,
and only the researcher can decide that is acceptable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

StepState = Literal["done", "current", "todo", "warn", "error", "optional"]

STEP_LABELS = (
    ("import", "Import"),
    ("calibrate", "Calibrate"),
    ("segment", "Segment"),
    ("review", "Review"),
    ("design", "Design"),
    ("export", "Export"),
)


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    state: StepState
    detail: str = ""


# --------------------------------------------------------------------------- #
# Per-image status
# --------------------------------------------------------------------------- #


def segmentation_fingerprint(session) -> str:
    """Hash of every input that determines an image's masks.

    Calibration and clustering are deliberately absent: they change what is
    *measured* from a mask, not the mask, and measurements are recomputed from
    them on demand.
    """
    nuclear_file = session.nuclear_image.sha256 if session.nuclear_image is not None else None
    values = [
        session.image.sha256 if session.image is not None else None,
        session.channel,
        nuclear_file,
        session.nuclear_image_channel if nuclear_file else None,
        session.preprocess_params.describe(),
        session.segmentation_params.describe(),
        session.split_params.describe(),
    ]
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def is_stale(session) -> bool:
    """True when the masks were made with settings that have since changed."""
    if not session.has_segmentation:
        return False
    recorded = getattr(session, "segmented_with", "")
    return bool(recorded) and recorded != segmentation_fingerprint(session)


def image_state(session) -> str:
    """One word for an image: failed, pending, stale, reviewed or segmented."""
    if session.error:
        return "failed"
    if not session.has_segmentation:
        return "pending"
    if is_stale(session):
        return "stale"
    if session.reviewed:
        return "reviewed"
    return "segmented"


def design_complete(session) -> bool:
    return all(str(v).strip() for v in (session.well, session.condition, session.replicate))


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #


def _plural(n: int, word: str) -> str:
    return "{} {}{}".format(n, word, "" if n == 1 else "s")


def workflow_state(batch) -> tuple[list[Step], str]:
    """The stepper, and one sentence saying what to do next."""
    images = list(batch.images) if batch is not None else []
    n = len(images)
    if not n:
        steps = [Step(k, label, "current" if k == "import" else "todo") for k, label in STEP_LABELS]
        return steps, "Add images on the Batch tab to start an experiment."

    from .review import validate_review

    for session in images:
        validate_review(session)

    calibrated = [s for s in images if s.calibration.is_calibrated]
    confirmed = bool(getattr(batch, "calibration_confirmed", False))
    failed = [s for s in images if s.error]
    segmented = [s for s in images if s.has_segmentation and not s.error]
    pending = [s for s in images if not s.has_segmentation and not s.error]
    stale = [s for s in segmented if is_stale(s)]
    unreviewed = [s for s in segmented if not s.reviewed]
    designed = [s for s in images if design_complete(s)]
    exported = getattr(batch, "exported_token", None)
    export_current = exported is not None and exported == change_token(batch)

    steps: list[Step] = [Step("import", "Import", "done", _plural(n, "image"))]

    if len(calibrated) == n:
        steps.append(Step("calibrate", "Calibrate", "done", batch.calibration.summary()))
    elif calibrated:
        steps.append(Step("calibrate", "Calibrate", "error",
                          "{} of {} calibrated: units cannot be pooled".format(len(calibrated), n)))
    elif confirmed:
        steps.append(Step("calibrate", "Calibrate", "warn", "pixel units"))
    else:
        steps.append(Step("calibrate", "Calibrate", "todo", "not calibrated"))

    if failed:
        steps.append(Step("segment", "Segment", "error", "{} failed".format(len(failed))))
    elif stale:
        steps.append(Step("segment", "Segment", "warn", "{} stale".format(len(stale))))
    elif pending:
        steps.append(Step("segment", "Segment", "todo" if not segmented else "current",
                          "{} / {}".format(len(segmented), n)))
    else:
        steps.append(Step("segment", "Segment", "done", "{} / {}".format(n, n)))

    if not segmented:
        steps.append(Step("review", "Review", "todo"))
    elif unreviewed:
        steps.append(Step("review", "Review", "current" if not pending else "todo",
                          "{} / {}".format(len(segmented) - len(unreviewed), len(segmented))))
    else:
        steps.append(Step("review", "Review", "done", "{} / {}".format(len(segmented), n)))

    if len(designed) == n:
        steps.append(Step("design", "Design", "done", "wells, conditions, replicates"))
    elif designed:
        steps.append(Step("design", "Design", "warn", "{} of {} labelled".format(len(designed), n)))
    else:
        steps.append(Step("design", "Design", "optional", "optional"))

    if export_current:
        steps.append(Step("export", "Export", "done", "archive is current"))
    elif segmented:
        steps.append(Step("export", "Export", "todo", "changed since export" if exported else ""))
    else:
        steps.append(Step("export", "Export", "todo"))

    # One instruction, the most important first.
    if not calibrated and not confirmed:
        action = ("Calibrate on the Batch tab (a scale bar or µm/px), or confirm pixel "
                  "units, before measuring.")
    elif calibrated and len(calibrated) != n:
        action = ("Calibrate the remaining {} so the batch shares one unit."
                  .format(_plural(n - len(calibrated), "image")))
    elif failed:
        action = "{} failed to segment. Check the error on Segment and retry.".format(
            _plural(len(failed), "image"))
    elif not segmented:
        action = ("Segment: preview one representative image, adjust settings, then run "
                  "the batch.")
    elif pending:
        action = "Segment the remaining {}.".format(_plural(len(pending), "image"))
    elif stale:
        action = ("{} {} segmented with settings that have since changed. Re-run them on "
                  "Segment, or restore the settings.").format(
                      _plural(len(stale), "image"), "was" if len(stale) == 1 else "were")
    elif unreviewed:
        action = "Review {} ({} left).".format(unreviewed[0].display_name, len(unreviewed))
    elif len(designed) != n:
        action = ("Optional: assign wells, conditions and biological replicates on "
                  "Quality & design for replicate-level summaries.")
    elif not export_current:
        action = "Export the analysis archive."
    else:
        action = "Analysis complete and exported."

    # Exactly one step is current: the first unfinished required one.
    current_index = next(
        (i for i, s in enumerate(steps) if s.state in ("todo", "current")), None
    )
    steps = [
        Step(s.key, s.label,
             "current" if i == current_index else ("todo" if s.state == "current" else s.state),
             s.detail)
        for i, s in enumerate(steps)
    ]
    return steps, action


def change_token(batch) -> tuple:
    """Changes whenever anything that affects the saved or exported result changes.

    Built from monotonic counters and frozen values only, so it is cheap to
    compute and two equal tokens mean nothing relevant happened in between.
    """
    parts: list = [
        batch.label, len(batch.images), batch.calibration,
        getattr(batch, "calibration_confirmed", False),
        getattr(batch, "design_version", 0),
    ]
    for s in batch.images:
        parts.append((
            s.analysis_id, s.segmentation_version, s.qc.version, s.calibration,
            s.cluster_params, s.reviewed, s.review_note, s.well, s.condition, s.replicate,
            s.channel, s.segmentation_params, s.preprocess_params, s.split_params,
            getattr(s, "qc_params", None), s.error, len(s.edit_log),
            id(s.nuclei_labels), len(s.nuclei_excluded), len(s.nuclei_log),
            s.nuclear_image.sha256 if s.nuclear_image is not None else None,
        ))
    return tuple(parts)
