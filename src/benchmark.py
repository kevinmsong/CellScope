"""Benchmarking against manual annotation (§20).

Built and tested, but deliberately not wired into the UI: these metrics are
meaningful only against real hand-annotated fluorescence images, which the MVP
does not ship. Synthetic shape tests validate *measurement*, never segmentation
accuracy, and §20 forbids conflating the two.

Single-cell performance and touching-cell separation are reported separately,
via :func:`stratify_by_contact`, because a segmenter can be excellent at
isolated cells and poor at doublets and a single pooled F1 hides that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class InstanceMetrics:
    """Instance-level agreement between a prediction and a ground truth."""

    n_true: int
    n_pred: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    mean_dice: float
    over_segmentation_rate: float
    under_segmentation_rate: float
    count_error: int
    count_error_rate: float
    matches: list[tuple[int, int, float]] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items() if k != "matches"
        }


def _contingency(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """counts[i, j] = pixels where ground truth is i and prediction is j."""
    true_flat = np.asarray(true_labels).ravel().astype(np.int64)
    pred_flat = np.asarray(pred_labels).ravel().astype(np.int64)
    n_true = int(true_flat.max()) + 1
    n_pred = int(pred_flat.max()) + 1
    combined = true_flat * n_pred + pred_flat
    counts = np.bincount(combined, minlength=n_true * n_pred)
    return counts.reshape(n_true, n_pred)


def label_ids(labels: np.ndarray) -> np.ndarray:
    """The label values actually present, excluding background.

    Indexing by ``range(1, max + 1)`` instead would invent objects wherever the
    numbering has gaps -- and gaps are normal here, because
    :func:`~src.segmentation.split_touching_cells` retires a parent label when
    it splits it. Those phantom objects would show up as false positives.
    """
    values = np.unique(np.asarray(labels))
    return values[values != 0].astype(np.int64)


def iou_matrix(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """IoU of every ground-truth object against every predicted object.

    Rows follow ``label_ids(true_labels)`` and columns ``label_ids(pred_labels)``,
    both in ascending order. Background is excluded from both axes.
    """
    counts = _contingency(true_labels, pred_labels)
    true_ids = label_ids(true_labels)
    pred_ids = label_ids(pred_labels)
    if not len(true_ids) or not len(pred_ids):
        return np.zeros((len(true_ids), len(pred_ids)), dtype=np.float64)

    intersection = counts[np.ix_(true_ids, pred_ids)].astype(np.float64)
    true_areas = counts[true_ids, :].sum(axis=1, keepdims=True).astype(np.float64)
    pred_areas = counts[:, pred_ids].sum(axis=0, keepdims=True).astype(np.float64)
    union = true_areas + pred_areas - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


def _split_merge_rates(true_labels, pred_labels, overlap_fraction: float = 0.2):
    """Over- and under-segmentation rates.

    A ground-truth object is *over-segmented* when two or more predictions each
    cover at least ``overlap_fraction`` of it. A prediction is *under-segmented*
    (a merge) when it covers that much of two or more ground-truth objects.
    """
    counts = _contingency(true_labels, pred_labels)
    true_ids = label_ids(true_labels)
    pred_ids = label_ids(pred_labels)
    if not len(true_ids) or not len(pred_ids):
        return 0.0, 0.0

    intersection = counts[np.ix_(true_ids, pred_ids)].astype(np.float64)
    true_areas = counts[true_ids, :].sum(axis=1).astype(np.float64)
    pred_areas = counts[:, pred_ids].sum(axis=0).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        share_of_true = np.where(true_areas[:, None] > 0, intersection / true_areas[:, None], 0.0)
        share_of_pred = np.where(pred_areas[None, :] > 0, intersection / pred_areas[None, :], 0.0)

    split = int(((share_of_true >= overlap_fraction).sum(axis=1) > 1).sum())
    merged = int(((share_of_pred >= overlap_fraction).sum(axis=0) > 1).sum())

    n_true = len(true_areas)
    n_pred = len(pred_areas)
    return (
        split / n_true if n_true else 0.0,
        merged / n_pred if n_pred else 0.0,
    )


def match_instances(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    iou_threshold: float = 0.5,
    overlap_fraction: float = 0.2,
) -> InstanceMetrics:
    """One-to-one IoU matching, then the §20 instance metrics.

    Matching is optimal (Hungarian) rather than greedy, so the score does not
    depend on the order labels happen to be numbered in.
    """
    scores = iou_matrix(true_labels, pred_labels)
    true_ids = label_ids(true_labels)
    pred_ids = label_ids(pred_labels)
    n_true, n_pred = len(true_ids), len(pred_ids)

    matches: list[tuple[int, int, float]] = []
    if n_true and n_pred:
        rows, cols = linear_sum_assignment(-scores)
        for row, col in zip(rows, cols):
            if scores[row, col] >= iou_threshold:
                matches.append(
                    (int(true_ids[row]), int(pred_ids[col]), float(scores[row, col]))
                )

    true_positives = len(matches)
    false_positives = n_pred - true_positives
    false_negatives = n_true - true_positives

    precision = true_positives / n_pred if n_pred else 0.0
    recall = true_positives / n_true if n_true else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    ious = [iou for _, _, iou in matches]
    mean_iou = float(np.mean(ious)) if ious else 0.0
    # Dice and IoU are monotonically related: D = 2I / (1 + I).
    mean_dice = float(np.mean([2 * i / (1 + i) for i in ious])) if ious else 0.0

    over_rate, under_rate = _split_merge_rates(true_labels, pred_labels, overlap_fraction)

    return InstanceMetrics(
        n_true=n_true,
        n_pred=n_pred,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=mean_iou,
        mean_dice=mean_dice,
        over_segmentation_rate=over_rate,
        under_segmentation_rate=under_rate,
        count_error=n_pred - n_true,
        count_error_rate=(n_pred - n_true) / n_true if n_true else 0.0,
        matches=matches,
    )


@dataclass
class AgreementMetrics:
    """Automated versus manually corrected morphology, for one metric (§20)."""

    n: int
    mae: float
    bias: float
    pearson_r: float
    spearman_r: float
    #: Bland-Altman limits of agreement: bias +/- 1.96 SD of the differences.
    sd_difference: float
    loa_lower: float
    loa_upper: float

    def describe(self) -> dict[str, Any]:
        return dict(self.__dict__)


def compare_morphology(automated: Sequence[float], manual: Sequence[float]) -> AgreementMetrics:
    """Agreement between paired automated and manual measurements.

    Bias is ``mean(automated - manual)``: a signed systematic offset, which MAE
    alone would hide. Bland-Altman limits of agreement come with it, because a
    correlation can be near 1 while the two methods still disagree materially.
    """
    auto = np.asarray(automated, dtype=float)
    ref = np.asarray(manual, dtype=float)
    if auto.shape != ref.shape:
        raise ValueError(
            "Paired inputs must be the same length, got {} and {}.".format(
                auto.shape, ref.shape
            )
        )

    finite = np.isfinite(auto) & np.isfinite(ref)
    auto, ref = auto[finite], ref[finite]
    n = int(auto.size)
    if n == 0:
        nan = float("nan")
        return AgreementMetrics(0, nan, nan, nan, nan, nan, nan, nan)

    differences = auto - ref
    bias = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if n > 1 else float("nan")

    def _correlation(function):
        if n < 2 or np.ptp(auto) == 0 or np.ptp(ref) == 0:
            return float("nan")
        return float(function(auto, ref))

    from scipy import stats

    return AgreementMetrics(
        n=n,
        mae=float(np.mean(np.abs(differences))),
        bias=bias,
        pearson_r=_correlation(lambda a, b: stats.pearsonr(a, b)[0]),
        spearman_r=_correlation(lambda a, b: stats.spearmanr(a, b)[0]),
        sd_difference=sd,
        loa_lower=bias - 1.96 * sd if np.isfinite(sd) else float("nan"),
        loa_upper=bias + 1.96 * sd if np.isfinite(sd) else float("nan"),
    )


def stratify_by_contact(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    contact_distance_px: int = 2,
    **kwargs,
) -> dict[str, InstanceMetrics]:
    """Score isolated cells and touching cells separately (§20).

    Ground-truth objects are partitioned by whether they touch a neighbour.
    Both sides are restricted, not just the ground truth: each predicted object
    is attributed to the stratum of the ground-truth object it overlaps most, so
    a stratum is scored only against the predictions that belong to it.
    Comparing a subset of the truth against the whole prediction would count
    every other stratum's objects as false positives.

    Predictions overlapping no ground truth cannot be attributed to a stratum
    and so appear only under ``"all"``, which is the unrestricted comparison.
    """
    from .clustering import build_cell_contact_graph

    edges = build_cell_contact_graph(true_labels, contact_distance_px)
    touching = {label for edge in edges for label in edge}
    all_true = {int(v) for v in label_ids(true_labels)}
    isolated = all_true - touching

    # Attribute each prediction to the ground-truth object it overlaps most.
    counts = _contingency(true_labels, pred_labels)
    true_ids = label_ids(true_labels)
    pred_ids = label_ids(pred_labels)
    owner: dict[int, int] = {}
    if len(true_ids) and len(pred_ids):
        overlap = counts[np.ix_(true_ids, pred_ids)]
        for column, pred_id in enumerate(pred_ids):
            best = int(np.argmax(overlap[:, column]))
            if overlap[best, column] > 0:
                owner[int(pred_id)] = int(true_ids[best])

    def subset(keep: set[int]) -> InstanceMetrics:
        keep_pred = {p for p, t in owner.items() if t in keep}
        true_subset = np.where(np.isin(true_labels, list(keep) or [-1]), true_labels, 0)
        pred_subset = np.where(
            np.isin(pred_labels, list(keep_pred) or [-1]), pred_labels, 0
        )
        return match_instances(true_subset, pred_subset, **kwargs)

    return {
        "isolated": subset(isolated),
        "touching": subset(touching),
        "all": match_instances(true_labels, pred_labels, **kwargs),
    }
