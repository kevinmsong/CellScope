"""Benchmark-metric tests (§20), on synthetic ground truth."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import separated_disks

from src.benchmark import (
    compare_morphology,
    iou_matrix,
    match_instances,
    stratify_by_contact,
)


@pytest.fixture
def truth():
    return separated_disks(n=4, radius=25, gap=30)


def test_perfect_prediction_scores_one(truth):
    metrics = match_instances(truth, truth.copy())
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.mean_iou == pytest.approx(1.0)
    assert metrics.mean_dice == pytest.approx(1.0)
    assert metrics.count_error == 0


def test_relabelling_does_not_change_the_score(truth):
    """Matching is optimal, so it cannot depend on label numbering."""
    shuffled = np.zeros_like(truth)
    remap = {1: 4, 2: 3, 3: 2, 4: 1}
    for old, new in remap.items():
        shuffled[truth == old] = new
    assert match_instances(truth, shuffled).f1 == 1.0


def test_missed_object_lowers_recall_only(truth):
    prediction = truth.copy()
    prediction[prediction == 2] = 0

    metrics = match_instances(truth, prediction)
    assert metrics.true_positives == 3
    assert metrics.false_negatives == 1
    assert metrics.false_positives == 0
    assert metrics.recall == pytest.approx(0.75)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.count_error == -1


def test_spurious_object_lowers_precision_only(truth):
    prediction = truth.copy()
    prediction[5:15, 5:15] = prediction.max() + 1

    metrics = match_instances(truth, prediction)
    assert metrics.false_positives == 1
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(0.8)
    assert metrics.count_error == 1


def test_split_ground_truth_object_is_over_segmentation(truth):
    """One true cell predicted as two halves."""
    prediction = truth.copy()
    columns = np.nonzero((truth == 1).any(axis=0))[0]
    midpoint = int(columns.mean())
    prediction[:, midpoint:][truth[:, midpoint:] == 1] = prediction.max() + 1

    metrics = match_instances(truth, prediction)
    assert metrics.over_segmentation_rate == pytest.approx(0.25)
    assert metrics.under_segmentation_rate == pytest.approx(0.0)


def test_merged_prediction_is_under_segmentation(truth):
    """Two true cells predicted as one object."""
    prediction = truth.copy()
    prediction[prediction == 2] = 1

    metrics = match_instances(truth, prediction)
    assert metrics.under_segmentation_rate > 0
    assert metrics.false_negatives >= 1


def test_iou_matrix_shape_and_values(truth):
    scores = iou_matrix(truth, truth.copy())
    assert scores.shape == (4, 4)
    assert np.allclose(np.diag(scores), 1.0)
    assert np.allclose(scores - np.diag(np.diag(scores)), 0.0)


def test_iou_threshold_is_respected(truth):
    """A poor-quality match counts as a hit only under a lenient threshold."""
    prediction = np.zeros_like(truth)
    rows, cols = np.nonzero(truth == 1)
    keep = cols < np.median(cols)  # roughly half the object
    prediction[rows[keep], cols[keep]] = 1

    assert match_instances(truth, prediction, iou_threshold=0.5).true_positives == 0
    assert match_instances(truth, prediction, iou_threshold=0.3).true_positives == 1


def test_empty_prediction_scores_zero(truth):
    metrics = match_instances(truth, np.zeros_like(truth))
    assert metrics.true_positives == 0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.count_error == -4


def test_label_gaps_do_not_invent_objects(truth):
    """CellScope's own masks have gaps: a split parent's ID is retired.

    Scoring must count the objects that exist, not every integer up to the
    maximum, or those gaps would register as false positives.
    """
    from src.benchmark import label_ids

    gapped = truth.copy()
    gapped[gapped == 2] = 7  # ids become 1, 3, 4, 7

    assert list(label_ids(gapped)) == [1, 3, 4, 7]
    metrics = match_instances(gapped, gapped.copy())
    assert metrics.n_true == 4 and metrics.n_pred == 4
    assert metrics.false_positives == 0
    assert metrics.f1 == 1.0


def test_matches_report_real_label_ids(truth):
    gapped = truth.copy()
    gapped[gapped == 2] = 9
    metrics = match_instances(gapped, gapped.copy())
    assert sorted(t for t, _, _ in metrics.matches) == [1, 3, 4, 9]


def test_stratify_scores_isolated_and_touching_separately():
    """§20: a segmenter good at singles and bad at doublets must show it."""
    truth = np.zeros((80, 260), dtype=np.int32)
    truth[20:60, 10:50] = 1     # isolated
    truth[20:60, 120:160] = 2   # touching pair
    truth[20:60, 161:200] = 3

    prediction = truth.copy()
    prediction[prediction == 3] = 2  # the doublet is merged

    scores = stratify_by_contact(truth, prediction, contact_distance_px=2)
    assert scores["isolated"].f1 == pytest.approx(1.0)
    assert scores["touching"].f1 < 1.0
    assert scores["all"].f1 < 1.0


# --------------------------------------------------------------------------- #
# Automated vs manual morphology
# --------------------------------------------------------------------------- #


def test_identical_measurements_show_no_error():
    values = [10.0, 20.0, 30.0, 40.0]
    agreement = compare_morphology(values, values)
    assert agreement.mae == pytest.approx(0.0)
    assert agreement.bias == pytest.approx(0.0)
    assert agreement.pearson_r == pytest.approx(1.0)
    assert agreement.loa_lower == pytest.approx(0.0)
    assert agreement.loa_upper == pytest.approx(0.0)


def test_bias_is_signed_and_mae_is_not():
    """A systematic overestimate must be visible as a positive bias."""
    manual = [10.0, 20.0, 30.0, 40.0]
    automated = [12.0, 22.0, 32.0, 42.0]

    agreement = compare_morphology(automated, manual)
    assert agreement.bias == pytest.approx(2.0)
    assert agreement.mae == pytest.approx(2.0)
    assert agreement.sd_difference == pytest.approx(0.0)


def test_bias_cancels_where_mae_does_not():
    """The reason both are reported: symmetric errors have zero bias."""
    agreement = compare_morphology([12.0, 18.0], [10.0, 20.0])
    assert agreement.bias == pytest.approx(0.0)
    assert agreement.mae == pytest.approx(2.0)


def test_bland_altman_limits():
    automated = [11.0, 19.0, 32.0, 39.0]
    manual = [10.0, 20.0, 30.0, 40.0]
    differences = np.array(automated) - np.array(manual)

    agreement = compare_morphology(automated, manual)
    assert agreement.bias == pytest.approx(differences.mean())
    assert agreement.sd_difference == pytest.approx(differences.std(ddof=1))
    assert agreement.loa_upper == pytest.approx(
        differences.mean() + 1.96 * differences.std(ddof=1)
    )


def test_non_finite_pairs_are_dropped():
    agreement = compare_morphology([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
    assert agreement.n == 2
    assert agreement.mae == pytest.approx(0.0)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        compare_morphology([1.0, 2.0], [1.0])


def test_empty_input_returns_nan_not_zero():
    agreement = compare_morphology([], [])
    assert agreement.n == 0
    assert np.isnan(agreement.mae)
    assert np.isnan(agreement.bias)
