from copy import deepcopy
import numpy as np
import pytest

from src.study.analysis import prediction_report, summarize, calibration
from src.evaluate.generative import polynomial_mmd, nearest_neighbours


def predictions():
    return [{"id": str(i), "group": str(i // 2), "cohort": "shenzhen" if i < 4 else "montgomery",
             "label": (i // 2) % 2, "probability": .9 if (i // 2) % 2 else .1} for i in range(8)]


def test_group_bootstrap_and_calibration_are_reproducible():
    first = prediction_report(predictions(), .5, bootstrap=20)
    assert first == prediction_report(predictions(), .5, bootstrap=20)
    assert first["ci95"]["recall"] == [1., 1.]
    assert first["group_count"] == 4
    assert first["calibration"]["brier"] == pytest.approx(.01)


def test_paired_analysis_refuses_misaligned_test_sets():
    base = dict(arm="A0", budget=.25, seed=42, publication=False, status="measured", manifest_digest="fixture",
                predictions=predictions(), threshold=.5, training_ids=["train"], config={"classifier": {"updates": 2}})
    other = deepcopy(base)
    other["arm"] = "A1"
    result = summarize([base, other], bootstrap=10)
    assert result["paired"][0]["metrics"]["recall"]["mean_difference"] == 0
    assert result["paired"][0]["metrics"]["recall"]["seed_bootstrap_ci95"] is None
    other["predictions"][0]["id"] = "different"
    with pytest.raises(ValueError, match="share real"):
        summarize([base, other], bootstrap=10)


def test_nearest_neighbours_find_identical_samples():
    real = np.array([[0., 0.], [1., 1.]])
    distances, indices = nearest_neighbours(real, np.array([[1., 1.], [.1, .1]]), chunk_size=1)
    assert indices == [1, 0]
    assert distances == pytest.approx([0., .1])


def test_kid_matches_direct_unbiased_estimator():
    a, b = np.array([[0.], [1.], [2.]]), np.array([[3.], [4.]])
    expected = sum((float(x @ y) + 1) ** 3 for i, x in enumerate(a) for j, y in enumerate(a) if i != j) / 6
    expected += sum((float(x @ y) + 1) ** 3 for i, x in enumerate(b) for j, y in enumerate(b) if i != j) / 2
    expected -= 2 * sum((float(x @ y) + 1) ** 3 for x in a for y in b) / 6
    assert polynomial_mmd(a, b) == pytest.approx(expected)


def test_analysis_does_not_mix_smoke_and_publication_results():
    with pytest.raises(ValueError, match="Cannot pool"):
        summarize([{"manifest_digest": "a", "publication": True}, {"manifest_digest": "a", "publication": False}])
