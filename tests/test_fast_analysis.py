from copy import deepcopy
import numpy as np
import pytest
from src.study.analysis import prediction_report as slow
from src.study.fast_analysis import prediction_report as fast


@pytest.mark.parametrize('seed', [42, 123, 999])
@pytest.mark.parametrize('ties', [False, True])
def test_identical_bootstrap_with_groups_and_ties(seed, ties):
    rng = np.random.default_rng(seed)
    records = [dict(id=str(i), group=str(i//2), cohort='a' if i < 12 else 'b',
                    label=(i//2) % 2, probability=float(rng.uniform())) for i in range(24)]
    if ties:
        for r in records:
            r['probability'] = round(r['probability'], 1)
    a, b = slow(records, .5, 50, seed), fast(records, .5, 50, seed)
    assert a['metrics'] == b['metrics'] and a['calibration'] == b['calibration']
    for key in a['ci95']:
        assert b['ci95'][key] == pytest.approx(a['ci95'][key], abs=1e-12)


@pytest.mark.parametrize('label', [0, 1])
def test_single_class_matches_undefined_metrics(label):
    records = [dict(id=str(i), group=str(i), cohort='a', label=label, probability=i/10) for i in range(5)]
    a, b = slow(records, .5, 20), fast(records, .5, 20)
    for key in a['ci95']:
        if a['ci95'][key] is None:
            assert b['ci95'][key] is None
        else:
            assert b['ci95'][key] == pytest.approx(a['ci95'][key], abs=1e-12)
