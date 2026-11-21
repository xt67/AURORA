"""Same stratified bootstrap draws/estimators, with batched binary metrics.

Training and its archived implementation remain untouched. This postprocessing
optimization is checked against the original sklearn-per-resample analysis.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from . import analysis as original
from .data import sha256
from .publication_analysis import analyze
from .runner import write_json
from src.evaluate.metrics import compute_metrics


def prediction_report(predictions, threshold, bootstrap=1000, seed=42):
    if bootstrap < 2 or len({r['id'] for r in predictions}) != len(predictions):
        raise ValueError('At least two replicates and unique prediction IDs required')
    y = np.array([r['label'] for r in predictions])
    p = np.array([r['probability'] for r in predictions])
    metrics = compute_metrics(y, (p >= threshold).astype(int), p)
    groups, strata = defaultdict(list), defaultdict(list)
    for i, row in enumerate(predictions):
        groups[row['group']].append(i)
    for indices in groups.values():
        labels = {predictions[i]['label'] for i in indices}
        if len(labels) != 1:
            raise ValueError('Prediction duplicate group has conflicting labels')
        key = (tuple(sorted({predictions[i]['cohort'] for i in indices})), tuple(labels))
        strata[key].append(indices)
    rng = np.random.default_rng(seed)
    weights = np.zeros((bootstrap, len(y)), dtype=float)
    # Preserve the original RNG calls and their exact order, including groups.
    for replicate in range(bootstrap):
        indices = []
        for stratum in strata.values():
            for index in rng.integers(0, len(stratum), size=len(stratum)):
                indices.extend(stratum[index])
        np.add.at(weights[replicate], indices, 1)
    positive = weights[:, y == 1].sum(axis=1)
    negative = weights[:, y == 0].sum(axis=1)
    tp = weights[:, (y == 1) & (p >= threshold)].sum(axis=1)
    fp = weights[:, (y == 0) & (p >= threshold)].sum(axis=1)
    def divide(a, b, fallback=0.):
        return np.divide(a, b, out=np.full_like(a, fallback, dtype=float), where=b != 0)
    draws = {'recall': divide(tp, positive), 'specificity': divide(negative-fp, negative, np.nan),
             'f1': divide(2*tp, positive+tp+fp)}
    order = np.argsort(-p, kind='stable')
    endpoints = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(p)-1]
    cum_tp = np.cumsum(weights[:, order] * y[order], axis=1)[:, endpoints]
    cum_fp = np.cumsum(weights[:, order] * (1-y[order]), axis=1)[:, endpoints]
    tpr = divide(cum_tp, positive[:, None], np.nan)
    fpr = divide(cum_fp, negative[:, None], np.nan)
    previous_tpr = np.c_[np.zeros(bootstrap), tpr[:, :-1]]
    previous_fpr = np.c_[np.zeros(bootstrap), fpr[:, :-1]]
    draws['roc_auc'] = ((fpr-previous_fpr) * (tpr+previous_tpr) / 2).sum(axis=1)
    draws['auprc'] = (divide(cum_tp, cum_tp+cum_fp) * (tpr-previous_tpr)).sum(axis=1)
    draws['auprc'][(positive == 0) | (negative == 0)] = np.nan
    ci = {k: np.quantile(v[np.isfinite(v)], [.025, .975]).tolist() if np.isfinite(v).any() else None
          for k, v in draws.items()}
    return {'metrics': metrics, 'calibration': original.calibration(y, p), 'ci95': ci,
            'ci_method': 'percentile bootstrap of duplicate groups within cohort-composition/label strata; conditional on trained model',
            'bootstrap_repetitions': bootstrap, 'bootstrap_seed': seed, 'group_count': len(groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    original.prediction_report = prediction_report
    report = analyze(args.run, args.output)
    report['analysis_implementation'] = {
        'method': 'Identical RNG order and bootstrap estimators; batched metric arithmetic validated against original sklearn implementation',
        'source_sha256': {p: sha256(p) for p in ['src/study/fast_analysis.py', 'src/study/analysis.py', 'src/study/publication_analysis.py']}}
    write_json(Path(args.output) / 'paper_results.json', report)


if __name__ == '__main__':
    main()
