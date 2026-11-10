"""Recompute metrics, grouped bootstrap intervals and paired seed summaries."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from src.evaluate.metrics import compute_metrics
from .runner import write_json

METRICS = ("recall", "auprc", "f1", "specificity", "roc_auc")


def calibration(labels, probabilities, bins=10):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    ece = 0.
    for index in range(bins):
        mask = np.minimum((probabilities * bins).astype(int), bins - 1) == index
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return {"brier": float(np.mean((probabilities - labels) ** 2)), "ece_10_bins": float(ece)}


def prediction_report(predictions, threshold, bootstrap=1000, seed=42):
    if bootstrap < 2:
        raise ValueError("At least two bootstrap repetitions required")
    if len({r["id"] for r in predictions}) != len(predictions):
        raise ValueError("Repeated prediction IDs")
    y = np.array([r["label"] for r in predictions])
    p = np.array([r["probability"] for r in predictions])
    metrics = compute_metrics(y, (p >= threshold).astype(int), p)
    groups = defaultdict(list)
    for i, row in enumerate(predictions):
        groups[row["group"]].append(i)
    strata = defaultdict(list)
    for indices in groups.values():
        labels = {predictions[i]["label"] for i in indices}
        if len(labels) != 1:
            raise ValueError("Prediction duplicate group has conflicting labels")
        key = (tuple(sorted({predictions[i]["cohort"] for i in indices})), tuple(labels))
        strata[key].append(indices)
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in METRICS}
    for _ in range(bootstrap):
        indices = []
        for stratum in strata.values():
            for index in rng.integers(0, len(stratum), size=len(stratum)):
                indices.extend(stratum[index])
        score = compute_metrics(y[indices], (p[indices] >= threshold).astype(int), p[indices])
        for metric in METRICS:
            if score[metric] is not None:
                draws[metric].append(score[metric])
    return {"metrics": metrics, "calibration": calibration(y, p),
            "ci95": {key: np.quantile(values, [.025, .975]).tolist() if values else None for key, values in draws.items()},
            "ci_method": "percentile bootstrap of duplicate groups within cohort-composition/label strata; conditional on trained model",
            "bootstrap_repetitions": bootstrap, "bootstrap_seed": seed, "group_count": len(groups)}


def summarize(results, bootstrap=1000, baseline="A0"):
    if not results:
        raise ValueError("No result artifacts provided")
    if len({r["manifest_digest"] for r in results}) != 1 or len({r["publication"] for r in results}) != 1:
        raise ValueError("Cannot pool different manifests or publication/smoke results")
    by_key = {}
    groups = defaultdict(list)
    per_run = []
    for run in results:
        if run["status"] != "measured":
            raise ValueError("Only completed measured runs can be summarized")
        key = (run["arm"], run["budget"], run["seed"])
        if key in by_key:
            raise ValueError("Duplicate arm/budget/seed result")
        by_key[key] = run
        report = prediction_report(run["predictions"], run["threshold"], bootstrap, run["seed"])
        cohort_reports = {}
        for cohort in sorted({p["cohort"] for p in run["predictions"]}):
            cohort_reports[cohort] = prediction_report([p for p in run["predictions"] if p["cohort"] == cohort], run["threshold"], bootstrap, run["seed"])
        per_run.append({"arm": key[0], "budget": key[1], "seed": key[2], **report, "cohorts": cohort_reports})
        groups[key[:2]].append(report["metrics"])
    aggregate = []
    for (arm, budget), scores in sorted(groups.items()):
        summary = {}
        for metric in METRICS:
            values = [r[metric] for r in scores if r[metric] is not None]
            summary[metric] = {"mean": float(np.mean(values)) if values else None,
                               "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                               "n": len(values)}
        aggregate.append({"arm": arm, "budget": budget, "metrics": summary})
    paired = []
    for arm, budget in sorted(groups):
        if arm == baseline:
            continue
        deltas = defaultdict(list)
        paired_seeds = []
        for (candidate_arm, candidate_budget, seed), run in by_key.items():
            if (candidate_arm, candidate_budget) != (arm, budget) or (baseline, budget, seed) not in by_key:
                continue
            base = by_key[(baseline, budget, seed)]
            signature = lambda x: sorted((p["id"], p["label"], p["cohort"], p["group"]) for p in x["predictions"])
            if signature(base) != signature(run) or base["training_ids"] != run["training_ids"]:
                raise ValueError("Paired runs must share real training budget and real test identities/labels")
            if base["config"]["classifier"] != run["config"]["classifier"]:
                raise ValueError("Paired runs have inconsistent classifier settings")
            if {baseline, arm} == {"A3", "G1"}:
                control, generated = (base, run) if baseline == "A3" else (run, base)
                if control["duplicate_count"] != generated["synthetic_count"]:
                    raise ValueError("A3/G1 comparison requires matched added counts")
            paired_seeds.append(seed)
            for metric in METRICS:
                a = next(r for r in per_run if (r["arm"], r["budget"], r["seed"]) == (baseline, budget, seed))["metrics"][metric]
                b = next(r for r in per_run if (r["arm"], r["budget"], r["seed"]) == (arm, budget, seed))["metrics"][metric]
                if a is not None and b is not None:
                    deltas[metric].append(b - a)
        comparisons = {}
        rng = np.random.default_rng(42)
        for metric, values in deltas.items():
            means = np.mean(rng.choice(values, size=(bootstrap, len(values)), replace=True), axis=1)
            comparisons[metric] = {"mean_difference": float(np.mean(values)),
                                   "seed_bootstrap_ci95": np.quantile(means, [.025, .975]).tolist() if len(values) > 1 else None}
        paired.append({"arm": arm, "baseline": baseline, "budget": budget, "paired_seeds": sorted(paired_seeds), "metrics": comparisons})
    return {"status": "analysis_only", "publication": results[0]["publication"], "aggregate": aggregate,
            "paired": paired, "per_run": per_run,
            "limitations": "Image-group intervals do not establish patient independence. Seed bootstrap intervals with few seeds are unstable; no hypothesis-test significance is claimed. Missing runs are not imputed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True, help="Explicit measured result.json paths")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--baseline", choices=["A0", "A1", "A2", "A3", "G1"], default="A0")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output exists; choose a new analysis artifact")
    results = [json.loads(Path(path).read_text()) for path in args.results]
    write_json(args.output, summarize(results, args.bootstrap, args.baseline))


if __name__ == "__main__":
    main()
