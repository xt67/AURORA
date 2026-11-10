"""Export measured study results for the manuscript and demo, without private rows."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import yaml

from .analysis import METRICS, summarize
from .data import sha256
from .runner import write_json


def build_report(results, config, protocol_digest, bootstrap=1000):
    expected = set(itertools.product(config["arms"], config["budgets"], config["seeds"]))
    observed = [(r["arm"], r["budget"], r["seed"]) for r in results]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Complete configured experiment grid required; no missing, extra or repeated runs")
    if not results or any(r.get("publication") is not True for r in results):
        raise ValueError("Fixture/exploratory results cannot be exported as study results")
    if any(r["config"] != config for r in results):
        raise ValueError("Results differ from the frozen configuration")
    summary = summarize(results, bootstrap=bootstrap)
    comparisons = summary["paired"]
    if "A3" in config["arms"] and "G1" in config["arms"]:
        comparisons += [r for r in summarize(results, bootstrap=bootstrap, baseline="A3")["paired"] if r["arm"] == "G1"]
    return {
        "status": "measured", "schema_version": 1,
        "protocol": "sha256:" + protocol_digest,
        "scope": "Experimental comparison under the recorded conditions; no universally best method or clinical efficacy is claimed.",
        "experiments": summary["aggregate"], "per_run": summary["per_run"], "comparisons": comparisons,
        "run_count": len(results), "seeds": config["seeds"],
        "source_commits": sorted({r["commit"] for r in results}),
        "limitations": summary["limitations"] + " Measured status is not independent scientific or clinical validation. Review before publication.",
    }


def markdown_table(report):
    lines = ["# AURORA experimental results", "", report["scope"], "",
             "Values are seed mean ± sample standard deviation; n counts defined seed metrics. NA means undefined.", "",
             "| Arm | Positive-data budget | " + " | ".join(METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    for row in report["experiments"]:
        values = []
        for name in METRICS:
            metric = row["metrics"][name]
            if metric["mean"] is None:
                values.append("NA (n=0)")
            else:
                std = f'{metric["std"]:.4f}' if metric["std"] is not None else "NA"
                values.append(f'{metric["mean"]:.4f} ± {std} (n={metric["n"]})')
        lines.append(f'| {row["arm"]} | {row["budget"]:.0%} | ' + " | ".join(values) + " |")
    lines += ["", report["limitations"], "", "Per-seed/cohort metrics, grouped intervals and paired contrasts are in paper_results.json.", ""]
    return "\n".join(lines)


def export_run(run_root, output, bootstrap=1000):
    root, output = Path(run_root), Path(output)
    provenance = json.loads((root / "run_provenance.json").read_text(encoding="utf-8"))
    for name, filename in (("config", "config.yaml"), ("protocol", "protocol.md"),
                           ("manifest", "frozen_manifest.json"), ("approval", "approval.json")):
        if sha256(root / filename) != provenance[name + "_sha256"]:
            raise ValueError(f"Frozen {name} hash mismatch")
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    approval = json.loads((root / "approval.json").read_text(encoding="utf-8"))
    for field in ("canonical_dataset_approved", "split_frozen", "protocol_approved", "publication_run_approved"):
        if approval.get(field) is not True:
            raise ValueError("Frozen approval is incomplete")
    for name in ("config", "protocol", "manifest"):
        if approval.get(name + "_sha256") != provenance[name + "_sha256"]:
            raise ValueError("Frozen approval does not match run provenance")
    manifest = json.loads((root / "frozen_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("fixture"):
        raise ValueError("Artificial fixture manifest cannot be published")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    results = []
    for fraction in config["budgets"]:
        for seed in config["seeds"]:
            for arm in config["arms"]:
                folder = root / f"{arm}_b{fraction}_s{seed}"
                if (folder / "failure.json").exists():
                    raise ValueError(f"Failed run remains: {folder.name}")
                result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
                if result["manifest_digest"] != digest:
                    raise ValueError("Result manifest differs from frozen manifest")
                results.append(result)
    report = build_report(results, config, provenance["protocol_sha256"], bootstrap)
    # Validate everything before creating a new output directory; never overwrite.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "paper_results.json", report)
    (output / "results_table.md").write_text(markdown_table(report), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed study run root")
    parser.add_argument("--output", required=True, help="New export directory")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    export_run(args.run, args.output, args.bootstrap)


if __name__ == "__main__":
    main()
