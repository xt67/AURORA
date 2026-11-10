"""Prepare an automated-exclusion dataset proposal; never authorize or run training."""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import random

from PIL import Image
from .data import COHORTS, LABELS, read_csv, sha256, validate_manifest, budget_rows


def select_rows(reference, pairs):
    by_id = {}
    for row in reference:
        cohort, filename = row["source_cohort"], row["filename"]
        if cohort not in COHORTS or Path(filename).name != filename or any(c in filename for c in "/\\:"):
            raise ValueError("Invalid reference identity")
        if row["label_inferred"] not in LABELS:
            raise ValueError("Unknown reference label")
        key = (cohort, filename)
        if key in by_id:
            raise ValueError("Repeated reference identity")
        by_id[key] = row
    if not by_id:
        raise ValueError("Empty reference")
    membership = defaultdict(list)
    pair_ids = set()
    for pair in pairs:
        pid = pair["pair_id"]
        if not pid or pid in pair_ids:
            raise ValueError("Empty or repeated pair ID")
        pair_ids.add(pid)
        endpoints = [(pair[f"cohort{i}"], pair[f"file{i}"]) for i in (1, 2)]
        if endpoints[0] == endpoints[1]:
            raise ValueError("Self-pair in audit")
        for i, key in enumerate(endpoints, 1):
            if key not in by_id:
                raise ValueError("Audit image absent from original reference")
            if pair[f"label{i}"] != by_id[key]["label_inferred"]:
                raise ValueError("Audit/reference label mismatch")
            membership[key].append(pid)
    kept, excluded = [], []
    for key, row in sorted(by_id.items()):
        if key in membership:
            excluded.append(dict(row, pair_ids=";".join(sorted(membership[key])),
                                 exclusion_reason="Automated similarity-based exclusion: image in flagged candidate pair; not a confirmed duplicate"))
        else:
            kept.append(dict(row))
    return kept, excluded


def propose_split(retained, seed=42):
    # Exact hashes must be unique for this image-level stratification. Refuse
    # unexpected repeats rather than silently splitting or relabelling them.
    if len({r["sha256"] for r in retained}) != len(retained):
        raise ValueError("Repeated retained hash requires an explicit grouping amendment")
    strata = defaultdict(list)
    for r in retained:
        strata[(r["source_cohort"], r["label_inferred"])].append(r)
    rng = random.Random(seed)
    rows = []
    for key, members in sorted(strata.items()):
        members = sorted(members, key=lambda r: r["filename"])
        rng.shuffle(members)
        targets = [len(members) * f for f in (.7, .15, .15)]
        counts = [math.floor(v) for v in targets]
        order = sorted(range(3), key=lambda i: (-(targets[i] - counts[i]), i))
        for i in order[:len(members) - sum(counts)]:
            counts[i] += 1
        offset = 0
        for split, count in zip(("train", "val", "test"), counts):
            for row in members[offset:offset + count]:
                identity = f'{row["source_cohort"]}/{row["filename"]}'
                rows.append(dict(id=identity, path=identity, cohort=row["source_cohort"],
                                 label=LABELS[row["label_inferred"]], sha256=row["sha256"],
                                 group=identity, split=split))
            offset += count
    manifest = dict(schema_version=1, status="proposal", selection_method="automated_similarity_exclusion",
                    human_duplicate_adjudication=False, patient_independence=False, proven_duplicate_free=False,
                    split_seed=seed, split_ratios=[.7, .15, .15], rows=sorted(rows, key=lambda r: r["id"]))
    validate_manifest(manifest)
    return manifest


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def prepare(reference_path, index_path, data_root, output_dir, seed=42):
    reference_path, index_path, root, out = map(Path, (reference_path, index_path, data_root, output_dir))
    source_hashes = {"original_reference": sha256(reference_path), "candidate_pair_index": sha256(index_path)}
    reference, pairs = read_csv(reference_path), read_csv(index_path)
    retained, excluded = select_rows(reference, pairs)
    # Verify ALL originals, including excluded images; only read them.
    for row in reference:
        path = (root / row["source_cohort"] / row["filename"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != row["sha256"]:
            raise ValueError("Original image path/hash mismatch")
        with Image.open(path) as image:
            if image.size != (int(row["width"]), int(row["height"])):
                raise ValueError("Original dimensions differ from reference")
            if image.mode in ("I", "F") or image.mode.startswith("I;16"):
                raise ValueError("High-bit-depth input needs explicit preprocessing")
            image.verify()
    manifest = propose_split(retained, seed)
    manifest["source_hashes"] = source_hashes
    original_counts = Counter((r["source_cohort"], r["label_inferred"]) for r in reference)
    kept_counts = Counter((r["source_cohort"], r["label_inferred"]) for r in retained)
    split_counts = Counter((r["cohort"], r["label"], r["split"]) for r in manifest["rows"])
    lines = ["# AURORA automated dataset selection — proposal v2", "",
             f"Verified {len(reference)} originals and {len(pairs)} flagged pairs: retained {len(retained)}, excluded {len(excluded)} unique images.",
             "No original image/reference/audit was modified. Exclusions are similarity-based, not confirmed duplicates. No human adjudication or training occurred.", "",
             "| Cohort / source label | Original | Retained | Excluded | Excluded % | Train | Val | Test |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (cohort, label), original in sorted(original_counts.items()):
        kept = kept_counts[(cohort, label)]
        counts = [split_counts[(cohort, LABELS[label], s)] for s in ("train", "val", "test")]
        lines.append(f"| {cohort} / {label} | {original} | {kept} | {original-kept} | {100*(original-kept)/original:.2f}% | " + " | ".join(map(str, counts)) + " |")
    lines += ["", "## Marginal exclusion rates", "", "| Group | Original | Retained | Excluded % |", "|---|---:|---:|---:|"]
    for field, values in (("source_cohort", sorted(COHORTS)), ("label_inferred", sorted(LABELS))):
        for value in values:
            a = sum(r[field] == value for r in reference)
            b = sum(r[field] == value for r in retained)
            lines.append(f"| {value} | {a} | {b} | {100*(a-b)/a:.2f}% |")
    totals = Counter(r["split"] for r in manifest["rows"])
    lines += ["", f"Split seed: {seed}. Counts: train {totals['train']}, validation {totals['val']}, test {totals['test']}.",
              "Within each cohort/label stratum: seeded shuffle, floor 70/15/15 targets, allocate remainders by descending fractional part (ties train, val, test). One fixed split; labels unchanged.",
              "", "## Nested TB training budgets", "", "| Budget | Normal training images | TB training images |", "|---|---:|---:|"]
    for fraction in (.25, .5, 1.):
        selected = budget_rows(manifest["rows"], fraction, 42)
        lines.append(f"| {fraction:.0%} | {sum(r['label']==0 for r in selected)} | {sum(r['label']==1 for r in selected)} |")
    lines += ["", "## Limitations and next boundary", "",
              "Exclusion may change acquisition, anatomy and disease distributions and preferentially remove common-looking images. Exclusion rates differ across cohorts/classes; findings apply to this selected subset. Retained images are not proven duplicate-free, and patient identifiers are unavailable: patient-level independence cannot be claimed. Source labels are not clinically revalidated. Montgomery holdouts are small; cohort results are descriptive and no independent cross-cohort experiment is required.",
              "", "The simplified four-condition protocol is proposed in docs/EXPERIMENT_PROTOCOL_SIMPLIFIED.md. Diffusion is deferred for scope/hardware reasons. Researcher review of these counts, split and protocol is required before any first training run. Earlier approval does not authorize this revised run.",
              "", "## Source hashes", "", *[f"- {k}: `{v}`" for k, v in source_hashes.items()], ""]
    files = {f"aurora_primary_manifest_{len(retained)}.csv": csv_text(retained, list(reference[0])),
             f"aurora_exclusion_manifest_{len(excluded)}.csv": csv_text(excluded, list(reference[0])+["pair_ids", "exclusion_reason"]),
             "aurora_primary_split_proposal.json": json.dumps(manifest, indent=2),
             "aurora_dataset_selection_report.md": "\n".join(lines)}
    snapshot = f"aurora_flagged_pairs_{len(pairs)}.csv"
    if any((out / name).exists() for name in (*files, snapshot)):
        raise FileExistsError("Selection outputs exist; choose a new directory instead of overwriting")
    if source_hashes != {"original_reference": sha256(reference_path), "candidate_pair_index": sha256(index_path)}:
        raise ValueError("Source metadata changed during preparation")
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        with (out / name).open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
    with (out / snapshot).open("xb") as stream:
        stream.write(index_path.read_bytes())
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare(args.reference, args.index, args.root, args.output, args.seed))


if __name__ == "__main__":
    main()
