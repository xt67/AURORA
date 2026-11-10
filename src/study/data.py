"""Review validation, immutable manifests and deterministic nested budgets."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import random

LABELS = {"normal": 0, "tuberculosis": 1}
COHORTS = {"shenzhen", "montgomery"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def validate_decisions(index, decisions):
    """Require complete identified human decisions; never infer them from scores."""
    pairs = {row["pair_id"]: row for row in index}
    if len(pairs) != len(index):
        raise ValueError("Duplicate pair IDs in review index")
    by_id = {row["pair_id"]: row for row in decisions}
    if len(by_id) != len(decisions) or set(by_id) != set(pairs):
        raise ValueError("Review must contain exactly one decision per indexed pair")
    for pid, pair in pairs.items():
        row = by_id[pid]
        if row.get("decision") not in {"CONFIRMED_DUPLICATE", "SIMILAR_BUT_DISTINCT"}:
            raise ValueError(f"{pid}: probable/unresolved/blank decisions block finalization")
        if row.get("label_conflict_status") != "none":
            raise ValueError(f"{pid}: unresolved label conflict")
        if not row.get("reviewer_identifier", "").strip():
            raise ValueError(f"{pid}: reviewer identifier required")
        try:
            stamp = datetime.fromisoformat(row.get("review_timestamp", "").replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("timezone required")
        except ValueError as exc:
            raise ValueError(f"{pid}: valid timestamp with timezone required") from exc
        if row.get("image_a_filename") != pair["file1"] or row.get("image_b_filename") != pair["file2"]:
            raise ValueError(f"{pid}: decision image identity mismatch")
        if pair.get("cross_class", "").lower() == "true":
            if not row.get("reviewer_notes", "").strip():
                raise ValueError(f"{pid}: document cross-class investigation in reviewer notes")
            if row["decision"] == "CONFIRMED_DUPLICATE":
                raise ValueError(f"{pid}: cross-label duplicate requires explicit label adjudication; no automatic relabelling")
    return by_id


def build_manifest(reference, index, decisions, seed=42):
    """Create a proposal only. Approval is separate and binds this file's hash."""
    reviewed = validate_decisions(index, decisions)
    rows = []
    for row in reference:
        cohort, filename = row["source_cohort"], row["filename"]
        if cohort not in COHORTS or Path(filename).name != filename or "/" in filename or "\\" in filename:
            raise ValueError("Invalid reference image identity")
        if row["label_inferred"] not in LABELS:
            raise ValueError("Unknown source label")
        rows.append({"id": f"{cohort}/{filename}", "path": f"{cohort}/{filename}",
                     "cohort": cohort, "label": LABELS[row["label_inferred"]], "sha256": row["sha256"]})
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows) or not rows:
        raise ValueError("Reference manifest is empty or has repeated identities")
    parent = {key: key for key in by_id}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        if a not in by_id or b not in by_id:
            raise ValueError("Review pair references an absent image")
        if by_id[a]["label"] != by_id[b]["label"]:
            raise ValueError("Duplicate group has conflicting labels")
        ra, rb = find(a), find(b)
        parent[max(ra, rb)] = min(ra, rb)

    hashes = {}
    for row in rows:
        if row["sha256"] in hashes:
            union(row["id"], hashes[row["sha256"]])
        hashes[row["sha256"]] = row["id"]
    for pair in index:
        a, b = f"{pair['cohort1']}/{pair['file1']}", f"{pair['cohort2']}/{pair['file2']}"
        if a not in by_id or b not in by_id:
            raise ValueError("Review index references an absent image")
        if reviewed[pair["pair_id"]]["decision"] == "CONFIRMED_DUPLICATE":
            union(a, b)
    groups = defaultdict(list)
    for row in rows:
        groups[find(row["id"])].append(row)
    strata = defaultdict(list)
    for gid, members in groups.items():
        stratum = (tuple(sorted({r["cohort"] for r in members})), members[0]["label"])
        strata[stratum].append((gid, members))
    rng = random.Random(seed)
    for stratum in sorted(strata):
        items = sorted(strata[stratum])
        rng.shuffle(items)
        total = sum(len(members) for _, members in items)
        remaining = dict(train=0.70 * total, val=0.15 * total, test=0.15 * total)
        for gid, members in items:
            split = max(remaining, key=remaining.get)
            remaining[split] -= len(members)
            for row in members:
                row.update(group=gid, split=split)
    return {"schema_version": 1, "status": "proposal", "patient_independence": False,
            "split_seed": seed, "split_ratios": [0.7, 0.15, 0.15], "rows": sorted(rows, key=lambda r: r["id"])}


def validate_manifest(manifest, root=None):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    rows = manifest.get("rows", [])
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty manifest or repeated IDs")
    groups, hashes, paths = {}, {}, set()
    counts = defaultdict(int)
    for row in rows:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or "\\" in row["path"] or ":" in row["path"]:
            raise ValueError("Manifest paths must stay within the dataset root")
        if row["path"] in paths:
            raise ValueError("Repeated image path")
        paths.add(row["path"])
        if row["cohort"] not in COHORTS or row["label"] not in (0, 1) or row["split"] not in ("train", "val", "test"):
            raise ValueError("Invalid cohort, label or split")
        if not row.get("group") or len(row["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["sha256"]):
            raise ValueError("Missing group or invalid SHA-256")
        for lookup, key in ((groups, row["group"]), (hashes, row["sha256"])):
            assignment = (row["split"], row["label"])
            if key in lookup and lookup[key] != assignment:
                raise ValueError("Duplicate group/hash crosses partitions or labels")
            lookup[key] = assignment
        counts[(row["split"], row["label"])] += 1
        if root is not None:
            resolved = (Path(root) / path).resolve()
            if not resolved.is_relative_to(Path(root).resolve()) or sha256(resolved) != row["sha256"]:
                raise ValueError(f"Image hash/path mismatch: {row['id']}")
            from PIL import Image
            with Image.open(resolved) as image:
                if image.mode in ("I", "F") or image.mode.startswith("I;16"):
                    raise ValueError("High-bit-depth input needs an approved intensity conversion")
                image.verify()
    if any(counts[(split, label)] == 0 for split in ("train", "val", "test") for label in (0, 1)):
        raise ValueError("Each partition must contain both classes; revise proposed split with researcher")
    return rows


def budget_rows(rows, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("Budget must be in (0, 1]")
    train = [r for r in rows if r["split"] == "train"]
    positive = sorted((r for r in train if r["label"] == 1), key=lambda r: r["id"])
    if not positive:
        raise ValueError("Training partition has no positive images")
    random.Random(seed).shuffle(positive)
    # Prefix selection yields exactly nested image budgets; duplicated images
    # remain in one partition but are counted as images, not independent patients.
    return sorted([r for r in train if r["label"] == 0] + positive[:max(1, math.floor(len(positive) * fraction))], key=lambda r: r["id"])


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_manifest(read_csv(args.reference), read_csv(args.index), read_csv(args.decisions), args.seed)
    manifest["source_hashes"] = {name: sha256(getattr(args, name)) for name in ("reference", "index", "decisions")}
    validate_manifest(manifest, args.root)
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print("Proposal written; researcher approval and a frozen protocol are still required.")


if __name__ == "__main__":
    main()
