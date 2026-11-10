"""Separate, fail-closed approval gate for the fixed v2 DCGAN feasibility run."""
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess

import yaml

from .data import budget_rows, validate_manifest
from .selection import select_rows, propose_split

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "aurora_v2_dcgan_feasibility_only"
FLAGS = ("dataset_approved", "split_approved", "protocol_approved", "feasibility_run_approved")
ARTIFACT_KEYS = ("protocol", "original_reference", "candidate_pairs", "retained_manifest",
                 "exclusion_manifest", "split_manifest")
IMPLEMENTATION_FILES = (
    "src/study/v2_gate.py", "src/study/feasibility.py", "src/study/selection.py", "src/study/data.py",
    "src/gan/train_gan.py", "src/gan/generator.py", "src/gan/discriminator.py", "src/gan/losses.py",
    "src/data_loader.py", "src/utils/seed.py", "src/utils/logger.py", "src/utils/visualize.py",
    "src/utils/error_guard.py",
)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def object_digest(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def implementation_version():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    changed = subprocess.check_output(["git", "status", "--porcelain", "--", *IMPLEMENTATION_FILES], cwd=ROOT, text=True)
    if changed.strip():
        raise ValueError("Feasibility implementation must be committed before approval/execution")
    hashes = {name: digest((ROOT / name).read_bytes()) for name in IMPLEMENTATION_FILES}
    return {"commit": commit, "source_sha256": object_digest(hashes)}


@dataclass(frozen=True)
class SourceImage:
    id: str
    path: str
    sha256: str
    split: str
    label: int


@dataclass(frozen=True)
class FeasibilityPlan:
    root: Path
    sources: tuple
    source_ids: tuple
    effective_config: dict
    bindings: dict
    artifacts: dict
    device: str
    approval_bytes: bytes = b""


def _csv(payload):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))


def _effective_config(config):
    if config.get("schema_version") != 2 or config.get("selection_method") != "automated_similarity_exclusion":
        raise ValueError("Expected the automated-exclusion v2 proposal")
    if config.get("training_authorized") is not False:
        raise ValueError("Publication authorization must remain false; use separate feasibility approval")
    if config.get("split_seed") != 42 or config.get("budget_seed") != 42:
        raise ValueError("Feasibility split/budget seeds must be 42")
    f = config["feasibility_check"]
    if (f.get("epochs"), f.get("budget"), f.get("seed")) != (2, .25, 42):
        raise ValueError("Only two epochs, 25% budget and seed 42 are supported")
    if f.get("training_partition_only") is not True or f.get("publication_result") is not False:
        raise ValueError("Feasibility must be train-only and non-publication")
    p = config["preprocessing"]
    if p.get("image_size") != 128 or p.get("grayscale") is not True or p.get("generator_range") != [-1, 1]:
        raise ValueError("Expected grayscale 128px GAN preprocessing in [-1,1]")
    if p.get("high_bit_depth") != "reject_without_explicit_conversion":
        raise ValueError("High-bit-depth conversion is not authorized")
    if config["conventional_augmentation"].get("horizontal_flip_probability") != 0:
        raise ValueError("Primary horizontal flips must remain disabled")
    g = config["generator"]
    expected = dict(architecture="dcgan", loss_type="bce", z_dim=100, ngf=64, ndf=64,
                    lr_g=.0002, lr_d=.0002, beta1=.5, beta2=.999, d_steps=1, batch_size=16, use_amp=False)
    if any(g.get(key) != value for key, value in expected.items()):
        raise ValueError("Generator settings differ from the bounded DCGAN feasibility specification")
    # Never inherit the later publication epoch count or allow arbitrary trainer options.
    return {**{k: v for k, v in expected.items() if k != "architecture"},
            "epochs": 2, "seed": 42, "image_size": 128, "checkpoint_every": 2, "label_smoothing": .9}


def _plan(config_path, device, expected_hashes=None):
    if device not in ("cuda", "cpu"):
        raise ValueError("Explicit approved device must be cuda or cpu; no automatic fallback")
    artifacts = {"config": Path(config_path).read_bytes()}
    if expected_hashes is not None and digest(artifacts["config"]) != expected_hashes.get("config"):
        raise ValueError("Approval hash mismatch: config")
    config = yaml.safe_load(artifacts["config"].decode("utf-8"))
    effective = _effective_config(config)
    for key in ARTIFACT_KEYS:
        artifacts[key] = Path(config[key]).read_bytes()
        if expected_hashes is not None and digest(artifacts[key]) != expected_hashes.get(key):
            raise ValueError(f"Approval hash mismatch: {key}")
    hashes = {key: digest(value) for key, value in artifacts.items()}
    if expected_hashes is not None and set(expected_hashes) != set(hashes):
        raise ValueError("Approval must bind exactly all seven input artifacts")
    reference, pairs, retained, excluded = [_csv(artifacts[k]) for k in
                                           ("original_reference", "candidate_pairs", "retained_manifest", "exclusion_manifest")]
    if (len(reference), len(pairs), len(retained), len(excluded)) != (800, 206, 590, 210):
        raise ValueError("Expected verified 800/206/590/210 original/pair/retained/excluded counts")
    expected_retained, expected_excluded = select_rows(reference, pairs)
    if retained != expected_retained:
        raise ValueError("Retained manifest differs from the complete flagged-image exclusion")
    # Existing immutable records retain their historical wording. Accept ONLY
    # that exact older phrase or the corrected phrase, never ignore other fields.
    corrected = []
    for row in excluded:
        copy = dict(row)
        old = "Automated similarity-based exclusion: participant in flagged candidate pair; not a confirmed duplicate"
        new = "Automated similarity-based exclusion: image in flagged candidate pair; not a confirmed duplicate"
        if copy.get("exclusion_reason") not in (old, new):
            raise ValueError("Unexpected exclusion reason")
        copy["exclusion_reason"] = new
        corrected.append(copy)
    if corrected != expected_excluded:
        raise ValueError("Exclusion identities, metadata or pair memberships differ from audit")
    if len({r["sha256"] for r in reference}) != 800:
        raise ValueError("Unexpected repeated original hash")
    manifest = json.loads(artifacts["split_manifest"])
    if manifest.get("fixture"):
        raise ValueError("Fixture manifest cannot authorize a real feasibility run")
    rebuilt = propose_split(retained, 42)
    rebuilt["source_hashes"] = {"original_reference": hashes["original_reference"], "candidate_pair_index": hashes["candidate_pairs"]}
    if manifest != rebuilt:
        raise ValueError("Split does not reconstruct from automated-exclusion provenance")
    validate_manifest(manifest)  # Metadata only: never open holdout image bytes.
    if Counter(r["split"] for r in manifest["rows"]) != {"train": 413, "val": 89, "test": 88}:
        raise ValueError("Unexpected split counts")
    if Counter((r["cohort"], r["label"]) for r in manifest["rows"]) != {
            ("shenzhen", 0): 214, ("shenzhen", 1): 249, ("montgomery", 0): 75, ("montgomery", 1): 52}:
        raise ValueError("Unexpected cohort/class counts")
    selected = [r for r in budget_rows(manifest["rows"], .25, 42) if r["label"] == 1]
    if len(selected) != 52 or any(r["split"] != "train" for r in selected):
        raise ValueError("Exactly 52 real TB training sources required")
    sources = tuple(SourceImage(**{k: r[k] for k in ("id", "path", "sha256", "split", "label")}) for r in selected)
    version = implementation_version()
    bindings = {"hashes": hashes, "source_ids": [r.id for r in sources],
                "selection_sha256": object_digest(selected), "effective_config_sha256": object_digest(effective),
                "implementation_commit": version["commit"], "implementation_sha256": version["source_sha256"]}
    return FeasibilityPlan(Path(config["data_root"]).resolve(), sources, tuple(bindings["source_ids"]),
                           effective, bindings, artifacts, device)


def approval_draft(config_path, device="cuda"):
    """Prepare exact bindings without granting approval or reading any image."""
    plan = _plan(config_path, device)
    return {"schema_version": 2, "scope": SCOPE, **{k: False for k in FLAGS},
            "approved_by": "", "approved_at": "", "device": device, **plan.bindings}


def feasibility_preflight(config_path, approval_path):
    if not Path(approval_path).is_file():
        raise ValueError("Explicit v2 feasibility approval is missing")
    payload = Path(approval_path).read_bytes()
    approval = json.loads(payload)
    if approval.get("schema_version") != 2 or approval.get("scope") != SCOPE:
        raise ValueError("Approval scope must be v2 feasibility only, not v1/publication")
    if any(approval.get(field) is not True for field in FLAGS):
        raise ValueError("Explicit final feasibility approval and all dataset/split/protocol flags are required")
    if not str(approval.get("approved_by", "")).strip():
        raise ValueError("Researcher identity is required")
    stamp = datetime.fromisoformat(approval.get("approved_at", "").replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Approval timestamp needs a timezone")
    if not isinstance(approval.get("hashes"), dict):
        raise ValueError("Approval hashes are required")
    plan = _plan(config_path, approval.get("device"), approval["hashes"])
    for field, value in plan.bindings.items():
        if approval.get(field) != value:
            raise ValueError(f"Approval mismatch: {field}")
    return FeasibilityPlan(plan.root, plan.sources, plan.source_ids, plan.effective_config,
                           plan.bindings, plan.artifacts, plan.device, payload)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Write an UNAPPROVED v2 feasibility template; never trains")
    parser.add_argument("--config", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    draft = approval_draft(args.config, args.device)
    with Path(args.draft).open("x", encoding="utf-8") as stream:
        json.dump(draft, stream, indent=2)
    print("Unapproved template written. All approval flags remain false. No image read or training occurred.")


if __name__ == "__main__":
    main()
