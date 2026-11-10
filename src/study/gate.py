"""Fail-closed checks for human-authorized publication runs."""
import json
import math
from datetime import datetime
from pathlib import Path
import yaml

from .data import sha256, read_csv, validate_decisions, validate_manifest


def publication_preflight(config_path, approval_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not config.get("arms") or len(set(config["arms"])) != len(config["arms"]) or not set(config["arms"]) <= {"A0", "A1", "A2", "A3", "G1"}:
        raise ValueError("Supported unique arms are A0, A1, A2, A3 and G1; G2 is not implemented")
    if not config.get("budgets") or not set(config["budgets"]) <= {0.25, 0.5, 1.0} or len(set(config["budgets"])) != len(config["budgets"]):
        raise ValueError("Use unique proposed budgets 0.25, 0.5 and/or 1.0")
    cfg = config["classifier"]
    if cfg["updates"] < 1 or cfg["eval_every"] < 1 or cfg["batch_size"] < 2 or cfg["lr"] <= 0 or cfg["weight_decay"] < 0:
        raise ValueError("Invalid classifier optimization settings")
    if not math.isfinite(config.get("synthetic_ratio", 1)) or config.get("synthetic_ratio", 1) < 0:
        raise ValueError("Synthetic ratio must be finite and nonnegative")
    approval = json.loads(Path(approval_path).read_text(encoding="utf-8"))
    for field in ("canonical_dataset_approved", "split_frozen", "protocol_approved", "publication_run_approved"):
        if approval.get(field) is not True:
            raise ValueError(f"Publication blocked: {field} requires explicit researcher approval")
    if not str(approval.get("approved_by", "")).strip():
        raise ValueError("Publication blocked: researcher identity missing")
    stamp = datetime.fromisoformat(approval.get("approved_at", "").replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Approval timestamp must include timezone")
    for field, path in (("config_sha256", config_path), ("manifest_sha256", config["manifest"]), ("protocol_sha256", config["protocol"])):
        if approval.get(field) != sha256(path):
            raise ValueError(f"Publication blocked: changed or unapproved {field}")
    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    if manifest.get("fixture"):
        raise ValueError("Artificial fixtures cannot be used for publication")
    for name in ("reference", "index", "decisions"):
        if manifest.get("source_hashes", {}).get(name) != sha256(config[name]):
            raise ValueError(f"Publication blocked: {name} differs from canonical provenance")
    validate_decisions(read_csv(config["index"]), read_csv(config["decisions"]))
    # Rebuild to ensure reviewed group assignments and seed actually generated
    # this manifest, rather than accepting an edited list of split labels.
    from .data import build_manifest
    rebuilt = build_manifest(read_csv(config["reference"]), read_csv(config["index"]),
                             read_csv(config["decisions"]), manifest["split_seed"])
    if rebuilt["rows"] != manifest["rows"]:
        raise ValueError("Publication blocked: manifest no longer matches reviewed split proposal")
    validate_manifest(manifest, config["data_root"])
    if len(set(config["seeds"])) < 3:
        raise ValueError("Publication config requires at least three distinct seeds")
    return config, manifest
