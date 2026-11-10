"""Artificial metadata/images only. Trainer is stubbed: no optimizer/training runs."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest
import torch
import yaml

from src.study import v2_gate as gate
from src.study import feasibility as feasibility
from src.study.data import budget_rows, sha256
from src.study.selection import csv_text, select_rows, propose_split


@pytest.fixture
def approved_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "implementation_version", lambda: {"commit": "a" * 40, "source_sha256": "b" * 64})
    rows, pairs = [], []
    for cohort, label, count, excluded_count in (("shenzhen", "normal", 326, 112),
            ("shenzhen", "tuberculosis", 336, 87), ("montgomery", "normal", 80, 5),
            ("montgomery", "tuberculosis", 58, 6)):
        for i in range(count):
            filename = f"{label}_{i:04d}.png"
            rows.append(dict(source_cohort=cohort, filename=filename, label_inferred=label,
                             sha256=hashlib.sha256(f"{cohort}/{filename}".encode()).hexdigest()))
        for i in range(excluded_count - 1):
            pairs.append(dict(pair_id=f"P{len(pairs):03d}", cohort1=cohort, cohort2=cohort,
                              file1=f"{label}_{i:04d}.png", file2=f"{label}_{i+1:04d}.png",
                              label1=label, label2=label, ncc_512="0.9"))
    retained, _ = select_rows(rows, pairs)
    manifest = propose_split(retained)
    selected = [r for r in budget_rows(manifest["rows"], .25, 42) if r["label"] == 1]
    root = tmp_path / "original"
    for i, row in enumerate(selected):
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (8, 8), i).save(path)
        source = next(r for r in rows if f'{r["source_cohort"]}/{r["filename"]}' == row["id"])
        source["sha256"] = sha256(path)
    retained, excluded = select_rows(rows, pairs)
    # Exercise historical immutable exclusion wording compatibility.
    for row in excluded:
        row["exclusion_reason"] = row["exclusion_reason"].replace("image in flagged", "participant in flagged")
    config = yaml.safe_load(Path("configs/study_simplified_proposal.yaml").read_text())
    for name, records in (("original_reference", rows), ("candidate_pairs", pairs),
                          ("retained_manifest", retained), ("exclusion_manifest", excluded)):
        path = tmp_path / (name + ".csv")
        path.write_text(csv_text(records, list(records[0])), encoding="utf-8")
        config[name] = str(path)
    manifest = propose_split(retained)
    manifest["source_hashes"] = {"original_reference": sha256(config["original_reference"]),
                                  "candidate_pair_index": sha256(config["candidate_pairs"])}
    split = tmp_path / "split.json"
    split.write_text(json.dumps(manifest))
    protocol = tmp_path / "protocol.md"
    protocol.write_bytes(b"ARTIFICIAL TEST ONLY\r\n")
    config.update(split_manifest=str(split), protocol=str(protocol), data_root=str(root))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config))
    approval = gate.approval_draft(cfg, "cpu")
    assert not any(approval[field] for field in gate.FLAGS)
    approval.update({field: True for field in gate.FLAGS})
    approval.update(approved_by="TEST_FIXTURE_NOT_REAL_APPROVAL", approved_at="2026-11-11T12:00:00+05:30")
    auth = tmp_path / "test_approval.json"
    auth.write_text(json.dumps(approval))
    return cfg, auth, approval, config, manifest


def test_missing_and_wrong_scope_approval_never_calls_trainer(approved_fixture, tmp_path, monkeypatch):
    cfg, auth, approval, _, _ = approved_fixture
    monkeypatch.setattr(feasibility, "train_gan", lambda *a: pytest.fail("Trainer must not be called"))
    with pytest.raises(ValueError, match="missing"):
        feasibility.run_feasibility(cfg, tmp_path / "absent", tmp_path / "run")
    approval["scope"] = "publication"
    auth.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="scope"):
        feasibility.run_feasibility(cfg, auth, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("flag", gate.FLAGS)
def test_every_explicit_approval_flag_required(approved_fixture, flag):
    cfg, auth, approval, _, _ = approved_fixture
    approval[flag] = False
    auth.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="Explicit final"):
        gate.feasibility_preflight(cfg, auth)


@pytest.mark.parametrize("artifact", ("config", *gate.ARTIFACT_KEYS))
def test_changed_input_hash_rejected(approved_fixture, artifact):
    cfg, auth, _, config, _ = approved_fixture
    path = cfg if artifact == "config" else Path(config[artifact])
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.feasibility_preflight(cfg, auth)


@pytest.mark.parametrize("field", ("source_ids", "selection_sha256", "effective_config_sha256",
                                  "implementation_commit", "implementation_sha256"))
def test_incorrect_source_or_implementation_bindings_rejected(approved_fixture, field):
    cfg, auth, approval, _, manifest = approved_fixture
    approval[field] = ([next(r["id"] for r in manifest["rows"] if r["split"] == "test")]
                       if field == "source_ids" else "wrong")
    auth.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match=field):
        gate.feasibility_preflight(cfg, auth)


def test_edited_split_rejected_even_if_file_hash_updated(approved_fixture):
    cfg, auth, approval, config, manifest = approved_fixture
    a = next(r for r in manifest["rows"] if r["split"] == "train")
    b = next(r for r in manifest["rows"] if r["split"] == "test" and r["label"] == a["label"])
    a["split"], b["split"] = b["split"], a["split"]
    path = Path(config["split_manifest"])
    path.write_text(json.dumps(manifest))
    approval["hashes"]["split_manifest"] = sha256(path)
    auth.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="reconstruct"):
        gate.feasibility_preflight(cfg, auth)


def test_holdout_and_altered_row_rejected_before_io(approved_fixture, monkeypatch):
    cfg, auth, _, _, manifest = approved_fixture
    plan = gate.feasibility_preflight(cfg, auth)
    reader = feasibility.ApprovedImageReader(plan)
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("Unapproved image was opened"))
    for split in ("val", "test"):
        holdout = next(r["id"] for r in manifest["rows"] if r["split"] == split)
        with pytest.raises(ValueError, match="before image IO"):
            reader.read(holdout)
    with pytest.raises(ValueError, match="Holdout"):
        feasibility.ApprovedImageReader(replace(plan, sources=(replace(plan.sources[0], split="test"), *plan.sources[1:])))
    with pytest.raises(ValueError, match="52 ordered"):
        feasibility.ApprovedImageReader(replace(plan, source_ids=plan.source_ids[::-1]))


def test_changed_selected_image_rejected(approved_fixture):
    cfg, auth, _, _, _ = approved_fixture
    plan = gate.feasibility_preflight(cfg, auth)
    (plan.root / plan.sources[0].path).write_bytes(b"altered")
    with pytest.raises(ValueError, match="image hash mismatch"):
        feasibility.ApprovedImageReader(plan).read(plan.source_ids[0])


def test_two_epochs_only_and_no_flip(approved_fixture):
    cfg, _, _, config, _ = approved_fixture
    config["feasibility_check"]["epochs"] = 200
    cfg.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="two epochs"):
        gate.approval_draft(cfg)
    config["feasibility_check"]["epochs"] = 2
    config["conventional_augmentation"]["horizontal_flip_probability"] = .5
    cfg.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="flips"):
        gate.approval_draft(cfg)


def test_check_only_reads_only_allowlist_and_never_trains(approved_fixture, tmp_path, monkeypatch):
    cfg, auth, approval, _, _ = approved_fixture
    reads = []
    original_read = Path.read_bytes
    def observed_read(path):
        if path.suffix == ".png":
            reads.append(path.parent.name + "/" + path.name)
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", observed_read)
    monkeypatch.setattr(feasibility, "train_gan", lambda *a: pytest.fail("Check-only cannot train"))
    result = feasibility.run_feasibility(cfg, auth, tmp_path / "unused", check_only=True)
    assert result["status"] == "preflight_passed_no_training"
    assert reads == approval["source_ids"] and len(reads) == 52
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_execution_provenance_with_stubbed_trainer(approved_fixture, tmp_path, monkeypatch, device):
    cfg, auth, approval, config, _ = approved_fixture
    approval["device"] = device
    auth.write_text(json.dumps(approval))
    if device == "cuda":
        # Exercise memory provenance without creating a CUDA context or training.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (1000, 2000))
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "TEST GPU")
        monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 300)
        monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: 400)
    inputs = [cfg, auth, *[Path(config[k]) for k in gate.ARTIFACT_KEYS]]
    before = {str(p): sha256(p) for p in inputs}
    calls = []
    def stub_trainer(config_path, loader, device, output_dir):
        effective = yaml.safe_load(Path(config_path).read_text())
        assert effective["epochs"] == 2 and effective["seed"] == 42 and effective["use_amp"] is False
        assert type(loader.dataset) is torch.utils.data.TensorDataset
        assert len(loader.dataset) == 52 and len(loader) == 4 and loader.num_workers == 0
        assert [len(labels) for _, labels in loader] == [16, 16, 16, 4]
        folder = Path(output_dir) / "models"
        folder.mkdir(parents=True)
        torch.save({"epoch": 2, "config": effective}, folder / "generator_latest.pth")
        calls.append(effective)
    monkeypatch.setattr(feasibility, "train_gan", stub_trainer)
    out = tmp_path / "run"
    result = feasibility.run_feasibility(cfg, auth, out)
    assert len(calls) == 1 and result["publication"] is False
    assert result["source_ids"] == result["actual_image_read_ids"]
    assert result["commit"] == "a" * 40 and result["configuration_sha256"] == sha256(cfg)
    assert result["runtime_seconds"] >= result["training_seconds"] >= 0
    assert result["classifier_updates"] == 0
    if device == "cuda":
        assert result["gpu_memory"] == dict(free_bytes_at_start=1000, total_bytes=2000, device_name="TEST GPU",
                                           peak_allocated_bytes=300, peak_reserved_bytes=400)
    else:
        assert result["gpu_memory"] is None
    assert {str(p): sha256(p) for p in inputs} == before
    with pytest.raises(ValueError, match="new and outside"):
        feasibility.run_feasibility(cfg, auth, out)


def test_failure_artifact_and_no_device_fallback(approved_fixture, tmp_path, monkeypatch):
    cfg, auth, approval, _, _ = approved_fixture
    approval["device"] = "cuda"
    auth.write_text(json.dumps(approval))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(feasibility, "train_gan", lambda *a: pytest.fail("Must not silently train on CPU"))
    out = tmp_path / "unavailable"
    with pytest.raises(RuntimeError, match="unavailable"):
        feasibility.run_feasibility(cfg, auth, out)
    failure = json.loads((out / "failure.json").read_text())
    assert failure["status"] == "failed" and failure["runtime_seconds"] >= 0
    assert failure["device"] == "cuda"
