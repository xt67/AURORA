import json
from pathlib import Path

import pytest
import torch
import yaml
import csv

from test_study_data import study_fixture
from src.study.data import validate_manifest
from src.study.data import sha256
from src.study.gate import publication_preflight
from src.study.runner import run_one, select_threshold, make_train_loader, ManifestDataset


def config_for(root):
    return {"data_root": str(root), "device": "cpu", "budget_seed": 42,
            "classifier": dict(pretrained=False, updates=2, eval_every=1, batch_size=2, lr=.001, weight_decay=0., use_amp=True),
            "generator": dict(z_dim=8, ngf=2, ndf=2, lr_g=.0002, lr_d=.0002, beta1=.5, beta2=.999,
                              epochs=1, d_steps=1, batch_size=4, loss_type="bce"), "synthetic_ratio": 1.}


@pytest.mark.parametrize("arm", ["A0", "A1", "A2", "G1"])
def test_each_arm_runs_on_artificial_fixture(study_fixture, arm):
    root, _, _, _, manifest = study_fixture
    validate_manifest(manifest, root)
    result = run_one(config_for(root), manifest, arm, .25, 42, root / arm)
    assert result["publication"] is False
    assert result["updates"] == 2
    assert set(result["metrics"]["cohorts"]) == {"shenzhen", "montgomery"}
    assert all(r["split"] == "train" for r in manifest["rows"] if r["id"] in result["training_ids"])
    assert (root / arm / "classifier.pth").is_file()
    if arm == "G1":
        provenance = json.loads((root / arm / "generator_provenance.json").read_text())
        assert set(provenance["training_ids"]) <= set(result["training_ids"])
        assert result["synthetic_count"] > 0


def test_threshold_uses_validation_predictions_only():
    assert select_threshold([0, 0, 1, 1], [.1, .3, .4, .8]) == .4
    with pytest.raises(ValueError):
        select_threshold([1, 1], [.2, .5])


def test_publication_refuses_unapproved_example(tmp_path):
    with pytest.raises(ValueError, match="explicit researcher approval"):
        publication_preflight("configs/study.yaml", "configs/approval.example.json")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fixed_update_classifier_on_cuda(study_fixture):
    root, _, _, _, manifest = study_fixture
    config = config_for(root)
    config["device"] = "cuda"
    result = run_one(config, manifest, "A0", .25, 42, root / "cuda")
    assert result["device"] == "cuda"
    assert result["updates"] == 2


def test_exact_update_counts_in_sampling(study_fixture):
    root, _, _, _, manifest = study_fixture
    dataset = ManifestDataset([r for r in manifest["rows"] if r["split"] == "train"], root)
    for arm in ("A0", "A1", "A2"):
        loader = make_train_loader(dataset, arm, 7, 4, 42)
        assert len(loader) == 7
        assert len(list(loader.sampler)) == 28


def test_approval_is_bound_to_review_manifest_config_and_protocol(study_fixture):
    # Entirely artificial review metadata to test validation, never an approval
    # for the real dataset. No publication training is launched by this test.
    root, reference, index, decisions, manifest = study_fixture
    config = config_for(root)
    for name, rows in (("reference", reference), ("index", index), ("decisions", decisions)):
        path = root / f"{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        config[name] = str(path)
    manifest["source_hashes"] = {name: sha256(config[name]) for name in ("reference", "index", "decisions")}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    protocol = root / "protocol.md"
    protocol.write_text("ARTIFICIAL SOFTWARE FIXTURE ONLY")
    config.update(manifest=str(path), protocol=str(protocol), seeds=[42, 123, 999], arms=["A0"], budgets=[.25])
    cfg = root / "study.yaml"
    cfg.write_text(yaml.safe_dump(config))
    approval = dict(canonical_dataset_approved=True, split_frozen=True, protocol_approved=True, publication_run_approved=True,
                    approved_by="unit-test-fixture", approved_at="2026-01-01T00:00:00+00:00",
                    manifest_sha256=sha256(path), config_sha256=sha256(cfg), protocol_sha256=sha256(protocol))
    approval_path = root / "approval.json"
    approval_path.write_text(json.dumps(approval))
    assert publication_preflight(cfg, approval_path)[1] == manifest
    protocol.write_text("changed")
    with pytest.raises(ValueError, match="protocol_sha256"):
        publication_preflight(cfg, approval_path)
