from copy import deepcopy
import hashlib
import json

import pytest
import yaml

from test_study_analysis import predictions
from src.study.data import sha256
from src.study.report import build_report, export_run, markdown_table


def report_fixture():
    config = {"arms": ["A0", "A3", "G1"], "budgets": [.25], "seeds": [42, 123, 999],
              "classifier": {"updates": 2}, "data_root": "PRIVATE_LOCAL_PATH"}
    results = []
    for arm in config["arms"]:
        for seed in config["seeds"]:
            results.append(dict(arm=arm, budget=.25, seed=seed, publication=True, status="measured",
                                manifest_digest="test", predictions=predictions(), threshold=.5,
                                training_ids=["PRIVATE_TRAIN_ID"], config=config, commit="test-only",
                                duplicate_count=3 if arm == "A3" else 0, synthetic_count=3 if arm == "G1" else 0))
    return config, results


def test_report_keeps_all_arms_and_removes_private_fields():
    config, results = report_fixture()
    report = build_report(results, config, "a" * 64, bootstrap=2)
    payload = json.dumps(report)
    assert "PRIVATE" not in payload and '"predictions"' not in payload
    assert report["run_count"] == 9
    assert {r["arm"] for r in report["experiments"]} == {"A0", "A3", "G1"}
    assert any(r["baseline"] == "A3" and r["arm"] == "G1" for r in report["comparisons"])
    assert "n=3" in markdown_table(report)


def test_report_refuses_incomplete_smoke_and_changed_config():
    config, results = report_fixture()
    with pytest.raises(ValueError, match="Complete configured"):
        build_report(results[:-1], config, "digest", bootstrap=2)
    bad = deepcopy(results)
    bad[0]["publication"] = False
    with pytest.raises(ValueError, match="Fixture/exploratory"):
        build_report(bad, config, "digest", bootstrap=2)
    with pytest.raises(ValueError, match="frozen configuration"):
        build_report(results, dict(config, synthetic_ratio=2), "digest", bootstrap=2)


def test_export_checks_frozen_hashes_and_never_overwrites(tmp_path):
    # Artificial metadata tests only; no real approval or result is generated.
    config, results = report_fixture()
    root = tmp_path / "run"
    root.mkdir()
    manifest = {"rows": []}
    (root / "frozen_manifest.json").write_text(json.dumps(manifest))
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    (root / "protocol.md").write_bytes(b"TEST ONLY\r\n")
    provenance = {name + "_sha256": sha256(root / filename) for name, filename in
                  (("config", "config.yaml"), ("manifest", "frozen_manifest.json"), ("protocol", "protocol.md"))}
    approval = dict(provenance, canonical_dataset_approved=True, split_frozen=True,
                    protocol_approved=True, publication_run_approved=True)
    (root / "approval.json").write_text(json.dumps(approval))
    provenance["approval_sha256"] = sha256(root / "approval.json")
    (root / "run_provenance.json").write_text(json.dumps(provenance))
    for result in results:
        result["manifest_digest"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        folder = root / f'{result["arm"]}_b0.25_s{result["seed"]}'
        folder.mkdir()
        (folder / "result.json").write_text(json.dumps(result))
    output = tmp_path / "export"
    export_run(root, output, bootstrap=2)
    assert (output / "paper_results.json").is_file()
    with pytest.raises(FileExistsError):
        export_run(root, output, bootstrap=2)
    (root / "protocol.md").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        export_run(root, tmp_path / "other", bootstrap=2)


def test_demo_displays_measured_report(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    config, results = report_fixture()
    path = tmp_path / "results.json"
    path.write_text(json.dumps(build_report(results, config, "a" * 64, bootstrap=2)))
    monkeypatch.setenv("AURORA_RESULTS", str(path))
    app = AppTest.from_file("app/streamlit_app.py").run()
    app.sidebar.radio[0].set_value("Results").run()
    assert not app.exception and not app.error
    assert len(app.json) == 1
