from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.study.data import build_manifest, validate_manifest, validate_decisions, budget_rows, sha256


@pytest.fixture
def study_fixture(tmp_path):
    reference = []
    rng = np.random.default_rng(42)
    for cohort in ("shenzhen", "montgomery"):
        (tmp_path / cohort).mkdir()
        for label in ("normal", "tuberculosis"):
            for i in range(12):
                filename = f"{label}_{i}.png"
                path = tmp_path / cohort / filename
                Image.fromarray(rng.integers(0, 256, (16, 16), dtype=np.uint8)).save(path)
                reference.append(dict(filename=filename, source_cohort=cohort, label_inferred=label, sha256=sha256(path)))
    index = [dict(pair_id="TEST-1", cohort1="shenzhen", cohort2="shenzhen",
                  file1="normal_0.png", file2="normal_1.png", cross_class="false")]
    decisions = [dict(pair_id="TEST-1", image_a_filename="normal_0.png", image_b_filename="normal_1.png",
                      decision="CONFIRMED_DUPLICATE", label_conflict_status="none", reviewer_identifier="unit-test-fixture",
                      review_timestamp="2026-01-01T00:00:00+00:00", reviewer_notes="Artificial fixture, not human evidence")]
    manifest = build_manifest(reference, index, decisions)
    return tmp_path, reference, index, decisions, manifest


def test_duplicate_grouping_hashes_and_nested_budgets(study_fixture):
    root, reference, index, decisions, manifest = study_fixture
    rows = validate_manifest(manifest, root)
    duplicate_rows = [r for r in rows if r["id"] in ("shenzhen/normal_0.png", "shenzhen/normal_1.png")]
    assert len({r["split"] for r in duplicate_rows}) == 1
    assert len({r["group"] for r in duplicate_rows}) == 1
    small, medium, full = [set(r["id"] for r in budget_rows(rows, x, 42)) for x in (.25, .5, 1)]
    assert small < medium < full
    assert build_manifest(reference, index, decisions) == manifest


@pytest.mark.parametrize("field,value", [("decision", "UNRESOLVED"), ("decision", "PROBABLE_DUPLICATE"),
    ("reviewer_identifier", ""), ("review_timestamp", "bad"), ("label_conflict_status", "potential_label_conflict"),
    ("image_a_filename", "wrong.png")])
def test_review_rejects_incomplete_or_mismatched_rows(study_fixture, field, value):
    _, _, index, decisions, _ = study_fixture
    decisions[0][field] = value
    with pytest.raises(ValueError):
        validate_decisions(index, decisions)


def test_manifest_rejects_leakage_and_modified_data(study_fixture):
    root, _, _, _, manifest = study_fixture
    bad = deepcopy(manifest)
    bad["rows"][1]["sha256"] = bad["rows"][0]["sha256"]
    bad["rows"][1]["split"] = "test" if bad["rows"][0]["split"] != "test" else "train"
    with pytest.raises(ValueError, match="crosses"):
        validate_manifest(bad)
    (root / manifest["rows"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash/path"):
        validate_manifest(manifest, root)


def test_missing_decisions_block_review(study_fixture):
    with pytest.raises(ValueError, match="exactly one"):
        validate_decisions(study_fixture[2], [])
