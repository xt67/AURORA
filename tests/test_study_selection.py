from copy import deepcopy
import csv
import json

import pytest
from PIL import Image

from src.study.data import budget_rows, sha256
from src.study.selection import prepare, select_rows, propose_split


@pytest.fixture
def selection_fixture(tmp_path):
    root = tmp_path / "original"
    rows = []
    for cohort in ("shenzhen", "montgomery"):
        (root / cohort).mkdir(parents=True)
        for label in ("normal", "tuberculosis"):
            for i in range(10):
                filename = f"{label}_{i}.png"
                image = root / cohort / filename
                Image.new("L", (8, 8), color=len(rows)).save(image)
                rows.append(dict(filename=filename, source_cohort=cohort, label_inferred=label,
                                 width="8", height="8", sha256=sha256(image)))
    pairs = [dict(pair_id="P1", cohort1="shenzhen", file1="normal_0.png", label1="normal",
                  cohort2="shenzhen", file2="tuberculosis_0.png", label2="tuberculosis", ncc_512="0.8"),
             dict(pair_id="P2", cohort1="shenzhen", file1="normal_0.png", label1="normal",
                  cohort2="montgomery", file2="normal_0.png", label2="normal", ncc_512="0.7")]
    paths = []
    for name, data in (("reference", rows), ("pairs", pairs)):
        path = tmp_path / (name + ".csv")
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        paths.append(path)
    return root, rows, pairs, *paths


def test_excludes_union_and_preserves_every_pair_membership(selection_fixture):
    _, rows, pairs, _, _ = selection_fixture
    kept, excluded = select_rows(rows, pairs)
    assert len(kept) == 37 and len(excluded) == 3
    row = next(r for r in excluded if r["source_cohort"] == "shenzhen" and r["label_inferred"] == "normal")
    assert row["pair_ids"] == "P1;P2"
    assert "not a confirmed duplicate" in row["exclusion_reason"]
    split = propose_split(kept)
    assert split == propose_split(list(reversed(kept)))
    selections = [budget_rows(split["rows"], b, 42) for b in (.25, .5, 1.)]
    positives = [{r["id"] for r in group if r["label"]} for group in selections]
    assert positives[0] <= positives[1] <= positives[2]
    assert len({tuple(r["id"] for r in group if not r["label"]) for group in selections}) == 1


def test_rejects_unaccounted_or_inconsistent_audit(selection_fixture):
    _, rows, pairs, _, _ = selection_fixture
    bad = deepcopy(pairs)
    bad[0]["file1"] = "absent.png"
    with pytest.raises(ValueError, match="absent"):
        select_rows(rows, bad)
    bad = deepcopy(pairs)
    bad[0]["label1"] = "tuberculosis"
    with pytest.raises(ValueError, match="label mismatch"):
        select_rows(rows, bad)
    with pytest.raises(ValueError, match="repeated pair"):
        select_rows(rows, pairs + pairs[:1])
    kept, _ = select_rows(rows, pairs)
    kept[0]["sha256"] = kept[1]["sha256"]
    with pytest.raises(ValueError, match="Repeated retained hash"):
        propose_split(kept)


def test_output_is_non_destructive_and_hash_bound(selection_fixture, tmp_path):
    root, _, _, reference, pairs = selection_fixture
    before = {p: sha256(p) for p in root.glob("*/*.png")}
    before.update({reference: sha256(reference), pairs: sha256(pairs)})
    output = tmp_path / "proposal"
    prepare(reference, pairs, root, output)
    assert (output / "aurora_flagged_pairs_2.csv").read_bytes() == pairs.read_bytes()
    split = json.loads((output / "aurora_primary_split_proposal.json").read_text())
    assert split["source_hashes"]["original_reference"] == before[reference]
    assert split["human_duplicate_adjudication"] is False
    assert all(sha256(p) == digest for p, digest in before.items())
    with pytest.raises(FileExistsError):
        prepare(reference, pairs, root, output)
    image = next(root.glob("*/*.png"))
    image.write_bytes(b"altered")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare(reference, pairs, root, tmp_path / "other")
    assert not (tmp_path / "other").exists()
