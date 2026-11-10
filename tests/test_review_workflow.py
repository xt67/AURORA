import importlib.util
from pathlib import Path
import json
import hashlib


def load_server():
    spec = importlib.util.spec_from_file_location("review_server", "tools/provenance_review/review_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_review_preserves_previous_decisions(tmp_path, monkeypatch):
    server = load_server()
    monkeypatch.setattr(server, "DECISIONS_CSV", tmp_path / "decisions.csv")
    first = {"TEST": {"pair_id": "TEST", "decision": "UNRESOLVED", "reviewer_identifier": "fixture"}}
    server.save_decisions(first)
    second = {"TEST": {"pair_id": "TEST", "decision": "SIMILAR_BUT_DISTINCT", "reviewer_identifier": "fixture"}}
    server.save_decisions(second)
    assert server.load_decisions()["TEST"]["decision"] == "SIMILAR_BUT_DISTINCT"
    history = [json.loads(line) for line in (tmp_path / "decisions.history.jsonl").read_text().splitlines()]
    assert history[-1]["previous"]["decision"] == "UNRESOLVED"


def test_local_review_rejects_path_traversal(tmp_path, monkeypatch):
    server = load_server()
    monkeypatch.setattr(server, "REF_ROOT", tmp_path)
    assert server.safe_image_path("shenzhen", "../private.png") is None
    assert server.safe_image_path("other", "image.png") is None


def test_ai_notes_are_separate_and_reject_stale_or_mismatched_sources(tmp_path, monkeypatch):
    server = load_server()
    index = tmp_path / "index.csv"
    reference = tmp_path / "reference.csv"
    notes = tmp_path / "ai.json"
    decisions = tmp_path / "human.csv"
    index.write_text("pair_id,file1,file2\nTEST,a.png,b.png\n")
    reference.write_text("fixture reference")
    monkeypatch.setattr(server, "INDEX_CSV", index)
    monkeypatch.setattr(server, "MANIFEST_CSV", reference)
    monkeypatch.setattr(server, "AI_REVIEW_JSON", notes)
    monkeypatch.setattr(server, "DECISIONS_CSV", decisions)
    bundle = {"status": "AI_ASSISTED_RECOMMENDATIONS_NOT_HUMAN_DECISIONS",
              "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
              "recommendations": [{"pair_id": "TEST", "image_a_filename": "a.png",
                                   "image_b_filename": "b.png", "recommendation": "UNRESOLVED"}]}
    notes.write_text(json.dumps(bundle))
    assert server.load_ai_recommendations()["TEST"]["recommendation"] == "UNRESOLVED"
    assert server.load_decisions() == {} and not decisions.exists()
    bundle["recommendations"][0]["image_a_filename"] = "wrong.png"
    notes.write_text(json.dumps(bundle))
    assert server.load_ai_recommendations() == {}
    bundle["recommendations"][0]["image_a_filename"] = "a.png"
    notes.write_text(json.dumps(bundle))
    reference.write_text("changed")
    assert server.load_ai_recommendations() == {}
    notes.write_text("invalid JSON")
    assert server.load_ai_recommendations() == {}
