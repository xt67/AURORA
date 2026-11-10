import importlib.util
from pathlib import Path
import json


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
