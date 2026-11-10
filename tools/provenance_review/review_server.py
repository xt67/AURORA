import csv, json, os, hashlib, mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REF_ROOT = ROOT / "data" / "reference" / "nlm_original"
INDEX_CSV = ROOT / "outputs" / "provenance" / "manual_review_index.csv"
METRICS_CSV = ROOT / "outputs" / "provenance" / "nlm_candidate_metrics.csv"
REVIEW_CSV = ROOT / "outputs" / "provenance" / "nlm_near_duplicate_review.csv"
MANIFEST_CSV = ROOT / "data" / "reference" / "nlm_original" / "reference_manifest.csv"
DECISIONS_CSV = ROOT / "outputs" / "provenance" / "manual_review_decisions.csv"
CONTACT_DIRS = [
    ROOT / "outputs" / "provenance" / "manual_review_contact_sheets",
    ROOT / "outputs" / "provenance" / "contact_sheets",
    ROOT / "outputs" / "provenance",
]

VALID_DECISIONS = {"CONFIRMED_DUPLICATE", "PROBABLE_DUPLICATE", "SIMILAR_BUT_DISTINCT", "UNRESOLVED"}
VALID_CONFLICTS = {"none", "potential_label_conflict", "unable_to_determine"}
VALID_CONFIDENCE = {"High", "Medium", "Low"}

def load_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def resolve_contact_sheet(name):
    if not name:
        return None
    for d in CONTACT_DIRS:
        p = d / name
        if p.exists():
            return p
        # also try basename under any contact dir
    for d in CONTACT_DIRS:
        for p in d.glob("*.png"):
            if p.name == Path(name).name:
                return p
    return None

def load_reference_manifest():
    out = {}
    for r in load_csv(MANIFEST_CSV):
        out[(r["source_cohort"], r["filename"])] = r
    return out

def load_metrics():
    out = {}
    for r in load_csv(METRICS_CSV):
        key = (r["cohort1"], r["file1"], r["cohort2"], r["file2"])
        out[key] = r
    return out

def load_review_statuses():
    out = {}
    for r in load_csv(REVIEW_CSV):
        key = (r["cohort1"], r["file1"], r["cohort2"], r["file2"])
        out[key] = r.get("verification_status", "")
    return out

REF_MANIFEST = load_reference_manifest()
METRICS = load_metrics()
REVIEW_STATUSES = load_review_statuses()

def image_meta(cohort, filename):
    r = REF_MANIFEST.get((cohort, filename), {})
    path = REF_ROOT / cohort / filename
    dims = r.get("dimensions", "")
    if not dims and path.exists():
        with Image.open(path) as im:
            dims = f"{im.width}x{im.height}"
    return {
        "cohort": cohort,
        "filename": filename,
        "dimensions": dims,
        "sha256": r.get("sha256", ""),
        "path": str(path),
    }

def build_pairs():
    pairs = []
    for r in load_csv(INDEX_CSV):
        key = (r["cohort1"], r["file1"], r["cohort2"], r["file2"])
        m = METRICS.get(key, {})
        a = image_meta(r["cohort1"], r["file1"])
        b = image_meta(r["cohort2"], r["file2"])
        cs = resolve_contact_sheet(r.get("contact_sheet", ""))
        pairs.append({
            "pair_id": r["pair_id"],
            "priority": r["priority"],
            "image_a": a,
            "image_b": b,
            "label_a": r["label1"],
            "label_b": r["label2"],
            "cross_class": r["cross_class"].lower() == "true",
            "phash_distance": m.get("phash_distance", r.get("phash_distance", "")),
            "ncc_512": m.get("ncc_512", r.get("ncc_512", "")),
            "ssim_512": m.get("ssim_512", r.get("ssim_512", "")),
            "ecc": m.get("ecc", r.get("ecc", "")),
            "existing_status": REVIEW_STATUSES.get(key, r.get("existing_verification_status", "")),
            "contact_sheet": str(cs) if cs else "",
            "contact_sheet_name": cs.name if cs else "",
        })
    return pairs

PAIRS = build_pairs()
PAIR_BY_ID = {p["pair_id"]: p for p in PAIRS}

DECISION_FIELDS = [
    "pair_id",
    "image_a_filename",
    "image_b_filename",
    "decision",
    "label_conflict_status",
    "reviewer_confidence",
    "reviewer_notes",
    "review_timestamp",
    "reviewer_identifier",
]

def load_decisions():
    decisions = {}
    if DECISIONS_CSV.exists():
        with DECISIONS_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                decisions[row["pair_id"]] = row
    return decisions

def save_decisions(decisions):
    DECISIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    previous = load_decisions()
    with DECISIONS_CSV.with_suffix(".history.jsonl").open("a", encoding="utf-8") as history:
        for pid, row in decisions.items():
            if previous.get(pid) != row:
                history.write(json.dumps({"pair_id": pid, "previous": previous.get(pid), "replacement": row}) + "\n")
    tmp = DECISIONS_CSV.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_FIELDS)
        writer.writeheader()
        for pid in sorted(decisions):
            writer.writerow({k: decisions[pid].get(k, "") for k in DECISION_FIELDS})
    os.replace(tmp, DECISIONS_CSV)

def safe_image_path(cohort, filename):
    if cohort not in ("shenzhen", "montgomery"):
        return None
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        return None
    path = (REF_ROOT / cohort / filename).resolve()
    try:
        path.relative_to((REF_ROOT / cohort).resolve())
    except ValueError:
        return None
    if not path.is_file():
        return None
    return path

def make_diff(pair_id):
    p = PAIR_BY_ID.get(pair_id)
    if not p:
        return None
    a_path = safe_image_path(p["image_a"]["cohort"], p["image_a"]["filename"])
    b_path = safe_image_path(p["image_b"]["cohort"], p["image_b"]["filename"])
    if not a_path or not b_path:
        return None
    with Image.open(a_path) as ia, Image.open(b_path) as ib:
        ia = ia.convert("L").resize((512, 512))
        ib = ib.convert("L").resize((512, 512))
        import array
        diff = Image.new("L", (512, 512))
        pa = list(ia.getdata())
        pb = list(ib.getdata())
        out = [abs(int(x) - int(y)) for x, y in zip(pa, pb)]
        diff.putdata(out)
    return diff

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path, content_type=None):
        path = Path(path)
        if not path.is_file():
            self.send_error(404)
            return
        if content_type is None:
            content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/":
            self.send_file(Path(__file__).parent / "index.html", "text/html; charset=utf-8")
            return
        if path in ("/styles.css", "/review.js"):
            self.send_file(Path(__file__).parent / path.lstrip("/"))
            return

        if path == "/api/pairs":
            decisions = load_decisions()
            out = []
            for p in PAIRS:
                d = decisions.get(p["pair_id"], {})
                out.append({
                    **p,
                    "decision": d.get("decision", ""),
                    "label_conflict_status": d.get("label_conflict_status", ""),
                    "reviewer_confidence": d.get("reviewer_confidence", ""),
                    "reviewer_notes": d.get("reviewer_notes", ""),
                    "review_timestamp": d.get("review_timestamp", ""),
                    "reviewer_identifier": d.get("reviewer_identifier", ""),
                })
            self.send_json(out)
            return

        if path == "/api/decisions":
            self.send_json(load_decisions())
            return

        if path == "/image":
            cohort = qs.get("cohort", [""])[0]
            filename = qs.get("file", [""])[0]
            p = safe_image_path(cohort, filename)
            if not p:
                self.send_error(404)
                return
            self.send_file(p)
            return

        if path == "/contact":
            pair_id = qs.get("pair", [""])[0]
            p = PAIR_BY_ID.get(pair_id)
            if not p:
                self.send_error(404)
                return
            cs = p.get("contact_sheet", "")
            if not cs:
                self.send_error(404)
                return
            resolved = None
            candidate = Path(cs)
            if candidate.is_file():
                resolved = candidate
            else:
                for d in CONTACT_DIRS:
                    q = d / Path(cs).name
                    if q.is_file():
                        resolved = q
                        break
            if not resolved:
                self.send_error(404)
                return
            self.send_file(resolved)
            return

        if path == "/diff":
            pair_id = qs.get("pair", [""])[0]
            img = make_diff(pair_id)
            if img is None:
                self.send_error(404)
                return
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/decision":
            self.send_error(404)
            return
        origin = self.headers.get("Origin")
        if origin and origin != "http://127.0.0.1:8765":
            self.send_json({"error": "local review origin required"}, 403)
            return
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.send_json({"error": "application/json required"}, 400)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= 65536:
            self.send_json({"error": "invalid request length"}, 400)
            return
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self.send_json({"error": "invalid json"}, 400)
            return
        if not isinstance(body, dict) or not str(body.get("reviewer_identifier", "")).strip():
            self.send_json({"error": "reviewer identifier required"}, 400)
            return
        pair_id = body.get("pair_id", "")
        if pair_id not in PAIR_BY_ID:
            self.send_json({"error": "unknown pair"}, 400)
            return
        decision = body.get("decision", "")
        if decision not in VALID_DECISIONS:
            self.send_json({"error": "invalid decision"}, 400)
            return
        conflict = body.get("label_conflict_status", "none")
        if conflict not in VALID_CONFLICTS:
            self.send_json({"error": "invalid label conflict status"}, 400)
            return
        confidence = body.get("reviewer_confidence", "Medium")
        if confidence not in VALID_CONFIDENCE:
            self.send_json({"error": "invalid confidence"}, 400)
            return
        p = PAIR_BY_ID[pair_id]
        decisions = load_decisions()
        decisions[pair_id] = {
            "pair_id": pair_id,
            "image_a_filename": p["image_a"]["filename"],
            "image_b_filename": p["image_b"]["filename"],
            "decision": decision,
            "label_conflict_status": conflict,
            "reviewer_confidence": confidence,
            "reviewer_notes": body.get("reviewer_notes", ""),
            "review_timestamp": datetime.now(timezone.utc).isoformat(),
            "reviewer_identifier": body.get("reviewer_identifier", ""),
        }
        save_decisions(decisions)
        self.send_json({"ok": True})

def main():
    host = "127.0.0.1"
    port = 8765
    # Single writer prevents concurrent read-modify-write saves losing decisions.
    httpd = HTTPServer((host, port), Handler)
    print(f"AURORA provenance review server running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        httpd.server_close()

if __name__ == "__main__":
    main()
