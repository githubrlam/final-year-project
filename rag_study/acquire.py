"""Acquire pinned FinanceBench metadata and the complete 26-report corpus."""
import hashlib
import json
from pathlib import Path
import requests
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]

def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

def download(url, path):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(response.content)
    temporary.replace(path)

def acquire():
    DATA.mkdir(exist_ok=True)
    source = DATA / "source"
    source.mkdir(exist_ok=True)
    provenance = source / "provenance.json"
    if provenance.exists():
        revision = json.loads(provenance.read_text())["revision"]
    else:
        response = requests.get("https://api.github.com/repos/patronus-ai/financebench/commits/main", timeout=60)
        response.raise_for_status()
        revision = response.json()["sha"]
        provenance.write_text(json.dumps({"repository": "https://github.com/patronus-ai/financebench", "revision": revision,
            "benchmark_license": "CC BY-NC 4.0; https://huggingface.co/datasets/PatronusAI/financebench",
            "filing_rights": "Issuer reports retain their respective rights; the benchmark license is not a blanket filing license."}, indent=2))
    base = f"https://raw.githubusercontent.com/patronus-ai/financebench/{revision}"
    for name in ("financebench_open_source.jsonl", "financebench_document_information.jsonl"):
        if not (source / name).exists():
            download(f"{base}/data/{name}", source / name)
    questions = read_jsonl(source / "financebench_open_source.jsonl")
    documents = {d["doc_name"]: d for d in read_jsonl(source / "financebench_document_information.jsonl")}
    test = [q for q in questions if documents[q["doc_name"]]["doc_type"] == "10k"
            and int(documents[q["doc_name"]]["doc_period"]) == 2022
            and all(e.get("doc_name", e.get("evidence_doc_name")) == q["doc_name"] for e in q["evidence"])]
    companies = {q["company"] for q in test}
    dev = sorted([q for q in questions if documents[q["doc_name"]]["doc_type"] == "10k"
                  and int(documents[q["doc_name"]]["doc_period"]) < 2022 and q["company"] not in companies
                  and all(e.get("doc_name", e.get("evidence_doc_name")) == q["doc_name"] for e in q["evidence"])],
                 key=lambda q: q["financebench_id"])[:10]
    assert len(test) == 52 and len(dev) == 10, "Released data changed: review the protocol before proceeding."
    assert len({q["doc_name"] for q in test}) == 18 and len({q["doc_name"] for q in dev}) == 8
    write_jsonl(DATA / "dev.jsonl", dev)
    write_jsonl(DATA / "test.jsonl", sorted(test, key=lambda q: q["financebench_id"]))
    pdfs = DATA / "pdfs"
    pdfs.mkdir(exist_ok=True)
    manifest = []
    for name in sorted({q["doc_name"] for q in dev + test}):
        path = pdfs / f"{name}.pdf"
        url = f"{base}/pdfs/{name}.pdf"
        if not path.exists():
            download(url, path)
        if not path.read_bytes().startswith(b"%PDF-"):
            raise ValueError(f"Not a PDF: {name}")
        page_count = len(PdfReader(path).pages)
        qs = [q for q in dev + test if q["doc_name"] == name]
        assert all(0 <= e["evidence_page_num"] < page_count for q in qs for e in q["evidence"])
        manifest.append({**documents[name], "download_url": url, "sha256": sha256(path), "bytes": path.stat().st_size,
                         "pdf_pages": page_count, "split": "dev" if name in {q["doc_name"] for q in dev} else "test"})
        print(name, page_count, "pages", flush=True)
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    import pandas as pd
    pd.DataFrame(manifest).to_csv(DATA / "manifest.csv", index=False)
    print("Acquired 10 development questions, 52 test questions and 26 full reports.")

if __name__ == "__main__":
    acquire()
