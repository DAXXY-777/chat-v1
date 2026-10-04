from __future__ import annotations
import argparse, json
from pathlib import Path
from app.rag2 import RAGService as CanonicalService

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="append", required=True)
    parser.add_argument("--output", default="benchmark/chunks.json")
    args = parser.parse_args()

    service = CanonicalService()
    for pdf_value in args.pdf:
        pdf_path = Path(pdf_value).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)
        print(f"\n[PREPARE] Ingesting {pdf_path.name}")
        service.ingest_pdf(pdf_path, pdf_path.name)

    chunks = sorted(service.chunks.values(), key=lambda c: (c["filename"], c["chunk_index"]))
    export = [{
        "chunk_id": c["chunk_id"],
        "filename": c["filename"],
        "chunk_index": c["chunk_index"],
        "pages": c.get("pages", []),
        "headings": c.get("headings", []),
        "text": c["text"],
    } for c in chunks]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[PREPARE] Exported {len(export)} chunks -> {output}")

if __name__ == "__main__":
    main()
