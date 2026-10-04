from __future__ import annotations

import argparse, csv, gc, json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from app.rag2 import (
    BM25_TOP_K, COLLECTION_NAME as CHUNK_COLLECTION, DENSE_TOP_K,
    RERANK_CANDIDATES, RRF_K, RAGService as HyDEService,
)
from app.rag3 import RAGService as HyPEService
from app.retrieval_utils import bm25_search, rrf_fuse
from benchmark.metrics import aggregate_query_rows, per_query_metrics

BENCHMARK_FINAL_TOP_K = 10

def load_queries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("queries.json must be a non-empty JSON list.")
    seen = set()
    for item in data:
        qid = str(item.get("query_id", "")).strip()
        question = str(item.get("question", "")).strip()
        refs = item.get("relevant_chunks")
        if not qid or qid in seen:
            raise ValueError(f"Missing/duplicate query_id: {qid!r}")
        seen.add(qid)
        if not question:
            raise ValueError(f"{qid}: empty question")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"{qid}: relevant_chunks must be non-empty")
        for ref in refs:
            if "filename" not in ref or "chunk_index" not in ref:
                raise ValueError(f"{qid}: relevant chunk needs filename + chunk_index")
    return data

def build_chunk_ref_map(chunks: dict[str, dict[str, Any]]) -> dict[tuple[str, int], str]:
    out = {}
    for cid, chunk in chunks.items():
        key = (str(chunk["filename"]), int(chunk["chunk_index"]))
        if key in out and out[key] != cid:
            raise RuntimeError(f"Duplicate canonical chunk reference: {key}")
        out[key] = cid
    return out

def resolve_relevant_ids(item: dict[str, Any], ref_map: dict[tuple[str, int], str]) -> set[str]:
    ids = set()
    for ref in item["relevant_chunks"]:
        key = (str(ref["filename"]), int(ref["chunk_index"]))
        if key not in ref_map:
            raise KeyError(f"{item['query_id']}: relevant chunk {key} not in indexed corpus")
        ids.add(ref_map[key])
    return ids

def naive_dense_search(service: HyDEService, query: str, top_k: int = DENSE_TOP_K):
    start = perf_counter()
    vector = service.embedder.encode(
        [query], prompt_name="query", normalize_embeddings=True, show_progress_bar=False
    )[0]
    embedding_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    response = service.qdrant.query_points(
        collection_name=CHUNK_COLLECTION,
        query=vector.tolist(),
        limit=top_k,
        with_payload=True,
    )
    vector_search_ms = (perf_counter() - start) * 1000

    results = []
    for rank, hit in enumerate(response.points, start=1):
        payload = hit.payload or {}
        cid = payload.get("chunk_id")
        if cid and cid in service.chunks:
            results.append({
                "chunk_id": cid,
                "rank": rank,
                "score": float(hit.score),
                "source": "dense",
            })
    return results, {
        "query_embedding_ms": round(embedding_ms, 2),
        "vector_search_ms": round(vector_search_ms, 2),
    }

def strategy_dense_search(service: Any, strategy: str, query: str):
    if strategy == "naive":
        return naive_dense_search(service, query, DENSE_TOP_K)
    if strategy == "hyde":
        return service._hyde_dense_search(query, DENSE_TOP_K)
    if strategy == "hype":
        return service._hype_dense_search(query, DENSE_TOP_K)
    raise ValueError(strategy)

def run_search(service: Any, strategy: str, mode: str, query: str, rerank_candidates: int):
    total_start = perf_counter()
    dense, metrics = strategy_dense_search(service, strategy, query)

    if mode == "pure":
        metrics = dict(metrics)
        metrics.update({
            "total_retrieval_ms": round((perf_counter() - total_start) * 1000, 2),
            "strategy": strategy,
            "mode": "pure",
            "dense_candidates": len(dense),
        })
        return dense, metrics

    start = perf_counter()
    lexical = bm25_search(service.bm25, service.bm25_chunk_ids, query, top_k=BM25_TOP_K)
    bm25_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    fused = rrf_fuse(dense, lexical, k=RRF_K)
    rrf_ms = (perf_counter() - start) * 1000

    start = perf_counter()
    final = service.reranker.rerank(
        query=query,
        candidates=fused,
        chunks=service.chunks,
        candidate_limit=rerank_candidates,
        top_k=BENCHMARK_FINAL_TOP_K,
    )
    rerank_ms = (perf_counter() - start) * 1000

    metrics = dict(metrics)
    metrics.update({
        "bm25_ms": round(bm25_ms, 2),
        "rrf_ms": round(rrf_ms, 2),
        "rerank_ms": round(rerank_ms, 2),
        "total_retrieval_ms": round((perf_counter() - total_start) * 1000, 2),
        "strategy": strategy,
        "mode": "hybrid",
        "dense_candidates": len(dense),
        "bm25_candidates": len(lexical),
        "fused_candidates": len(fused),
        "final_candidates": len(final),
    })
    return final, metrics

def sum_ingest_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    keys = set()
    for item in items:
        keys.update(k for k, v in item.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
    totals = {}
    for key in sorted(keys):
        if key == "vectors_per_chunk":
            continue
        totals[key] = round(sum(float(item.get(key, 0.0)) for item in items), 3)
    chunks = totals.get("chunks", 0.0)
    qvecs = totals.get("question_vectors")
    if chunks and qvecs is not None:
        totals["vectors_per_chunk"] = round(qvecs / chunks, 3)
    return totals

def ingest_corpus(service: Any, pdfs: list[Path]) -> dict[str, Any]:
    metrics = []
    for pdf in pdfs:
        print(f"\n[BENCH] Ingesting {pdf.name}")
        result = service.ingest_pdf(pdf, pdf.name)
        if result.get("metrics"):
            metrics.append(dict(result["metrics"]))
    return sum_ingest_metrics(metrics)

def benchmark_strategy(service: Any, strategy: str, mode: str, queries, ref_map, rerank_candidates, warmup: bool):
    if warmup:
        print(f"[BENCH] Warm-up: {strategy}/{mode}")
        run_search(service, strategy, mode, queries[0]["question"], rerank_candidates)

    rows = []
    for i, item in enumerate(queries, start=1):
        print(f"[BENCH] {strategy}/{mode} {i}/{len(queries)}: {item['query_id']}")
        relevant_ids = resolve_relevant_ids(item, ref_map)
        results, timing = run_search(service, strategy, mode, item["question"], rerank_candidates)
        ranked_ids = [r["chunk_id"] for r in results]
        rows.append({
            "strategy": strategy,
            "mode": mode,
            "query_id": item["query_id"],
            "question": item["question"],
            "category": item.get("category", ""),
            "relevant_chunk_ids": sorted(relevant_ids),
            "retrieved_chunk_ids": ranked_ids,
            **per_query_metrics(ranked_ids, relevant_ids),
            **timing,
        })
    return rows

def release_strategy_models(service: Any) -> None:
    # Keep the shared llama.cpp instance cached, but release strategy-specific
    # SentenceTransformer / CrossEncoder models before creating the next service.
    if hasattr(service, "_embedder"):
        service._embedder = None
    reranker = getattr(service, "reranker", None)
    if reranker is not None and hasattr(reranker, "_model"):
        reranker._model = None
    qdrant = getattr(service, "qdrant", None)
    if qdrant is not None and hasattr(qdrant, "close"):
        qdrant.close()

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def aggregate_and_append(summaries, strategy, mode, rows):
    summaries.append({"strategy": strategy, "mode": mode, **aggregate_query_rows(rows)})

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = [k for k, v in rows[0].items() if not isinstance(v, (list, dict))]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="append", required=True)
    parser.add_argument("--queries", default="benchmark/queries.json")
    parser.add_argument("--mode", choices=["pure", "hybrid", "both"], default="pure")
    parser.add_argument("--output-dir", default="benchmark/results")
    parser.add_argument("--rerank-candidates", type=int, default=RERANK_CANDIDATES)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    pdfs = [Path(p).resolve() for p in args.pdf]
    for pdf in pdfs:
        if not pdf.exists():
            raise FileNotFoundError(pdf)

    queries = load_queries(Path(args.queries))
    modes = ["pure", "hybrid"] if args.mode == "both" else [args.mode]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_rows, summaries = [], []
    ingestion = {}

    print("\n[BENCH] Building shared Naive/HyDE chunk index...")
    hyde_service = HyDEService()
    hyde_ingest = ingest_corpus(hyde_service, pdfs)
    ref_map = build_chunk_ref_map(hyde_service.chunks)

    ingestion["naive"] = {**hyde_ingest, "vectors_per_chunk": 1.0}
    ingestion["hyde"] = {**hyde_ingest, "vectors_per_chunk": 1.0}

    for strategy in ("naive", "hyde"):
        for mode in modes:
            rows = benchmark_strategy(
                hyde_service, strategy, mode, queries, ref_map,
                args.rerank_candidates, not args.no_warmup
            )
            all_rows.extend(rows)
            aggregate_and_append(summaries, strategy, mode, rows)

    release_strategy_models(hyde_service)
    del hyde_service
    gc.collect()

    print("\n[BENCH] Building HyPE question-vector index...")
    hype_service = HyPEService()
    ingestion["hype"] = ingest_corpus(hype_service, pdfs)
    hype_ref_map = build_chunk_ref_map(hype_service.chunks)

    if set(ref_map.keys()) != set(hype_ref_map.keys()):
        raise RuntimeError("Naive/HyDE and HyPE produced different canonical chunks.")
    for key, cid in ref_map.items():
        if hype_ref_map[key] != cid:
            raise RuntimeError(f"Stable chunk ID mismatch for {key}")

    for mode in modes:
        rows = benchmark_strategy(
            hype_service, "hype", mode, queries, hype_ref_map,
            args.rerank_candidates, not args.no_warmup
        )
        all_rows.extend(rows)
        aggregate_and_append(summaries, "hype", mode, rows)

    release_strategy_models(hype_service)
    del hype_service
    gc.collect()

    naive_vectors = float(ingestion["naive"].get("vectors", ingestion["naive"].get("chunks", 0)))
    for strategy, item in ingestion.items():
        vector_count = float(
            item.get("question_vectors", 0) if strategy == "hype"
            else item.get("vectors", item.get("chunks", 0))
        )
        item["vector_count"] = int(vector_count)
        item["raw_vector_bytes_estimate"] = int(vector_count * 1024 * 4)
        item["vector_count_multiplier_vs_naive"] = (
            round(vector_count / naive_vectors, 4) if naive_vectors else None
        )

    payload = {
        "run_id": run_id,
        "pdfs": [str(p) for p in pdfs],
        "query_file": str(Path(args.queries)),
        "query_count": len(queries),
        "mode": args.mode,
        "warmup": not args.no_warmup,
        "rerank_candidates": args.rerank_candidates,
        "benchmark_final_top_k": BENCHMARK_FINAL_TOP_K,
        "summaries": summaries,
        "ingestion": ingestion,
        "storage_note": (
            "Qdrant is in-memory; vector count and raw-vector byte estimates are reported, "
            "not measured on-disk index size."
        ),
    }

    summary_path = output_dir / f"{run_id}_summary.json"
    details_path = output_dir / f"{run_id}_queries.json"
    csv_path = output_dir / f"{run_id}_queries.csv"

    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    details_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, all_rows)

    print("\n[BENCH] COMPLETE")
    print(f"[BENCH] Summary -> {summary_path}")
    print(f"[BENCH] Query details -> {details_path}")
    print(f"[BENCH] CSV -> {csv_path}")
    for summary in summaries:
        print(summary)

if __name__ == "__main__":
    main()
