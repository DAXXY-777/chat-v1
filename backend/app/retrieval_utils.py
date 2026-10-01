from __future__ import annotations

import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"


def _sha256_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def stable_document_id(path: Path) -> str:
    return str(uuid5(NAMESPACE_URL, f"chat-v1:document:{_sha256_file(path)}"))


def stable_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    content_hash = sha256(text.encode("utf-8")).hexdigest()
    return str(
        uuid5(
            NAMESPACE_URL,
            f"chat-v1:chunk:{document_id}:{chunk_index}:{content_hash}",
        )
    )


def stable_question_id(chunk_id: str, question_index: int, question: str) -> str:
    normalized = " ".join(question.lower().split())
    return str(
        uuid5(
            NAMESPACE_URL,
            f"chat-v1:hype-question:{chunk_id}:{question_index}:{normalized}",
        )
    )


def tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def build_bm25(
    chunks: dict[str, dict[str, Any]],
) -> tuple[BM25Okapi | None, list[str]]:
    chunk_ids = list(chunks.keys())
    if not chunk_ids:
        return None, []

    corpus = [tokenize_bm25(chunks[chunk_id]["text"]) for chunk_id in chunk_ids]
    return BM25Okapi(corpus), chunk_ids


def bm25_search(
    bm25: BM25Okapi | None,
    chunk_ids: list[str],
    query: str,
    top_k: int = 30,
) -> list[dict[str, Any]]:
    if bm25 is None or not chunk_ids:
        return []

    scores = bm25.get_scores(tokenize_bm25(query))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    results: list[dict[str, Any]] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            continue

        results.append(
            {
                "chunk_id": chunk_ids[index],
                "rank": len(results) + 1,
                "score": score,
                "source": "bm25",
            }
        )

        if len(results) >= top_k:
            break

    return results


def rrf_fuse(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for result_list in (dense_results, bm25_results):
        for result in result_list:
            chunk_id = result["chunk_id"]
            source = result["source"]

            item = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                },
            )

            item["rrf_score"] += 1.0 / (k + int(result["rank"]))
            item[f"{source}_rank"] = int(result["rank"])
            item[f"{source}_score"] = float(result["score"])

            if "matched_question" in result:
                item["matched_question"] = result["matched_question"]

    ranked = sorted(
        fused.values(),
        key=lambda item: (-item["rrf_score"], item["chunk_id"]),
    )

    for rank, item in enumerate(ranked, start=1):
        item["rrf_rank"] = rank

    return ranked


class LocalReranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            print(f"[RERANK] Loading {self.model_name}...")
            self._model = CrossEncoder(
                self.model_name,
                device=os.getenv("RERANK_DEVICE", "cpu"),
            )
            print("[RERANK] Ready")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        chunks: dict[str, dict[str, Any]],
        candidate_limit: int = 20,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        candidates = candidates[:candidate_limit]
        if not candidates:
            return []

        pairs = [
            (query, chunks[candidate["chunk_id"]]["text"])
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        reranked: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["reranker_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: (-item["reranker_score"], item["chunk_id"])
        )

        for rank, item in enumerate(reranked, start=1):
            item["final_rank"] = rank

        return reranked[:top_k]
