# Benchmark quick start

Place this `benchmark/` directory directly under `backend/`.

## 1. Export chunks for annotation

```powershell
uv run python -m benchmark.prepare --pdf "path\to\document.pdf"
```

This creates `benchmark/chunks.json`.

## 2. Create `queries.json`

Copy `queries.example.json` to `queries.json`.

For each natural question, inspect `chunks.json` and record all relevant chunks
using `filename` + `chunk_index`.

Do not reuse the exact HyPE-generated questions as benchmark queries.

## 3. Run pure retrieval first

```powershell
uv run python -m benchmark.run `
  --pdf "path\to\document.pdf" `
  --queries benchmark\queries.json `
  --mode pure
```

This benchmarks Naive, HyDE and HyPE and writes JSON/CSV to `benchmark/results/`.

## 4. Run hybrid

```powershell
uv run python -m benchmark.run `
  --pdf "path\to\document.pdf" `
  --queries benchmark\queries.json `
  --mode hybrid
```

The runner warms each strategy before measured queries. It uses the same BM25,
RRF and reranker for all strategies.

## GPU note

The benchmark imports `rag2.py` and `rag3.py` directly, so the `get_rag_service`
import in `app/main.py` does not matter during benchmarking.

In the current repo, embeddings and the reranker are hard-coded to CUDA.
`llama.cpp` uses `n_gpu_layers=-1`, but your package index currently points to
the Vulkan llama-cpp wheel source. Docling is still created with plain
`DocumentConverter()`, so its accelerator is not explicitly forced to CUDA.
