from functools import lru_cache
from pathlib import Path

from llama_cpp import Llama


@lru_cache(maxsize=1)
def get_llm() -> Llama:
    """Return the single shared llama.cpp model instance used by chat, HyDE and HyPE."""
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "gguf-models" / "Qwen3.5-9B-Q4_K_M.gguf"

    return Llama(
        model_path=str(model_path),
        n_gpu_layers=-1,
        n_ctx=8192,
        verbose=False,
    )
