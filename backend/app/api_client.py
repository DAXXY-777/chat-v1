import os
from pathlib import Path
import httpx


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)


def stream_llama(prompt: str):
    """
    Sends a prompt to the FastAPI backend
    and yields streamed text chunks.
    """

    payload = {
        "prompt": prompt,
    }

    with httpx.stream(
        "POST",
        f"{API_BASE_URL}/llama",
        json=payload,
        timeout=None,
    ) as response:

        response.raise_for_status()

        for chunk in response.iter_text():
            if chunk:
                yield chunk

def ingest_pdf(file_path: str):

    path = Path(file_path)

    with path.open("rb") as file:

        response = httpx.post(
            f"{API_BASE_URL}/ingest",
            files={
                "file": (
                    path.name,
                    file,
                    "application/pdf",
                )
            },
            timeout=None,
        )

    response.raise_for_status()

    return response.json()