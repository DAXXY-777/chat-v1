from fastapi import FastAPI,File,HTTPException,UploadFile
from pathlib import Path
from app.snippets import posts
from app.base import ChatRequest as chatrequest
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

import shutil
import tempfile
from app.rag import get_rag_service

from llama_cpp import Llama



system_prompt = """
You are a helpful, accurate, and thoughtful AI assistant.

Your goals are:

- Answer questions clearly and correctly.
- Ask clarifying questions when needed.
- Explain complex topics in simple language.
- Be honest about uncertainty and avoid making up facts.
- Adapt your tone to the user's preferences.
- Provide step-by-step guidance for technical or complex tasks.
- Prioritize safety, privacy, and user well-being.
- Format responses with headings, bullet points, and examples when helpful.
- If you don't know something, say so and suggest ways to find the answer.
- Avoid unnecessary verbosity while remaining complete.
"""


app = FastAPI()

rag = get_rag_service()

# Directory containing this Python file
BASE_DIR = Path(__file__).resolve().parent

# Your existing local model
MODEL_PATH = BASE_DIR / "gguf-models" / "Qwen3.5-9B-Q4_K_M.gguf"

llm = Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=-1,   # offloaded model layers to gpu
    n_ctx=8192,         #total context length
    verbose=False,
)


@app.get("/")
def home():
    return {"message": "hello world"}


@app.get("/posts")
def get_posts():
    return posts



@app.post("/ingest")
def ingest_pdf(
    file: UploadFile = File(...),
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    # Prevent strange filenames/path traversal
    filename = Path(file.filename).name

    with tempfile.TemporaryDirectory() as temp_dir:

        pdf_path = (
            Path(temp_dir)
            / filename
        )

        with pdf_path.open("wb") as output:

            shutil.copyfileobj(
                file.file,
                output,
            )

        result = rag.ingest_pdf(
            pdf_path=pdf_path,
            filename=filename,
        )

    return {
        "status": "ok",
        **result,
    }


@app.post("/llama")
def chat(request: chatrequest):

    def stream_response():

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": rag.build_prompt(request.prompt)
                },
            ],

            stream=True,
            max_tokens=2048,

            # Optional generation settings
            temperature=0.7,
            top_p=0.8,
        )

        for chunk in response:
            delta = chunk["choices"][0]["delta"]

            content = delta.get("content")

            if content:
                yield content

    return StreamingResponse(
        stream_response(),
        media_type="text/plain; charset=utf-8",
    )


@app.post("/cloud")
async def llama(request: chatrequest):


    stream = await llama_client.chat.completions.create(
        model = "qwen3-8b",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": request.prompt,
            },
        ],
        stream=True,
    )

    async def stream_response():
        try:
            async for chunk in stream:

                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    yield content

        finally:
            await stream.close()

    return StreamingResponse(
        stream_response(),
        media_type="text/plain",
    )    