from fastapi import FastAPI
import ollama
from app.snippets import posts
from app.base import ChatRequest as chatrequest
from fastapi.responses import StreamingResponse

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

@app.get("/")
def home():
    return {"message": "hello world"}

@app.get("/posts")
def get_posts():
    return posts

@app.post("/chat")
def chat(request: chatrequest):
    response = ollama.chat(
        model="qwen3:8b",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {request.prompt}"},
        ],
    )
    def stream_response():
        for chunk in response:
            yield chunk["message"]["content"]

    return StreamingResponse(stream_response(), media_type="text/plain")
