import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


from .core import get_response, load_default_model, ModelLoadError
from .dev_dummy import DummyTokenizer, DummyModel

MODEL_DIR = os.environ.get("MODEL_DIR", "distilgpt2")
USE_DUMMY = os.environ.get("USE_DUMMY_MODEL", "0") == "1"

app = FastAPI(title="Realtime Chatbot")


class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.7


# Lazy load
_runtime = {"tokenizer": None, "model": None, "device": None}


@app.on_event("startup")
async def load_model_event():
    try:
        if USE_DUMMY:
            tokenizer, model, device = DummyTokenizer(), DummyModel(), None
        else:
            tokenizer, model, device = load_default_model(MODEL_DIR)
        _runtime.update({"tokenizer": tokenizer, "model": model, "device": device})
    except ModelLoadError:
        # keep runtime empty; endpoint will return 503 until user injects a model
        pass


@app.post("/chat")
async def chat(req: ChatRequest):
    tokenizer = _runtime.get("tokenizer")
    model = _runtime.get("model")
    device = _runtime.get("device")
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        text, latency_ms = get_response(
            prompt=req.prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            device=device,
        )
        return {"response": text, "latency_ms": latency_ms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
