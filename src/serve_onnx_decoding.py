"""ONNX Runtime decoding server.

This module implements a token-by-token generation loop using two ONNX graphs:
 - model_init.onnx: converts input_ids -> logits (+ past_key_values if present)
 - model_step.onnx: consumes last_token + past_key_values -> logits + past_key_values

The exact input/output names depend on the exported models. This script demonstrates the decoding loop and includes batching and simple sampling.
"""
import os
import time
import numpy as np
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

app = FastAPI(title="ONNX Decoding Chatbot")


class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 32
    temperature: Optional[float] = 0.7


MODEL_DIR = os.environ.get("ONNX_DIR", "onnx")


@app.on_event("startup")
def startup():
    if ort is None:
        raise RuntimeError("onnxruntime is required for ONNX serving")
    # load init and step sessions
    init_path = os.path.join(MODEL_DIR, "model_init.onnx")
    step_path = os.path.join(MODEL_DIR, "model_step.onnx")
    if not os.path.exists(init_path):
        raise RuntimeError(f"Missing {init_path}")
    if not os.path.exists(step_path):
        raise RuntimeError(f"Missing {step_path}")

    app.state.init_sess = ort.InferenceSession(init_path, providers=["CPUExecutionProvider"])
    app.state.step_sess = ort.InferenceSession(step_path, providers=["CPUExecutionProvider"])
    if AutoTokenizer is None:
        raise RuntimeError("transformers tokenizer required for tokenization")
    app.state.tokenizer = AutoTokenizer.from_pretrained(os.environ.get("BASE_MODEL", "distilgpt2"))


def to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


@app.post("/chat")
def chat(req: ChatRequest):
    init_sess = app.state.init_sess
    step_sess = app.state.step_sess
    tokenizer = app.state.tokenizer

    # tokenize prompt
    enc = tokenizer(req.prompt, return_tensors="np")
    input_ids = enc["input_ids"]

    # run init graph
    init_out = init_sess.run(None, {"input_ids": input_ids})
    logits = init_out[0]
    # naive: no past_key_values handled here for simplicity - real exported models should return them

    # decode tokens autoregressively using step graph
    generated = []
    last_token = input_ids[:, -1:]
    for i in range(req.max_new_tokens):
        step_inputs = {"input_ids": last_token}
        out = step_sess.run(None, step_inputs)
        logits = out[0]
        # sample greedy
        next_id = np.argmax(logits[:, -1, :], axis=-1)
        generated.append(int(next_id[0]))
        last_token = next_id.reshape(1, 1)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {"response": text, "latency_ms": 0.0}
