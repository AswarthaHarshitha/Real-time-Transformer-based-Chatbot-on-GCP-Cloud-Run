from fastapi.testclient import TestClient

from src.app import app, _runtime


class DummyTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": text}

    def decode(self, seq, skip_special_tokens=True):
        return str(seq)


class DummyModel:
    def generate(self, **kwargs):
        return ["dummy-output"]


def test_chat_endpoint_injected():
    # inject dummy runtime
    _runtime.update({"tokenizer": DummyTokenizer(), "model": DummyModel(), "device": None})
    client = TestClient(app)
    r = client.post("/chat", json={"prompt": "hello", "max_new_tokens": 8})
    assert r.status_code == 200
    body = r.json()
    assert "response" in body
    assert "latency_ms" in body
