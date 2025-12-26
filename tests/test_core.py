import time

from src.core import get_response


class DummyTokenizer:
    def __call__(self, text, return_tensors=None):
        # return a simple dict to simulate token tensors
        return {"input_ids": text}

    def decode(self, sequence, skip_special_tokens=True):
        return str(sequence)


class DummyModel:
    def __init__(self, reply="dummy reply"):
        self.reply = reply

    def __call__(self, prompt):
        return self.reply


def test_get_response_basic():
    tok = DummyTokenizer()
    model = DummyModel("hello from dummy")
    text, latency = get_response("hi", model=model, tokenizer=tok, max_new_tokens=10, temperature=0.5, device=None)
    assert isinstance(text, str)
    assert latency >= 0
    assert "dummy" in text
