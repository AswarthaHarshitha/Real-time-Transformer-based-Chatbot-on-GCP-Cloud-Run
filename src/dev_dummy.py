class DummyTokenizer:
    def __call__(self, text, return_tensors=None):
        # return a simple mapping compatible with get_response
        return {"input_ids": text}

    def decode(self, seq, skip_special_tokens=True):
        # if seq is a list of ids, join them; otherwise return str
        if isinstance(seq, (list, tuple)):
            return " ".join(str(x) for x in seq)
        return str(seq)


class DummyModel:
    def generate(self, **kwargs):
        # return a nested list to mimic token ids
        return [["dummy-response"]]

    def __call__(self, prompt):
        return "dummy-response"
