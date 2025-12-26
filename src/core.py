import time
from typing import Tuple, Optional

class ModelLoadError(Exception):
    pass


def load_default_model(model_dir: str):
    """Try to lazily import transformers and torch and load a model.

    Returns (tokenizer, model, device) or raises ModelLoadError.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except Exception as e:
        raise ModelLoadError("ML dependencies not available: " + str(e))

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        raise ModelLoadError(str(e))


def get_response(prompt: str, model, tokenizer, max_new_tokens: int = 64, temperature: float = 0.7, device: Optional[str] = None) -> Tuple[str, float]:
    """Run a generation and return (text, latency_ms).

    This function is intentionally small so it can be unit tested with a dummy model/tokenizer.
    """
    start = time.time()
    # The real model/tokenizer will accept these calls. For tests, provide dummy objects.
    inputs = tokenizer(prompt, return_tensors="pt")
    # move tensors if torch available and device passed
    try:
        import torch
        if device is not None and isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        pass

    # Use model.generate if exists, otherwise call model.__call__ for dummy
    try:
        # prefer generate when available
        gen = getattr(model, "generate", None)
        if gen is not None:
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, pad_token_id=getattr(tokenizer, 'eos_token_id', None))
            text = tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            # fallback: the model returns text directly
            text = model(prompt)
    except Exception:
        # fallback for dummy objects
        try:
            text = model(prompt)
        except Exception:
            text = ""

    latency_ms = (time.time() - start) * 1000
    return text, latency_ms
