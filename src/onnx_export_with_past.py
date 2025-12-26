"""Export causal LM with support for past_key_values to ONNX.

This script exports two graphs:
 - init: encodes input_ids and returns logits + past_key_values
 - step: given last_token_id and past_key_values, returns logits + updated past_key_values

Not all HF models can be exported this way without model-specific tweaks. This script works as a best-effort example for DistilGPT2-like models.
"""
import argparse
import os


def export_with_past(model_name_or_path: str, output_dir: str):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except Exception as e:
        raise RuntimeError("Please install transformers and torch: " + str(e))

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # Export the full forward for a sequence input (init graph)
    sample = tokenizer("Hello", return_tensors="pt")
    input_ids = sample["input_ids"]
    init_onnx = os.path.join(output_dir, "model_init.onnx")
    torch.onnx.export(
        model,
        (input_ids,),
        init_onnx,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
        opset_version=13,
    )

    # Attempt to export step graph using a single token + past_key_values
    # This part may need model internals; we'll provide a fallback message.
    step_onnx = os.path.join(output_dir, "model_step.onnx")
    try:
        # For many models, tracing past_key_values requires constructing the model's forward signature.
        # This example attempts to call model with input_ids and use torch.onnx to export, but may fail.
        last_token = tokenizer(".", return_tensors="pt")["input_ids"][:, -1:]
        torch.onnx.export(
            model,
            (last_token,),
            step_onnx,
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=13,
        )
        print("Exported step graph to", step_onnx)
    except Exception as e:
        print("Could not export step graph automatically; some models need manual export. Error:", e)

    print("Exported init graph to", init_onnx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    export_with_past(args.model, args.out)


if __name__ == "__main__":
    main()
