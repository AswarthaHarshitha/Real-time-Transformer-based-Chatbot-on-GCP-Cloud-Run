"""Quantize an ONNX model to int8 using onnxruntime.quantization (dynamic quantization).

Usage:
    python src/quantize_onnx.py --input onnx/model.onnx --output onnx/model.quant.onnx
"""
import argparse
import os


def quantize(input_model: str, output_model: str):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as e:
        raise RuntimeError("Please install onnxruntime-tools/onnxruntime: " + str(e))

    quantize_dynamic(input_model, output_model, weight_type=QuantType.QInt8)
    print("Wrote quantized model to", output_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    quantize(args.input, args.output)


if __name__ == "__main__":
    main()
