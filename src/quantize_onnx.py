"""
Dynamic INT8 quantization for the all-MiniLM-L6-v2 ONNX embedding model.

Reads models/all-MiniLM-L6-v2.onnx and writes models/all-MiniLM-L6-v2-int8.onnx.

Usage:
    python -m src.quantize_onnx
"""

from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from src.config import MODELS_DIR

INPUT_MODEL = MODELS_DIR / "all-MiniLM-L6-v2.onnx"
OUTPUT_MODEL = MODELS_DIR / "all-MiniLM-L6-v2-int8.onnx"
INPUT_FALLBACK = MODELS_DIR / "all-MiniLM-L6-v2-onnx" / "model.onnx"


def resolve_input_model() -> Path:
    """Return the FP32 ONNX model path, copying from export folder if needed."""
    if INPUT_MODEL.exists():
        return INPUT_MODEL
    if INPUT_FALLBACK.exists():
        INPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        INPUT_MODEL.write_bytes(INPUT_FALLBACK.read_bytes())
        return INPUT_MODEL
    raise FileNotFoundError(
        f"Input model not found at {INPUT_MODEL} or {INPUT_FALLBACK}. "
        "Run export_onnx.py first."
    )


def main() -> None:
    input_path = resolve_input_model()
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    print(f"Quantizing {input_path} -> {OUTPUT_MODEL} (weight_type=QUInt8) …")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(OUTPUT_MODEL),
        weight_type=QuantType.QUInt8,
    )
    print(f"Success! Quantized model saved to {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
