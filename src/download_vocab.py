"""
Download vocab.txt for sentence-transformers/all-MiniLM-L6-v2.

Saves models/vocab.txt for the C++ WordPiece tokenizer.

Usage:
    python -m src.download_vocab
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from src.config import MODELS_DIR

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_PATH = MODELS_DIR / "vocab.txt"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading tokenizer vocab for {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    inv_vocab = {idx: token for token, idx in tokenizer.vocab.items()}
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for idx in range(len(inv_vocab)):
            f.write(inv_vocab[idx] + "\n")

    print(f"Success! Saved {len(inv_vocab)} tokens to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
