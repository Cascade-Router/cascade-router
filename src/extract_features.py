"""
Feature extraction pipeline for cascade-router routing classifier training.

Reads normalized prompts and computes token counts, structural complexity
heuristics, and local sentence-transformer embeddings.

Usage:
    python -m src.extract_features
    python -m src.extract_features --limit 5
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import tiktoken
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    EMBEDDING_MODEL,
    MAX_ROUTING_SEQ_LEN,
    NORMALIZED_PROMPTS_PATH,
    PROCESS_LIMIT_ROWS,
    PROMPT_FEATURES_PATH,
    TIKTOKEN_ENCODING,
    ensure_data_dir,
)
from src.parquet_io import apply_row_limit
from src.schemas import PromptFeatures

logger = logging.getLogger(__name__)

# Characters that signal structure beyond plain prose.
_PUNCTUATION = re.compile(r"[.,;:!?]")
_BRACKETS = re.compile(r"[\[\]{}()<>`]")
_NEWLINES = re.compile(r"\n")
_TABS = re.compile(r"\t")
_CODE_HINTS = re.compile(r"[\\|/@#$%^&*~_+=]")


def load_normalized_prompts(path: Path | None = None) -> pd.DataFrame:
    """Load normalized prompts parquet."""
    source = path or NORMALIZED_PROMPTS_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"Normalized prompts not found at {source}. Run `python -m src.ingest` first."
        )
    return pd.read_parquet(source, engine="pyarrow")


def count_tokens(text: str, encoding_name: str = TIKTOKEN_ENCODING) -> int:
    """Exact token count using tiktoken (cl100k_base by default)."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def truncate_for_embedding(
    text: str,
    *,
    max_tokens: int = MAX_ROUTING_SEQ_LEN,
    encoding_name: str = TIKTOKEN_ENCODING,
) -> str:
    """Truncate prompt to max_tokens for routing embedding (matches C++ ONNX path)."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])


def structural_complexity(text: str) -> float:
    """
    Heuristic structural complexity score in [0, ~1].

    Counts punctuation, newlines, brackets/braces, and other syntactic
    characters, then divides by total prompt length. Higher values indicate
    more structured or code-like prompts vs simple sentences.
    """
    length = len(text)
    if length == 0:
        return 0.0

    structural_hits = (
        len(_PUNCTUATION.findall(text))
        + len(_BRACKETS.findall(text))
        + len(_NEWLINES.findall(text))
        + len(_TABS.findall(text))
        + len(_CODE_HINTS.findall(text))
    )
    return round(structural_hits / length, 6)


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load a local sentence-transformer model (CPU-friendly)."""
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def extract_features(
    prompts_df: pd.DataFrame,
    *,
    model: SentenceTransformer | None = None,
    encoding_name: str = TIKTOKEN_ENCODING,
) -> pd.DataFrame:
    """
    Build feature rows for each prompt.

    Returns a DataFrame with columns:
        prompt_id, token_count, structural_complexity, embedding
    """
    if prompts_df.empty:
        return pd.DataFrame(
            columns=["prompt_id", "token_count", "structural_complexity", "embedding"]
        )

    embedder = model or load_embedding_model()
    prompt_texts = prompts_df["prompt_text"].astype(str).tolist()

    logger.info("Computing token counts and structural complexity (full prompts) …")
    token_counts = [count_tokens(text, encoding_name) for text in prompt_texts]
    complexities = [structural_complexity(text) for text in prompt_texts]

    truncated_texts = [
        truncate_for_embedding(text, max_tokens=MAX_ROUTING_SEQ_LEN, encoding_name=encoding_name)
        for text in prompt_texts
    ]
    logger.info(
        "Encoding %d prompts with %s (truncated to %d tokens) …",
        len(truncated_texts),
        EMBEDDING_MODEL,
        MAX_ROUTING_SEQ_LEN,
    )
    vectors = embedder.encode(
        truncated_texts,
        show_progress_bar=len(truncated_texts) > 10,
        convert_to_numpy=True,
    )

    rows: list[dict] = []
    for idx, row in prompts_df.reset_index(drop=True).iterrows():
        embedding = vectors[idx].tolist()
        record = {
            "prompt_id": row["prompt_id"],
            "token_count": token_counts[idx],
            "structural_complexity": complexities[idx],
            "embedding": embedding,
        }
        rows.append(PromptFeatures(**record).model_dump())

    return pd.DataFrame(rows)


def save_features(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist feature rows to parquet."""
    target = path or PROMPT_FEATURES_PATH
    ensure_data_dir()
    df.to_parquet(target, index=False, engine="pyarrow")
    logger.info("Saved %d feature rows to %s", len(df), target)
    return target


def print_features(df: pd.DataFrame) -> None:
    """Print a concise summary of extracted features to the terminal."""
    print("\n" + "=" * 72)
    print("PROMPT FEATURES")
    print("=" * 72)
    for _, row in df.iterrows():
        embedding = row["embedding"]
        emb_len = len(embedding) if isinstance(embedding, (list, tuple)) else 0
        print(f"\nprompt_id:             {row['prompt_id']}")
        print(f"token_count:           {row['token_count']}")
        print(f"structural_complexity: {row['structural_complexity']}")
        print(f"embedding_dim:         {emb_len}")
    print("\n" + "=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract routing features from prompts.")
    parser.add_argument(
        "--limit",
        type=int,
        default=PROCESS_LIMIT_ROWS,
        help="Process only the first N prompts (overrides PROCESS_LIMIT_ROWS).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROMPT_FEATURES_PATH,
        help="Output parquet path (default: data/prompt_features.parquet).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    args = parse_args()
    prompts = apply_row_limit(load_normalized_prompts(), args.limit)
    logger.info("Processing %d prompts (limit=%s)", len(prompts), args.limit)

    features = extract_features(prompts)
    output = save_features(features, args.output)
    print_features(features)
    print(f"\nFeatures file: {output} ({len(features)} rows)")


if __name__ == "__main__":
    main()
