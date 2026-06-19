"""
Ingestion pipeline: download Hugging Face datasets, normalize, and save parquet.

Usage:
    python -m src.ingest
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import pandas as pd
from datasets import load_dataset

from src.config import (
    NORMALIZED_PROMPTS_PATH,
    RANDOM_SEED,
    TARGET_TOTAL_ROWS,
    ensure_data_dir,
)
from src.schemas import NormalizedPrompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category mapping helpers
# ---------------------------------------------------------------------------

DOLLY_CATEGORY_MAP: dict[str, str] = {
    "information_extraction": "extraction",
    "summarization": "summarization",
    "closed_qa": "qa",
    "open_qa": "qa",
    "general_qa": "qa",
    "classification": "extraction",
    "creative_writing": "creative",
    "brainstorming": "creative",
}


def _new_prompt_id() -> str:
    return str(uuid.uuid4())


def _build_prompt_text(instruction: str, context: str | None = None) -> str:
    """Combine Dolly instruction + optional context into a single user prompt."""
    instruction = (instruction or "").strip()
    context = (context or "").strip()
    if context:
        return f"{instruction}\n\nContext:\n{context}"
    return instruction


def _sample_dataframe(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _validate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate each row with Pydantic; skip invalid rows with a warning."""
    validated: list[dict[str, Any]] = []
    for record in records:
        try:
            validated.append(NormalizedPrompt(**record).model_dump())
        except Exception as exc:
            logger.warning("Skipping invalid record: %s", exc)
    return validated


# ---------------------------------------------------------------------------
# Per-dataset extractors
# ---------------------------------------------------------------------------


def ingest_dolly(n: int, seed: int) -> pd.DataFrame:
    """
    Load databricks-dolly-15k and extract extraction / summarization prompts.

    Dolly rows include an instruction, optional context, and a category label.
    """
    logger.info("Loading databricks/databricks-dolly-15k …")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    rows: list[dict[str, Any]] = []
    for item in dataset:
        raw_category = (item.get("category") or "").strip().lower()
        mapped = DOLLY_CATEGORY_MAP.get(raw_category)
        if mapped not in ("extraction", "summarization"):
            continue

        prompt_text = _build_prompt_text(
            instruction=item.get("instruction", ""),
            context=item.get("context"),
        )
        if not prompt_text:
            continue

        rows.append(
            {
                "prompt_id": _new_prompt_id(),
                "source_dataset": "databricks/databricks-dolly-15k",
                "category": mapped,
                "prompt_text": prompt_text,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("Dolly: %d extraction/summarization candidates", len(df))
    return _sample_dataframe(df, n, seed)


def ingest_humaneval(n: int, seed: int) -> pd.DataFrame:
    """Load OpenAI HumanEval coding prompts."""
    logger.info("Loading openai/openai_humaneval …")
    dataset = load_dataset("openai/openai_humaneval", split="test")

    rows: list[dict[str, Any]] = []
    for item in dataset:
        prompt_text = (item.get("prompt") or "").strip()
        if not prompt_text:
            continue

        rows.append(
            {
                "prompt_id": _new_prompt_id(),
                "source_dataset": "openai/openai_humaneval",
                "category": "coding",
                "prompt_text": prompt_text,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("HumanEval: %d coding prompts", len(df))
    return _sample_dataframe(df, n, seed)


def ingest_no_robots(n: int, seed: int) -> pd.DataFrame:
    """
    Load HuggingFaceH4/no_robots for general reasoning prompts.

    The dataset may expose `prompt` directly or nested `messages` content.
    """
    logger.info("Loading HuggingFaceH4/no_robots …")
    dataset = load_dataset("HuggingFaceH4/no_robots", split="train")

    rows: list[dict[str, Any]] = []
    for item in dataset:
        prompt_text = _extract_no_robots_prompt(item)
        if not prompt_text:
            continue

        rows.append(
            {
                "prompt_id": _new_prompt_id(),
                "source_dataset": "HuggingFaceH4/no_robots",
                "category": "general_reasoning",
                "prompt_text": prompt_text,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("no_robots: %d general reasoning prompts", len(df))
    return _sample_dataframe(df, n, seed)


def _extract_no_robots_prompt(item: dict[str, Any]) -> str:
    """Resolve prompt text from no_robots row (schema varies by revision)."""
    if prompt := (item.get("prompt") or "").strip():
        return prompt

    messages = item.get("messages")
    if isinstance(messages, list) and messages:
        parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = (msg.get("role") or "user").strip()
            content = (msg.get("content") or "").strip()
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts).strip()

    return ""


# ---------------------------------------------------------------------------
# Main ingest orchestration
# ---------------------------------------------------------------------------


def ingest_all(
    target_rows: int = TARGET_TOTAL_ROWS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Pull, sample, merge, and validate prompts from all configured datasets.

    Returns a consolidated DataFrame with schema:
        prompt_id, source_dataset, category, prompt_text

    When a source has fewer rows than its quota (e.g. HumanEval ~164), all
    available rows are kept and the shortfall is logged; total may be < target.
    """
    per_dataset = target_rows // 3
    remainder = target_rows - (per_dataset * 3)

    dolly_df = ingest_dolly(per_dataset + remainder, seed)
    humaneval_df = ingest_humaneval(per_dataset, seed + 1)
    no_robots_df = ingest_no_robots(per_dataset, seed + 2)

    frames = [dolly_df, humaneval_df, no_robots_df]
    combined = pd.concat(frames, ignore_index=True)

    # If we are under target, top up from no_robots (largest pool) when possible.
    shortfall = target_rows - len(combined)
    if shortfall > 0:
        logger.warning(
            "Only %d rows after initial sampling (target %d). "
            "Attempting top-up from no_robots …",
            len(combined),
            target_rows,
        )
        extra_seed = seed + 100
        extra = ingest_no_robots(per_dataset + shortfall, extra_seed)
        existing_texts = set(combined["prompt_text"])
        extra = extra[~extra["prompt_text"].isin(existing_texts)]
        if not extra.empty:
            take = min(shortfall, len(extra))
            combined = pd.concat(
                [combined, extra.head(take)],
                ignore_index=True,
            )

    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    validated = _validate_records(combined.to_dict(orient="records"))
    result = pd.DataFrame(validated)

    logger.info(
        "Ingestion complete: %d rows (target ~%d). Category breakdown:\n%s",
        len(result),
        target_rows,
        result["category"].value_counts().to_string(),
    )
    return result


def save_normalized_prompts(df: pd.DataFrame, path: str | None = None) -> None:
    """Persist normalized prompts to parquet."""
    output = path or str(NORMALIZED_PROMPTS_PATH)
    ensure_data_dir()
    df.to_parquet(output, index=False, engine="pyarrow")
    logger.info("Saved %d rows to %s", len(df), output)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    df = ingest_all()
    save_normalized_prompts(df)
    print(f"Wrote {len(df)} normalized prompts to {NORMALIZED_PROMPTS_PATH}")


if __name__ == "__main__":
    main()
