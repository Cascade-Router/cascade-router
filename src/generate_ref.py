"""
Async reference-answer generation pipeline using the OpenAI API.

Usage:
    python -m src.generate_ref
    python -m src.generate_ref --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

from src.config import (
    BATCH_SAVE_SIZE,
    MAX_CONCURRENT_REQUESTS,
    NORMALIZED_PROMPTS_PATH,
    OPENAI_API_KEY,
    PROCESS_LIMIT_ROWS,
    REFERENCE_ANSWERS_PATH,
    REFERENCE_MODEL,
)
from src.llm_async import chat_completion
from src.parquet_io import (
    append_results_to_parquet,
    apply_row_limit,
    load_completed_prompt_ids,
)
from src.schemas import ReferenceAnswer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant. Provide a clear, complete reference "
    "answer to the user's prompt. Match the expected format for the task type "
    "(e.g., code for coding tasks, concise summaries for summarization)."
)


def load_normalized_prompts(path: Path | None = None) -> pd.DataFrame:
    """Load the normalized prompts parquet produced by ingest.py."""
    source = path or NORMALIZED_PROMPTS_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"Normalized prompts not found at {source}. Run `python -m src.ingest` first."
        )
    return pd.read_parquet(source, engine="pyarrow")


def _validate_reference(row: dict[str, Any]) -> dict[str, Any]:
    return ReferenceAnswer(**row).model_dump()


async def _process_row(
    client: AsyncOpenAI,
    row: pd.Series,
    semaphore: asyncio.Semaphore,
    model: str,
) -> dict[str, Any]:
    """Generate a reference answer for a single prompt row."""
    answer = await chat_completion(
        client,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt_text"]},
        ],
        semaphore=semaphore,
    )
    return {
        "prompt_id": row["prompt_id"],
        "prompt_text": row["prompt_text"],
        "category": row["category"],
        "reference_model": model,
        "reference_answer": answer,
    }


async def generate_references_async(
    prompts_df: pd.DataFrame,
    *,
    model: str = REFERENCE_MODEL,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    batch_save_size: int = BATCH_SAVE_SIZE,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Process pending prompts asynchronously with incremental parquet saves."""
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it or add it to a .env file."
        )

    target = output_path or REFERENCE_ANSWERS_PATH
    completed_ids = load_completed_prompt_ids(target)
    pending = prompts_df[~prompts_df["prompt_id"].astype(str).isin(completed_ids)].copy()

    if pending.empty:
        logger.info("All prompts already have reference answers. Nothing to do.")
        if target.exists():
            return pd.read_parquet(target, engine="pyarrow")
        return pd.DataFrame()

    logger.info(
        "Generating references for %d prompts (%d already done, concurrency=%d)",
        len(pending),
        len(completed_ids),
        max_concurrent,
    )

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(max_concurrent)
    batch_buffer: list[dict[str, Any]] = []
    processed = 0
    start = time.perf_counter()

    tasks = [
        _process_row(client, row, semaphore, model)
        for _, row in pending.iterrows()
    ]

    with tqdm(total=len(tasks), desc="Generating references") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                batch_buffer.append(result)
                processed += 1
                pbar.update(1)

                if len(batch_buffer) >= batch_save_size:
                    append_results_to_parquet(
                        batch_buffer, target, validator=_validate_reference
                    )
                    batch_buffer.clear()

            except Exception as exc:
                if batch_buffer:
                    append_results_to_parquet(
                        batch_buffer, target, validator=_validate_reference
                    )
                    batch_buffer.clear()
                logger.error("Generation failed after %d successes: %s", processed, exc)
                raise

    if batch_buffer:
        append_results_to_parquet(batch_buffer, target, validator=_validate_reference)

    elapsed = time.perf_counter() - start
    logger.info(
        "Finished %d reference answers in %.1fs (%.2f req/s)",
        processed,
        elapsed,
        processed / elapsed if elapsed > 0 else 0,
    )
    return pd.read_parquet(target, engine="pyarrow")


def generate_references(
    prompts_df: pd.DataFrame | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Synchronous entry point wrapping the async pipeline."""
    df = prompts_df if prompts_df is not None else load_normalized_prompts()
    return asyncio.run(generate_references_async(df, **kwargs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reference answers via OpenAI.")
    parser.add_argument(
        "--limit",
        type=int,
        default=PROCESS_LIMIT_ROWS,
        help="Process only the first N prompts (overrides PROCESS_LIMIT_ROWS).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    args = parse_args()
    prompts = apply_row_limit(load_normalized_prompts(), args.limit)
    logger.info("Loaded %d prompts (limit=%s)", len(prompts), args.limit)
    result = generate_references(prompts)
    print(f"Reference answers file: {REFERENCE_ANSWERS_PATH} ({len(result)} rows)")


if __name__ == "__main__":
    main()
