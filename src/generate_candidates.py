"""
Async candidate-answer generation using a smaller/cheaper OpenAI model.

Mirrors generate_ref.py: same concurrency, backoff, resume-safe batching.

Usage:
    python -m src.generate_candidates
    python -m src.generate_candidates --limit 5
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
    CANDIDATE_ANSWERS_PATH,
    CANDIDATE_MODEL,
    MAX_CONCURRENT_REQUESTS,
    NORMALIZED_PROMPTS_PATH,
    OPENAI_API_KEY,
    PROCESS_LIMIT_ROWS,
)
from src.generate_ref import load_normalized_prompts
from src.llm_async import chat_completion
from src.parquet_io import (
    append_results_to_parquet,
    apply_row_limit,
    load_completed_prompt_ids,
)
from src.schemas import CandidateAnswer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Provide a clear answer to the user's prompt. "
    "Match the expected format for the task type when applicable."
)


def _validate_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return CandidateAnswer(**row).model_dump()


async def _process_row(
    client: AsyncOpenAI,
    row: pd.Series,
    semaphore: asyncio.Semaphore,
    model: str,
) -> dict[str, Any]:
    """Generate a candidate answer for a single prompt row."""
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
        "candidate_model": model,
        "candidate_answer": answer,
    }


async def generate_candidates_async(
    prompts_df: pd.DataFrame,
    *,
    model: str = CANDIDATE_MODEL,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    batch_save_size: int = BATCH_SAVE_SIZE,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Process pending prompts asynchronously with incremental parquet saves."""
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it or add it to a .env file."
        )

    target = output_path or CANDIDATE_ANSWERS_PATH
    completed_ids = load_completed_prompt_ids(target)
    pending = prompts_df[~prompts_df["prompt_id"].astype(str).isin(completed_ids)].copy()

    if pending.empty:
        logger.info("All prompts already have candidate answers. Nothing to do.")
        if target.exists():
            return pd.read_parquet(target, engine="pyarrow")
        return pd.DataFrame()

    logger.info(
        "Generating candidates for %d prompts (%d already done, concurrency=%d)",
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

    with tqdm(total=len(tasks), desc="Generating candidates") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                batch_buffer.append(result)
                processed += 1
                pbar.update(1)

                if len(batch_buffer) >= batch_save_size:
                    append_results_to_parquet(
                        batch_buffer, target, validator=_validate_candidate
                    )
                    batch_buffer.clear()

            except Exception as exc:
                if batch_buffer:
                    append_results_to_parquet(
                        batch_buffer, target, validator=_validate_candidate
                    )
                    batch_buffer.clear()
                logger.error("Generation failed after %d successes: %s", processed, exc)
                raise

    if batch_buffer:
        append_results_to_parquet(batch_buffer, target, validator=_validate_candidate)

    elapsed = time.perf_counter() - start
    logger.info(
        "Finished %d candidate answers in %.1fs (%.2f req/s)",
        processed,
        elapsed,
        processed / elapsed if elapsed > 0 else 0,
    )
    return pd.read_parquet(target, engine="pyarrow")


def generate_candidates(
    prompts_df: pd.DataFrame | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Synchronous entry point wrapping the async pipeline."""
    df = prompts_df if prompts_df is not None else load_normalized_prompts()
    return asyncio.run(generate_candidates_async(df, **kwargs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate answers via OpenAI.")
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
    result = generate_candidates(prompts)
    print(f"Candidate answers file: {CANDIDATE_ANSWERS_PATH} ({len(result)} rows)")


if __name__ == "__main__":
    main()
