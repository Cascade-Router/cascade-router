"""
Agreement evaluator: compare candidate answers against reference answers.

Joins reference and candidate parquet files on prompt_id, calls OpenAI with
structured outputs (EvaluationResult), and persists judgments.

Usage:
    python -m src.judge
    python -m src.judge --limit 5
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
    EVALUATION_RESULTS_PATH,
    JUDGE_MODEL,
    MAX_CONCURRENT_REQUESTS,
    OPENAI_API_KEY,
    PROCESS_LIMIT_ROWS,
    REFERENCE_ANSWERS_PATH,
)
from src.llm_async import chat_completion
from src.parquet_io import (
    append_results_to_parquet,
    apply_row_limit,
    load_completed_prompt_ids,
)
from src.schemas import EvaluationRecord, EvaluationResult

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = (
    "You are an expert technical evaluator. Compare the Candidate Answer to the "
    "Reference Answer based on the Original Prompt. Does the Candidate Answer "
    "achieve the exact same semantic intent, logical correctness, and factual "
    "accuracy as the Reference Answer? Ignore stylistic or formatting differences. "
    "Output your reasoning, then output a pass_score of 1 if functionally "
    "equivalent, or 0 if the Candidate failed."
)


def load_joined_pairs(
    reference_path: Path | None = None,
    candidate_path: Path | None = None,
) -> pd.DataFrame:
    """Inner-join reference and candidate answers on prompt_id."""
    ref_path = reference_path or REFERENCE_ANSWERS_PATH
    cand_path = candidate_path or CANDIDATE_ANSWERS_PATH

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference answers not found at {ref_path}")
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate answers not found at {cand_path}")

    references = pd.read_parquet(ref_path, engine="pyarrow")
    candidates = pd.read_parquet(cand_path, engine="pyarrow")

    joined = references.merge(candidates, on="prompt_id", how="inner", suffixes=("_ref", "_cand"))
    if joined.empty:
        raise ValueError("No overlapping prompt_id rows between reference and candidate files.")

    logger.info("Joined %d prompt pairs for evaluation", len(joined))
    return joined


def _build_user_message(row: pd.Series) -> str:
    return (
        f"Original Prompt:\n{row['prompt_text']}\n\n"
        f"Reference Answer:\n{row['reference_answer']}\n\n"
        f"Candidate Answer:\n{row['candidate_answer']}"
    )


def _validate_evaluation(row: dict[str, Any]) -> dict[str, Any]:
    return EvaluationRecord(**row).model_dump()


async def _judge_row(
    client: AsyncOpenAI,
    row: pd.Series,
    semaphore: asyncio.Semaphore,
    model: str,
) -> dict[str, Any]:
    """Run structured-output judge for a single joined row."""
    result = await chat_completion(
        client,
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(row)},
        ],
        semaphore=semaphore,
        response_format=EvaluationResult,
    )
    if not isinstance(result, EvaluationResult):
        result = EvaluationResult.model_validate(result)

    return {
        "prompt_id": row["prompt_id"],
        "reasoning": result.reasoning,
        "pass_score": result.pass_score,
    }


async def judge_async(
    pairs_df: pd.DataFrame,
    *,
    model: str = JUDGE_MODEL,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    batch_save_size: int = BATCH_SAVE_SIZE,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Evaluate all pending pairs asynchronously with incremental saves."""
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it or add it to a .env file."
        )

    target = output_path or EVALUATION_RESULTS_PATH
    completed_ids = load_completed_prompt_ids(target)
    pending = pairs_df[~pairs_df["prompt_id"].astype(str).isin(completed_ids)].copy()

    if pending.empty:
        logger.info("All pairs already evaluated. Nothing to do.")
        if target.exists():
            return pd.read_parquet(target, engine="pyarrow")
        return pd.DataFrame()

    logger.info(
        "Judging %d pairs (%d already done, concurrency=%d)",
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
        _judge_row(client, row, semaphore, model)
        for _, row in pending.iterrows()
    ]

    with tqdm(total=len(tasks), desc="Judging") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                batch_buffer.append(result)
                processed += 1
                pbar.update(1)

                if len(batch_buffer) >= batch_save_size:
                    append_results_to_parquet(
                        batch_buffer, target, validator=_validate_evaluation
                    )
                    batch_buffer.clear()

            except Exception as exc:
                if batch_buffer:
                    append_results_to_parquet(
                        batch_buffer, target, validator=_validate_evaluation
                    )
                    batch_buffer.clear()
                logger.error("Judging failed after %d successes: %s", processed, exc)
                raise

    if batch_buffer:
        append_results_to_parquet(batch_buffer, target, validator=_validate_evaluation)

    elapsed = time.perf_counter() - start
    logger.info(
        "Finished %d evaluations in %.1fs (%.2f req/s)",
        processed,
        elapsed,
        processed / elapsed if elapsed > 0 else 0,
    )
    return pd.read_parquet(target, engine="pyarrow")


def judge(pairs_df: pd.DataFrame | None = None, **kwargs: Any) -> pd.DataFrame:
    """Synchronous entry point wrapping the async judge pipeline."""
    df = pairs_df if pairs_df is not None else load_joined_pairs()
    return asyncio.run(judge_async(df, **kwargs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge candidate vs reference answers.")
    parser.add_argument(
        "--limit",
        type=int,
        default=PROCESS_LIMIT_ROWS,
        help="Evaluate only the first N joined pairs (overrides PROCESS_LIMIT_ROWS).",
    )
    return parser.parse_args()


def print_results(results: pd.DataFrame) -> None:
    """Pretty-print evaluation outcomes to the terminal."""
    print("\n" + "=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)
    for _, row in results.iterrows():
        status = "PASS" if row["pass_score"] == 1 else "FAIL"
        print(f"\nprompt_id: {row['prompt_id']}")
        print(f"pass_score: {row['pass_score']} ({status})")
        print(f"reasoning: {row['reasoning']}")
    print("\n" + "=" * 72)
    passed = int((results["pass_score"] == 1).sum())
    print(f"Summary: {passed}/{len(results)} passed")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    args = parse_args()
    pairs = apply_row_limit(load_joined_pairs(), args.limit)
    logger.info("Evaluating %d pairs (limit=%s)", len(pairs), args.limit)
    judge(pairs)
    target_ids = set(pairs["prompt_id"].astype(str))
    results = pd.read_parquet(EVALUATION_RESULTS_PATH, engine="pyarrow")
    results = results[results["prompt_id"].astype(str).isin(target_ids)]
    print_results(results)
    print(f"\nEvaluation results file: {EVALUATION_RESULTS_PATH} ({len(results)} rows)")


if __name__ == "__main__":
    main()
