"""
Shared parquet persistence helpers for generation pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.config import ensure_data_dir


def load_completed_prompt_ids(path: Path) -> set[str]:
    """Return prompt_ids already present in a parquet output file (for resume)."""
    if not path.exists():
        return set()
    existing = pd.read_parquet(path, engine="pyarrow", columns=["prompt_id"])
    return set(existing["prompt_id"].astype(str))


def append_results_to_parquet(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Append validated result rows to a parquet file, deduplicating by prompt_id."""
    if not rows:
        return

    ensure_data_dir()
    validated = [validator(row) for row in rows]
    new_df = pd.DataFrame(validated)

    if path.exists():
        existing = pd.read_parquet(path, engine="pyarrow")
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["prompt_id"], keep="last")
        combined.to_parquet(path, index=False, engine="pyarrow")
    else:
        new_df.to_parquet(path, index=False, engine="pyarrow")


def apply_row_limit(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Return the first ``limit`` rows when set; otherwise the full frame."""
    if limit is None or limit <= 0:
        return df
    return df.head(limit).copy()
