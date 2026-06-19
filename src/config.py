"""
Central configuration for the cascade-router data pipeline.

Loads settings from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root when present (does not override existing env vars).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = _PROJECT_ROOT / "data"
NORMALIZED_PROMPTS_PATH: Path = DATA_DIR / "normalized_prompts.parquet"
REFERENCE_ANSWERS_PATH: Path = DATA_DIR / "reference_answers.parquet"
CANDIDATE_ANSWERS_PATH: Path = DATA_DIR / "candidate_answers.parquet"
EVALUATION_RESULTS_PATH: Path = DATA_DIR / "evaluation_results.parquet"
PROMPT_FEATURES_PATH: Path = DATA_DIR / "prompt_features.parquet"

MODELS_DIR: Path = _PROJECT_ROOT / "models"
TINY_ROUTER_MODEL_PATH: Path = MODELS_DIR / "tiny_router.pkl"

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
TARGET_TOTAL_ROWS: int = int(os.getenv("TARGET_TOTAL_ROWS", "10000"))
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# Per-dataset row targets (roughly equal split across three sources).
ROWS_PER_DATASET: int = TARGET_TOTAL_ROWS // 3

# ---------------------------------------------------------------------------
# LLM generation (Gemini via OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
REFERENCE_MODEL: str = os.getenv("REFERENCE_MODEL", "gemini-2.5-flash")
CANDIDATE_MODEL: str = os.getenv("CANDIDATE_MODEL", "gemini-2.5-flash")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gemini-2.5-flash")
MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
BATCH_SAVE_SIZE: int = int(os.getenv("BATCH_SAVE_SIZE", "500"))

# Optional cap for micro-tests / debugging (None = process all rows).
_limit_raw = os.getenv("PROCESS_LIMIT_ROWS", "")
PROCESS_LIMIT_ROWS: int | None = int(_limit_raw) if _limit_raw.strip() else None

# Retry / backoff
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "6"))
INITIAL_BACKOFF_SECONDS: float = float(os.getenv("INITIAL_BACKOFF_SECONDS", "1.0"))
MAX_BACKOFF_SECONDS: float = float(os.getenv("MAX_BACKOFF_SECONDS", "60.0"))

# OpenAI request tuning
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120.0"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
TIKTOKEN_ENCODING: str = os.getenv("TIKTOKEN_ENCODING", "cl100k_base")
MAX_ROUTING_SEQ_LEN: int = int(os.getenv("MAX_ROUTING_SEQ_LEN", "16"))


def ensure_data_dir() -> Path:
    """Create the data directory if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def ensure_models_dir() -> Path:
    """Create the models directory if it does not exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR
