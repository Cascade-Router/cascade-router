"""
Shared async OpenAI helpers: backoff, retries, and chat completion.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

from src.config import (
    GEMINI_BASE_URL,
    INITIAL_BACKOFF_SECONDS,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MAX_BACKOFF_SECONDS,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


def create_async_llm_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointed at Gemini's OpenAI-compatible API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Export it or add it to a .env file."
        )
    return AsyncOpenAI(
        api_key=api_key,
        base_url=GEMINI_BASE_URL,
    )


def compute_backoff(attempt: int, retry_after: float | None = None) -> float:
    """Exponential backoff with jitter; honors Retry-After when provided."""
    if retry_after is not None and retry_after > 0:
        base = min(retry_after, MAX_BACKOFF_SECONDS)
    else:
        base = min(INITIAL_BACKOFF_SECONDS * (2**attempt), MAX_BACKOFF_SECONDS)
    jitter = random.uniform(0, base * 0.1)
    return base + jitter


def extract_retry_after(exc: APIStatusError) -> float | None:
    """Parse Retry-After header from a rate-limit response if present."""
    headers = getattr(exc, "response", None)
    if headers is None:
        return None
    header_map = getattr(headers, "headers", None)
    if not header_map:
        return None
    value = header_map.get("retry-after") or header_map.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def chat_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    semaphore: asyncio.Semaphore,
    response_format: Any | None = None,
) -> str | Any:
    """
    Call OpenAI chat completions with concurrency limiting and retries.

    When ``response_format`` is a Pydantic model, uses structured parsing and
    returns the parsed object. Otherwise returns plain text content.
    """
    async with semaphore:
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                if response_format is not None:
                    response = await client.beta.chat.completions.parse(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        temperature=LLM_TEMPERATURE,
                        max_tokens=LLM_MAX_TOKENS,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                    parsed = response.choices[0].message.parsed
                    if parsed is None:
                        raise ValueError("Structured output parsing returned None")
                    return parsed

                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("Empty completion content from API")
                return content.strip()

            except RateLimitError as exc:
                last_error = exc
                wait = compute_backoff(attempt, extract_retry_after(exc))
                logger.warning(
                    "Rate limited (attempt %d/%d). Sleeping %.1fs …",
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)

            except APIStatusError as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status == 429:
                    wait = compute_backoff(attempt, extract_retry_after(exc))
                    logger.warning(
                        "HTTP 429 (attempt %d/%d). Sleeping %.1fs …",
                        attempt + 1,
                        MAX_RETRIES,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                if status is not None and status >= 500:
                    wait = compute_backoff(attempt)
                    logger.warning(
                        "Server error %s (attempt %d/%d). Sleeping %.1fs …",
                        status,
                        attempt + 1,
                        MAX_RETRIES,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

            except APIConnectionError as exc:
                last_error = exc
                wait = compute_backoff(attempt)
                logger.warning(
                    "Connection error (attempt %d/%d). Sleeping %.1fs …",
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(f"Failed after {MAX_RETRIES} retries") from last_error
