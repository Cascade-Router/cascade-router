"""
Quick health check for Gemini OpenAI-compatible API connectivity.

Usage:
    python -m src.test_gemini
"""

from __future__ import annotations

import asyncio
import sys
import traceback

from src.config import GEMINI_BASE_URL, REFERENCE_MODEL, REQUEST_TIMEOUT_SECONDS
from src.llm_async import create_async_llm_client

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

HEALTH_PROMPT = "Return the word 'SUCCESS' if you can read this."


async def run_health_check() -> int:
    """Send one chat completion and return exit code (0 = ok, 1 = fail)."""
    print(f"Endpoint : {GEMINI_BASE_URL}")
    print(f"Model    : {REFERENCE_MODEL}")
    print(f"Prompt   : {HEALTH_PROMPT!r}")
    print()

    try:
        client = create_async_llm_client()
        response = await client.chat.completions.create(
            model=REFERENCE_MODEL,
            messages=[{"role": "user", "content": HEALTH_PROMPT}],
            max_tokens=16,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        content = (response.choices[0].message.content or "").strip()
        print(f"{GREEN}SUCCESS — API responded:{RESET}")
        print(f"{GREEN}{content}{RESET}")
        return 0

    except Exception as exc:
        print(f"{RED}FAILED — Gemini health check error:{RESET}")
        print(f"{RED}{type(exc).__name__}: {exc}{RESET}")
        status = getattr(exc, "status_code", None)
        if status is not None:
            print(f"{RED}HTTP status: {status}{RESET}")
        body = getattr(exc, "body", None)
        if body is not None:
            print(f"{RED}Response body: {body}{RESET}")
        print(f"{RED}{traceback.format_exc()}{RESET}")
        return 1


def main() -> None:
    exit_code = asyncio.run(run_health_check())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
