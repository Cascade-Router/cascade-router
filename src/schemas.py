"""
Pydantic models for validating pipeline records at ingestion and generation time.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator


Category = Literal[
    "coding",
    "extraction",
    "summarization",
    "general_reasoning",
    "qa",
    "creative",
    "other",
]


class NormalizedPrompt(BaseModel):
    """A single normalized prompt ready for reference-answer generation."""

    prompt_id: str = Field(..., description="Stable UUID for this prompt.")
    source_dataset: str = Field(..., description="Hugging Face dataset identifier.")
    category: Category
    prompt_text: str = Field(..., min_length=1)

    @field_validator("prompt_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        uuid.UUID(value)
        return value

    @field_validator("prompt_text")
    @classmethod
    def strip_prompt(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt_text must not be empty after stripping.")
        return stripped


class ReferenceAnswer(BaseModel):
    """Reference answer produced by a frontier LLM for a normalized prompt."""

    prompt_id: str
    prompt_text: str
    category: Category
    reference_model: str
    reference_answer: str = Field(..., min_length=1)

    @field_validator("prompt_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        uuid.UUID(value)
        return value


class CandidateAnswer(BaseModel):
    """Answer from a smaller/cheaper candidate model."""

    prompt_id: str
    candidate_model: str
    candidate_answer: str = Field(..., min_length=1)

    @field_validator("prompt_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        uuid.UUID(value)
        return value


class EvaluationResult(BaseModel):
    """Structured judge output comparing candidate vs reference answers."""

    reasoning: str = Field(..., min_length=1)
    pass_score: Literal[0, 1] = Field(
        ...,
        description="1 if functionally equivalent, 0 if the candidate failed.",
    )


class EvaluationRecord(BaseModel):
    """Persisted evaluation row written to parquet."""

    prompt_id: str
    reasoning: str
    pass_score: Literal[0, 1]

    @field_validator("prompt_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        uuid.UUID(value)
        return value


class PromptFeatures(BaseModel):
    """Extracted routing features for a normalized prompt."""

    prompt_id: str
    token_count: int = Field(..., ge=0)
    structural_complexity: float = Field(..., ge=0.0)
    embedding: list[float] = Field(..., min_length=1)

    @field_validator("prompt_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        uuid.UUID(value)
        return value
