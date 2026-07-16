"""Independent annotation schema used by humans and model simulation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pipeline.step2.schema import PatentClassification


class IndependentAnnotation(PatentClassification):
    """A Step 2-compatible decision made without seeing the Step 2 result."""


class CodexAnnotationItem(IndependentAnnotation):
    """One batch item bound to a local sample ID."""

    sample_id: str = Field(min_length=1, max_length=80)


class CodexAnnotationBatch(BaseModel):
    """Strict final response from one local Codex batch."""

    model_config = ConfigDict(extra="forbid")

    annotations: list[CodexAnnotationItem] = Field(min_length=1, max_length=25)
