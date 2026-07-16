"""Independent annotation schema used by humans and model simulation."""

from __future__ import annotations

from pipeline.step2.schema import PatentClassification


class IndependentAnnotation(PatentClassification):
    """A Step 2-compatible decision made without seeing the Step 2 result."""
