"""Validated final three-class patent labels."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Subtype = Literal[
    "privacy_protection",
    "privacy_computing",
    "data_access_control",
    "data_confidentiality",
    "data_integrity",
    "data_availability",
    "data_governance",
    "other_data_security",
    "network_security",
    "system_security",
    "application_security",
    "device_security",
    "communication_security",
    "transaction_security",
    "physical_safety",
    "other_security",
    "unrelated",
]

CAT_SUBTYPES = {
    1: {
        "privacy_protection",
        "privacy_computing",
        "data_access_control",
        "data_confidentiality",
        "data_integrity",
        "data_availability",
        "data_governance",
        "other_data_security",
    },
    2: {
        "network_security",
        "system_security",
        "application_security",
        "device_security",
        "communication_security",
        "transaction_security",
        "physical_safety",
        "other_security",
    },
    3: {"unrelated"},
}


class PatentClassification(BaseModel):
    """Machine label contract; uncertainty is represented by review fields, not class 4."""

    model_config = ConfigDict(extra="forbid")

    cat: Literal[1, 2, 3]
    confidence: float = Field(ge=0, le=1)
    subtype: Subtype
    evidence: list[str] = Field(min_length=1, max_length=3)
    reason: str = Field(min_length=1, max_length=800)
    review_flag: bool
    review_reason: str = Field(max_length=400)

    @model_validator(mode="after")
    def validate_subtype_for_category(self) -> PatentClassification:
        if self.subtype not in CAT_SUBTYPES[self.cat]:
            raise ValueError(f"subtype {self.subtype!r} is invalid for cat {self.cat}")
        if self.review_flag and not self.review_reason.strip():
            raise ValueError("review_reason is required when review_flag is true")
        return self
