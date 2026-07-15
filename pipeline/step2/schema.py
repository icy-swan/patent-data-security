"""Strict binary output schema for Step 2."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Label = Literal["DATA_SECURITY", "OTHER"]
ScopeBasis = Literal[
    "legal_data_security",
    "personal_information_protection",
    "cryptography",
    "data_confidentiality",
    "data_integrity",
    "data_availability",
    "access_control_authentication",
    "privacy_enhancing_technology",
    "secure_computation",
    "security_audit_provenance",
    "data_governance_compliance",
    "security_monitoring_response",
    "other_data_security",
    "other",
]
EvidenceField = Literal["title", "abstract", "claim"]
ProcessingActivity = Literal[
    "collection",
    "storage",
    "use",
    "processing",
    "transmission",
    "provision",
    "disclosure",
    "other",
]
IndustrySector = Literal[
    "industry",
    "telecommunications",
    "transportation",
    "finance",
    "natural_resources",
    "healthcare",
    "education",
    "science_technology",
    "other",
]


class PatentEvidence(BaseModel):
    """A verbatim patent quote; identifiers and IPC cannot be substantive evidence."""

    model_config = ConfigDict(extra="forbid")

    field: EvidenceField
    quote: str = Field(min_length=1, max_length=600)


class PatentClassification(BaseModel):
    """Binary decision, two downstream analysis dimensions and a review flag."""

    model_config = ConfigDict(extra="forbid")

    label: Label
    confidence: float = Field(ge=0, le=1)
    scope_basis: list[ScopeBasis] = Field(min_length=1, max_length=3)
    processing_activities: list[ProcessingActivity] = Field(min_length=1, max_length=8)
    industry_sectors: list[IndustrySector] = Field(min_length=1, max_length=9)
    technical_scope: str = Field(min_length=1, max_length=800)
    legal_scope: str = Field(min_length=1, max_length=800)
    evidence: list[PatentEvidence] = Field(min_length=1, max_length=3)
    reason: str = Field(min_length=1, max_length=1000)
    review_flag: bool
    review_reason: str = Field(max_length=500)

    @model_validator(mode="after")
    def validate_label_contract(self) -> PatentClassification:
        unique_basis = list(dict.fromkeys(self.scope_basis))
        if len(unique_basis) != len(self.scope_basis):
            raise ValueError("scope_basis must not contain duplicates")
        if self.label == "OTHER" and self.scope_basis != ["other"]:
            raise ValueError("OTHER requires scope_basis=['other']")
        if self.label == "DATA_SECURITY" and "other" in self.scope_basis:
            raise ValueError("DATA_SECURITY cannot use scope_basis='other'")
        for field_name in ("processing_activities", "industry_sectors"):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
            if "other" in values and values != ["other"]:
                raise ValueError(f"{field_name} cannot combine 'other' with specific values")
        if self.label == "OTHER" and self.processing_activities != ["other"]:
            raise ValueError("OTHER requires processing_activities=['other']")
        if self.label == "OTHER" and self.industry_sectors != ["other"]:
            raise ValueError("OTHER requires industry_sectors=['other']")
        if self.review_flag and not self.review_reason.strip():
            raise ValueError("review_reason is required when review_flag=true")
        if not self.review_flag and self.review_reason.strip():
            raise ValueError("review_reason must be empty when review_flag=false")
        return self
