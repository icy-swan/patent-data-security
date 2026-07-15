"""Load and validate the versioned Step 1 resources."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.common.io import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
ALLOWED_MATCH_MODES = {"standalone", "cooccurrence", "phrase_family"}


@dataclass(frozen=True)
class KeywordBundle:
    """Validated keyword, source and validation resources."""

    keywords: dict[str, Any]
    sources: dict[str, Any]
    validation_protocol: dict[str, Any]
    resource_dir: Path
    hashes: dict[str, str]

    @property
    def keyword_version(self) -> str:
        return str(self.keywords["keyword_version"])

    @property
    def methodology_version(self) -> str:
        return str(self.keywords["methodology_version"])

    @property
    def source_manifest_version(self) -> str:
        return str(self.sources["source_manifest_version"])


def load_keyword_bundle(directory: str | Path = DEFAULT_RESOURCE_DIR) -> KeywordBundle:
    """Read resources and fail early on ambiguous or untraceable rules."""

    root = Path(directory).resolve()
    paths = {
        "keywords": root / "keywords.json",
        "sources": root / "sources.json",
        "validation_protocol": root / "validation_protocol.json",
    }
    resource_paths = {**paths, "changelog": root / "CHANGELOG.md"}
    values = {name: _read_json(path) for name, path in paths.items()}
    _validate_resources(
        values["keywords"],
        values["sources"],
        values["validation_protocol"],
    )
    return KeywordBundle(
        keywords=values["keywords"],
        sources=values["sources"],
        validation_protocol=values["validation_protocol"],
        resource_dir=root,
        hashes={name: sha256_file(path) for name, path in resource_paths.items()},
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Resource must contain a JSON object: {path}")
    return value


def _validate_resources(
    keywords: dict[str, Any],
    sources: dict[str, Any],
    validation_protocol: dict[str, Any],
) -> None:
    source_ids = {source["id"] for source in sources.get("sources", [])}
    if not source_ids:
        raise ValueError("sources.json must define at least one source")
    if len(source_ids) != len(sources["sources"]):
        raise ValueError("sources.json contains duplicate source IDs")

    seen_variants: dict[str, str] = {}
    concept_ids: set[str] = set()
    context_ids = {item["id"] for item in keywords.get("context_lexicons", [])}
    for concept in keywords.get("concepts", []):
        concept_id = concept["concept_id"]
        if concept_id in concept_ids:
            raise ValueError(f"Duplicate concept_id: {concept_id}")
        concept_ids.add(concept_id)
        mode = concept["match_policy"]["mode"]
        if mode not in ALLOWED_MATCH_MODES:
            raise ValueError(f"Unsupported match mode for {concept_id}: {mode}")
        if concept.get("category") not in {"descriptive", "technical"}:
            raise ValueError(f"Unsupported category for {concept_id}")
        referenced_contexts = set(concept["match_policy"].get("required_any", [])) | set(
            concept["match_policy"].get("required_all", [])
        )
        if missing_contexts := referenced_contexts - context_ids:
            raise ValueError(f"Unknown context IDs for {concept_id}: {sorted(missing_contexts)}")
        variants = concept.get("variants", [])
        if not variants:
            raise ValueError(f"Concept has no variants: {concept_id}")
        for variant in variants:
            normalized = _normalize_variant(str(variant))
            previous = seen_variants.get(normalized)
            if previous is not None:
                raise ValueError(
                    f"Keyword variant {variant!r} appears in both {previous} and {concept_id}"
                )
            seen_variants[normalized] = concept_id
        _validate_source_ids(concept.get("source_ids", []), source_ids, concept_id)

    for section in ("context_lexicons", "diagnostic_patterns", "ipc_audit_rules"):
        for item in keywords.get(section, []):
            _validate_source_ids(item.get("source_ids", []), source_ids, item["id"])

    configured = keywords.get("matching", {}).get("context_window_chars")
    candidates = validation_protocol.get("context_window_candidates", [])
    if configured not in candidates:
        raise ValueError("Configured context window must appear in validation candidates")
    if configured != validation_protocol.get("provisional_context_window"):
        raise ValueError("Keyword and validation resources disagree on the pilot window")
    _validate_source_ids(
        validation_protocol.get("method_sources", []),
        source_ids,
        "validation_protocol.method_sources",
    )


def _validate_source_ids(values: list[str], known: set[str], owner: str) -> None:
    if not values:
        raise ValueError(f"Rule has no source IDs: {owner}")
    missing = set(values) - known
    if missing:
        raise ValueError(f"Unknown source IDs for {owner}: {sorted(missing)}")


def _normalize_variant(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
