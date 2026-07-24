"""Public Step 1 reference implementation for deterministic S/E patent routing.

The code exposes the method, not the project's production taxonomy. The supplied rule
template is deliberately empty; users must provide a reviewable public rule file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROUTES = ("S", "E")
LABEL_BY_ROUTE = {"S": "DATA_SECURITY", "E": "OTHER"}
ALLOWED_MATCH_MODES = {"standalone", "cooccurrence", "phrase_family"}
SENTENCE_BOUNDARIES = "。！？!?；;\n\r"
ASCII_TERM = re.compile(r"^[a-z0-9][a-z0-9 ._+/-]*$", re.IGNORECASE)
CONNECTOR_TRANSLATION = str.maketrans(
    {
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
    }
)
OUTPUT_FIELDS = (
    "dataset_id",
    "patent_id",
    "source_row_number",
    "association_count",
    "title",
    "abstract",
    "claim",
    "ipc",
    "route",
    "step1_label",
    "selected_for_step2",
    "selection_group",
    "selection_probability",
    "sample_weight",
    "sample_seed",
    "matched_concepts",
    "keyword_hits",
    "diagnostic_hits",
    "ipc_audit_hits",
)


@dataclass(frozen=True)
class PublicStep1Outputs:
    result: Path
    manifest: Path


@dataclass
class _SelectedRecord:
    row: dict[str, Any]
    route_rank: int
    quality_score: int
    association_count: int = 1


class PublicKeywordMatcher:
    """Transparent matcher supporting standalone and local-context rules."""

    def __init__(self, rules: Mapping[str, Any]) -> None:
        validate_rules(rules)
        matching = rules.get("matching", {})
        self.fields = tuple(matching.get("fields", ("claim", "abstract", "title")))
        self.context_window = int(matching.get("context_window_chars", 48))
        self.concepts = tuple(rules.get("concepts", ()))
        self.contexts = tuple(rules.get("context_lexicons", ()))
        self.diagnostics = tuple(rules.get("diagnostic_patterns", ()))
        self.ipc_rules = tuple(rules.get("ipc_audit_rules", ()))

    def match(self, record: Mapping[str, str]) -> dict[str, Any]:
        keyword_hits: list[dict[str, Any]] = []
        diagnostic_hits: list[dict[str, Any]] = []
        for field in self.fields:
            text = normalize_text(record.get(field, ""))
            if not text:
                continue
            keyword_hits.extend(self._match_keywords(text, field))
            diagnostic_hits.extend(self._match_diagnostics(text, field))
        keyword_hits.sort(key=lambda hit: (self.fields.index(hit["field"]), hit["start"]))
        diagnostic_hits.sort(
            key=lambda hit: (self.fields.index(hit["field"]), hit["start"])
        )
        route = "S" if keyword_hits else "E"
        return {
            "route": route,
            "matched_concepts": sorted({hit["concept_id"] for hit in keyword_hits}),
            "keyword_hits": keyword_hits,
            "diagnostic_hits": diagnostic_hits,
            "ipc_audit_hits": self._match_ipc(record.get("ipc", "")),
        }

    def _match_keywords(self, text: str, field: str) -> list[dict[str, Any]]:
        candidates: list[tuple[int, int, str, Mapping[str, Any]]] = []
        for concept in self.concepts:
            for variant in concept.get("variants", ()):
                normalized_variant = normalize_text(str(variant))
                for start, end in find_term_occurrences(text, normalized_variant):
                    candidates.append((start, end, normalized_variant, concept))
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))

        hits: list[dict[str, Any]] = []
        cursor = 0
        for start, end, _variant, concept in candidates:
            if start < cursor:
                continue
            if _inside_excluded_phrase(text, start, end, concept):
                continue
            scope_start, scope_end, scope_name = context_scope(
                text,
                start,
                end,
                field,
                self.context_window,
            )
            context_hits = self._context_hits(
                text,
                keyword_start=start,
                keyword_end=end,
                scope_start=scope_start,
                scope_end=scope_end,
            )
            context_ids = {hit["context_id"] for hit in context_hits}
            policy = concept.get("match_policy", {"mode": "standalone"})
            if policy.get("mode") == "cooccurrence":
                required_any = set(policy.get("required_any", ()))
                required_all = set(policy.get("required_all", ()))
                if required_any and not (required_any & context_ids):
                    continue
                if required_all and not required_all.issubset(context_ids):
                    continue
            hits.append(
                {
                    "concept_id": concept["concept_id"],
                    "category": concept.get("category", ""),
                    "canonical_term": concept.get("canonical_term", ""),
                    "matched_text": text[start:end],
                    "field": field,
                    "start": start,
                    "end": end,
                    "match_policy": policy.get("mode", "standalone"),
                    "context_scope": scope_name,
                    "context_snippet": text[scope_start:scope_end],
                    "context_hits": context_hits,
                    "public_source_ids": list(concept.get("public_source_ids", ())),
                }
            )
            cursor = end
        return hits

    def _context_hits(
        self,
        text: str,
        *,
        keyword_start: int,
        keyword_end: int,
        scope_start: int,
        scope_end: int,
    ) -> list[dict[str, Any]]:
        scope = text[scope_start:scope_end]
        hits: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for context in self.contexts:
            for variant in context.get("variants", ()):
                normalized_variant = normalize_text(str(variant))
                for relative_start, relative_end in find_term_occurrences(
                    scope,
                    normalized_variant,
                ):
                    start = scope_start + relative_start
                    end = scope_start + relative_end
                    if spans_overlap(start, end, keyword_start, keyword_end):
                        continue
                    identity = (context["context_id"], normalized_variant)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    hits.append(
                        {
                            "context_id": context["context_id"],
                            "kind": context.get("kind", ""),
                            "matched_text": text[start:end],
                            "start": start,
                            "end": end,
                            "distance": span_distance(
                                start,
                                end,
                                keyword_start,
                                keyword_end,
                            ),
                        }
                    )
        hits.sort(key=lambda hit: (hit["start"], hit["context_id"]))
        return hits

    def _match_diagnostics(self, text: str, field: str) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for pattern in self.diagnostics:
            found = False
            for variant in pattern.get("variants", ()):
                normalized_variant = normalize_text(str(variant))
                occurrences = find_term_occurrences(text, normalized_variant)
                if not occurrences:
                    continue
                start, end = occurrences[0]
                hits.append(
                    {
                        "pattern_id": pattern["pattern_id"],
                        "matched_text": text[start:end],
                        "field": field,
                        "start": start,
                        "end": end,
                    }
                )
                found = True
                break
            if found:
                continue
        return hits

    def _match_ipc(self, value: str) -> list[dict[str, Any]]:
        normalized = normalize_text(value).upper().replace(" ", "")
        hits = []
        for rule in self.ipc_rules:
            symbol = normalize_text(str(rule.get("symbol", ""))).upper().replace(" ", "")
            if symbol and symbol in normalized:
                hits.append(
                    {
                        "rule_id": rule["rule_id"],
                        "symbol": rule["symbol"],
                        "audit_only": True,
                    }
                )
        return hits


def run_public_step1(
    config_path: str | Path,
    *,
    input_override: str | Path | None = None,
    output_override: str | Path | None = None,
    overwrite: bool = False,
) -> PublicStep1Outputs:
    """Run public Step 1 without loading any private project resource."""

    config_file = Path(config_path).resolve()
    config = read_json(config_file)
    input_value = str(input_override or config.get("input_csv", "")).strip()
    if not input_value:
        raise ValueError(
            "input_csv is intentionally blank in the public template; "
            "provide --input or fill your local config"
        )
    input_path = resolve_from(config_file.parent, input_value)
    output_value = str(output_override or config.get("output_dir", "output")).strip()
    output_dir = resolve_from(config_file.parent, output_value)
    rules_path = resolve_from(
        config_file.parent,
        str(config.get("rules_file", "rules.template.json")),
    )
    rules = read_json(rules_path)
    matcher = PublicKeywordMatcher(rules)
    encoding = str(config.get("encoding", "utf-8-sig"))
    columns = dict(config.get("columns", {}))
    e_sample_rate = float(config.get("e_sample_rate", 0.02))
    e_sample_seed = str(config.get("e_sample_seed", "public-step1-e-sample-v1"))
    if not 0 < e_sample_rate <= 1:
        raise ValueError("e_sample_rate must be in (0, 1]")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    dataset_id = str(config.get("dataset_id", "")).strip() or input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.csv"
    manifest_path = output_dir / "manifest.json"
    if not overwrite and (result_path.exists() or manifest_path.exists()):
        raise FileExistsError("Public Step 1 outputs already exist; pass --overwrite")

    selected: dict[str, _SelectedRecord] = {}
    raw_rows = 0
    with input_path.open(encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header")
        for source_row_number, source_row in enumerate(reader, start=2):
            raw_rows += 1
            record = project_record(source_row, columns)
            patent_id = record["patent_id"] or synthetic_patent_id(
                record,
                source_row_number,
            )
            record["patent_id"] = patent_id
            match = matcher.match(record)
            row = {
                **record,
                **match,
                "source_row_number": source_row_number,
            }
            quality_score = (
                (1 if match["route"] == "S" else 0) * 1_000_000
                + bool(record["claim"]) * 100_000
                + bool(record["abstract"]) * 10_000
                + min(len(record["claim"]), 9_999)
                + min(len(record["abstract"]), 9_999)
            )
            current = selected.get(patent_id)
            if current is None:
                selected[patent_id] = _SelectedRecord(
                    row=row,
                    route_rank=1 if match["route"] == "S" else 0,
                    quality_score=quality_score,
                )
                continue
            current.association_count += 1
            candidate_rank = 1 if match["route"] == "S" else 0
            if (candidate_rank, quality_score) > (
                current.route_rank,
                current.quality_score,
            ):
                current.row = row
                current.route_rank = candidate_rank
                current.quality_score = quality_score

    route_counts: Counter[str] = Counter()
    selection_counts: Counter[str] = Counter()
    result_rows = []
    for patent_id, item in sorted(
        selected.items(),
        key=lambda pair: (pair[1].row["source_row_number"], pair[0]),
    ):
        row = item.row
        route = row["route"]
        route_counts[route] += 1
        if route == "S":
            selected_for_step2 = True
            selection_group = "S_all"
            probability = 1.0
            sample_weight = 1.0
            sample_seed = ""
        else:
            selected_for_step2 = stable_sample(
                f"{dataset_id}|{patent_id}",
                e_sample_rate,
                e_sample_seed,
            )
            selection_group = (
                "E_random" if selected_for_step2 else "E_not_selected"
            )
            probability = e_sample_rate
            sample_weight = 1 / e_sample_rate if selected_for_step2 else ""
            sample_seed = e_sample_seed
        if selected_for_step2:
            selection_counts[selection_group] += 1
        result_rows.append(
            {
                "dataset_id": dataset_id,
                **row,
                "association_count": item.association_count,
                "step1_label": LABEL_BY_ROUTE[route],
                "selected_for_step2": str(selected_for_step2).lower(),
                "selection_group": selection_group,
                "selection_probability": f"{probability:.12g}",
                "sample_weight": (
                    f"{sample_weight:.12g}"
                    if isinstance(sample_weight, float)
                    else ""
                ),
                "sample_seed": sample_seed,
            }
        )
    atomic_write_csv(result_path, result_rows)

    manifest = {
        "public_method_version": "public-step1-se-routing-v1",
        "dataset_id": dataset_id,
        "input_source": "",
        "input_file_name": input_path.name,
        "input_sha256": sha256_file(input_path),
        "config_sha256": sha256_file(config_file),
        "rules_sha256": sha256_file(rules_path),
        "rules_schema_version": rules.get("schema_version", ""),
        "stats": {
            "input_rows": raw_rows,
            "unique_patents": len(result_rows),
            "duplicate_association_rows": raw_rows - len(result_rows),
            "route_counts": {route: route_counts[route] for route in ROUTES},
            "selected_for_step2": dict(sorted(selection_counts.items())),
        },
        "sampling": {
            "S_probability": 1.0,
            "E_probability": e_sample_rate,
            "E_seed": e_sample_seed,
        },
        "output": {
            "result_file": result_path.name,
            "result_sha256": sha256_file(result_path),
        },
        "llm_requests_executed": 0,
        "disclosure": {
            "production_taxonomy_included": False,
            "expert_lexicon_included": False,
            "raw_data_location_included": False,
            "credentials_included": False,
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(manifest_path, manifest)
    return PublicStep1Outputs(result=result_path, manifest=manifest_path)


def validate_rules(rules: Mapping[str, Any]) -> None:
    matching = rules.get("matching", {})
    fields = matching.get("fields", ("claim", "abstract", "title"))
    if not fields or not all(isinstance(field, str) and field for field in fields):
        raise ValueError("matching.fields must contain at least one field")
    if int(matching.get("context_window_chars", 48)) < 0:
        raise ValueError("context_window_chars cannot be negative")

    context_ids = {
        str(item.get("context_id", ""))
        for item in rules.get("context_lexicons", ())
    }
    seen_ids: set[str] = set()
    seen_variants: set[str] = set()
    for concept in rules.get("concepts", ()):
        concept_id = str(concept.get("concept_id", ""))
        if not concept_id or concept_id in seen_ids:
            raise ValueError(f"Missing or duplicate concept_id: {concept_id!r}")
        seen_ids.add(concept_id)
        variants = [normalize_text(str(value)) for value in concept.get("variants", ())]
        if not variants or any(not value for value in variants):
            raise ValueError(f"Concept has no usable variants: {concept_id}")
        overlap = seen_variants & set(variants)
        if overlap:
            raise ValueError(f"Keyword variants must be globally unique: {sorted(overlap)}")
        seen_variants.update(variants)
        policy = concept.get("match_policy", {"mode": "standalone"})
        mode = policy.get("mode")
        if mode not in ALLOWED_MATCH_MODES:
            raise ValueError(f"Unsupported match mode for {concept_id}: {mode}")
        referenced = set(policy.get("required_any", ())) | set(
            policy.get("required_all", ())
        )
        if missing := referenced - context_ids:
            raise ValueError(f"Unknown context IDs for {concept_id}: {sorted(missing)}")


def project_record(
    source_row: Mapping[str, str],
    columns: Mapping[str, str],
) -> dict[str, str]:
    fields = (
        "patent_id",
        "title",
        "abstract",
        "claim",
        "ipc",
        "applicant",
        "application_date",
    )
    return {
        field: str(source_row.get(columns.get(field, field), "") or "").strip()
        for field in fields
    }


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    normalized = normalized.translate(CONNECTOR_TRANSLATION)
    normalized = re.sub(r"[\t\u3000 ]+", " ", normalized)
    return normalized.strip()


def find_term_occurrences(text: str, term: str) -> list[tuple[int, int]]:
    if not term:
        return []
    occurrences = []
    start = text.find(term)
    while start >= 0:
        end = start + len(term)
        if not needs_ascii_boundary(term) or has_ascii_boundaries(text, start, end):
            occurrences.append((start, end))
        start = text.find(term, start + 1)
    return occurrences


def needs_ascii_boundary(term: str) -> bool:
    return bool(ASCII_TERM.fullmatch(term)) and any(char.isascii() for char in term)


def has_ascii_boundaries(text: str, start: int, end: int) -> bool:
    left_ok = start == 0 or not text[start - 1].isascii() or not text[start - 1].isalnum()
    right_ok = end == len(text) or not text[end].isascii() or not text[end].isalnum()
    return left_ok and right_ok


def context_scope(
    text: str,
    start: int,
    end: int,
    field: str,
    window: int,
) -> tuple[int, int, str]:
    if field == "title":
        return 0, len(text), "title"
    left_boundary = max(text.rfind(mark, 0, start) for mark in SENTENCE_BOUNDARIES)
    right_boundaries = [
        position
        for mark in SENTENCE_BOUNDARIES
        if (position := text.find(mark, end)) >= 0
    ]
    if left_boundary >= 0 or right_boundaries:
        left = left_boundary + 1 if left_boundary >= 0 else 0
        right = min(right_boundaries) if right_boundaries else len(text)
        return left, right, "sentence"
    return max(0, start - window), min(len(text), end + window), "window"


def _inside_excluded_phrase(
    text: str,
    start: int,
    end: int,
    concept: Mapping[str, Any],
) -> bool:
    for value in concept.get("excluded_phrases", ()):
        phrase = normalize_text(str(value))
        search_start = max(0, start - len(phrase) + 1)
        position = text.find(phrase, search_start, min(len(text), end + len(phrase)))
        while position >= 0:
            if position <= start and position + len(phrase) >= end:
                return True
            position = text.find(
                phrase,
                position + 1,
                min(len(text), end + len(phrase)),
            )
    return False


def spans_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def span_distance(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    if spans_overlap(left_start, left_end, right_start, right_end):
        return 0
    return min(abs(left_end - right_start), abs(right_end - left_start))


def synthetic_patent_id(record: Mapping[str, str], source_row_number: int) -> str:
    identity = "\x1f".join(
        (
            normalize_text(record.get("title", "")),
            normalize_text(record.get("applicant", "")),
            normalize_text(record.get("application_date", "")),
        )
    )
    if not identity.replace("\x1f", ""):
        identity = f"source-row-{source_row_number}"
    return "synthetic-" + hashlib.sha256(identity.encode()).hexdigest()[:32]


def stable_sample(identity: str, probability: float, seed: str) -> bool:
    digest = hashlib.sha256(f"{seed}|{identity}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < probability


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON resource must contain an object: {path}")
    return value


def resolve_from(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: (
                    json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if key
                    in {
                        "matched_concepts",
                        "keyword_hits",
                        "diagnostic_hits",
                        "ipc_audit_hits",
                    }
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow({field: serialized.get(field, "") for field in OUTPUT_FIELDS})
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.example.json"),
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = run_public_step1(
        args.config,
        input_override=args.input,
        output_override=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "result": str(outputs.result),
                "manifest": str(outputs.manifest),
                "llm_requests_executed": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
