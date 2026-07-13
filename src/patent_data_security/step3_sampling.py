"""Reproducible Step 3 sampling for blinded Human Gold annotation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from patent_data_security.step2_prompt import PROMPT_VERSION

SAMPLING_VERSION = "step3-gold-v1.0.0"
METHODOLOGY_VERSION = "1.0.0"
ELIGIBLE_LEVELS = ("S", "W", "R")
RISK_GROUP_ORDER = (
    "cat2_or_review",
    "retry_success",
    "rare_subtype",
    "high_conflict",
    "swr_to_cat3",
)
RISK_GROUP_WEIGHTS = {
    "cat2_or_review": 0.36,
    "retry_success": 0.12,
    "rare_subtype": 0.12,
    "high_conflict": 0.24,
    "swr_to_cat3": 0.16,
}

AUDIT_FIELDS = (
    "sample_id",
    "task_id",
    "dataset_id",
    "year",
    "patent_id",
    "source_row_number",
    "sample_stage",
    "sampling_stratum",
    "representative_stratum",
    "risk_group",
    "risk_flags",
    "stratum_population",
    "stratum_sample_size",
    "core_draws_in_stratum",
    "risk_draws_in_stratum",
    "inclusion_probability",
    "evaluation_weight",
    "random_seed",
    "sample_rank_in_stratum",
    "keyword_level",
    "transition_type",
    "selection_group",
    "step2_selection_probability",
    "step2_sample_weight",
    "requested_model",
    "actual_model",
    "prompt_version",
    "response_id",
    "glm_cat",
    "raw_confidence",
    "confidence_bin",
    "subtype",
    "core_invention",
    "protected_object_or_activity",
    "security_goal_or_risk",
    "technical_mechanism",
    "causal_centrality",
    "missing_or_ambiguous_link",
    "glm_evidence",
    "glm_reason",
    "review_flag",
    "review_reason",
    "attempts",
    "elapsed_seconds",
    "completed_at",
    "text_hash",
    "exact_text_group_size",
    "title",
    "abstract",
    "claim",
    "ipc",
    "main_ipc",
)

BLINDED_FIELDS = (
    "sample_id",
    "patent_id",
    "year",
    "title",
    "abstract",
    "claim",
    "ipc",
    "main_ipc",
    "annotator_slot",
    "annotator_id",
    "annotation_status",
    "human_cat",
    "human_confidence",
    "human_core_invention",
    "human_protected_object_or_activity",
    "human_security_goal_or_risk",
    "human_technical_mechanism",
    "human_causal_centrality",
    "human_missing_or_ambiguous_link",
    "human_evidence",
    "human_reason",
    "human_review_state",
    "human_review_reason",
    "duplicate_or_unreadable",
    "annotation_notes",
    "submitted_at",
)

ADJUDICATION_FIELDS = (
    "sample_id",
    "patent_id",
    "year",
    "title",
    "abstract",
    "claim",
    "ipc",
    "main_ipc",
    "annotator_a_cat",
    "annotator_a_review_state",
    "annotator_a_reason",
    "annotator_b_cat",
    "annotator_b_review_state",
    "annotator_b_reason",
    "conflict_type",
    "adjudicator_id",
    "adjudication_status",
    "final_human_cat",
    "human_review_state",
    "adjudication_reason",
    "adjudicated_at",
)

STRATA_FIELDS = (
    "sampling_stratum",
    "year",
    "keyword_level",
    "glm_cat",
    "confidence_bin",
    "risk_group",
    "population_size",
    "core_draws",
    "risk_draws",
    "sample_size",
    "inclusion_probability",
    "evaluation_weight",
)


@dataclass(frozen=True)
class GoldSamplingConfig:
    target_size: int = 2_000
    core_size: int = 1_500
    seed: str = "step3-gold-v1"
    rare_subtype_max_population: int = 100
    high_confidence_threshold: float = 0.90

    @property
    def risk_size(self) -> int:
        return self.target_size - self.core_size


@dataclass(frozen=True)
class GoldSamplingPaths:
    audit: Path
    blinded: Path
    annotator_a: Path
    annotator_b: Path
    adjudication: Path
    strata: Path
    report: Path
    schema: Path


def step3_paths(output_dir: str | Path) -> GoldSamplingPaths:
    root = Path(output_dir).resolve()
    return GoldSamplingPaths(
        audit=root / "gold_sample_audit.csv",
        blinded=root / "gold_sample_blinded.csv",
        annotator_a=root / "annotation_round1_A.csv",
        annotator_b=root / "annotation_round1_B.csv",
        adjudication=root / "adjudication_template.csv",
        strata=root / "sampling_strata.csv",
        report=root / "sampling_report.json",
        schema=root / "annotation_schema.json",
    )


def sample_gold_corpus(
    databases: list[str | Path],
    output_dir: str | Path,
    *,
    config: GoldSamplingConfig | None = None,
    rebuild: bool = False,
) -> tuple[GoldSamplingPaths, dict[str, Any]]:
    """Sample an auditable S/W/R Gold candidate set and write blinded templates."""

    config = config or GoldSamplingConfig()
    _validate_config(config)
    database_paths = sorted({Path(path).resolve() for path in databases})
    if not database_paths:
        raise ValueError("At least one Step 2 database is required")
    missing = [path for path in database_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Step 2 databases: {', '.join(map(str, missing))}")

    paths = step3_paths(output_dir)
    _prepare_output_directory(paths, rebuild=rebuild)
    records, input_manifest, excluded = _load_step2_records(database_paths)
    if config.target_size > len(records):
        raise ValueError(
            f"target_size={config.target_size} exceeds S/W/R population={len(records)}"
        )

    subtype_counts = Counter(record["subtype"] for record in records)
    text_counts = Counter(_text_hash(record) for record in records)
    for record in records:
        record["confidence_bin"] = _confidence_bin(record["confidence"])
        record["transition_type"] = (
            f"{record['keyword_level']}_to_cat{record['cat']}"
        )
        record["risk_flags"] = _risk_flags(record, subtype_counts, config)
        record["risk_group"] = _primary_risk_group(record["risk_flags"])
        record["representative_stratum"] = _join_stratum(
            record["year"],
            record["keyword_level"],
            f"cat{record['cat']}",
            record["confidence_bin"],
        )
        record["sampling_stratum"] = _join_stratum(
            record["representative_stratum"], record["risk_group"]
        )
        record["text_hash"] = _text_hash(record)
        record["exact_text_group_size"] = text_counts[record["text_hash"]]

    allocations = _build_allocations(records, config)
    selected = _draw_records(records, allocations, config)
    _write_outputs(paths, selected, allocations, config)

    report = _sampling_report(
        records,
        selected,
        allocations,
        input_manifest,
        excluded,
        config,
        paths,
        subtype_counts,
    )
    _write_json(paths.report, report)
    return paths, report


def discover_step2_databases(step2_dir: str | Path) -> list[Path]:
    return sorted(Path(step2_dir).resolve().glob("classification_state_*.sqlite3"))


def _validate_config(config: GoldSamplingConfig) -> None:
    if config.target_size < 1:
        raise ValueError("target_size must be positive")
    if not 0 <= config.core_size <= config.target_size:
        raise ValueError("core_size must be between zero and target_size")
    if config.rare_subtype_max_population < 1:
        raise ValueError("rare_subtype_max_population must be positive")
    if not 0 <= config.high_confidence_threshold <= 1:
        raise ValueError("high_confidence_threshold must be in [0, 1]")


def _prepare_output_directory(paths: GoldSamplingPaths, *, rebuild: bool) -> None:
    root = paths.report.parent
    existing = [path for path in asdict(paths).values() if Path(path).exists()]
    if existing and not rebuild:
        raise FileExistsError(
            "Step 3 outputs already exist; use --rebuild to replace the frozen sample: "
            + ", ".join(map(str, existing))
        )
    if rebuild and root.exists():
        for path in asdict(paths).values():
            Path(path).unlink(missing_ok=True)
    root.mkdir(parents=True, exist_ok=True)


def _load_step2_records(
    databases: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    input_manifest: list[dict[str, Any]] = []
    excluded = {"E": 0, "non_succeeded": 0}
    seen_task_ids: set[str] = set()
    seen_patent_ids: set[str] = set()
    for database in databases:
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            connection.close()
            raise ValueError(f"SQLite integrity check failed for {database}: {integrity}")
        rows = connection.execute("SELECT * FROM tasks ORDER BY dataset_id, task_id")
        database_total = 0
        database_eligible = 0
        for row in rows:
            database_total += 1
            if row["keyword_level"] == "E":
                excluded["E"] += 1
                continue
            if row["keyword_level"] not in ELIGIBLE_LEVELS:
                connection.close()
                raise ValueError(
                    f"Unknown keyword_level={row['keyword_level']!r} in {database}"
                )
            if row["status"] != "succeeded":
                excluded["non_succeeded"] += 1
                continue
            if row["task_id"] in seen_task_ids:
                connection.close()
                raise ValueError(f"Duplicate task_id across databases: {row['task_id']}")
            if row["patent_id"] in seen_patent_ids:
                connection.close()
                raise ValueError(
                    "Gold sampling requires globally unique patent_id values; duplicate: "
                    f"{row['patent_id']}"
                )
            result = json.loads(row["result_json"])
            payload = json.loads(row["payload_json"])
            evidence_chain = result.get("evidence_chain", {})
            dataset_id = str(row["dataset_id"])
            records.append(
                {
                    "task_id": row["task_id"],
                    "dataset_id": dataset_id,
                    "year": dataset_id,
                    "patent_id": row["patent_id"],
                    "source_row_number": row["source_row_number"],
                    "keyword_level": row["keyword_level"],
                    "selection_group": row["selection_group"],
                    "step2_selection_probability": row["selection_probability"],
                    "step2_sample_weight": row["sample_weight"],
                    "requested_model": row["requested_model"] or "",
                    "actual_model": row["actual_model"] or "",
                    "response_id": row["response_id"] or "",
                    "cat": int(result["cat"]),
                    "confidence": float(result["confidence"]),
                    "subtype": result["subtype"],
                    "core_invention": result["core_invention"],
                    "protected_object_or_activity": evidence_chain[
                        "protected_object_or_activity"
                    ],
                    "security_goal_or_risk": evidence_chain["security_goal_or_risk"],
                    "technical_mechanism": evidence_chain["technical_mechanism"],
                    "causal_centrality": evidence_chain["causal_centrality"],
                    "missing_or_ambiguous_link": evidence_chain[
                        "missing_or_ambiguous_link"
                    ],
                    "glm_evidence": result.get("evidence", []),
                    "glm_reason": result["reason"],
                    "review_flag": bool(result["review_flag"]),
                    "review_reason": result.get("review_reason", ""),
                    "attempts": int(row["attempts"]),
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                    "completed_at": row["completed_at"] or "",
                    "title": payload.get("title", ""),
                    "abstract": payload.get("abstract", ""),
                    "claim": payload.get("claim", ""),
                    "ipc": payload.get("ipc", ""),
                    "main_ipc": payload.get("main_ipc", ""),
                }
            )
            seen_task_ids.add(row["task_id"])
            seen_patent_ids.add(row["patent_id"])
            database_eligible += 1
        connection.close()
        input_manifest.append(
            {
                "path": str(database),
                "size_bytes": database.stat().st_size,
                "sha256": _file_sha256(database),
                "total_tasks": database_total,
                "eligible_swr_tasks": database_eligible,
            }
        )
    if excluded["non_succeeded"]:
        raise ValueError(
            "All eligible S/W/R tasks must have succeeded before Step 3; "
            f"found {excluded['non_succeeded']} incomplete tasks"
        )
    return records, input_manifest, excluded


def _confidence_bin(confidence: float) -> str:
    if confidence < 0.80:
        return "lt_0.80"
    if confidence < 0.90:
        return "0.80_to_lt_0.90"
    if confidence < 0.98:
        return "0.90_to_lt_0.98"
    return "0.98_to_1.00"


def _risk_flags(
    record: dict[str, Any],
    subtype_counts: Counter[str],
    config: GoldSamplingConfig,
) -> list[str]:
    flags = []
    if record["cat"] == 2 or record["review_flag"]:
        flags.append("cat2_or_review")
    if record["attempts"] > 1:
        flags.append("retry_success")
    if subtype_counts[record["subtype"]] <= config.rare_subtype_max_population:
        flags.append("rare_subtype")
    if (
        record["cat"] == 3
        and record["keyword_level"] in {"S", "W"}
        and record["confidence"] >= config.high_confidence_threshold
    ):
        flags.append("high_conflict")
    if record["cat"] == 3:
        flags.append("swr_to_cat3")
    return flags


def _primary_risk_group(flags: list[str]) -> str:
    for group in RISK_GROUP_ORDER:
        if group in flags:
            return group
    return "none"


def _build_allocations(
    records: list[dict[str, Any]], config: GoldSamplingConfig
) -> dict[str, dict[str, int]]:
    fine_counts = Counter(record["sampling_stratum"] for record in records)
    base_counts = Counter(record["representative_stratum"] for record in records)
    fine_to_base = {
        record["sampling_stratum"]: record["representative_stratum"]
        for record in records
    }
    fine_to_risk = {
        record["sampling_stratum"]: record["risk_group"] for record in records
    }
    base_core = _allocate(
        weights={key: float(value) for key, value in base_counts.items()},
        capacities=dict(base_counts),
        total=config.core_size,
        minimum=1 if config.core_size >= len(base_counts) else 0,
    )
    core_fine = {key: 0 for key in fine_counts}
    by_base: dict[str, list[str]] = defaultdict(list)
    for fine, base in fine_to_base.items():
        by_base[base].append(fine)
    for base, quota in base_core.items():
        keys = by_base[base]
        allocation = _allocate(
            weights={key: float(fine_counts[key]) for key in keys},
            capacities={key: fine_counts[key] for key in keys},
            total=quota,
        )
        core_fine.update(allocation)

    remaining_capacity = {
        fine: fine_counts[fine] - core_fine[fine] for fine in fine_counts
    }
    group_capacities = {
        group: sum(
            capacity
            for fine, capacity in remaining_capacity.items()
            if fine_to_risk[fine] == group
        )
        for group in RISK_GROUP_ORDER
    }
    available_groups = {
        group: RISK_GROUP_WEIGHTS[group]
        for group in RISK_GROUP_ORDER
        if group_capacities[group] > 0
    }
    if config.risk_size > sum(group_capacities.values()):
        raise ValueError(
            f"risk_size={config.risk_size} exceeds remaining risk capacity="
            f"{sum(group_capacities.values())}"
        )
    group_risk = _allocate(
        weights=available_groups,
        capacities={group: group_capacities[group] for group in available_groups},
        total=config.risk_size,
    )
    risk_fine = {key: 0 for key in fine_counts}
    for group, quota in group_risk.items():
        keys = [
            fine
            for fine in fine_counts
            if fine_to_risk[fine] == group and remaining_capacity[fine] > 0
        ]
        allocation = _allocate(
            weights={key: float(remaining_capacity[key]) for key in keys},
            capacities={key: remaining_capacity[key] for key in keys},
            total=quota,
        )
        risk_fine.update(allocation)

    allocations = {
        fine: {
            "population": fine_counts[fine],
            "core": core_fine[fine],
            "risk": risk_fine[fine],
            "sample": core_fine[fine] + risk_fine[fine],
        }
        for fine in sorted(fine_counts)
    }
    if sum(item["sample"] for item in allocations.values()) != config.target_size:
        raise AssertionError("Internal allocation error: target size was not met")
    return allocations


def _allocate(
    *,
    weights: dict[str, float],
    capacities: dict[str, int],
    total: int,
    minimum: int = 0,
) -> dict[str, int]:
    keys = sorted(weights)
    if total < 0 or total > sum(capacities.get(key, 0) for key in keys):
        raise ValueError("Allocation total is outside available capacity")
    allocation = {
        key: min(minimum, capacities.get(key, 0)) if weights[key] > 0 else 0
        for key in keys
    }
    if sum(allocation.values()) > total:
        allocation = {key: 0 for key in keys}
    remaining = total - sum(allocation.values())
    while remaining:
        eligible = [
            key
            for key in keys
            if weights[key] > 0 and allocation[key] < capacities.get(key, 0)
        ]
        if not eligible:
            raise ValueError("Unable to allocate requested total within capacities")
        weight_sum = sum(weights[key] for key in eligible)
        ideals = {key: remaining * weights[key] / weight_sum for key in eligible}
        added = 0
        for key in eligible:
            increment = min(
                int(ideals[key]), capacities[key] - allocation[key], remaining - added
            )
            allocation[key] += increment
            added += increment
            if added == remaining:
                break
        remaining -= added
        if not remaining:
            break
        remainder_order = sorted(
            eligible,
            key=lambda key: (-(ideals[key] - int(ideals[key])), key),
        )
        for key in remainder_order:
            if allocation[key] >= capacities[key]:
                continue
            allocation[key] += 1
            remaining -= 1
            if not remaining:
                break
    return allocation


def _draw_records(
    records: list[dict[str, Any]],
    allocations: dict[str, dict[str, int]],
    config: GoldSamplingConfig,
) -> list[dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stratum[record["sampling_stratum"]].append(record)
    selected: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        allocation = allocations[stratum]
        ordered = sorted(
            by_stratum[stratum],
            key=lambda record: _stable_digest(
                config.seed, "draw", stratum, record["task_id"]
            ),
        )
        drawn = [dict(record) for record in ordered[: allocation["sample"]]]
        phase_order = sorted(
            range(len(drawn)),
            key=lambda index: _stable_digest(
                config.seed, "phase", stratum, drawn[index]["task_id"]
            ),
        )
        core_indexes = set(phase_order[: allocation["core"]])
        probability = allocation["sample"] / allocation["population"]
        for rank, record in enumerate(drawn, start=1):
            record["sample_id"] = "GOLD-" + _stable_digest(
                config.seed, "sample-id", record["task_id"]
            )[:16]
            record["sample_stage"] = (
                "representative_core" if rank - 1 in core_indexes else "risk_addition"
            )
            record["stratum_population"] = allocation["population"]
            record["stratum_sample_size"] = allocation["sample"]
            record["core_draws_in_stratum"] = allocation["core"]
            record["risk_draws_in_stratum"] = allocation["risk"]
            record["inclusion_probability"] = probability
            record["evaluation_weight"] = 1 / probability
            record["random_seed"] = config.seed
            record["sample_rank_in_stratum"] = rank
            selected.append(record)
    if len(selected) != config.target_size:
        raise AssertionError("Internal sampling error: selected size differs from target")
    if len({record["sample_id"] for record in selected}) != len(selected):
        raise AssertionError("Stable sample_id collision detected")
    return selected


def _write_outputs(
    paths: GoldSamplingPaths,
    selected: list[dict[str, Any]],
    allocations: dict[str, dict[str, int]],
    config: GoldSamplingConfig,
) -> None:
    audit_rows = [_audit_row(record) for record in _canonical_order(selected)]
    _write_csv(paths.audit, AUDIT_FIELDS, audit_rows)
    canonical_blinded = [_blinded_row(record, "") for record in _canonical_order(selected)]
    _write_csv(paths.blinded, BLINDED_FIELDS, canonical_blinded)
    annotator_a = [
        _blinded_row(record, "A")
        for record in _annotator_order(selected, config.seed, "A")
    ]
    annotator_b = [
        _blinded_row(record, "B")
        for record in _annotator_order(selected, config.seed, "B")
    ]
    _write_csv(paths.annotator_a, BLINDED_FIELDS, annotator_a)
    _write_csv(paths.annotator_b, BLINDED_FIELDS, annotator_b)
    adjudication_rows = [
        _adjudication_row(record) for record in _canonical_order(selected)
    ]
    _write_csv(paths.adjudication, ADJUDICATION_FIELDS, adjudication_rows)
    record_by_stratum = {
        record["sampling_stratum"]: record for record in selected
    }
    strata_rows = []
    for stratum, allocation in allocations.items():
        example = record_by_stratum.get(stratum)
        if example is None:
            parts = stratum.split("|")
            year, level, cat, confidence_bin, risk_group = parts
        else:
            year = example["year"]
            level = example["keyword_level"]
            cat = f"cat{example['cat']}"
            confidence_bin = example["confidence_bin"]
            risk_group = example["risk_group"]
        probability = allocation["sample"] / allocation["population"]
        strata_rows.append(
            {
                "sampling_stratum": stratum,
                "year": year,
                "keyword_level": level,
                "glm_cat": cat.removeprefix("cat"),
                "confidence_bin": confidence_bin,
                "risk_group": risk_group,
                "population_size": allocation["population"],
                "core_draws": allocation["core"],
                "risk_draws": allocation["risk"],
                "sample_size": allocation["sample"],
                "inclusion_probability": _format_float(probability),
                "evaluation_weight": (
                    _format_float(1 / probability) if probability else ""
                ),
            }
        )
    _write_csv(paths.strata, STRATA_FIELDS, strata_rows)
    _write_json(paths.schema, _annotation_schema())


def _audit_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "task_id": record["task_id"],
        "dataset_id": record["dataset_id"],
        "year": record["year"],
        "patent_id": record["patent_id"],
        "source_row_number": record["source_row_number"],
        "sample_stage": record["sample_stage"],
        "sampling_stratum": record["sampling_stratum"],
        "representative_stratum": record["representative_stratum"],
        "risk_group": record["risk_group"],
        "risk_flags": json.dumps(record["risk_flags"], ensure_ascii=False),
        "stratum_population": record["stratum_population"],
        "stratum_sample_size": record["stratum_sample_size"],
        "core_draws_in_stratum": record["core_draws_in_stratum"],
        "risk_draws_in_stratum": record["risk_draws_in_stratum"],
        "inclusion_probability": _format_float(record["inclusion_probability"]),
        "evaluation_weight": _format_float(record["evaluation_weight"]),
        "random_seed": record["random_seed"],
        "sample_rank_in_stratum": record["sample_rank_in_stratum"],
        "keyword_level": record["keyword_level"],
        "transition_type": record["transition_type"],
        "selection_group": record["selection_group"],
        "step2_selection_probability": record["step2_selection_probability"],
        "step2_sample_weight": record["step2_sample_weight"],
        "requested_model": record["requested_model"],
        "actual_model": record["actual_model"],
        "prompt_version": PROMPT_VERSION,
        "response_id": record["response_id"],
        "glm_cat": record["cat"],
        "raw_confidence": record["confidence"],
        "confidence_bin": record["confidence_bin"],
        "subtype": record["subtype"],
        "core_invention": record["core_invention"],
        "protected_object_or_activity": record["protected_object_or_activity"],
        "security_goal_or_risk": record["security_goal_or_risk"],
        "technical_mechanism": record["technical_mechanism"],
        "causal_centrality": record["causal_centrality"],
        "missing_or_ambiguous_link": record["missing_or_ambiguous_link"],
        "glm_evidence": json.dumps(record["glm_evidence"], ensure_ascii=False),
        "glm_reason": record["glm_reason"],
        "review_flag": str(record["review_flag"]).lower(),
        "review_reason": record["review_reason"],
        "attempts": record["attempts"],
        "elapsed_seconds": round(record["elapsed_seconds"], 3),
        "completed_at": record["completed_at"],
        "text_hash": record["text_hash"],
        "exact_text_group_size": record["exact_text_group_size"],
        "title": record["title"],
        "abstract": record["abstract"],
        "claim": record["claim"],
        "ipc": record["ipc"],
        "main_ipc": record["main_ipc"],
    }


def _blinded_row(record: dict[str, Any], slot: str) -> dict[str, Any]:
    row = {field: "" for field in BLINDED_FIELDS}
    for field in (
        "sample_id",
        "patent_id",
        "year",
        "title",
        "abstract",
        "claim",
        "ipc",
        "main_ipc",
    ):
        row[field] = record[field]
    row["annotator_slot"] = slot
    return row


def _adjudication_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {field: "" for field in ADJUDICATION_FIELDS}
    for field in (
        "sample_id",
        "patent_id",
        "year",
        "title",
        "abstract",
        "claim",
        "ipc",
        "main_ipc",
    ):
        row[field] = record[field]
    return row


def _sampling_report(
    population: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    allocations: dict[str, dict[str, int]],
    input_manifest: list[dict[str, Any]],
    excluded: dict[str, int],
    config: GoldSamplingConfig,
    paths: GoldSamplingPaths,
    subtype_counts: Counter[str],
) -> dict[str, Any]:
    return {
        "sampling_version": SAMPLING_VERSION,
        "methodology_version": METHODOLOGY_VERSION,
        "created_at": _now(),
        "seed": config.seed,
        "target_size": config.target_size,
        "core_target": config.core_size,
        "risk_target": config.risk_size,
        "eligible_population": len(population),
        "selected": len(selected),
        "excluded": excluded,
        "eligibility": {
            "included_keyword_levels": list(ELIGIBLE_LEVELS),
            "E_policy": "excluded_from_regular_Gold; reserved for final exclusion certification",
            "required_step2_status": "succeeded",
        },
        "representative_strata": [
            "year",
            "keyword_level",
            "glm_cat",
            "confidence_bin",
        ],
        "confidence_bins": [
            "lt_0.80",
            "0.80_to_lt_0.90",
            "0.90_to_lt_0.98",
            "0.98_to_1.00",
        ],
        "risk_design": {
            "priority_order": list(RISK_GROUP_ORDER),
            "target_weights": RISK_GROUP_WEIGHTS,
            "rare_subtype_max_population": config.rare_subtype_max_population,
            "high_confidence_threshold": config.high_confidence_threshold,
            "definitions": {
                "cat2_or_review": "glm_cat=2 or review_flag=true",
                "retry_success": "attempts>1",
                "rare_subtype": "eligible subtype population at or below threshold",
                "high_conflict": "S/W route -> glm_cat=3 with confidence at or above threshold",
                "swr_to_cat3": "remaining S/W/R -> glm_cat=3 transitions",
            },
        },
        "probability_contract": (
            "A single SRSWOR draw is made within each mutually exclusive fine stratum; "
            "inclusion_probability=n_h/N_h and evaluation_weight=N_h/n_h."
        ),
        "population_distribution": {
            "keyword_level": dict(sorted(Counter(r["keyword_level"] for r in population).items())),
            "glm_cat": _string_key_counter(r["cat"] for r in population),
            "subtype": dict(sorted(subtype_counts.items())),
        },
        "sample_distribution": {
            "stage": dict(sorted(Counter(r["sample_stage"] for r in selected).items())),
            "representative_core_by_risk_group": dict(
                sorted(
                    Counter(
                        r["risk_group"]
                        for r in selected
                        if r["sample_stage"] == "representative_core"
                    ).items()
                )
            ),
            "risk_addition_by_group": dict(
                sorted(
                    Counter(
                        r["risk_group"]
                        for r in selected
                        if r["sample_stage"] == "risk_addition"
                    ).items()
                )
            ),
            "keyword_level": dict(sorted(Counter(r["keyword_level"] for r in selected).items())),
            "glm_cat": _string_key_counter(r["cat"] for r in selected),
            "risk_group": dict(sorted(Counter(r["risk_group"] for r in selected).items())),
            "attempts_gt_1": sum(r["attempts"] > 1 for r in selected),
            "review_flag": sum(r["review_flag"] for r in selected),
        },
        "strata": {
            "population_strata": len(allocations),
            "sampled_strata": sum(item["sample"] > 0 for item in allocations.values()),
        },
        "input_databases": input_manifest,
        "outputs": {key: str(value) for key, value in asdict(paths).items()},
        "blinding": {
            "round1_files_exclude": [
                "keyword_level",
                "G0 category/confidence/subtype/evidence/reason",
                "risk flags and sampling stage",
            ],
            "audit_file_access": "research coordinator only; do not give to annotators",
        },
    }


def _annotation_schema() -> dict[str, Any]:
    return {
        "schema_version": "human-gold-annotation-v1.0.0",
        "round1": {
            "human_cat": {"allowed": [1, 2, 3], "required_when_completed": True},
            "human_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "annotation_status": {"allowed": ["completed", "skipped"]},
            "human_review_state": {
                "allowed": ["resolved", "needs_adjudication", "unresolved"]
            },
            "duplicate_or_unreadable": {"allowed": ["false", "duplicate", "unreadable"]},
            "human_evidence": {
                "format": "JSON array of verbatim evidence snippets from the patent"
            },
        },
        "adjudication": {
            "final_human_cat": {"allowed": [1, 3], "nullable_when_unresolved": True},
            "human_review_state": {"allowed": ["resolved", "unresolved"]},
            "adjudication_status": {"allowed": ["completed", "not_required"]},
        },
        "blinding_rule": (
            "Annotators must not receive gold_sample_audit.csv or any G0/keyword/sampling fields "
            "before both independent submissions are frozen."
        ),
    }


def _canonical_order(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: record["sample_id"])


def _annotator_order(
    records: list[dict[str, Any]], seed: str, slot: str
) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: _stable_digest(
            seed, "annotator-order", slot, record["sample_id"]
        ),
    )


def _text_hash(record: dict[str, Any]) -> str:
    parts = []
    for field in ("title", "abstract", "claim"):
        normalized = unicodedata.normalize("NFKC", str(record.get(field, "")))
        parts.append(" ".join(normalized.split()))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _stable_digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _join_stratum(*parts: Any) -> str:
    return "|".join(map(str, parts))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _string_key_counter(values: Any) -> dict[str, int]:
    return {
        str(key): value for key, value in sorted(Counter(values).items(), key=lambda x: x[0])
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()
