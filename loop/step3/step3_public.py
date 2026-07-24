"""公开版 Step 3：从冻结的 Step 2 任务池确定性抽取人工复核样本。

本模块只负责分层抽样和审计输出，不包含任何自动模型复核、模型调用或评测实现。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

METHOD_VERSION = "public-step3-50000-to-10000-dual-cohort-v1"
COHORTS = ("positive_priority", "negative_priority")
SAMPLING_GROUPS = ("positive", "hard_negative", "easy_negative")
LABELS = {"DATA_SECURITY", "OTHER"}
OUTPUT_FIELDS = (
    "sample_id",
    "dataset_id",
    "patent_id",
    "application_date",
    "application_year",
    "title",
    "abstract",
    "claim",
    "ipc",
    "sample_cohort",
    "sampling_group",
    "step1_route",
    "step2_label",
    "step2_reason",
    "step2_evidence",
    "sampling_stratum",
    "cohort_population",
    "cohort_sample_size",
    "cohort_inclusion_probability",
    "step3_inclusion_probability",
    "step3_sample_weight",
    "combined_step2_inclusion_probability",
    "combined_inclusion_probability",
    "combined_sample_weight",
    "human_review_label",
    "human_reason",
)


@dataclass(frozen=True)
class CohortSpec:
    name: str
    seed: str
    group_targets: dict[str, int]

    @property
    def target_size(self) -> int:
        return sum(self.group_targets.values())


@dataclass(frozen=True)
class PublicStep3Paths:
    root: Path
    positive_review: Path
    negative_review: Path
    combined_sample: Path
    manifest: Path


def public_step3_paths(output_dir: str | Path) -> PublicStep3Paths:
    root = Path(output_dir).resolve()
    return PublicStep3Paths(
        root=root,
        positive_review=root / "need_manual_review_positive.csv",
        negative_review=root / "need_manual_review_negative.csv",
        combined_sample=root / "review_sample.csv",
        manifest=root / "manifest.json",
    )


def prepare_public_step3(
    config_path: str | Path,
    *,
    step2_override: str | Path | None = None,
    output_override: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[PublicStep3Paths, dict[str, Any]]:
    """从完整 Step 2 结果抽取两个互斥队列，不执行任何复核。"""

    config_file = Path(config_path).resolve()
    config = read_json(config_file)
    step2_value = str(step2_override or config.get("step2_result", "")).strip()
    if not step2_value:
        raise ValueError(
            "step2_result 在公开模板中故意留空；请传入 --step2-result "
            "或填写本地配置"
        )
    step2_path = resolve_from(config_file.parent, step2_value)
    if not step2_path.is_file():
        raise FileNotFoundError(f"Step 2 结果不存在：{step2_path}")

    output_value = str(output_override or config.get("output_dir", "output"))
    paths = public_step3_paths(resolve_from(config_file.parent, output_value))
    encoding = str(config.get("encoding", "utf-8-sig"))
    expected_population = int(config.get("expected_population_size", 50_000))
    if expected_population < 1:
        raise ValueError("expected_population_size 必须为正整数")
    cohort_specs = parse_cohort_specs(config.get("cohorts"))

    records = read_step2_population(step2_path, encoding=encoding)
    if len(records) != expected_population:
        raise ValueError(
            "Step 3 必须使用完整、冻结的 Step 2 样本框："
            f"预期={expected_population}，实际={len(records)}"
        )
    patent_ids = [row["patent_id"] for row in records]
    if len(patent_ids) != len(set(patent_ids)):
        raise ValueError("Step 2 结果中存在重复 patent_id")

    remaining = list(records)
    selected_by_cohort: dict[str, list[dict[str, Any]]] = {}
    cohort_reports: list[dict[str, Any]] = []
    for spec in cohort_specs:
        selected, strata = select_cohort(remaining, spec)
        selected_ids = {row["patent_id"] for row in selected}
        remaining = [
            row for row in remaining if row["patent_id"] not in selected_ids
        ]
        selected_by_cohort[spec.name] = selected
        cohort_reports.append(
            {
                "name": spec.name,
                "seed": spec.seed,
                "target_size": spec.target_size,
                "group_targets": dict(sorted(spec.group_targets.items())),
                "sample_by_group": dict(
                    sorted(Counter(row["sampling_group"] for row in selected).items())
                ),
                "strata": strata,
            }
        )

    combined = [
        row
        for cohort in COHORTS
        for row in selected_by_cohort[cohort]
    ]
    if len({row["patent_id"] for row in combined}) != len(combined):
        raise AssertionError("两个 Step 3 队列出现重复专利")
    add_final_probabilities(records, combined)

    paths.root.mkdir(parents=True, exist_ok=True)
    managed = (
        paths.positive_review,
        paths.negative_review,
        paths.combined_sample,
        paths.manifest,
    )
    if not overwrite and any(path.exists() for path in managed):
        raise FileExistsError("Step 3 输出已存在；如需替换请传入 --overwrite")
    if overwrite:
        for path in managed:
            path.unlink(missing_ok=True)

    write_csv(
        paths.positive_review,
        sorted(selected_by_cohort["positive_priority"], key=lambda row: row["sample_id"]),
    )
    write_csv(
        paths.negative_review,
        sorted(selected_by_cohort["negative_priority"], key=lambda row: row["sample_id"]),
    )
    write_csv(
        paths.combined_sample,
        sorted(combined, key=lambda row: (COHORTS.index(row["sample_cohort"]), row["sample_id"])),
    )

    population_by_group = Counter(row["sampling_group"] for row in records)
    sample_by_group = Counter(row["sampling_group"] for row in combined)
    population_strata = Counter(
        (row["application_year"], row["sampling_group"]) for row in records
    )
    sample_strata = Counter(
        (row["application_year"], row["sampling_group"]) for row in combined
    )
    final_strata = [
        {
            "application_year": year,
            "sampling_group": group,
            "population": population,
            "sample": sample_strata[(year, group)],
            "inclusion_probability": sample_strata[(year, group)] / population,
        }
        for (year, group), population in sorted(population_strata.items())
        if sample_strata[(year, group)]
    ]
    manifest = {
        "public_method_version": METHOD_VERSION,
        "input": {
            "source_address": "",
            "file_name": step2_path.name,
            "sha256": sha256_file(step2_path),
            "records": len(records),
        },
        "expected_population_size": expected_population,
        "target_size": len(combined),
        "definitions": {
            "positive": "step2_label=DATA_SECURITY",
            "hard_negative": "step1_route=S and step2_label=OTHER",
            "easy_negative": "step1_route=E and step2_label=OTHER",
        },
        "year_allocation": (
            "equal_within_cohort_and_sampling_group_with_capacity_redistribution"
        ),
        "unknown_year_policy": "treat_as_separate_UNKNOWN_stratum",
        "population_by_group": dict(sorted(population_by_group.items())),
        "sample_by_group": dict(sorted(sample_by_group.items())),
        "cohorts": cohort_reports,
        "final_strata": final_strata,
        "outputs": {
            "positive_review": file_record(paths.positive_review),
            "negative_review": file_record(paths.negative_review),
            "combined_sample": file_record(paths.combined_sample),
        },
        "review_policy": {
            "human_fields_initially_blank": [
                "human_review_label",
                "human_reason",
            ],
            "automated_review_included": False,
        },
        "model_requests_executed": 0,
        "created_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(paths.manifest, manifest)
    return paths, manifest


def parse_cohort_specs(value: Any) -> tuple[CohortSpec, ...]:
    if not isinstance(value, Mapping):
        raise ValueError("cohorts 必须是对象")
    specs = []
    for name in COHORTS:
        raw = value.get(name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"cohorts.{name} 必须是对象")
        seed = str(raw.get("seed", "")).strip()
        if not seed:
            raise ValueError(f"cohorts.{name}.seed 不能为空")
        targets_raw = raw.get("group_targets")
        if not isinstance(targets_raw, Mapping):
            raise ValueError(f"cohorts.{name}.group_targets 必须是对象")
        unknown = set(targets_raw) - set(SAMPLING_GROUPS)
        if unknown:
            raise ValueError(f"cohorts.{name} 包含未知抽样组：{sorted(unknown)}")
        targets = {
            group: int(targets_raw.get(group, 0))
            for group in SAMPLING_GROUPS
        }
        if any(target < 0 for target in targets.values()):
            raise ValueError(f"cohorts.{name} 的组目标不能为负数")
        if sum(targets.values()) < 1:
            raise ValueError(f"cohorts.{name} 的总目标必须为正整数")
        specs.append(CohortSpec(name=name, seed=seed, group_targets=targets))
    return tuple(specs)


def read_step2_population(path: Path, *, encoding: str) -> list[dict[str, Any]]:
    required = {
        "dataset_id",
        "patent_id",
        "title",
        "abstract",
        "claim",
        "ipc",
        "step1_route",
        "step2_label",
        "step2_reason",
        "step2_evidence",
    }
    records = []
    with path.open(encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or ())
        if missing := required - fields:
            raise ValueError(f"{path.name} 缺少字段：{sorted(missing)}")
        for row_number, source in enumerate(reader, start=2):
            patent_id = str(source.get("patent_id", "")).strip()
            if not patent_id:
                raise ValueError(f"第 {row_number} 行 patent_id 为空")
            route = str(source.get("step1_route", "")).strip().upper()
            label = str(source.get("step2_label", "")).strip().upper()
            group = sampling_group(route, label)
            application_date = str(source.get("application_date", "") or "").strip()
            application_year = normalize_application_year(
                str(source.get("application_year", "") or ""),
                application_date=application_date,
                fallback=str(source.get("dataset_id", "") or ""),
            )
            evidence = normalize_evidence(
                source.get("step2_evidence", ""),
                row_number=row_number,
            )
            prior_probability = optional_probability(
                source.get("combined_step2_inclusion_probability", ""),
                field="combined_step2_inclusion_probability",
                row_number=row_number,
            )
            dataset_id = str(source.get("dataset_id", "") or "").strip()
            records.append(
                {
                    "sample_id": sample_id(dataset_id, patent_id),
                    "dataset_id": dataset_id,
                    "patent_id": patent_id,
                    "application_date": application_date,
                    "application_year": application_year,
                    "title": str(source.get("title", "") or ""),
                    "abstract": str(source.get("abstract", "") or ""),
                    "claim": str(source.get("claim", "") or ""),
                    "ipc": str(source.get("ipc", "") or ""),
                    "sampling_group": group,
                    "step1_route": route,
                    "step2_label": label,
                    "step2_reason": str(source.get("step2_reason", "") or ""),
                    "step2_evidence": evidence,
                    "combined_step2_inclusion_probability": prior_probability,
                    "human_review_label": "",
                    "human_reason": "",
                }
            )
    return records


def sampling_group(route: str, label: str) -> str:
    if route not in {"S", "E"}:
        raise ValueError(f"非法 step1_route：{route!r}")
    if label not in LABELS:
        raise ValueError(f"非法 step2_label：{label!r}")
    if label == "DATA_SECURITY":
        return "positive"
    return "hard_negative" if route == "S" else "easy_negative"


def select_cohort(
    remaining: Sequence[Mapping[str, Any]],
    spec: CohortSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for group in SAMPLING_GROUPS:
        target = spec.group_targets[group]
        if target == 0:
            continue
        group_rows = [
            row for row in remaining if row["sampling_group"] == group
        ]
        capacities = Counter(row["application_year"] for row in group_rows)
        quotas = balanced_capacity_allocation(
            capacities,
            target,
            seed=f"{spec.seed}|{group}",
        )
        by_year: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_year[row["application_year"]].append(row)
        for year in sorted(by_year):
            population = by_year[year]
            quota = quotas[year]
            if quota == 0:
                continue
            ranked = sorted(
                population,
                key=lambda row: (
                    stable_score(spec.seed, row["dataset_id"], row["patent_id"]),
                    row["patent_id"],
                ),
            )
            probability = quota / len(population)
            strata.append(
                {
                    "application_year": year,
                    "sampling_group": group,
                    "remaining_population": len(population),
                    "sample": quota,
                    "conditional_inclusion_probability": probability,
                }
            )
            for source in ranked[:quota]:
                row = dict(source)
                row.update(
                    {
                        "sample_cohort": spec.name,
                        "sampling_stratum": (
                            f"cohort={spec.name}|year={year}|sampling_group={group}"
                        ),
                        "cohort_population": len(population),
                        "cohort_sample_size": quota,
                        "cohort_inclusion_probability": probability,
                    }
                )
                selected.append(row)
    if len(selected) != spec.target_size:
        raise AssertionError(
            f"{spec.name} 预期抽取 {spec.target_size} 条，实际 {len(selected)} 条"
        )
    return selected, strata


def balanced_capacity_allocation(
    capacities: Mapping[str, int],
    target: int,
    *,
    seed: str,
) -> dict[str, int]:
    if sum(capacities.values()) < target:
        raise ValueError(
            f"分层目标无法满足：target={target}，available={sum(capacities.values())}"
        )
    quotas = {year: 0 for year in capacities}
    tie_order = {
        year: stable_score(seed, year, "capacity-allocation")
        for year in capacities
    }
    for _ in range(target):
        eligible = [
            year
            for year, capacity in capacities.items()
            if quotas[year] < capacity
        ]
        chosen = min(eligible, key=lambda year: (quotas[year], tie_order[year]))
        quotas[chosen] += 1
    return quotas


def add_final_probabilities(
    population: Sequence[Mapping[str, Any]],
    selected: Sequence[dict[str, Any]],
) -> None:
    population_strata = Counter(
        (row["application_year"], row["sampling_group"]) for row in population
    )
    sample_strata = Counter(
        (row["application_year"], row["sampling_group"]) for row in selected
    )
    for row in selected:
        stratum = (row["application_year"], row["sampling_group"])
        probability = sample_strata[stratum] / population_strata[stratum]
        row["step3_inclusion_probability"] = probability
        row["step3_sample_weight"] = 1 / probability
        prior = row["combined_step2_inclusion_probability"]
        if prior is None:
            row["combined_inclusion_probability"] = None
            row["combined_sample_weight"] = None
        else:
            combined = prior * probability
            row["combined_inclusion_probability"] = combined
            row["combined_sample_weight"] = 1 / combined


def normalize_application_year(
    value: str,
    *,
    application_date: str,
    fallback: str,
) -> str:
    for candidate in (value, application_date, fallback):
        if match := re.search(r"(?<!\d)((?:18|19|20|21)\d{2})(?!\d)", candidate):
            return match.group(1)
    return "UNKNOWN"


def normalize_evidence(value: Any, *, row_number: int) -> str:
    try:
        evidence = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"第 {row_number} 行 step2_evidence 不是合法 JSON") from error
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"第 {row_number} 行 step2_evidence 必须是非空数组")
    return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))


def optional_probability(
    value: Any,
    *,
    field: str,
    row_number: int,
) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"第 {row_number} 行 {field} 不是数值") from error
    if not 0 < probability <= 1:
        raise ValueError(f"第 {row_number} 行 {field} 必须位于 (0, 1]")
    return probability


def sample_id(dataset_id: str, patent_id: str) -> str:
    digest = hashlib.sha256(
        f"{METHOD_VERSION}|{dataset_id}|{patent_id}".encode()
    ).hexdigest()[:24]
    return f"public-step3-{digest}"


def stable_score(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: serialize_output_value(row.get(field))
                    for field in OUTPUT_FIELDS
                }
            )
    os.replace(temporary, path)


def serialize_output_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.12g}"
    if value is None:
        return ""
    return value


def file_record(path: Path) -> dict[str, Any]:
    return {
        "file_name": path.name,
        "sha256": sha256_file(path),
        "records": count_csv_rows(path),
        "fields": list(OUTPUT_FIELDS),
    }


def count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 文件必须是对象：{path}")
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
    parser.add_argument("--step2-result", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths, manifest = prepare_public_step3(
        args.config,
        step2_override=args.step2_result,
        output_override=args.output_dir,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "paths": {
                    key: str(value)
                    for key, value in vars(paths).items()
                },
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
