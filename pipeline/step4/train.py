"""Fine-tune and evaluate the paper-style RoBERTa binary classifier."""

from __future__ import annotations

import csv
import inspect
import json
import math
import os
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pipeline.common.io import atomic_json_write, sha256_file
from pipeline.step4.data import LABEL_TO_ID, Step4Paths
from pipeline.step4.metrics import binary_metrics

DEFAULT_MODEL = "hfl/chinese-roberta-wwm-ext"
ID_TO_LABEL = {identifier: label for label, identifier in LABEL_TO_ID.items()}
TEXT_FIELDS = ("title", "abstract", "claim")


def train_roberta(
    paths: Step4Paths,
    *,
    model_name: str = DEFAULT_MODEL,
    text_fields: Sequence[str] = ("abstract",),
    max_length: int = 512,
    epochs: float = 4,
    learning_rate: float = 2e-5,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    fp16: bool = False,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    resume_from_checkpoint: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Select the best checkpoint by validation accuracy, then evaluate test once."""

    if fp16 and bf16:
        raise ValueError("fp16 and bf16 are mutually exclusive")
    selected_fields = tuple(text_fields)
    invalid_fields = sorted(set(selected_fields) - set(TEXT_FIELDS))
    if not selected_fields or invalid_fields:
        raise ValueError(f"text_fields must be drawn from {TEXT_FIELDS}; invalid={invalid_fields}")
    if max_length < 32:
        raise ValueError("max_length must be at least 32")

    _require_training_inputs(paths)
    dependencies = _load_training_dependencies()
    _prepare_training_output(paths, overwrite=overwrite, resume=resume_from_checkpoint is not None)
    np = dependencies["numpy"]
    Dataset = dependencies["Dataset"]
    AutoModelForSequenceClassification = dependencies["AutoModelForSequenceClassification"]
    AutoTokenizer = dependencies["AutoTokenizer"]
    DataCollatorWithPadding = dependencies["DataCollatorWithPadding"]
    Trainer = dependencies["Trainer"]
    TrainingArguments = dependencies["TrainingArguments"]

    source_records = {
        "train": _read_jsonl(paths.classifier_train),
        "validation": _read_jsonl(paths.classifier_validation),
        "test": _read_jsonl(paths.classifier_test),
    }
    datasets = {
        split: Dataset.from_list(
            [
                {
                    "text": compose_text(row, selected_fields),
                    "label": int(row["label_id"]),
                }
                for row in rows
            ]
        )
        for split, rows in source_records.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = {
        split: dataset.map(tokenize, batched=True, remove_columns=["text"])
        for split, dataset in datasets.items()
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        logits = prediction.predictions
        labels = prediction.label_ids.tolist()
        predicted = np.argmax(logits, axis=-1).tolist()
        probabilities = _positive_probabilities(logits, np=np)
        values = binary_metrics(labels, predicted, positive_scores=probabilities)
        return {"accuracy": float(values["accuracy"])}

    training_kwargs: dict[str, Any] = {
        "output_dir": str(paths.state),
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 25,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "seed": seed,
        "data_seed": seed,
        "report_to": "none",
        "fp16": fp16,
        "bf16": bf16,
        "gradient_checkpointing": gradient_checkpointing,
        "overwrite_output_dir": overwrite,
    }
    arguments_parameters = inspect.signature(TrainingArguments.__init__).parameters
    strategy_name = (
        "eval_strategy" if "eval_strategy" in arguments_parameters else "evaluation_strategy"
    )
    training_kwargs[strategy_name] = "epoch"
    training_arguments = TrainingArguments(**training_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_arguments,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    resume_value = str(resume_from_checkpoint) if resume_from_checkpoint else None
    train_result = trainer.train(resume_from_checkpoint=resume_value)
    trainer.save_model(str(paths.model))
    tokenizer.save_pretrained(str(paths.model))

    outputs = {
        split: trainer.predict(tokenized[split], metric_key_prefix=f"{split}_final")
        for split in ("train", "validation", "test")
    }
    evaluation: dict[str, dict[str, Any]] = {}
    prediction_values: dict[str, tuple[list[int], list[int], list[float]]] = {}
    for split, output in outputs.items():
        labels = output.label_ids.tolist()
        predictions = np.argmax(output.predictions, axis=-1).tolist()
        scores = _positive_probabilities(output.predictions, np=np)
        evaluation[split] = binary_metrics(labels, predictions, positive_scores=scores)
        prediction_values[split] = (labels, predictions, scores)

    paths.reports.mkdir(parents=True, exist_ok=True)
    validation_labels, validation_predictions, validation_scores = prediction_values["validation"]
    _write_predictions(
        paths.reports / "validation_predictions.csv",
        source_records["validation"],
        validation_labels,
        validation_predictions,
        validation_scores,
    )
    test_labels, test_predictions, test_scores = prediction_values["test"]
    _write_predictions(
        paths.reports / "test_predictions.csv",
        source_records["test"],
        test_labels,
        test_predictions,
        test_scores,
    )
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "method": "supervised_roberta_sequence_classification",
        "paper_alignment": {
            "default_input": "patent_abstract",
            "loss": "unweighted_cross_entropy",
            "validation_selection_metric": "accuracy",
            "prediction_rule": "softmax_argmax",
            "validation_selects_best_checkpoint": True,
            "test_used_after_selection": True,
        },
        "base_model": model_name,
        "text_fields": list(selected_fields),
        "label_to_id": LABEL_TO_ID,
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "max_length": max_length,
            "loss": "unweighted_cross_entropy",
            "seed": seed,
            "fp16": fp16,
            "bf16": bf16,
            "gradient_checkpointing": gradient_checkpointing,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_metric": trainer.state.best_metric,
            "train_metrics": _json_safe(train_result.metrics),
        },
        "primary_metric": "accuracy",
        "metrics_by_split": evaluation,
        "implementation_parameters_not_reported_by_paper": [
            "base_model_checkpoint",
            "epochs",
            "learning_rate",
            "batch_sizes",
            "weight_decay",
            "warmup_ratio",
            "random_seed",
        ],
        "input_manifest": str(paths.manifest),
        "input_manifest_sha256": sha256_file(paths.manifest),
        "model_path": str(paths.model),
        "package_versions": _package_versions(
            ("accelerate", "datasets", "numpy", "torch", "transformers")
        ),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    atomic_json_write(paths.reports / "metrics.json", report)
    return report


def compose_text(row: dict[str, Any], fields: Sequence[str]) -> str:
    labels = {"title": "专利名称", "abstract": "摘要", "claim": "主权项"}
    parts = [f"{labels[field]}：{str(row[field]).strip()}" for field in fields]
    if any(not part.split("：", 1)[1] for part in parts):
        raise ValueError("Selected classifier text field is empty")
    return "\n".join(parts)


def _positive_probabilities(logits: Any, *, np: Any) -> list[float]:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    probabilities = exponent / np.sum(exponent, axis=-1, keepdims=True)
    return probabilities[:, LABEL_TO_ID["DATA_SECURITY"]].tolist()


def _require_training_inputs(paths: Step4Paths) -> None:
    missing = [
        path
        for path in (
            paths.classifier_train,
            paths.classifier_validation,
            paths.classifier_test,
            paths.manifest,
        )
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Run Step 4 prepare first; missing: {missing}")


def _prepare_training_output(paths: Step4Paths, *, overwrite: bool, resume: bool) -> None:
    if overwrite and resume:
        raise ValueError("overwrite and resume_from_checkpoint cannot be used together")
    existing = [path for path in (paths.model, paths.state, paths.reports) if path.exists()]
    if existing and not overwrite and not resume:
        raise FileExistsError(f"RoBERTa outputs already exist: {existing}")
    if overwrite:
        for path in (paths.model, paths.state, paths.reports):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    paths.model.mkdir(parents=True, exist_ok=True)
    paths.state.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")
    return rows


def _write_predictions(
    path: Path,
    records: list[dict[str, Any]],
    labels: list[int],
    predictions: list[int],
    scores: list[float],
) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    fields = (
        "sample_id",
        "patent_id",
        "true_label",
        "predicted_label",
        "data_security_probability",
        "correct",
    )
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record, label, predicted, score in zip(
            records, labels, predictions, scores, strict=True
        ):
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "patent_id": record["patent_id"],
                    "true_label": ID_TO_LABEL[label],
                    "predicted_label": ID_TO_LABEL[predicted],
                    "data_security_probability": f"{score:.12f}",
                    "correct": label == predicted,
                }
            )
    os.replace(temporary, path)


def _load_training_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install Step 4 training dependencies with: pip install -e '.[step4]'"
        ) from exc
    return {
        "numpy": np,
        "torch": torch,
        "Dataset": Dataset,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def _package_versions(packages: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
