from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from pipeline.step4.data import step4_paths
from pipeline.step4.metrics import binary_metrics, labels_from_names
from pipeline.step4.train import compose_text, train_roberta


def test_binary_metrics_include_both_classes_and_ranking_metrics() -> None:
    values = binary_metrics(
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        positive_scores=[0.1, 0.4, 0.6, 0.9],
    )

    assert values["accuracy"] == 1
    assert values["macro_f1"] == 1
    assert values["roc_auc"] == 1
    assert values["average_precision"] == 1
    assert values["per_class"]["OTHER"]["recall"] == 1
    assert values["per_class"]["DATA_SECURITY"]["recall"] == 1


def test_label_mapping_is_deterministic() -> None:
    assert labels_from_names(["OTHER", "DATA_SECURITY"]) == [0, 1]


def test_paper_default_text_can_use_only_the_abstract() -> None:
    row = {"title": "名称", "abstract": "摘要内容", "claim": "主权项"}

    assert compose_text(row, ("abstract",)) == "摘要：摘要内容"


def test_paper_training_api_has_no_robustness_model_controls() -> None:
    parameters = inspect.signature(train_roberta).parameters

    assert "class_weighting" not in parameters
    assert "early_stopping_patience" not in parameters


def test_missing_training_dependencies_do_not_create_output_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = step4_paths(tmp_path / "step4")
    for path in (
        paths.classifier_train,
        paths.classifier_validation,
        paths.classifier_test,
        paths.manifest,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    def missing() -> dict[str, object]:
        raise RuntimeError("missing dependencies")

    monkeypatch.setattr("pipeline.step4.train._load_training_dependencies", missing)
    with pytest.raises(RuntimeError, match="missing dependencies"):
        train_roberta(paths)

    assert not paths.model.exists()
    assert not paths.state.exists()
    assert not paths.reports.exists()
