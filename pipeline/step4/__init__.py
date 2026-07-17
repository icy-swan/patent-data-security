"""Step 4 dataset preparation and RoBERTa classification."""

from __future__ import annotations

from typing import Any

__all__ = ["Step4Paths", "prepare_datasets", "step4_paths"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from pipeline.step4.data import Step4Paths, prepare_datasets, step4_paths

        return {
            "Step4Paths": Step4Paths,
            "prepare_datasets": prepare_datasets,
            "step4_paths": step4_paths,
        }[name]
    raise AttributeError(name)
