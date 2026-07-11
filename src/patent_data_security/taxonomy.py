"""Load the versioned DOCS and IPC routing taxonomies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY_DIR = PROJECT_ROOT / "config" / "taxonomy"


@dataclass(frozen=True)
class TaxonomyBundle:
    docs: dict[str, Any]
    ipc: dict[str, Any]

    @property
    def version(self) -> str:
        return f"docs-{self.docs['taxonomy_version']}__ipc-{self.ipc['taxonomy_version']}"


def load_taxonomies(directory: str | Path = DEFAULT_TAXONOMY_DIR) -> TaxonomyBundle:
    """Load both taxonomies from a directory."""

    root = Path(directory)
    return TaxonomyBundle(
        docs=_read_json(root / "docs_taxonomy.json"),
        ipc=_read_json(root / "ipc_taxonomy.json"),
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)
