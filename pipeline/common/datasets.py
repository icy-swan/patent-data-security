"""Dataset discovery and collision-safe identifiers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def dataset_id(path: str | Path) -> str:
    """Return a stable dataset token, preferring a four-digit year."""

    source = Path(path)
    match = YEAR_PATTERN.search(source.stem)
    if match:
        return match.group(1)
    token = "-".join(part for part in re.split(r"[^\w]+", source.stem) if part)
    return token.casefold() or "dataset"


def discover_files(
    explicit_paths: Iterable[str | Path] | None,
    directory: str | Path,
    pattern: str,
) -> tuple[Path, ...]:
    """Resolve explicit inputs or discover matching files in a directory."""

    if explicit_paths:
        paths = tuple(Path(path).expanduser().resolve() for path in explicit_paths)
    else:
        root = Path(directory).expanduser().resolve()
        paths = tuple(path.resolve() for path in root.glob(pattern) if path.is_file())
    paths = tuple(sorted(paths, key=lambda path: (dataset_id(path), str(path))))
    if not paths:
        raise FileNotFoundError(f"No input files found for {pattern!r} in {directory}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files do not exist: {', '.join(map(str, missing))}")
    _validate_unique_dataset_ids(paths)
    return paths


def _validate_unique_dataset_ids(paths: tuple[Path, ...]) -> None:
    by_id: dict[str, list[Path]] = {}
    for path in paths:
        by_id.setdefault(dataset_id(path), []).append(path)
    collisions = {key: values for key, values in by_id.items() if len(values) > 1}
    if collisions:
        details = "; ".join(
            f"{key}: {', '.join(path.name for path in values)}"
            for key, values in collisions.items()
        )
        raise ValueError(f"Dataset IDs collide: {details}")

