"""
Lightweight, file-based experiment manifest helpers.

PyYAML is not a project dependency, so manifests are stored as JSON
(`experiment.json`) to avoid adding an extra package.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
MANIFEST_NAME = "experiment.json"


@dataclass
class Experiment:
    """In-memory view of an experiment manifest."""

    slug: str
    purpose: str
    commit: str
    created_at: str
    upstream: str | None
    params: dict[str, Any]
    artifacts: dict[str, str]
    path: Path

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_NAME


def _experiment_dir(slug: str) -> Path:
    return EXPERIMENTS_DIR / slug


def load_experiment(path: Path) -> Experiment:
    """Read and validate ``experiment.json`` in ``path``."""
    manifest = path / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest}")

    data = json.loads(manifest.read_text(encoding="utf-8"))

    return Experiment(
        slug=data.get("slug", path.name),
        purpose=data.get("purpose", ""),
        commit=data.get("commit", ""),
        created_at=data.get("created_at", ""),
        upstream=data.get("upstream"),
        params=data.get("params", {}),
        artifacts=data.get("artifacts", {}),
        path=path,
    )


def require_experiment_dir(path: Path) -> Experiment:
    """Raise if ``path`` is not a directory containing ``experiment.json``."""
    if not path.is_dir():
        raise NotADirectoryError(f"Experiment directory does not exist: {path}")
    return load_experiment(path)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` atomically via a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_filename = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=".tmp",
        dir=path.parent,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary_filename, path)

    except Exception:
        try:
            os.unlink(temporary_filename)
        except FileNotFoundError:
            pass
        raise


def log_artifact(slug: str, key: str, filename: str) -> None:
    """Record ``filename`` under ``artifacts.<key>`` of the experiment manifest."""
    exp_dir = _experiment_dir(slug)
    manifest = exp_dir / MANIFEST_NAME

    if not manifest.is_file():
        raise FileNotFoundError(f"Cannot log artifact; manifest missing: {manifest}")

    data = json.loads(manifest.read_text(encoding="utf-8"))
    data.setdefault("artifacts", {})[key] = filename

    _atomic_write_json(manifest, data)


def list_experiments() -> list[Experiment]:
    """Return every valid experiment under ``experiments/``."""
    if not EXPERIMENTS_DIR.is_dir():
        return []

    experiments: list[Experiment] = []
    for path in sorted(EXPERIMENTS_DIR.iterdir()):
        if not path.is_dir():
            continue

        try:
            experiments.append(load_experiment(path))
        except (OSError, json.JSONDecodeError, KeyError):
            continue

    return experiments


def current_git_sha() -> str:
    """Return the current git commit hash, or ``"unknown"`` if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"
