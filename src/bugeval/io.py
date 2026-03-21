"""YAML I/O and checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from bugeval.models import TestCase
from bugeval.result_models import ToolResult
from bugeval.score_models import CaseScore


def save_case(case: TestCase, path: Path) -> None:
    """Serialize a TestCase to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(case.model_dump(mode="json"), f, sort_keys=False)


def load_case(path: Path) -> TestCase:
    """Load a TestCase from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return TestCase(**data)


def load_cases(cases_dir: Path, *, include_excluded: bool = False) -> list[TestCase]:
    """Load TestCase YAMLs from a directory tree.

    By default, cases with ``excluded=True`` are filtered out.
    Pass ``include_excluded=True`` for curation/dashboard views.
    """
    cases: list[TestCase] = []
    for p in sorted(cases_dir.rglob("*.yaml")):
        case = load_case(p)
        if not include_excluded and case.excluded:
            continue
        cases.append(case)
    return cases


def save_result(result: ToolResult, path: Path) -> None:
    """Serialize a ToolResult to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(result.model_dump(mode="json"), f, sort_keys=False)


def load_result(path: Path) -> ToolResult:
    """Load a ToolResult from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return ToolResult(**data)


def save_score(score: CaseScore, path: Path) -> None:
    """Serialize a CaseScore to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(score.model_dump(mode="json"), f, sort_keys=False)


def load_score(path: Path) -> CaseScore:
    """Load a CaseScore from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return CaseScore(**data)


def load_checkpoint(path: Path) -> set[str]:
    """Load a checkpoint file (JSON set of completed IDs)."""
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return set(data)


def save_checkpoint(done: set[str], path: Path) -> None:
    """Save a checkpoint file (JSON list of completed IDs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(done), indent=2))


def write_run_metadata(
    run_dir: Path,
    tool: str,
    context_level: str,
    cases_dir: Path,
    *,
    model: str = "",
    thinking_budget: int = 0,
    timeout: int = 300,
) -> None:
    """Write run_metadata.json for reproducibility."""
    import hashlib
    import subprocess
    import sys
    from datetime import UTC, datetime

    meta: dict[str, Any] = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "tool": tool,
        "context_level": context_level,
        "model": model,
        "thinking_budget": thinking_budget,
        "timeout": timeout,
        "cases_dir": str(cases_dir),
    }
    # Git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            meta["code_commit"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Config hash
    config_path = Path("config/config.yaml")
    if config_path.exists():
        meta["config_sha256"] = hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest()
    # Case count
    if cases_dir.exists():
        meta["total_cases"] = sum(1 for _ in cases_dir.rglob("*.yaml"))
    meta["python_version"] = sys.version.split()[0]
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2)
    )
