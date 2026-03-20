"""Pydantic models and persistence for dashboard state."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel


class RunNote(BaseModel):
    timestamp: str
    text: str


class GoldenEntry(BaseModel):
    case_id: str
    status: str = "unreviewed"  # unreviewed | confirmed | disputed
    reviewer: str = ""
    timestamp: str = ""
    notes: str = ""


class HumanScore(BaseModel):
    case_id: str
    tool: str
    detection_score: int = 0  # 0-3
    review_quality: int = 0  # 0-4
    notes: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Run notes persistence — results/run-{date}/.notes.json
# ---------------------------------------------------------------------------


def _notes_path(run_dir: Path) -> Path:
    return run_dir / ".notes.json"


def load_run_notes(run_dir: Path) -> list[RunNote]:
    """Load run notes from JSON sidecar."""
    path = _notes_path(run_dir)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [RunNote(**n) for n in data]


def save_run_notes(run_dir: Path, notes: list[RunNote]) -> None:
    """Persist run notes to JSON sidecar."""
    path = _notes_path(run_dir)
    path.write_text(json.dumps([n.model_dump(mode="json") for n in notes], indent=2))


def add_run_note(run_dir: Path, text: str) -> RunNote:
    """Append a timestamped note to a run."""
    notes = load_run_notes(run_dir)
    note = RunNote(timestamp=datetime.now(UTC).isoformat(), text=text)
    notes.append(note)
    save_run_notes(run_dir, notes)
    return note


# ---------------------------------------------------------------------------
# Golden set persistence — cases/.golden_set.json
# ---------------------------------------------------------------------------


def _golden_path(cases_dir: Path) -> Path:
    return cases_dir / ".golden_set.json"


def load_golden_set(cases_dir: Path) -> dict[str, GoldenEntry]:
    """Load golden set entries from JSON sidecar."""
    path = _golden_path(cases_dir)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {k: GoldenEntry(**v) for k, v in data.items()}


def save_golden_set(
    cases_dir: Path, entries: dict[str, GoldenEntry]
) -> None:
    """Persist golden set entries to JSON sidecar."""
    path = _golden_path(cases_dir)
    path.write_text(
        json.dumps(
            {k: v.model_dump(mode="json") for k, v in entries.items()},
            indent=2,
        )
    )


def set_golden_status(
    cases_dir: Path,
    case_id: str,
    status: str,
    reviewer: str = "",
    notes: str = "",
) -> GoldenEntry:
    """Set the golden-set status for a single case."""
    entries = load_golden_set(cases_dir)
    entry = entries.get(case_id, GoldenEntry(case_id=case_id))
    entry.status = status
    entry.reviewer = reviewer
    entry.notes = notes
    entry.timestamp = datetime.now(UTC).isoformat()
    entries[case_id] = entry
    save_golden_set(cases_dir, entries)
    return entry


# ---------------------------------------------------------------------------
# Human scores persistence
# ---------------------------------------------------------------------------


def _human_scores_dir(run_dir: Path) -> Path:
    return run_dir / "human_scores"


def load_human_score(
    run_dir: Path, case_id: str, tool: str
) -> HumanScore | None:
    """Load a single human score."""
    safe_case = case_id.replace("/", "_")
    safe_tool = tool.replace("/", "_")
    path = _human_scores_dir(run_dir) / f"{safe_case}__{safe_tool}.yaml"
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text())
    if not data:
        return None
    return HumanScore(**data)


def save_human_score(run_dir: Path, score: HumanScore) -> None:
    """Persist a human score to YAML."""
    hs_dir = _human_scores_dir(run_dir)
    hs_dir.mkdir(parents=True, exist_ok=True)
    safe_case = score.case_id.replace("/", "_")
    safe_tool = score.tool.replace("/", "_")
    path = hs_dir / f"{safe_case}__{safe_tool}.yaml"
    path.write_text(
        yaml.safe_dump(score.model_dump(mode="json"), sort_keys=False)
    )


# ---------------------------------------------------------------------------
# Experiment grouping models and persistence
# ---------------------------------------------------------------------------


class Experiment(BaseModel):
    id: str
    name: str
    status: str = "active"  # "active" | "archived"
    runs: list[str] = []  # run directory names
    notes: str = ""
    created: str = ""  # ISO date


class ExperimentStore(BaseModel):
    experiments: list[Experiment] = []


def slugify(name: str) -> str:
    """Convert a name to a URL-safe slug."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def load_experiments(results_dir: Path) -> ExperimentStore:
    """Load experiments from results/experiments.yaml."""
    path = results_dir / "experiments.yaml"
    if not path.exists():
        return ExperimentStore()
    data = yaml.safe_load(path.read_text())
    if not data:
        return ExperimentStore()
    return ExperimentStore(**data)


def save_experiments(
    results_dir: Path, store: ExperimentStore
) -> None:
    """Write experiments to results/experiments.yaml."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "experiments.yaml"
    path.write_text(
        yaml.safe_dump(store.model_dump(mode="json"), sort_keys=False)
    )


def current_date_iso() -> str:
    """Return today's date as ISO string."""
    return date.today().isoformat()
