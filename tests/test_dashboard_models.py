"""Tests for dashboard_models persistence layer."""

from __future__ import annotations

from pathlib import Path

from bugeval.dashboard_models import (
    Experiment,
    ExperimentStore,
    GoldenEntry,
    HumanScore,
    RunNote,
    add_run_note,
    load_experiments,
    load_golden_set,
    load_human_score,
    load_run_notes,
    save_experiments,
    save_golden_set,
    save_human_score,
    save_run_notes,
    set_golden_status,
    slugify,
)

# ---------------------------------------------------------------------------
# RunNote
# ---------------------------------------------------------------------------


class TestRunNote:
    def test_create(self) -> None:
        note = RunNote(timestamp="2025-01-01T00:00:00Z", text="hello")
        assert note.timestamp == "2025-01-01T00:00:00Z"
        assert note.text == "hello"

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        notes = [
            RunNote(timestamp="2025-01-01T00:00:00Z", text="first"),
            RunNote(timestamp="2025-01-02T00:00:00Z", text="second"),
        ]
        save_run_notes(tmp_path, notes)
        loaded = load_run_notes(tmp_path)
        assert len(loaded) == 2
        assert loaded[0].text == "first"
        assert loaded[1].text == "second"

    def test_load_empty_dir(self, tmp_path: Path) -> None:
        loaded = load_run_notes(tmp_path)
        assert loaded == []

    def test_add_run_note(self, tmp_path: Path) -> None:
        note = add_run_note(tmp_path, "note one")
        assert note.text == "note one"
        assert note.timestamp  # non-empty

        add_run_note(tmp_path, "note two")
        loaded = load_run_notes(tmp_path)
        assert len(loaded) == 2
        assert loaded[0].text == "note one"
        assert loaded[1].text == "note two"


# ---------------------------------------------------------------------------
# GoldenEntry
# ---------------------------------------------------------------------------


class TestGoldenEntry:
    def test_defaults(self) -> None:
        entry = GoldenEntry(case_id="leo-001")
        assert entry.status == "unreviewed"
        assert entry.reviewer == ""
        assert entry.timestamp == ""
        assert entry.notes == ""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        entries = {
            "leo-001": GoldenEntry(
                case_id="leo-001", status="confirmed", reviewer="alice"
            ),
        }
        save_golden_set(tmp_path, entries)
        loaded = load_golden_set(tmp_path)
        assert "leo-001" in loaded
        assert loaded["leo-001"].status == "confirmed"
        assert loaded["leo-001"].reviewer == "alice"

    def test_load_empty_dir(self, tmp_path: Path) -> None:
        loaded = load_golden_set(tmp_path)
        assert loaded == {}

    def test_set_golden_status(self, tmp_path: Path) -> None:
        entry = set_golden_status(
            tmp_path, "leo-001", "confirmed", reviewer="bob", notes="looks good"
        )
        assert entry.status == "confirmed"
        assert entry.reviewer == "bob"
        assert entry.timestamp  # non-empty

        # Verify persisted
        loaded = load_golden_set(tmp_path)
        assert loaded["leo-001"].status == "confirmed"

    def test_set_golden_status_overwrites(self, tmp_path: Path) -> None:
        set_golden_status(tmp_path, "leo-001", "confirmed")
        set_golden_status(tmp_path, "leo-001", "disputed", reviewer="eve")
        loaded = load_golden_set(tmp_path)
        assert loaded["leo-001"].status == "disputed"
        assert loaded["leo-001"].reviewer == "eve"


# ---------------------------------------------------------------------------
# HumanScore
# ---------------------------------------------------------------------------


class TestHumanScore:
    def test_defaults(self) -> None:
        hs = HumanScore(case_id="leo-001", tool="copilot")
        assert hs.detection_score == 0
        assert hs.review_quality == 0
        assert hs.notes == ""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        hs = HumanScore(
            case_id="leo-001",
            tool="copilot",
            detection_score=3,
            review_quality=4,
            notes="great",
            timestamp="2025-01-01T00:00:00Z",
        )
        save_human_score(tmp_path, hs)
        loaded = load_human_score(tmp_path, "leo-001", "copilot")
        assert loaded is not None
        assert loaded.detection_score == 3
        assert loaded.review_quality == 4
        assert loaded.notes == "great"

    def test_load_missing(self, tmp_path: Path) -> None:
        assert load_human_score(tmp_path, "no-case", "no-tool") is None

    def test_load_all_multiple(self, tmp_path: Path) -> None:
        save_human_score(
            tmp_path,
            HumanScore(case_id="leo-001", tool="copilot", detection_score=2),
        )
        save_human_score(
            tmp_path,
            HumanScore(case_id="leo-002", tool="agent", detection_score=3),
        )
        # Verify both load independently
        s1 = load_human_score(tmp_path, "leo-001", "copilot")
        s2 = load_human_score(tmp_path, "leo-002", "agent")
        assert s1 is not None and s1.detection_score == 2
        assert s2 is not None and s2.detection_score == 3


# ---------------------------------------------------------------------------
# Experiment + ExperimentStore
# ---------------------------------------------------------------------------


class TestExperiment:
    def test_create(self) -> None:
        exp = Experiment(id="baseline-v1", name="Baseline v1")
        assert exp.status == "active"
        assert exp.runs == []
        assert exp.notes == ""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        store = ExperimentStore(
            experiments=[
                Experiment(
                    id="exp-1",
                    name="Experiment 1",
                    runs=["run-2025-01-01"],
                    notes="first run",
                    created="2025-01-01",
                ),
            ]
        )
        save_experiments(tmp_path, store)
        loaded = load_experiments(tmp_path)
        assert len(loaded.experiments) == 1
        assert loaded.experiments[0].id == "exp-1"
        assert loaded.experiments[0].runs == ["run-2025-01-01"]

    def test_load_empty_dir(self, tmp_path: Path) -> None:
        loaded = load_experiments(tmp_path)
        assert loaded.experiments == []

    def test_load_empty_file(self, tmp_path: Path) -> None:
        (tmp_path / "experiments.yaml").write_text("")
        loaded = load_experiments(tmp_path)
        assert loaded.experiments == []


class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("Hello World") == "hello-world"

    def test_special_chars(self) -> None:
        assert slugify("Test! @#$ Run") == "test-run"

    def test_multiple_spaces(self) -> None:
        assert slugify("  lots   of   spaces  ") == "lots-of-spaces"

    def test_already_slug(self) -> None:
        assert slugify("already-a-slug") == "already-a-slug"

    def test_empty_after_strip(self) -> None:
        assert slugify("!!!") == ""
