"""Tests for YAML I/O round-trips."""

import json
from pathlib import Path

import pytest

from bugeval.io import (
    load_all_cases,
    load_candidates,
    load_case,
    save_candidates,
    save_case,
    write_run_metadata,
)
from bugeval.models import (
    Candidate,
    CaseStats,
    Category,
    Difficulty,
    ExpectedFinding,
    PRSize,
    Severity,
    TestCase,
)


def make_test_case(id_: str = "test-001") -> TestCase:
    return TestCase(
        id=id_,
        repo="provable-org/aleo-lang",
        base_commit="abc123def456abc123def456abc123def456abc1",
        head_commit="def456abc123def456abc123def456abc123def4",
        fix_commit="ghi789ghi789ghi789ghi789ghi789ghi789ghi7",
        category=Category.logic,
        difficulty=Difficulty.medium,
        severity=Severity.high,
        language="rust",
        pr_size=PRSize.small,
        description="A logic bug in the type checker",
        expected_findings=[ExpectedFinding(file="src/main.rs", line=42, summary="Off-by-one")],
        stats=None,
    )


def make_candidate(pr_number: int = 1) -> Candidate:
    return Candidate(
        repo="provable-org/aleo-lang",
        pr_number=pr_number,
        fix_commit="abc123def456abc123def456abc123def456abc1",
        confidence=0.7,
        signals=["has_bug_label"],
        title="Fix bug",
        body="Fixes #1",
        labels=["bug"],
        files_changed=["src/main.rs"],
        diff_stats=CaseStats(lines_added=5, lines_deleted=3, files_changed=1, hunks=2),
        expected_findings=[],
        language="rust",
        pr_size=PRSize.tiny,
    )


class TestSaveLoadCase:
    def test_round_trip(self, tmp_path: Path) -> None:
        case = make_test_case()
        path = tmp_path / "test-001.yaml"
        save_case(case, path)
        loaded = load_case(path)
        assert loaded == case

    def test_saves_yaml_file(self, tmp_path: Path) -> None:
        case = make_test_case()
        path = tmp_path / "case.yaml"
        save_case(case, path)
        assert path.exists()

    def test_stats_round_trip(self, tmp_path: Path) -> None:
        case = make_test_case()
        case = case.model_copy(
            update={"stats": CaseStats(lines_added=10, lines_deleted=5, files_changed=2, hunks=3)}
        )
        path = tmp_path / "case-stats.yaml"
        save_case(case, path)
        loaded = load_case(path)
        assert loaded.stats is not None
        assert loaded.stats.lines_added == 10

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_case(tmp_path / "nonexistent.yaml")


class TestSaveLoadCandidates:
    def test_round_trip(self, tmp_path: Path) -> None:
        candidates = [make_candidate(1), make_candidate(2)]
        path = tmp_path / "candidates.yaml"
        save_candidates(candidates, path)
        loaded = load_candidates(path)
        assert len(loaded) == 2
        assert loaded[0].pr_number == 1
        assert loaded[1].pr_number == 2

    def test_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        save_candidates([], path)
        loaded = load_candidates(path)
        assert loaded == []

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_candidates(tmp_path / "nonexistent.yaml")


class TestLoadAllCases:
    def test_loads_multiple_files(self, tmp_path: Path) -> None:
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        save_case(make_test_case("a-001"), cases_dir / "a-001.yaml")
        save_case(make_test_case("a-002"), cases_dir / "a-002.yaml")
        loaded = load_all_cases(cases_dir)
        assert len(loaded) == 2
        ids = {c.id for c in loaded}
        assert ids == {"a-001", "a-002"}

    def test_empty_dir(self, tmp_path: Path) -> None:
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        loaded = load_all_cases(cases_dir)
        assert loaded == []


class TestWriteRunMetadata:
    def test_creates_metadata_file(self, tmp_path: Path) -> None:
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        write_run_metadata(tmp_path, ["tool-a"], "diff-only", cases_dir)
        assert (tmp_path / "run_metadata.json").exists()

    def test_contains_required_fields(self, tmp_path: Path) -> None:
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        write_run_metadata(tmp_path, ["tool-a", "tool-b"], "diff+repo", cases_dir, limit=5)
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert data["tools"] == ["tool-a", "tool-b"]
        assert data["context_level"] == "diff+repo"
        assert data["cases_dir"] == str(cases_dir)
        assert data["limit"] == 5
        assert "git_sha" in data
        assert "created_at" in data

    def test_dataset_commit_is_hex_or_empty(self, tmp_path: Path) -> None:
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        write_run_metadata(tmp_path, ["tool-a"], "diff-only", cases_dir)
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        dc = data["dataset_commit"]
        assert dc == "" or (len(dc) == 40 and all(c in "0123456789abcdef" for c in dc))

    def test_total_cases_matches_case_count(self, tmp_path: Path) -> None:
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        save_case(make_test_case("case-001"), cases_dir / "case-001.yaml")
        save_case(make_test_case("case-002"), cases_dir / "case-002.yaml")
        write_run_metadata(tmp_path, ["tool-a"], "diff-only", cases_dir)
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert data["total_cases"] == 2

    def test_patches_dir_stored(self, tmp_path: Path) -> None:
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        patches_dir = tmp_path / "patches"
        write_run_metadata(tmp_path, ["tool-a"], "diff-only", cases_dir, patches_dir=patches_dir)
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert data["patches_dir"] == str(patches_dir)

    def test_limit_zero_by_default(self, tmp_path: Path) -> None:
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        write_run_metadata(tmp_path, ["tool-a"], "diff-only", cases_dir)
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert data["limit"] == 0

    def test_context_level_specific_prompt_resolved(self, tmp_path: Path) -> None:
        """agent_prompt_hash should reflect the context-level-specific file, not the generic one."""
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "agent_prompt.md").write_text("generic prompt")
        (config_dir / "agent_prompt_diff+repo.md").write_text("diff+repo prompt")

        orig = Path.cwd()
        import os
        os.chdir(tmp_path)
        try:
            write_run_metadata(tmp_path, ["tool-a"], "diff+repo", cases_dir)
        finally:
            os.chdir(orig)

        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert data["agent_prompt_file"].endswith("agent_prompt_diff+repo.md")
        assert data["agent_prompt_hash"].startswith("sha256:")
        # Snapshot should contain the context-specific content
        snapshot = (tmp_path / "agent_prompt_snapshot.md").read_text()
        assert snapshot == "diff+repo prompt"

    def test_generic_prompt_fallback_when_no_context_specific(self, tmp_path: Path) -> None:
        """Falls back to agent_prompt.md when no context-level-specific file exists."""
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "agent_prompt.md").write_text("generic prompt")

        import os
        orig = Path.cwd()
        os.chdir(tmp_path)
        try:
            write_run_metadata(tmp_path, ["tool-a"], "diff+repo", cases_dir)
        finally:
            os.chdir(orig)

        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert data["agent_prompt_file"].endswith("agent_prompt.md")
        snapshot = (tmp_path / "agent_prompt_snapshot.md").read_text()
        assert snapshot == "generic prompt"

    def test_no_prompt_file_graceful(self, tmp_path: Path) -> None:
        """No config dir at all: hash is empty string, no snapshot written."""
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()

        import os
        orig = Path.cwd()
        os.chdir(tmp_path)
        try:
            write_run_metadata(tmp_path, ["tool-a"], "diff-only", cases_dir)
        finally:
            os.chdir(orig)

        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert data["agent_prompt_hash"] == ""
        assert data["agent_prompt_file"] == ""
        assert not (tmp_path / "agent_prompt_snapshot.md").exists()

    def test_agent_prompt_file_in_metadata(self, tmp_path: Path) -> None:
        """agent_prompt_file key is always present in metadata."""
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        write_run_metadata(tmp_path, ["tool-a"], "diff-only", cases_dir)
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert "agent_prompt_file" in data
        assert "agent_prompt_hash" in data
