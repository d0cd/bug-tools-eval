"""Tests for validate-cases command using real tmp git repos."""

import subprocess
from pathlib import Path

from click.testing import CliRunner

from bugeval.io import save_case
from bugeval.models import (
    Category,
    Difficulty,
    ExpectedFinding,
    PRSize,
    Severity,
    TestCase,
)
from bugeval.validate_cases import validate_cases


def make_repo(path: Path) -> Path:
    """Create a git repo with two commits and return path."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"], cwd=path, check=True, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "T"], cwd=path, check=True, capture_output=True)
    (path / "main.rs").write_text("fn main() {}\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=path, check=True, capture_output=True)
    return path


def get_sha(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def add_commit(repo: Path, content: str) -> str:
    (repo / "main.rs").write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "fix"], cwd=repo, check=True, capture_output=True)
    return get_sha(repo)


class TestValidateCasesHelp:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(validate_cases, ["--help"])
        assert result.exit_code == 0
        assert "--repo-dir" in result.output
        assert "--cases-dir" in result.output


class TestValidateCasesValid:
    def test_valid_case_passes(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        base = get_sha(repo)
        head = add_commit(repo, "fn main() { let x = 1; }\n")

        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        case = TestCase(
            id="test-001",
            repo="foo/bar",
            base_commit=base,
            head_commit=head,
            fix_commit=head,
            category=Category.logic,
            difficulty=Difficulty.easy,
            severity=Severity.low,
            language="rust",
            pr_size=PRSize.tiny,
            description="Test case",
            expected_findings=[ExpectedFinding(file="main.rs", line=1, summary="change")],
        )
        save_case(case, cases_dir / "test-001.yaml")

        runner = CliRunner()
        result = runner.invoke(
            validate_cases,
            ["--repo-dir", str(repo), "--cases-dir", str(cases_dir)],
        )
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_stats_populated(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        base = get_sha(repo)
        head = add_commit(repo, "fn main() { let x = 1; }\n")

        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        case = TestCase(
            id="test-002",
            repo="foo/bar",
            base_commit=base,
            head_commit=head,
            fix_commit=head,
            category=Category.logic,
            difficulty=Difficulty.easy,
            severity=Severity.low,
            language="rust",
            pr_size=PRSize.tiny,
            description="Test case",
            expected_findings=[],
        )
        save_case(case, cases_dir / "test-002.yaml")

        runner = CliRunner()
        runner.invoke(
            validate_cases,
            ["--repo-dir", str(repo), "--cases-dir", str(cases_dir), "--update-stats"],
        )
        # Reload the case and check stats were populated
        from bugeval.io import load_case

        updated = load_case(cases_dir / "test-002.yaml")
        assert updated.stats is not None
        assert updated.stats.files_changed >= 1


class TestValidateCasesEmptyDiff:
    def test_same_commit_fails(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        sha = get_sha(repo)

        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        case = TestCase(
            id="empty-001",
            repo="foo/bar",
            base_commit=sha,
            head_commit=sha,  # same as base → empty diff
            fix_commit=sha,
            category=Category.logic,
            difficulty=Difficulty.easy,
            severity=Severity.low,
            language="rust",
            pr_size=PRSize.tiny,
            description="Empty diff case",
            expected_findings=[],
        )
        save_case(case, cases_dir / "empty-001.yaml")

        runner = CliRunner()
        result = runner.invoke(
            validate_cases,
            ["--repo-dir", str(repo), "--cases-dir", str(cases_dir)],
        )
        assert result.exit_code == 1
        assert "FAIL" in result.output


class TestValidateCasesDryRun:
    def test_validate_cases_dry_run_no_stats_written(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        base = get_sha(repo)
        head = add_commit(repo, "fn main() { let x = 1; }\n")

        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        case = TestCase(
            id="dr-001",
            repo="foo/bar",
            base_commit=base,
            head_commit=head,
            fix_commit=head,
            category=Category.logic,
            difficulty=Difficulty.easy,
            severity=Severity.low,
            language="rust",
            pr_size=PRSize.tiny,
            description="Test case",
            expected_findings=[],
        )
        save_case(case, cases_dir / "dr-001.yaml")
        original_mtime = (cases_dir / "dr-001.yaml").stat().st_mtime

        runner = CliRunner()
        result = runner.invoke(
            validate_cases,
            ["--repo-dir", str(repo), "--cases-dir", str(cases_dir), "--update-stats", "--dry-run"],
        )
        assert result.exit_code == 0
        # File should not have been modified
        assert (cases_dir / "dr-001.yaml").stat().st_mtime == original_mtime


class TestValidateCasesPatchesDir:
    def _make_case(self, repo: Path, cases_dir: Path, case_id: str) -> tuple[str, str]:
        base = get_sha(repo)
        head = add_commit(repo, f"fn main() {{ let x = {case_id}; }}\n")
        case = TestCase(
            id=case_id,
            repo="foo/bar",
            base_commit=base,
            head_commit=head,
            fix_commit=head,
            category=Category.logic,
            difficulty=Difficulty.easy,
            severity=Severity.low,
            language="rust",
            pr_size=PRSize.tiny,
            description="Test",
            expected_findings=[],
        )
        save_case(case, cases_dir / f"{case_id}.yaml")
        return base, head

    def test_patches_dir_missing_patch_fails(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        patches_dir = tmp_path / "patches"
        patches_dir.mkdir()
        self._make_case(repo, cases_dir, "p-001")

        runner = CliRunner()
        result = runner.invoke(
            validate_cases,
            [
                "--repo-dir",
                str(repo),
                "--cases-dir",
                str(cases_dir),
                "--patches-dir",
                str(patches_dir),
            ],
        )
        assert result.exit_code == 1
        assert "patch file missing" in result.output

    def test_patches_dir_present_patch_passes(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        patches_dir = tmp_path / "patches"
        patches_dir.mkdir()
        self._make_case(repo, cases_dir, "p-002")
        # Create a stub patch file
        (patches_dir / "p-002.patch").write_text("--- stub patch ---\n")

        runner = CliRunner()
        result = runner.invoke(
            validate_cases,
            [
                "--repo-dir",
                str(repo),
                "--cases-dir",
                str(cases_dir),
                "--patches-dir",
                str(patches_dir),
            ],
        )
        assert result.exit_code == 0
        assert "PASS" in result.output


class TestValidateCasesFindingFileCheck:
    def test_finding_file_not_in_diff_warns(self, tmp_path: Path) -> None:
        """Expected finding references a file not in the diff → warning printed."""
        repo = make_repo(tmp_path / "repo")
        base = get_sha(repo)
        head = add_commit(repo, "fn main() { let x = 1; }\n")

        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        case = TestCase(
            id="ff-001",
            repo="foo/bar",
            base_commit=base,
            head_commit=head,
            fix_commit=head,
            category=Category.logic,
            difficulty=Difficulty.easy,
            severity=Severity.low,
            language="rust",
            pr_size=PRSize.tiny,
            description="Test case",
            expected_findings=[ExpectedFinding(file="nonexistent.rs", line=1, summary="phantom")],
        )
        save_case(case, cases_dir / "ff-001.yaml")

        runner = CliRunner()
        result = runner.invoke(
            validate_cases,
            ["--repo-dir", str(repo), "--cases-dir", str(cases_dir)],
        )
        # Should warn but not hard-fail
        assert "nonexistent.rs" in result.output or "not in diff" in result.output

    def test_finding_file_in_diff_passes_cleanly(self, tmp_path: Path) -> None:
        """Expected finding references a file that IS in the diff → no warning."""
        repo = make_repo(tmp_path / "repo")
        base = get_sha(repo)
        head = add_commit(repo, "fn main() { let x = 1; }\n")

        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        case = TestCase(
            id="ff-002",
            repo="foo/bar",
            base_commit=base,
            head_commit=head,
            fix_commit=head,
            category=Category.logic,
            difficulty=Difficulty.easy,
            severity=Severity.low,
            language="rust",
            pr_size=PRSize.tiny,
            description="Test case",
            expected_findings=[ExpectedFinding(file="main.rs", line=1, summary="valid")],
        )
        save_case(case, cases_dir / "ff-002.yaml")

        runner = CliRunner()
        result = runner.invoke(
            validate_cases,
            ["--repo-dir", str(repo), "--cases-dir", str(cases_dir)],
        )
        assert result.exit_code == 0
        assert "PASS" in result.output
        assert "not in diff" not in result.output


class TestValidateCasesBadCommit:
    def test_bad_commit_fails(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        base = get_sha(repo)

        cases_dir = tmp_path / "cases"
        cases_dir.mkdir()
        case = TestCase(
            id="bad-001",
            repo="foo/bar",
            base_commit=base,
            head_commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            fix_commit="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            category=Category.logic,
            difficulty=Difficulty.easy,
            severity=Severity.low,
            language="rust",
            pr_size=PRSize.tiny,
            description="Bad case",
            expected_findings=[],
        )
        save_case(case, cases_dir / "bad-001.yaml")

        runner = CliRunner()
        result = runner.invoke(
            validate_cases,
            ["--repo-dir", str(repo), "--cases-dir", str(cases_dir)],
        )
        assert result.exit_code == 1
