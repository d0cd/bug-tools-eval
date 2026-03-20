"""Tests for the v2 dashboard."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from bugeval.dashboard import _invalidate_cache, create_app, dashboard_cmd


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear dashboard cache before each test to avoid cross-test pollution."""
    _invalidate_cache()


@pytest.fixture()
def cases_dir(tmp_path: Path) -> Path:
    """Create a minimal cases directory with v2 TestCase YAMLs."""
    repo_dir = tmp_path / "cases" / "ProvableHQ_leo"
    repo_dir.mkdir(parents=True)

    bug_case = {
        "id": "leo-001",
        "repo": "ProvableHQ/leo",
        "kind": "bug",
        "language": "rust",
        "base_commit": "abc123",
        "fix_commit": "def456",
        "category": "logic",
        "difficulty": "medium",
        "severity": "high",
        "bug_description": "Off-by-one in parser",
        "truth": {
            "introducing_commit": "aaa111",
            "blame_confidence": "A",
            "buggy_lines": [
                {"file": "src/parser.rs", "line": 42, "content": "x + 1"},
            ],
            "fix_summary": "Changed x+1 to x",
        },
        "validation": {
            "claude_verdict": "confirmed",
            "gemini_verdict": "confirmed",
            "agreement": True,
        },
    }
    (repo_dir / "leo-001.yaml").write_text(
        yaml.safe_dump(bug_case, sort_keys=False)
    )

    clean_case = {
        "id": "leo-002",
        "repo": "ProvableHQ/leo",
        "kind": "clean",
        "language": "rust",
        "base_commit": "ccc333",
    }
    (repo_dir / "leo-002.yaml").write_text(
        yaml.safe_dump(clean_case, sort_keys=False)
    )

    return tmp_path / "cases"


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    """Create a minimal results directory."""
    rd = tmp_path / "results"
    rd.mkdir()
    return rd


@pytest.fixture()
def run_dir_with_scores(results_dir: Path) -> Path:
    """Create a run directory with scores and results."""
    run_dir = results_dir / "run-2026-03-20"
    run_dir.mkdir()

    scores_dir = run_dir / "scores"
    scores_dir.mkdir()
    score = {
        "case_id": "leo-001",
        "tool": "agent",
        "caught": True,
        "detection_score": 3,
        "review_quality": 3,
        "tp_count": 1,
        "fp_count": 0,
        "false_alarm": False,
        "potentially_contaminated": False,
    }
    (scores_dir / "leo-001--agent.yaml").write_text(
        yaml.safe_dump(score, sort_keys=False)
    )

    clean_score = {
        "case_id": "leo-002",
        "tool": "agent",
        "caught": False,
        "detection_score": 0,
        "review_quality": 2,
        "false_alarm": False,
    }
    (scores_dir / "leo-002--agent.yaml").write_text(
        yaml.safe_dump(clean_score, sort_keys=False)
    )

    results_sub = run_dir / "results"
    results_sub.mkdir()
    result = {
        "case_id": "leo-001",
        "tool": "agent",
        "comments": [{"file": "src/parser.rs", "line": 42, "body": "Bug here"}],
        "time_seconds": 12.5,
        "cost_usd": 0.03,
    }
    (results_sub / "leo-001--agent.yaml").write_text(
        yaml.safe_dump(result, sort_keys=False)
    )

    return run_dir


@pytest.fixture()
def client(cases_dir: Path, results_dir: Path):
    """Flask test client."""
    app = create_app(cases_dir, results_dir)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture()
def client_with_scores(cases_dir: Path, run_dir_with_scores: Path):
    """Flask test client with a run that has scores."""
    results_dir = run_dir_with_scores.parent
    app = create_app(cases_dir, results_dir)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHome:
    def test_home_returns_200(self, client) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Dashboard" in resp.data

    def test_home_shows_counts(self, client) -> None:
        resp = client.get("/")
        html = resp.data.decode()
        assert "2" in html  # total cases
        assert "ProvableHQ/leo" in html


class TestCasesAPI:
    def test_cases_api_returns_json(self, client) -> None:
        resp = client.get("/api/cases")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 2
        assert len(data["cases"]) == 2

    def test_cases_api_has_v2_fields(self, client) -> None:
        resp = client.get("/api/cases")
        data = resp.get_json()
        case = next(c for c in data["cases"] if c["id"] == "leo-001")
        assert case["kind"] == "bug"
        assert case["blame_confidence"] == "A"
        assert case["validation_status"] == "agreed"

    def test_cases_api_filter_kind(self, client) -> None:
        resp = client.get("/api/cases?kind=clean")
        data = resp.get_json()
        assert data["total"] == 1
        assert data["cases"][0]["kind"] == "clean"

    def test_cases_api_filter_repo(self, client) -> None:
        resp = client.get("/api/cases?repo=ProvableHQ/leo")
        data = resp.get_json()
        assert data["total"] == 2

    def test_cases_api_search(self, client) -> None:
        resp = client.get("/api/cases?q=parser")
        data = resp.get_json()
        assert data["total"] == 1

    def test_cases_api_clean_no_blame(self, client) -> None:
        data = client.get("/api/cases?kind=clean").get_json()
        assert data["cases"][0]["blame_confidence"] == ""

    def test_cases_api_filters_field(self, client) -> None:
        data = client.get("/api/cases").get_json()
        assert "filters" in data
        assert "repos" in data["filters"]
        assert "kinds" in data["filters"]

    def test_cases_api_filter_nonexistent_repo(self, client) -> None:
        data = client.get("/api/cases?repo=nonexistent").get_json()
        assert data["total"] == 0


class TestCaseDetail:
    def test_case_detail_200(self, client) -> None:
        resp = client.get("/cases/leo-001")
        assert resp.status_code == 200
        assert b"leo-001" in resp.data

    def test_case_detail_404(self, client) -> None:
        resp = client.get("/cases/nonexistent-999")
        assert resp.status_code == 404

    def test_case_list_200(self, client) -> None:
        resp = client.get("/cases")
        assert resp.status_code == 200

    def test_case_detail_shows_v2_fields(self, client) -> None:
        resp = client.get("/cases/leo-001")
        html = resp.data.decode()
        assert "src/parser.rs" in html  # buggy line
        assert "Tier" not in html or "blame_confidence" not in html or "A" in html


class TestExperiments:
    def test_create_experiment(self, client) -> None:
        resp = client.post(
            "/api/experiments",
            json={"name": "Test Exp", "notes": "Testing"},
            content_type="application/json",
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["id"] == "test-exp"
        assert data["name"] == "Test Exp"

    def test_create_duplicate(self, client) -> None:
        client.post(
            "/api/experiments",
            json={"name": "Dup"},
            content_type="application/json",
        )
        resp = client.post(
            "/api/experiments",
            json={"name": "Dup"},
            content_type="application/json",
        )
        assert resp.status_code == 409

    def test_update_experiment(self, client) -> None:
        client.post(
            "/api/experiments",
            json={"name": "Updatable"},
            content_type="application/json",
        )
        resp = client.put(
            "/api/experiments/updatable",
            json={"notes": "Updated notes"},
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json()["notes"] == "Updated notes"

    def test_archive_experiment(self, client) -> None:
        client.post(
            "/api/experiments",
            json={"name": "Archivable"},
            content_type="application/json",
        )
        resp = client.post("/api/experiments/archivable/archive")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "archived"

        # Toggle back
        resp = client.post("/api/experiments/archivable/archive")
        assert resp.get_json()["status"] == "active"

    def test_list_experiments(self, client) -> None:
        client.post(
            "/api/experiments",
            json={"name": "Listed"},
            content_type="application/json",
        )
        resp = client.get("/api/experiments")
        data = resp.get_json()
        assert len(data["experiments"]) == 1

    def test_create_empty_name(self, client) -> None:
        resp = client.post(
            "/api/experiments",
            json={"name": ""},
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_update_not_found(self, client) -> None:
        resp = client.put(
            "/api/experiments/nonexistent",
            json={"notes": "nope"},
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_archive_not_found(self, client) -> None:
        resp = client.post("/api/experiments/nonexistent/archive")
        assert resp.status_code == 404


class TestGoldenSet:
    def test_golden_page_200(self, client) -> None:
        resp = client.get("/golden")
        assert resp.status_code == 200

    def test_golden_shows_cases(self, client) -> None:
        resp = client.get("/golden")
        html = resp.data.decode()
        assert "leo-001" in html

    def test_golden_toggle(self, client) -> None:
        resp = client.post(
            "/golden/leo-001",
            data={"status": "confirmed"},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"confirmed" in resp.data

    def test_golden_dispute(self, client) -> None:
        resp = client.post(
            "/golden/leo-001",
            data={"status": "disputed"},
            follow_redirects=False,
        )
        assert resp.status_code == 302

    def test_golden_invalid_status(self, client) -> None:
        resp = client.post(
            "/golden/leo-001",
            data={"status": "invalid"},
        )
        assert resp.status_code == 400


class TestMetrics:
    def test_metrics_list_200(self, client) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_with_scores(self, client_with_scores) -> None:
        resp = client_with_scores.get("/metrics/run-2026-03-20")
        assert resp.status_code == 200
        html = resp.data.decode()
        assert "agent" in html
        assert "Catch Rate" in html

    def test_metrics_404(self, client) -> None:
        resp = client.get("/metrics/nonexistent")
        assert resp.status_code == 404


class TestRuns:
    def test_runs_page_200(self, client) -> None:
        resp = client.get("/runs")
        assert resp.status_code == 200

    def test_run_detail_404(self, client) -> None:
        resp = client.get("/runs/run-nonexistent")
        assert resp.status_code == 404

    def test_run_detail_with_scores(self, client_with_scores) -> None:
        resp = client_with_scores.get("/runs/run-2026-03-20")
        assert resp.status_code == 200

    def test_run_notes(self, client_with_scores) -> None:
        resp = client_with_scores.post(
            "/runs/run-2026-03-20/notes",
            data={"text": "Test note"},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert b"Test note" in resp.data

    def test_run_notes_404(self, client) -> None:
        resp = client.post(
            "/runs/run-nonexistent/notes",
            data={"text": "test"},
        )
        assert resp.status_code == 404


class TestAddCase:
    def test_add_case_missing_url(self, client) -> None:
        resp = client.post(
            "/api/add-case",
            json={},
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "pr_url" in data["error"]

    def test_add_case_empty_url(self, client) -> None:
        resp = client.post(
            "/api/add-case",
            json={"pr_url": "  "},
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_add_case_success(self, client, monkeypatch) -> None:
        from bugeval.models import TestCase

        fake_case = TestCase(
            id="leo-099",
            repo="ProvableHQ/leo",
            kind="bug",
            base_commit="abc123",
        )

        def mock_add(pr_url: str, cases_dir, repo_dir):
            assert pr_url == "https://github.com/ProvableHQ/leo/pull/99"
            return fake_case

        monkeypatch.setattr(
            "bugeval.dashboard.add_case_from_pr", mock_add
        )
        resp = client.post(
            "/api/add-case",
            json={"pr_url": "https://github.com/ProvableHQ/leo/pull/99"},
            content_type="application/json",
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["case_id"] == "leo-099"
        assert data["repo"] == "ProvableHQ/leo"

    def test_add_case_duplicate(self, client, monkeypatch) -> None:
        def mock_add(pr_url: str, cases_dir, repo_dir):
            return None

        monkeypatch.setattr(
            "bugeval.dashboard.add_case_from_pr", mock_add
        )
        resp = client.post(
            "/api/add-case",
            json={"pr_url": "https://github.com/ProvableHQ/leo/pull/1"},
            content_type="application/json",
        )
        assert resp.status_code == 409
        data = resp.get_json()
        assert "Duplicate" in data["error"]

    def test_add_case_error(self, client, monkeypatch) -> None:
        def mock_add(pr_url: str, cases_dir, repo_dir):
            raise RuntimeError("gh CLI failed")

        monkeypatch.setattr(
            "bugeval.dashboard.add_case_from_pr", mock_add
        )
        resp = client.post(
            "/api/add-case",
            json={"pr_url": "https://github.com/ProvableHQ/leo/pull/1"},
            content_type="application/json",
        )
        assert resp.status_code == 500
        data = resp.get_json()
        assert "gh CLI failed" in data["error"]


class TestDashboardCli:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(dashboard_cmd, ["--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--cases-dir" in result.output
