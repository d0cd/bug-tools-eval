"""Tests for the dashboard Flask app."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from bugeval.dashboard import (
    _parse_case_form,
    create_app,
    is_reviewed,
    load_all_cases,
    load_review_state,
    mark_reviewed,
    save_review_state,
)
from bugeval.models import Category, Difficulty, ExpectedFinding, PRSize, Severity, TestCase

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_case(case_id: str = "leo-001", repo: str = "leo") -> TestCase:
    return TestCase(
        id=case_id,
        repo=repo,
        base_commit="abc123",
        head_commit="def456",
        fix_commit="ghi789",
        category=Category.logic,
        difficulty=Difficulty.medium,
        severity=Severity.high,
        language="rust",
        pr_size=PRSize.small,
        description="A test case",
        expected_findings=[ExpectedFinding(file="src/main.rs", line=42, summary="off-by-one")],
        needs_manual_review=True,
    )


@pytest.fixture()
def cases_dir(tmp_path: Path) -> Path:
    c_dir = tmp_path / "cases" / "final"
    repo_dir = c_dir / "leo"
    repo_dir.mkdir(parents=True)
    case = _make_case()
    (repo_dir / "leo-001.yaml").write_text(
        yaml.safe_dump(case.model_dump(mode="json"), sort_keys=False)
    )
    return c_dir


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    r_dir = tmp_path / "results"
    r_dir.mkdir()
    return r_dir


@pytest.fixture()
def app(cases_dir: Path, results_dir: Path):  # type: ignore[no-untyped-def]
    app = create_app(cases_dir, results_dir)
    app.config["TESTING"] = True
    return app


@pytest.fixture()
def client(app):  # type: ignore[no-untyped-def]
    return app.test_client()


# ---------------------------------------------------------------------------
# Sidecar review-state
# ---------------------------------------------------------------------------


def test_load_review_state_empty(cases_dir: Path) -> None:
    state = load_review_state(cases_dir, "leo")
    assert state == {}


def test_save_and_load_review_state(cases_dir: Path) -> None:
    save_review_state(cases_dir, "leo", {"leo-001": {"reviewed": True}})
    state = load_review_state(cases_dir, "leo")
    assert state["leo-001"]["reviewed"] is True


def test_mark_reviewed(cases_dir: Path) -> None:
    case = _make_case()
    mark_reviewed(cases_dir, case, reviewer="tester")
    assert is_reviewed(cases_dir, "leo-001", "leo")


def test_is_reviewed_false_initially(cases_dir: Path) -> None:
    assert not is_reviewed(cases_dir, "leo-001", "leo")


def test_is_reviewed_missing_repo(cases_dir: Path) -> None:
    assert not is_reviewed(cases_dir, "leo-001", "nonexistent-repo")


# ---------------------------------------------------------------------------
# load_all_cases
# ---------------------------------------------------------------------------


def test_load_all_cases(cases_dir: Path) -> None:
    cases = load_all_cases(cases_dir)
    assert len(cases) == 1
    assert cases[0].id == "leo-001"


def test_load_all_cases_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert load_all_cases(empty) == []


def test_load_all_cases_skips_invalid_yaml(cases_dir: Path) -> None:
    (cases_dir / "leo" / "bad.yaml").write_text("not: valid: yaml: ::::")
    # Should not raise; just skip bad file
    cases = load_all_cases(cases_dir)
    assert len(cases) == 1  # bad file skipped


# ---------------------------------------------------------------------------
# Flask routes — GET returns 200
# ---------------------------------------------------------------------------


def test_index_returns_200(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Dashboard" in resp.data


def test_case_list_returns_200(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/cases")
    assert resp.status_code == 200
    assert b"leo-001" in resp.data


def test_case_detail_returns_200(client) -> None:  # type: ignore[no-untyped-def]
    with patch("bugeval.dashboard.fetch_diff", return_value="--- a/src/main.rs\n+diff"):
        resp = client.get("/cases/leo-001")
    assert resp.status_code == 200
    assert b"leo-001" in resp.data


def test_case_detail_404_for_unknown(client) -> None:  # type: ignore[no-untyped-def]
    with patch("bugeval.dashboard.fetch_diff", return_value=""):
        resp = client.get("/cases/does-not-exist")
    assert resp.status_code == 404


def test_human_judge_returns_200(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/human-judge")
    assert resp.status_code == 200
    assert b"Human Judge" in resp.data


def test_dx_page_returns_200(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/dx")
    assert resp.status_code == 200
    assert b"DX Assessment" in resp.data


def test_metrics_list_returns_200(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"Metrics" in resp.data


def test_metrics_detail_404_for_unknown(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/metrics/nonexistent-run")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Case edit POST
# ---------------------------------------------------------------------------


def test_case_edit_post_saves(client, cases_dir: Path) -> None:  # type: ignore[no-untyped-def]
    with patch("bugeval.dashboard.fetch_diff", return_value=""):
        resp = client.post(
            "/cases/leo-001",
            data={
                "action": "save",
                "category": "memory",
                "difficulty": "hard",
                "severity": "critical",
                "pr_size": "large",
                "visibility": "public",
                "base_commit": "abc123",
                "head_commit": "def456",
                "description": "Updated description",
                "ef_file_0": "src/lib.rs",
                "ef_line_0": "10",
                "ef_summary_0": "overflow",
            },
            follow_redirects=False,
        )
    assert resp.status_code == 302  # redirect after save

    # Verify YAML was updated
    yaml_path = cases_dir / "leo" / "leo-001.yaml"
    data = yaml.safe_load(yaml_path.read_text())
    assert data["category"] == "memory"
    assert data["difficulty"] == "hard"
    assert data["description"] == "Updated description"
    assert data["expected_findings"][0]["file"] == "src/lib.rs"


def test_parse_case_form_preserves_verified_fields() -> None:
    original = _make_case()
    original = original.model_copy(update={"verified": True, "verified_by": "alice"})
    form = {
        "base_commit": original.base_commit,
        "head_commit": original.head_commit,
        "category": original.category.value,
        "difficulty": original.difficulty.value,
        "severity": original.severity.value,
        "pr_size": original.pr_size.value,
        "visibility": original.visibility.value,
        "description": original.description,
    }
    result = _parse_case_form(form, original)
    assert result.verified is True
    assert result.verified_by == "alice"


def test_case_accept_marks_reviewed(client, cases_dir: Path) -> None:  # type: ignore[no-untyped-def]
    resp = client.post(
        "/cases/leo-001",
        data={"action": "accept"},
        follow_redirects=False,
    )
    assert resp.status_code == 302
    assert is_reviewed(cases_dir, "leo-001", "leo")


def test_case_skip_redirects_to_next(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.post(
        "/cases/leo-001",
        data={"action": "skip"},
        follow_redirects=False,
    )
    assert resp.status_code == 302


# ---------------------------------------------------------------------------
# Case list filtering
# ---------------------------------------------------------------------------


def test_case_list_filter_by_repo(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/cases?repo=leo")
    assert resp.status_code == 200
    assert b"leo-001" in resp.data


def test_case_list_filter_no_match(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/cases?repo=nonexistent")
    assert resp.status_code == 200
    assert b"leo-001" not in resp.data


# ---------------------------------------------------------------------------
# Diff fetching (mocked)
# ---------------------------------------------------------------------------


def test_fetch_diff_mock(client) -> None:  # type: ignore[no-untyped-def]
    with patch("bugeval.dashboard.fetch_diff", return_value="+added line\n-removed line") as mock:
        resp = client.get("/cases/leo-001")
    assert resp.status_code == 200
    assert b"+added line" in resp.data
    mock.assert_called_once_with("leo", "def456")


# ---------------------------------------------------------------------------
# Human judge scoring POST
# ---------------------------------------------------------------------------


def test_human_judge_score_post(client, results_dir: Path) -> None:  # type: ignore[no-untyped-def]
    run_dir = results_dir / "run-2026-03-07"
    run_dir.mkdir()

    resp = client.post(
        "/human-judge/score",
        data={
            "run_id": "run-2026-03-07",
            "case_id": "leo-001",
            "tool": "greptile",
            "human_score": "2",
            "notes": "looks right",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302

    saved = yaml.safe_load((run_dir / "human_judge" / "leo-001-greptile.yaml").read_text())
    assert saved["human_score"] == 2
    assert saved["notes"] == "looks right"


def test_human_judge_score_invalid_score(client, results_dir: Path) -> None:  # type: ignore[no-untyped-def]
    run_dir = results_dir / "run-2026-03-07"
    run_dir.mkdir()

    resp = client.post(
        "/human-judge/score",
        data={
            "run_id": "run-2026-03-07",
            "case_id": "leo-001",
            "tool": "greptile",
            "human_score": "5",  # out of range
        },
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# DX Assessment POST
# ---------------------------------------------------------------------------


def test_dx_save(client, results_dir: Path, cases_dir: Path) -> None:  # type: ignore[no-untyped-def]
    run_dir = results_dir / "run-2026-03-07"
    run_dir.mkdir()

    # Write a fake normalized result
    nr = {
        "test_case_id": "leo-001",
        "tool": "greptile",
        "context_level": "diff-only",
        "comments": [],
        "metadata": {"tokens": 0, "cost_usd": 0.0, "time_seconds": 0.0},
        "dx": None,
    }
    (run_dir / "leo-001--greptile.yaml").write_text(yaml.safe_dump(nr, sort_keys=False))

    resp = client.post(
        "/dx/save",
        data={
            "run_id": "run-2026-03-07",
            "tool": "greptile",
            "actionability": "4",
            "false_positive_burden": "3",
            "integration_friction": "2",
            "response_latency": "5",
            "notes": "decent",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 302

    # Verify dx saved in YAML
    data = yaml.safe_load((run_dir / "leo-001--greptile.yaml").read_text())
    assert data["dx"]["actionability"] == 4
    assert data["dx"]["notes"] == "decent"


# ---------------------------------------------------------------------------
# CLI command help
# ---------------------------------------------------------------------------


def test_dashboard_cli_help() -> None:
    from click.testing import CliRunner

    from bugeval.dashboard import dashboard

    runner = CliRunner()
    result = runner.invoke(dashboard, ["--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--cases-dir" in result.output
    assert "--results-dir" in result.output


# ---------------------------------------------------------------------------
# Index page stats
# ---------------------------------------------------------------------------


def test_index_shows_counts(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"1" in resp.data  # total cases


def test_index_shows_needs_review_count(client) -> None:  # type: ignore[no-untyped-def]
    resp = client.get("/")
    # leo-001 has needs_manual_review=True
    assert b"Needs Manual Review" in resp.data


# ---------------------------------------------------------------------------
# Metrics with run data
# ---------------------------------------------------------------------------


def test_metrics_detail_with_run(client, results_dir: Path) -> None:  # type: ignore[no-untyped-def]
    run_dir = results_dir / "run-2026-03-07"
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(parents=True)

    score = {
        "test_case_id": "leo-001",
        "tool": "greptile",
        "score": 2,
        "votes": [2, 2, 2],
        "reasoning": "correct",
        "noise": {"total_comments": 1, "relevant_comments": 1, "snr": 1.0},
    }
    (scores_dir / "leo-001--greptile.yaml").write_text(yaml.safe_dump(score, sort_keys=False))

    resp = client.get("/metrics/run-2026-03-07")
    assert resp.status_code == 200
    assert b"greptile" in resp.data


# ---------------------------------------------------------------------------
# Case list — sorting
# ---------------------------------------------------------------------------


def test_case_list_sort_by_id(client, cases_dir: Path) -> None:
    """Adding a second case and sorting by ID returns both in order."""
    (cases_dir / "leo" / "leo-002.yaml").write_text(
        yaml.safe_dump(_make_case("leo-002").model_dump(mode="json"), sort_keys=False)
    )
    resp = client.get("/cases?sort=id")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert body.index("leo-001") < body.index("leo-002")


def test_case_list_sort_reverse(client, cases_dir: Path) -> None:
    """Reverse sort (-id) puts leo-002 before leo-001."""
    (cases_dir / "leo" / "leo-002.yaml").write_text(
        yaml.safe_dump(_make_case("leo-002").model_dump(mode="json"), sort_keys=False)
    )
    resp = client.get("/cases?sort=-id")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert body.index("leo-002") < body.index("leo-001")


def test_case_list_sort_link_clean_url(client) -> None:
    """Sort links must not carry page= — they should reset to page 1."""
    resp = client.get("/cases?sort=id&page=3")
    assert resp.status_code == 200
    # When no active filters, sort link for repo should be just ?sort=repo
    assert b'href="/cases?sort=repo"' in resp.data


def test_case_list_sort_link_preserves_filters(client) -> None:
    """Sort links should keep active filter params."""
    resp = client.get("/cases?sort=id&repo=leo")
    assert resp.status_code == 200
    assert b"sort=repo&amp;repo=leo" in resp.data or b"sort=repo&repo=leo" in resp.data


# ---------------------------------------------------------------------------
# Case list — pagination
# ---------------------------------------------------------------------------


def test_case_list_page_2_no_error(client) -> None:
    """Requesting page 2 with fewer than per_page cases returns 200 cleanly."""
    resp = client.get("/cases?page=2")
    assert resp.status_code == 200


def test_case_list_pagination_shown_with_many_cases(client, cases_dir: Path) -> None:
    """Pagination controls appear when cases exceed per_page (50)."""
    from unittest.mock import patch as mock_patch
    many = [_make_case(f"leo-{i:03d}") for i in range(51)]
    with mock_patch("bugeval.dashboard.load_all_cases", return_value=many):
        resp = client.get("/cases")
    assert resp.status_code == 200
    assert b"Next" in resp.data


def test_case_list_pagination_next_link_no_page_duplication(client, cases_dir: Path) -> None:
    """Pagination next-page link must not duplicate the page= parameter."""
    from unittest.mock import patch as mock_patch
    many = [_make_case(f"leo-{i:03d}") for i in range(51)]
    with mock_patch("bugeval.dashboard.load_all_cases", return_value=many):
        resp = client.get("/cases?sort=id")
    assert resp.status_code == 200
    body = resp.data.decode()
    # The next-page link should contain page=2 exactly once
    import re
    next_href = re.search(r'href="([^"]*page=2[^"]*)"', body)
    assert next_href is not None
    assert next_href.group(1).count("page=") == 1


# ---------------------------------------------------------------------------
# Metrics — cost data
# ---------------------------------------------------------------------------

_SCORE_YAML = {
    "test_case_id": "leo-001",
    "tool": "greptile",
    "score": 2,
    "votes": [2, 2, 2],
    "reasoning": "correct",
    "noise": {"total_comments": 1, "relevant_comments": 1, "snr": 1.0},
}


def _write_score(run_dir: Path, data: dict = _SCORE_YAML) -> None:
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    (scores_dir / "leo-001--greptile.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def test_metrics_detail_with_cost_data(client, results_dir: Path) -> None:
    run_dir = results_dir / "run-2026-03-07"
    _write_score(run_dir)
    nr = {
        "test_case_id": "leo-001",
        "tool": "greptile",
        "context_level": "diff-only",
        "comments": [],
        "metadata": {"tokens": 500, "cost_usd": 0.0500, "time_seconds": 1.5},
        "dx": None,
    }
    (run_dir / "leo-001--greptile.yaml").write_text(yaml.safe_dump(nr, sort_keys=False))

    resp = client.get("/metrics/run-2026-03-07")
    assert resp.status_code == 200
    assert b"Cost" in resp.data
    assert b"0.05" in resp.data


def test_metrics_detail_with_zero_cost_hides_cost_table(client, results_dir: Path) -> None:
    """Cost table is suppressed when all tools have zero cost."""
    run_dir = results_dir / "run-2026-03-07"
    _write_score(run_dir)
    nr = {
        "test_case_id": "leo-001",
        "tool": "greptile",
        "context_level": "diff-only",
        "comments": [],
        "metadata": {"tokens": 0, "cost_usd": 0.0, "time_seconds": 0.0},
        "dx": None,
    }
    (run_dir / "leo-001--greptile.yaml").write_text(yaml.safe_dump(nr, sort_keys=False))

    resp = client.get("/metrics/run-2026-03-07")
    assert resp.status_code == 200
    # Cost section is only shown if any_cost is truthy
    assert b"Per Review" not in resp.data


# ---------------------------------------------------------------------------
# Metrics — DX summary
# ---------------------------------------------------------------------------


def test_metrics_detail_with_dx_summary(client, results_dir: Path) -> None:
    run_dir = results_dir / "run-2026-03-07"
    _write_score(run_dir)
    nr = {
        "test_case_id": "leo-001",
        "tool": "greptile",
        "context_level": "diff-only",
        "comments": [],
        "metadata": {"tokens": 0, "cost_usd": 0.0, "time_seconds": 0.0},
        "dx": {
            "actionability": 4,
            "false_positive_burden": 3,
            "integration_friction": 2,
            "response_latency": 5,
            "notes": "solid",
        },
    }
    (run_dir / "leo-001--greptile.yaml").write_text(yaml.safe_dump(nr, sort_keys=False))

    resp = client.get("/metrics/run-2026-03-07")
    assert resp.status_code == 200
    assert b"DX Assessment" in resp.data
    assert b"greptile" in resp.data


# ---------------------------------------------------------------------------
# Index — pipeline status with run dir
# ---------------------------------------------------------------------------


def test_index_shows_pipeline_run(client, results_dir: Path) -> None:
    run_dir = results_dir / "run-2026-03-07"
    run_dir.mkdir()
    checkpoint = {"leo-001::greptile": {"case_id": "leo-001", "tool": "greptile", "status": "done"}}
    (run_dir / "checkpoint.yaml").write_text(yaml.safe_dump(checkpoint, sort_keys=False))
    nr = {
        "test_case_id": "leo-001", "tool": "greptile", "context_level": "diff-only",
        "comments": [], "metadata": {"tokens": 0, "cost_usd": 0.0, "time_seconds": 0.0}, "dx": None,
    }
    (run_dir / "leo-001--greptile.yaml").write_text(yaml.safe_dump(nr, sort_keys=False))

    resp = client.get("/")
    assert resp.status_code == 200
    assert b"run-2026-03-07" in resp.data
    assert b"greptile" in resp.data


def test_index_pipeline_status_normalize_count(client, results_dir: Path) -> None:
    run_dir = results_dir / "run-2026-03-07"
    run_dir.mkdir()
    nr = {
        "test_case_id": "leo-001", "tool": "greptile", "context_level": "diff-only",
        "comments": [], "metadata": {"tokens": 0, "cost_usd": 0.0, "time_seconds": 0.0}, "dx": None,
    }
    (run_dir / "leo-001--greptile.yaml").write_text(yaml.safe_dump(nr, sort_keys=False))

    resp = client.get("/")
    assert b"normalize 1/" in resp.data


# ---------------------------------------------------------------------------
# Human judge — page with scores
# ---------------------------------------------------------------------------


def test_human_judge_page_with_run_scores(client, results_dir: Path) -> None:
    run_dir = results_dir / "run-2026-03-07"
    _write_score(run_dir)

    resp = client.get("/human-judge?run=run-2026-03-07")
    assert resp.status_code == 200
    assert b"run-2026-03-07" in resp.data
    assert b"leo-001" in resp.data
    assert b"greptile" in resp.data


def test_human_judge_page_no_scores_shows_message(client, results_dir: Path) -> None:
    run_dir = results_dir / "run-2026-03-07"
    run_dir.mkdir()

    resp = client.get("/human-judge?run=run-2026-03-07")
    assert resp.status_code == 200
    assert b"No judge scores found" in resp.data


# ---------------------------------------------------------------------------
# Metrics — run selector renders runs
# ---------------------------------------------------------------------------


def test_metrics_run_selector_lists_runs(client, results_dir: Path) -> None:
    run_dir = results_dir / "run-2026-03-07"
    run_dir.mkdir()

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"run-2026-03-07" in resp.data
