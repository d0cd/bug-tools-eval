"""Local Flask dashboard for experiment management."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import click
import yaml
from flask import Flask, redirect, render_template, request, url_for
from pydantic import ValidationError

from bugeval.analyze import aggregate_scores, compute_catch_rate, compute_snr
from bugeval.human_judge import compute_kappa_report
from bugeval.judge_models import JudgeScore
from bugeval.models import (
    Category,
    Difficulty,
    ExpectedFinding,
    PRSize,
    Severity,
    TestCase,
    Visibility,
)
from bugeval.pr_eval_models import default_scoring
from bugeval.result_models import DxAssessment, NormalizedResult

# ---------------------------------------------------------------------------
# Sidecar review-state helpers
# ---------------------------------------------------------------------------

def _sidecar_path(cases_dir: Path, repo: str) -> Path:
    return cases_dir / repo / ".review_state.json"


def load_review_state(cases_dir: Path, repo: str) -> dict[str, Any]:
    path = _sidecar_path(cases_dir, repo)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_review_state(cases_dir: Path, repo: str, state: dict[str, Any]) -> None:
    path = _sidecar_path(cases_dir, repo)
    path.write_text(json.dumps(state, indent=2, default=str))


def mark_reviewed(cases_dir: Path, case: TestCase, reviewer: str = "human") -> None:
    state = load_review_state(cases_dir, case.repo)
    state[case.id] = {
        "reviewed": True,
        "reviewer": reviewer,
        "timestamp": datetime.utcnow().isoformat(),
    }
    save_review_state(cases_dir, case.repo, state)


def is_reviewed(cases_dir: Path, case_id: str, repo: str) -> bool:
    state = load_review_state(cases_dir, repo)
    return bool(state.get(case_id, {}).get("reviewed", False))


# ---------------------------------------------------------------------------
# Case YAML helpers
# ---------------------------------------------------------------------------

def _find_case_yaml(cases_dir: Path, case_id: str) -> Path | None:
    for path in cases_dir.rglob("*.yaml"):
        if path.stem == case_id:
            return path
    return None


def load_all_cases(cases_dir: Path) -> list[TestCase]:
    if not cases_dir.exists():
        return []
    cases = []
    for repo_dir in sorted(cases_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        for yaml_path in sorted(repo_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(yaml_path.read_text()) or {}
                cases.append(TestCase(**data))
            except (ValidationError, TypeError, yaml.YAMLError):
                pass
    return cases


def save_case(cases_dir: Path, case: TestCase) -> None:
    yaml_path = _find_case_yaml(cases_dir, case.id)
    if yaml_path is None:
        return
    yaml_path.write_text(yaml.safe_dump(case.model_dump(mode="json"), sort_keys=False))


# ---------------------------------------------------------------------------
# Diff fetching
# ---------------------------------------------------------------------------

def fetch_diff(repo: str, head_commit: str) -> str:
    """Fetch commit diff via gh CLI. Returns diff text or error message."""
    owner_repo = repo if "/" in repo else f"provable-labs/{repo}"
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{owner_repo}/commits/{head_commit}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return f"Error fetching diff: {result.stderr.strip()}"
        data = json.loads(result.stdout)
        files = data.get("files", [])
        parts = []
        for f in files:
            patch = f.get("patch", "")
            if patch:
                parts.append(f"--- {f['filename']}\n{patch}")
        return "\n\n".join(parts) or "(no diff available)"
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Pipeline progress helpers
# ---------------------------------------------------------------------------

def _run_dirs(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run-")])


def _pipeline_status(run_dir: Path, total_cases: int) -> dict[str, Any]:
    checkpoint_path = run_dir / "checkpoint.yaml"
    checkpoint: dict[str, Any] = {}
    if checkpoint_path.exists():
        checkpoint = yaml.safe_load(checkpoint_path.read_text()) or {}

    normalized_count = sum(1 for p in run_dir.glob("*.yaml") if p.name != "checkpoint.yaml")
    scores_dir = run_dir / "scores"
    judged_count = len(list(scores_dir.glob("*.yaml"))) if scores_dir.exists() else 0
    analysis_dir = run_dir / "analysis"
    has_analysis = analysis_dir.exists()

    tools_run = sorted({k.split("::")[1] for k in checkpoint if "::" in k})

    return {
        "name": run_dir.name,
        "tools_run": tools_run,
        "normalized": normalized_count,
        "judged": judged_count,
        "has_analysis": has_analysis,
        "expected": total_cases * max(len(tools_run), 1),
        "checkpoint_entries": len(checkpoint),
    }


# ---------------------------------------------------------------------------
# Score loading helpers
# ---------------------------------------------------------------------------

def _load_scores(run_dir: Path) -> list[JudgeScore]:
    scores_dir = run_dir / "scores"
    scores = []
    if not scores_dir.exists():
        return scores
    for path in sorted(scores_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        try:
            scores.append(JudgeScore(**data))
        except (ValidationError, TypeError):
            pass
    return scores


def _load_normalized(run_dir: Path) -> dict[tuple[str, str], NormalizedResult]:
    lookup: dict[tuple[str, str], NormalizedResult] = {}
    for path in run_dir.glob("*.yaml"):
        if path.name == "checkpoint.yaml":
            continue
        data = yaml.safe_load(path.read_text()) or {}
        try:
            r = NormalizedResult(**data)
            lookup[(r.test_case_id, r.tool)] = r
        except (ValidationError, TypeError):
            pass
    return lookup


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------

def create_app(cases_dir: Path, results_dir: Path) -> Flask:
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["CASES_DIR"] = cases_dir
    app.config["RESULTS_DIR"] = results_dir

    # ------------------------------------------------------------------
    # Dashboard home
    # ------------------------------------------------------------------

    @app.route("/")
    def index() -> str:
        c_dir: Path = app.config["CASES_DIR"]
        r_dir: Path = app.config["RESULTS_DIR"]
        cases = load_all_cases(c_dir)

        # Review state
        reviewed_ids: set[str] = set()
        for case in cases:
            if is_reviewed(c_dir, case.id, case.repo):
                reviewed_ids.add(case.id)

        # Dataset stats
        total = len(cases)
        needs_review = sum(1 for c in cases if c.needs_manual_review)
        reviewed = len(reviewed_ids)

        by_repo: dict[str, dict[str, Any]] = {}
        for c in cases:
            repo = c.repo
            if repo not in by_repo:
                by_repo[repo] = {"count": 0, "reviewed": 0}
            by_repo[repo]["count"] += 1
            if c.id in reviewed_ids:
                by_repo[repo]["reviewed"] += 1

        # Category/difficulty/severity distribution
        cat_dist = _count_field(cases, "category")
        diff_dist = _count_field(cases, "difficulty")
        sev_dist = _count_field(cases, "severity")

        # Pipeline progress
        pipeline_statuses = [_pipeline_status(d, total) for d in _run_dirs(r_dir)]

        return render_template(
            "index.html",
            total=total,
            needs_review=needs_review,
            reviewed=reviewed,
            by_repo=by_repo,
            cat_dist=cat_dist,
            diff_dist=diff_dist,
            sev_dist=sev_dist,
            pipeline_statuses=pipeline_statuses,
        )

    # ------------------------------------------------------------------
    # Case list
    # ------------------------------------------------------------------

    @app.route("/cases")
    def case_list() -> str:
        c_dir: Path = app.config["CASES_DIR"]
        cases = load_all_cases(c_dir)

        # Filters
        f_repo = request.args.get("repo", "")
        f_cat = request.args.get("category", "")
        f_diff = request.args.get("difficulty", "")
        f_sev = request.args.get("severity", "")
        f_nmr = request.args.get("needs_manual_review", "")
        f_reviewed = request.args.get("reviewed", "")
        sort_by = request.args.get("sort", "id")
        page = max(1, int(request.args.get("page", 1)))
        per_page = 50

        # Build review set
        reviewed_ids: set[str] = set()
        for case in cases:
            if is_reviewed(c_dir, case.id, case.repo):
                reviewed_ids.add(case.id)

        def match(c: TestCase) -> bool:
            if f_repo and c.repo != f_repo:
                return False
            if f_cat and c.category.value != f_cat:
                return False
            if f_diff and c.difficulty.value != f_diff:
                return False
            if f_sev and c.severity.value != f_sev:
                return False
            if f_nmr == "true" and not c.needs_manual_review:
                return False
            if f_nmr == "false" and c.needs_manual_review:
                return False
            if f_reviewed == "true" and c.id not in reviewed_ids:
                return False
            if f_reviewed == "false" and c.id in reviewed_ids:
                return False
            return True

        filtered = [c for c in cases if match(c)]

        # Sort
        reverse = sort_by.startswith("-")
        sort_key = sort_by.lstrip("-")
        if sort_key in ("id", "repo", "category", "difficulty", "severity"):
            filtered.sort(key=lambda c: str(getattr(c, sort_key, "")), reverse=reverse)

        total_filtered = len(filtered)
        start = (page - 1) * per_page
        page_cases = filtered[start : start + per_page]

        # Next unreviewed
        next_unreviewed = next(
            (c.id for c in filtered if c.needs_manual_review and c.id not in reviewed_ids),
            None,
        )

        repos = sorted({c.repo for c in cases})
        filter_query = urlencode(
            {k: v for k, v in {
                "repo": f_repo, "category": f_cat, "difficulty": f_diff,
                "severity": f_sev, "needs_manual_review": f_nmr, "reviewed": f_reviewed,
            }.items() if v}
        )
        return render_template(
            "case_list.html",
            cases=page_cases,
            reviewed_ids=reviewed_ids,
            total=total_filtered,
            page=page,
            per_page=per_page,
            total_pages=(total_filtered + per_page - 1) // per_page,
            repos=repos,
            categories=[e.value for e in Category],
            difficulties=[e.value for e in Difficulty],
            severities=[e.value for e in Severity],
            filters={
                "repo": f_repo,
                "category": f_cat,
                "difficulty": f_diff,
                "severity": f_sev,
                "needs_manual_review": f_nmr,
                "reviewed": f_reviewed,
                "sort": sort_by,
            },
            filter_query=filter_query,
            next_unreviewed=next_unreviewed,
        )

    # ------------------------------------------------------------------
    # Case detail + edit
    # ------------------------------------------------------------------

    @app.route("/cases/<case_id>", methods=["GET", "POST"])
    def case_detail(case_id: str) -> Any:
        c_dir: Path = app.config["CASES_DIR"]
        yaml_path = _find_case_yaml(c_dir, case_id)
        if yaml_path is None:
            return f"Case {case_id} not found", 404

        data = yaml.safe_load(yaml_path.read_text()) or {}
        try:
            case = TestCase(**data)
        except ValidationError as exc:
            return f"Invalid case YAML: {exc}", 500

        reviewed = is_reviewed(c_dir, case_id, case.repo)

        if request.method == "POST":
            action = request.form.get("action", "save")

            if action == "accept":
                mark_reviewed(c_dir, case)
                return redirect(url_for("case_detail", case_id=case_id))

            if action == "skip":
                # Navigate to next
                cases = load_all_cases(c_dir)
                ids = [c.id for c in cases]
                idx = ids.index(case_id) if case_id in ids else -1
                next_id = ids[idx + 1] if idx + 1 < len(ids) else ids[0]
                return redirect(url_for("case_detail", case_id=next_id))

            # Save edits
            updated = _parse_case_form(request.form, case)
            save_case(c_dir, updated)
            if request.form.get("mark_reviewed"):
                mark_reviewed(c_dir, updated)
            return redirect(url_for("case_detail", case_id=case_id))

        # GET — fetch diff
        diff_text = fetch_diff(case.repo, case.head_commit)

        # Prev/next navigation
        cases = load_all_cases(c_dir)
        ids = [c.id for c in cases]
        idx = ids.index(case_id) if case_id in ids else -1
        prev_id = ids[idx - 1] if idx > 0 else None
        next_id = ids[idx + 1] if idx + 1 < len(ids) else None

        return render_template(
            "case_detail.html",
            case=case,
            reviewed=reviewed,
            diff_text=diff_text,
            prev_id=prev_id,
            next_id=next_id,
            categories=[e.value for e in Category],
            difficulties=[e.value for e in Difficulty],
            severities=[e.value for e in Severity],
            pr_sizes=[e.value for e in PRSize],
            visibilities=[e.value for e in Visibility],
        )

    # ------------------------------------------------------------------
    # Human judge calibration
    # ------------------------------------------------------------------

    @app.route("/human-judge")
    def human_judge_page() -> str:
        r_dir: Path = app.config["RESULTS_DIR"]
        run_id = request.args.get("run", "")

        runs = _run_dirs(r_dir)
        selected_run: Path | None = None
        if run_id:
            candidate = r_dir / run_id
            if candidate.exists():
                selected_run = candidate
        elif runs:
            selected_run = runs[-1]

        kappa_report: dict[str, Any] = {}
        scores: list[JudgeScore] = []
        normalized: dict[tuple[str, str], NormalizedResult] = {}
        tool_map: dict[str, str] = {}

        if selected_run is not None:
            kappa_report = compute_kappa_report(selected_run)
            scores = _load_scores(selected_run)
            normalized = _load_normalized(selected_run)

            tool_map_path = selected_run / "human_judge" / "tool_map.yaml"
            if tool_map_path.exists():
                tool_map = yaml.safe_load(tool_map_path.read_text()) or {}

        _sc = default_scoring()
        _badge_classes = {0: "badge-red", 1: "badge-yellow", 2: "badge-blue", 3: "badge-green"}
        return render_template(
            "human_judge.html",
            runs=[d.name for d in runs],
            selected_run=selected_run.name if selected_run else "",
            kappa_report=kappa_report,
            scores=scores,
            normalized=normalized,
            tool_map=tool_map,
            scoring_scale=_sc.scale,
            scoring_badge_classes=_badge_classes,
        )

    @app.route("/human-judge/score", methods=["POST"])
    def human_judge_score() -> Any:
        """Save a single human score directly to the run's human_judge/ dir."""
        r_dir: Path = app.config["RESULTS_DIR"]
        run_id = request.form.get("run_id", "")
        case_id = request.form.get("case_id", "")
        tool = request.form.get("tool", "")
        human_score_str = request.form.get("human_score", "")
        notes = request.form.get("notes", "")

        if not (run_id and case_id and tool and human_score_str):
            return "Missing fields", 400

        try:
            human_score = int(human_score_str)
        except ValueError:
            return "Invalid score", 400

        if human_score not in default_scoring().scale:
            return "Score out of range", 400

        run_dir = r_dir / run_id
        if not run_dir.exists():
            return "Run not found", 404

        hj_dir = run_dir / "human_judge"
        hj_dir.mkdir(exist_ok=True)

        # Load LLM score for kappa
        scores = _load_scores(run_dir)
        llm_score = 0
        for s in scores:
            if s.test_case_id == case_id and s.tool == tool:
                llm_score = s.score
                break

        out: dict[str, Any] = {
            "test_case_id": case_id,
            "tool": tool,
            "human_score": human_score,
            "llm_score": llm_score,
            "notes": notes,
        }
        safe_case = case_id.replace("/", "_")
        safe_tool = tool.replace("/", "_")
        out_path = hj_dir / f"{safe_case}-{safe_tool}.yaml"
        out_path.write_text(yaml.safe_dump(out, sort_keys=False))

        return redirect(url_for("human_judge_page", run=run_id))

    # ------------------------------------------------------------------
    # DxAssessment entry
    # ------------------------------------------------------------------

    @app.route("/dx")
    def dx_page() -> str:
        r_dir: Path = app.config["RESULTS_DIR"]
        run_id = request.args.get("run", "")

        runs = _run_dirs(r_dir)
        selected_run: Path | None = None
        if run_id:
            candidate = r_dir / run_id
            if candidate.exists():
                selected_run = candidate
        elif runs:
            selected_run = runs[-1]

        tools: list[str] = []
        dx_by_tool: dict[str, DxAssessment | None] = {}

        if selected_run is not None:
            normalized = _load_normalized(selected_run)
            tools = sorted({tool for (_, tool) in normalized})
            for tool in tools:
                # Find first result for tool with dx data
                dx_val: DxAssessment | None = None
                for (_, t), r in normalized.items():
                    if t == tool and r.dx is not None:
                        dx_val = r.dx
                        break
                dx_by_tool[tool] = dx_val

        return render_template(
            "dx_assessment.html",
            runs=[d.name for d in runs],
            selected_run=selected_run.name if selected_run else "",
            tools=tools,
            dx_by_tool=dx_by_tool,
        )

    @app.route("/dx/save", methods=["POST"])
    def dx_save() -> Any:
        r_dir: Path = app.config["RESULTS_DIR"]
        run_id = request.form.get("run_id", "")
        tool = request.form.get("tool", "")

        if not (run_id and tool):
            return "Missing fields", 400

        run_dir = r_dir / run_id
        if not run_dir.exists():
            return "Run not found", 404

        def _int_field(name: str) -> int:
            try:
                val = int(request.form.get(name, "3"))
                return max(1, min(5, val))
            except ValueError:
                return 3

        dx = DxAssessment(
            actionability=_int_field("actionability"),
            false_positive_burden=_int_field("false_positive_burden"),
            integration_friction=_int_field("integration_friction"),
            response_latency=_int_field("response_latency"),
            notes=request.form.get("notes", ""),
        )

        # Save to all NormalizedResult YAMLs for this tool
        for path in run_dir.glob("*.yaml"):
            if path.name == "checkpoint.yaml":
                continue
            data = yaml.safe_load(path.read_text()) or {}
            if data.get("tool") == tool:
                data["dx"] = dx.model_dump(mode="json")
                path.write_text(yaml.safe_dump(data, sort_keys=False))

        return redirect(url_for("dx_page", run=run_id))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @app.route("/metrics")
    def metrics_list() -> str:
        r_dir: Path = app.config["RESULTS_DIR"]
        runs = _run_dirs(r_dir)
        _sc = default_scoring()
        return render_template(
            "metrics.html",
            runs=[d.name for d in runs],
            run_data=None,
            scoring_scale=_sc.scale,
            scoring_labels=_sc.labels,
        )

    @app.route("/metrics/<run_id>")
    def metrics_detail(run_id: str) -> Any:
        r_dir: Path = app.config["RESULTS_DIR"]
        run_dir = r_dir / run_id
        if not run_dir.exists():
            return f"Run {run_id} not found", 404

        scores = _load_scores(run_dir)
        normalized = _load_normalized(run_dir)
        agg = aggregate_scores(scores) if scores else {}
        kappa_report = compute_kappa_report(run_dir)

        # Per-tool DX averages
        dx_summary = _compute_dx_summary(normalized)

        # Cost data
        _scoring = default_scoring()
        cost_by_tool: dict[str, dict[str, float]] = {}
        for tool in {s.tool for s in scores}:
            tool_scores = [s for s in scores if s.tool == tool]
            total_cost = sum(
                normalized[(s.test_case_id, s.tool)].metadata.cost_usd
                for s in tool_scores
                if (s.test_case_id, s.tool) in normalized
            )
            n = len(tool_scores)
            detections = sum(1 for s in tool_scores if s.score >= _scoring.catch_threshold)
            cost_by_tool[tool] = {
                "total": total_cost,
                "per_review": total_cost / n if n else 0.0,
                "per_detection": total_cost / detections if detections else 0.0,
            }

        runs = _run_dirs(r_dir)
        _sc = default_scoring()
        return render_template(
            "metrics.html",
            runs=[d.name for d in runs],
            selected_run=run_id,
            scoring_scale=_sc.scale,
            scoring_labels=_sc.labels,
            run_data={
                "agg": agg,
                "scores": scores,
                "kappa_report": kappa_report,
                "dx_summary": dx_summary,
                "cost_by_tool": cost_by_tool,
                "catch_rate": compute_catch_rate(scores),
                "avg_snr": compute_snr(scores),
            },
        )

    return app


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _count_field(cases: list[TestCase], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in cases:
        val = getattr(c, field)
        key = val.value if hasattr(val, "value") else str(val)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _parse_case_form(form: Any, original: TestCase) -> TestCase:
    """Parse POST form data into an updated TestCase."""
    # Collect expected findings from indexed form fields
    findings: list[ExpectedFinding] = []
    i = 0
    while True:
        f_file = form.get(f"ef_file_{i}", "").strip()
        f_line = form.get(f"ef_line_{i}", "").strip()
        f_summary = form.get(f"ef_summary_{i}", "").strip()
        if not (f_file or f_summary):
            break
        try:
            findings.append(
                ExpectedFinding(file=f_file, line=int(f_line or "0"), summary=f_summary)
            )
        except (ValueError, TypeError):
            pass
        i += 1

    return TestCase(
        id=original.id,
        repo=original.repo,
        base_commit=form.get("base_commit", original.base_commit).strip(),
        head_commit=form.get("head_commit", original.head_commit).strip(),
        fix_commit=original.fix_commit,
        category=Category(form.get("category", original.category.value)),
        difficulty=Difficulty(form.get("difficulty", original.difficulty.value)),
        severity=Severity(form.get("severity", original.severity.value)),
        language=original.language,
        pr_size=PRSize(form.get("pr_size", original.pr_size.value)),
        description=form.get("description", original.description).strip(),
        expected_findings=findings if findings else original.expected_findings,
        stats=original.stats,
        visibility=Visibility(form.get("visibility", original.visibility.value)),
        needs_manual_review="needs_manual_review" in form,
        verified=original.verified,
        verified_by=original.verified_by,
    )


def _compute_dx_summary(
    normalized: dict[tuple[str, str], NormalizedResult],
) -> dict[str, dict[str, float]]:
    dims = ("actionability", "false_positive_burden", "integration_friction", "response_latency")
    by_tool: dict[str, list[DxAssessment]] = {}
    for (_, tool), r in normalized.items():
        if r.dx is not None:
            by_tool.setdefault(tool, []).append(r.dx)
    result: dict[str, dict[str, float]] = {}
    for tool, dxs in sorted(by_tool.items()):
        result[tool] = {dim: sum(getattr(d, dim) for d in dxs) / len(dxs) for dim in dims}
    return result


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------

@click.command("dashboard")
@click.option("--port", default=5000, show_default=True, help="Port to listen on")
@click.option(
    "--cases-dir",
    default="cases/final",
    show_default=True,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Directory containing curated case YAML files",
)
@click.option(
    "--results-dir",
    default="results",
    show_default=True,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Root directory for run outputs",
)
@click.option("--debug", is_flag=True, default=False, help="Enable Flask debug mode")
def dashboard(port: int, cases_dir: str, results_dir: str, debug: bool) -> None:
    """Launch the local review dashboard."""
    app = create_app(Path(cases_dir), Path(results_dir))
    click.echo(f"Dashboard → http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=debug)
