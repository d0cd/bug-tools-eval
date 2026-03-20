"""Local Flask dashboard for experiment management (v2)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import click
import yaml
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from pydantic import ValidationError

from bugeval.add_case import add_case_from_pr
from bugeval.analyze import (
    build_comparison_table,
    compute_catch_rate,
    false_alarm_rate,
    load_scores,
    signal_to_noise,
)
from bugeval.dashboard_models import (
    Experiment,
    add_run_note,
    current_date_iso,
    load_experiments,
    load_golden_set,
    load_run_notes,
    save_experiments,
    set_golden_status,
    slugify,
)
from bugeval.io import load_cases
from bugeval.models import CaseKind, TestCase
from bugeval.result_models import ToolResult
from bugeval.score_models import CaseScore

# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------

_TTL_SECONDS = 300


class _CachedResult:
    def __init__(self, value: Any, expires: float):
        self.value = value
        self.expires = expires

    @property
    def valid(self) -> bool:
        return time.monotonic() < self.expires


_cache: dict[str, _CachedResult] = {}


def _cached(key: str, loader: Any) -> Any:
    entry = _cache.get(key)
    if entry and entry.valid:
        return entry.value
    value = loader()
    _cache[key] = _CachedResult(value, time.monotonic() + _TTL_SECONDS)
    return value


def _invalidate_cache(prefix: str = "") -> None:
    keys = [k for k in _cache if k.startswith(prefix)] if prefix else list(_cache)
    for k in keys:
        del _cache[k]


# ---------------------------------------------------------------------------
# Case YAML helpers
# ---------------------------------------------------------------------------


def _find_case_yaml(cases_dir: Path, case_id: str) -> Path | None:
    for repo_dir in cases_dir.iterdir():
        if not repo_dir.is_dir():
            continue
        candidate = repo_dir / f"{case_id}.yaml"
        if candidate.exists():
            return candidate
    return None


def load_all_cases(cases_dir: Path) -> list[TestCase]:
    """Load all TestCase YAMLs from the cases directory tree."""
    if not cases_dir.exists():
        return []
    return load_cases(cases_dir)


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def _run_dirs(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run-")]
    )


def _run_summary(run_dir: Path) -> dict[str, Any]:
    results_dir = run_dir / "results"
    results_count = (
        len(list(results_dir.glob("*.yaml"))) if results_dir.exists() else 0
    )
    scores_dir = run_dir / "scores"
    scores_count = (
        len(list(scores_dir.glob("*.yaml"))) if scores_dir.exists() else 0
    )
    has_charts = (run_dir / "charts").exists()
    return {
        "name": run_dir.name,
        "results_count": results_count,
        "scores_count": scores_count,
        "has_charts": has_charts,
    }


# ---------------------------------------------------------------------------
# Score/result loading
# ---------------------------------------------------------------------------


def _load_run_scores(run_dir: Path) -> list[CaseScore]:
    return _cached(f"scores:{run_dir}", lambda: _load_scores_uncached(run_dir))


def _load_scores_uncached(run_dir: Path) -> list[CaseScore]:
    scores_dir = run_dir / "scores"
    if not scores_dir.exists():
        return []
    return load_scores(scores_dir)


def _load_run_results(run_dir: Path) -> list[ToolResult]:
    return _cached(f"results:{run_dir}", lambda: _load_results_uncached(run_dir))


def _load_results_uncached(run_dir: Path) -> list[ToolResult]:
    results_dir = run_dir / "results"
    if not results_dir.exists():
        return []
    from bugeval.io import load_result

    results: list[ToolResult] = []
    for p in sorted(results_dir.glob("*.yaml")):
        try:
            results.append(load_result(p))
        except (ValidationError, TypeError, yaml.YAMLError):
            pass
    return results


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------


def create_app(cases_dir: Path, results_dir: Path) -> Flask:
    """Create and configure the Flask dashboard app."""
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["CASES_DIR"] = cases_dir
    app.config["RESULTS_DIR"] = results_dir

    # ------------------------------------------------------------------
    # Home
    # ------------------------------------------------------------------

    @app.route("/")
    def index() -> str:
        c_dir: Path = app.config["CASES_DIR"]
        cases: list[TestCase] = _cached(f"cases:{c_dir}", lambda: load_all_cases(c_dir))
        total = len(cases)
        bug_count = sum(1 for c in cases if c.kind == CaseKind.bug)
        clean_count = sum(1 for c in cases if c.kind == CaseKind.clean)
        by_repo = _count_field(cases, "repo")

        # Blame confidence tiers (bug cases only)
        by_blame: dict[str, int] = {}
        for c in cases:
            if c.truth and c.truth.blame_confidence:
                tier = c.truth.blame_confidence
                by_blame[tier] = by_blame.get(tier, 0) + 1
        by_blame = dict(sorted(by_blame.items()))

        # Validation status
        by_val: dict[str, int] = {}
        for c in cases:
            if c.validation:
                status = "agreed" if c.validation.agreement else "disagreed"
                by_val[status] = by_val.get(status, 0) + 1
            else:
                by_val["unvalidated"] = by_val.get("unvalidated", 0) + 1
        by_val = dict(sorted(by_val.items()))

        return render_template(
            "index.html",
            total=total,
            bug_count=bug_count,
            clean_count=clean_count,
            by_repo=by_repo,
            by_blame_confidence=by_blame,
            by_validation=by_val,
        )

    # ------------------------------------------------------------------
    # Experiments API
    # ------------------------------------------------------------------

    @app.route("/api/experiments")
    def api_experiments() -> Any:
        r_dir: Path = app.config["RESULTS_DIR"]
        store = load_experiments(r_dir)
        assigned: set[str] = set()
        for exp in store.experiments:
            assigned.update(exp.runs)

        run_dir_list = _run_dirs(r_dir)
        summaries = {d.name: _run_summary(d) for d in run_dir_list}

        experiments_out = []
        for exp in store.experiments:
            runs_data = [summaries[r] for r in exp.runs if r in summaries]
            experiments_out.append({
                "id": exp.id,
                "name": exp.name,
                "status": exp.status,
                "notes": exp.notes,
                "created": exp.created,
                "runs": runs_data,
            })

        ungrouped = [
            summaries[d.name] for d in run_dir_list if d.name not in assigned
        ]

        return jsonify({"experiments": experiments_out, "ungrouped": ungrouped})

    @app.route("/api/experiments", methods=["POST"])
    def api_create_experiment() -> Any:
        r_dir: Path = app.config["RESULTS_DIR"]
        data = request.get_json(silent=True) or {}
        name = data.get("name", "").strip()
        if not name:
            return jsonify({"error": "name is required"}), 400

        exp_id = slugify(name)
        if not exp_id:
            return jsonify({"error": "invalid name"}), 400

        store = load_experiments(r_dir)
        if any(e.id == exp_id for e in store.experiments):
            return jsonify({"error": "experiment already exists"}), 409

        exp = Experiment(
            id=exp_id,
            name=name,
            runs=data.get("runs", []),
            notes=data.get("notes", ""),
            created=current_date_iso(),
        )
        store.experiments.append(exp)
        save_experiments(r_dir, store)
        return jsonify(exp.model_dump(mode="json")), 201

    @app.route("/api/experiments/<exp_id>", methods=["PUT"])
    def api_update_experiment(exp_id: str) -> Any:
        r_dir: Path = app.config["RESULTS_DIR"]
        store = load_experiments(r_dir)
        exp = next((e for e in store.experiments if e.id == exp_id), None)
        if exp is None:
            return jsonify({"error": "not found"}), 404

        data = request.get_json(silent=True) or {}
        if "name" in data:
            exp.name = data["name"]
        if "runs" in data:
            exp.runs = data["runs"]
        if "notes" in data:
            exp.notes = data["notes"]
        if "status" in data:
            exp.status = data["status"]
        save_experiments(r_dir, store)
        return jsonify(exp.model_dump(mode="json"))

    @app.route("/api/experiments/<exp_id>/archive", methods=["POST"])
    def api_archive_experiment(exp_id: str) -> Any:
        r_dir: Path = app.config["RESULTS_DIR"]
        store = load_experiments(r_dir)
        exp = next((e for e in store.experiments if e.id == exp_id), None)
        if exp is None:
            return jsonify({"error": "not found"}), 404

        exp.status = "active" if exp.status == "archived" else "archived"
        save_experiments(r_dir, store)
        return jsonify(exp.model_dump(mode="json"))

    # ------------------------------------------------------------------
    # Runs list + detail
    # ------------------------------------------------------------------

    @app.route("/runs")
    def runs_page() -> str:
        r_dir: Path = app.config["RESULTS_DIR"]
        runs = [_run_summary(d) for d in _run_dirs(r_dir)]
        return render_template("runs.html", runs=runs)

    @app.route("/runs/<run_id>")
    def run_detail(run_id: str) -> Any:
        r_dir: Path = app.config["RESULTS_DIR"]
        run_dir = r_dir / run_id
        if not run_dir.exists():
            return f"Run {run_id} not found", 404

        status = _run_summary(run_dir)
        notes = load_run_notes(run_dir)
        return render_template(
            "run_detail.html",
            run_id=run_id,
            status=status,
            notes=notes,
        )

    @app.route("/runs/<run_id>/notes", methods=["POST"])
    def add_run_note_route(run_id: str) -> Any:
        r_dir: Path = app.config["RESULTS_DIR"]
        run_dir = r_dir / run_id
        if not run_dir.exists():
            return f"Run {run_id} not found", 404
        text = request.form.get("text", "").strip()
        if text:
            add_run_note(run_dir, text)
        return redirect(url_for("run_detail", run_id=run_id))

    # ------------------------------------------------------------------
    # Cases API — paginated, filtered JSON
    # ------------------------------------------------------------------

    @app.route("/api/cases")
    def api_cases() -> Any:
        c_dir: Path = app.config["CASES_DIR"]
        cases: list[TestCase] = _cached(
            f"cases:{c_dir}", lambda: load_all_cases(c_dir)
        )

        f_repo = request.args.get("repo", "")
        f_kind = request.args.get("kind", "")
        f_cat = request.args.get("category", "")
        f_diff = request.args.get("difficulty", "")
        f_blame = request.args.get("blame_confidence", "")
        f_validated = request.args.get("validated", "")
        q = request.args.get("q", "").strip().lower()
        sort_by = request.args.get("sort", "id")
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(max(1, int(request.args.get("per_page", 50))), 200)

        def _match(c: TestCase) -> bool:
            if f_repo and c.repo != f_repo:
                return False
            if f_kind and c.kind.value != f_kind:
                return False
            if f_cat and c.category != f_cat:
                return False
            if f_diff and c.difficulty != f_diff:
                return False
            if f_blame:
                bc = c.truth.blame_confidence if c.truth else ""
                if bc != f_blame:
                    return False
            if f_validated == "true" and (
                c.validation is None or not c.validation.agreement
            ):
                return False
            if f_validated == "false" and (
                c.validation is not None and c.validation.agreement
            ):
                return False
            if q:
                haystack = (
                    c.id + " " + c.bug_description + " " + c.category
                ).lower()
                if q not in haystack:
                    return False
            return True

        filtered = [c for c in cases if _match(c)]

        # Sort
        reverse = sort_by.startswith("-")
        sort_key = sort_by.lstrip("-")
        if sort_key in ("id", "repo", "kind", "category", "difficulty"):
            filtered.sort(
                key=lambda c: str(getattr(c, sort_key, "")),
                reverse=reverse,
            )

        # Paginate
        total = len(filtered)
        total_pages = max(1, (total + per_page - 1) // per_page)
        start = (page - 1) * per_page
        page_cases = filtered[start : start + per_page]

        items = []
        for c in page_cases:
            bc = c.truth.blame_confidence if c.truth else ""
            val_status = ""
            if c.validation:
                val_status = "agreed" if c.validation.agreement else "disagreed"
            items.append({
                "id": c.id,
                "repo": c.repo,
                "kind": c.kind.value,
                "category": c.category,
                "difficulty": c.difficulty,
                "severity": c.severity,
                "blame_confidence": bc,
                "validation_status": val_status,
                "language": c.language,
                "bug_description": c.bug_description[:120],
            })

        repos = sorted({c.repo for c in cases})
        kinds = [e.value for e in CaseKind]

        return jsonify(
            cases=items,
            total=total,
            page=page,
            pages=total_pages,
            filters={"repos": repos, "kinds": kinds},
        )

    # ------------------------------------------------------------------
    # Case list (HTML shell)
    # ------------------------------------------------------------------

    @app.route("/cases")
    def case_list() -> str:
        return render_template("case_list.html")

    # ------------------------------------------------------------------
    # Case detail
    # ------------------------------------------------------------------

    @app.route("/cases/<case_id>")
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

        # Prev/next
        all_cases: list[TestCase] = _cached(
            f"cases:{c_dir}", lambda: load_all_cases(c_dir)
        )
        ids = [c.id for c in all_cases]
        idx = ids.index(case_id) if case_id in ids else -1
        prev_id = ids[idx - 1] if idx > 0 else None
        next_id = ids[idx + 1] if idx + 1 < len(ids) else None

        return render_template(
            "case_detail.html",
            case=case,
            prev_id=prev_id,
            next_id=next_id,
        )

    # ------------------------------------------------------------------
    # Golden set
    # ------------------------------------------------------------------

    @app.route("/golden")
    def golden_page() -> str:
        c_dir: Path = app.config["CASES_DIR"]
        cases: list[TestCase] = _cached(
            f"cases:{c_dir}", lambda: load_all_cases(c_dir)
        )
        golden = load_golden_set(c_dir)

        filter_status = request.args.get("status", "")
        filter_repo = request.args.get("repo", "")
        page = max(1, int(request.args.get("page", 1)))
        per_page = 50

        items: list[dict[str, str]] = []
        for c in cases:
            entry = golden.get(c.id)
            g_status = entry.status if entry else "unreviewed"
            reviewer = entry.reviewer if entry else ""
            if filter_status and g_status != filter_status:
                continue
            if filter_repo and c.repo != filter_repo:
                continue
            items.append({
                "case_id": c.id,
                "repo": c.repo,
                "kind": c.kind.value,
                "category": c.category,
                "severity": c.severity,
                "golden_status": g_status,
                "reviewer": reviewer,
            })

        total = len(items)
        total_pages = max(1, (total + per_page - 1) // per_page)
        start = (page - 1) * per_page
        page_items = items[start : start + per_page]

        confirmed = sum(1 for e in golden.values() if e.status == "confirmed")
        disputed = sum(1 for e in golden.values() if e.status == "disputed")
        unreviewed_count = len(cases) - confirmed - disputed
        repos = sorted({c.repo for c in cases})
        reviewed_ids = {
            e.case_id for e in golden.values() if e.status != "unreviewed"
        }
        next_unreviewed = next(
            (c.id for c in cases if c.id not in reviewed_ids), None
        )

        return render_template(
            "golden.html",
            items=page_items,
            total=len(cases),
            confirmed=confirmed,
            disputed=disputed,
            unreviewed=unreviewed_count,
            repos=repos,
            filter_status=filter_status,
            filter_repo=filter_repo,
            page=page,
            pages=total_pages,
            next_unreviewed=next_unreviewed,
        )

    @app.route("/golden/<case_id>", methods=["POST"])
    def golden_set_status(case_id: str) -> Any:
        c_dir: Path = app.config["CASES_DIR"]
        status = request.form.get("status", "")
        if status not in ("confirmed", "disputed", "unreviewed"):
            return "Invalid status", 400
        set_golden_status(c_dir, case_id, status, reviewer="dashboard")
        return redirect(url_for("golden_page"))

    # ------------------------------------------------------------------
    # Add case from PR URL
    # ------------------------------------------------------------------

    @app.route("/api/add-case", methods=["POST"])
    def api_add_case() -> Any:
        body = request.get_json(silent=True) or {}
        pr_url = body.get("pr_url", "").strip()
        if not pr_url:
            return jsonify({"error": "pr_url is required"}), 400

        cases_dir: Path = app.config["CASES_DIR"]
        repo_dir = app.config.get("REPO_DIR")

        try:
            case = add_case_from_pr(
                pr_url, cases_dir, repo_dir or Path()
            )
            if case is None:
                return (
                    jsonify({"error": "Duplicate: case with this PR already exists"}),
                    409,
                )
            _invalidate_cache("cases:")
            return jsonify({"case_id": case.id, "repo": case.repo}), 201
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @app.route("/metrics")
    def metrics_list() -> str:
        r_dir: Path = app.config["RESULTS_DIR"]
        runs = [d.name for d in _run_dirs(r_dir)]
        return render_template("metrics.html", runs=runs, run_data=None)

    @app.route("/metrics/<run_id>")
    def metrics_detail(run_id: str) -> Any:
        r_dir: Path = app.config["RESULTS_DIR"]
        c_dir: Path = app.config["CASES_DIR"]
        run_dir = r_dir / run_id
        if not run_dir.exists():
            return f"Run {run_id} not found", 404

        scores = _load_run_scores(run_dir)
        results = _load_run_results(run_dir)
        cases: list[TestCase] = _cached(
            f"cases:{c_dir}", lambda: load_all_cases(c_dir)
        )

        # Group by tool
        from collections import defaultdict

        scores_by_tool: dict[str, list[CaseScore]] = defaultdict(list)
        for s in scores:
            scores_by_tool[s.tool].append(s)
        results_by_tool: dict[str, list[ToolResult]] = defaultdict(list)
        for r in results:
            results_by_tool[r.tool].append(r)

        table = build_comparison_table(
            dict(scores_by_tool), dict(results_by_tool), cases
        )

        # Summary stats
        bug_scores = [
            s for s in scores
            if any(c.id == s.case_id and c.kind == CaseKind.bug for c in cases)
        ]
        cr = compute_catch_rate(bug_scores)
        far = false_alarm_rate(scores, cases)
        snr = signal_to_noise(scores)
        caught_count = sum(1 for s in bug_scores if s.caught)
        contaminated = sum(1 for s in scores if s.potentially_contaminated)
        judge_failures = sum(1 for s in scores if s.judge_failed)

        # Charts
        charts_dir = run_dir / "charts"
        has_catch_rate_chart = (charts_dir / "catch_rate.png").exists()
        has_detection_dist_chart = (charts_dir / "detection_dist.png").exists()

        runs = [d.name for d in _run_dirs(r_dir)]
        return render_template(
            "metrics.html",
            runs=runs,
            selected_run=run_id,
            run_data={
                "table": table,
                "catch_rate": round(cr, 4),
                "false_alarm_rate": round(far, 4),
                "snr": round(snr, 4),
                "total_scores": len(scores),
                "caught": caught_count,
                "contaminated": contaminated,
                "judge_failures": judge_failures,
            },
            has_catch_rate_chart=has_catch_rate_chart,
            has_detection_dist_chart=has_detection_dist_chart,
        )

    @app.route("/metrics/<run_id>/chart/<filename>")
    def serve_chart(run_id: str, filename: str) -> Any:
        allowed = {"catch_rate.png", "detection_dist.png"}
        if filename not in allowed:
            return "Not found", 404
        r_dir: Path = app.config["RESULTS_DIR"]
        chart_path = r_dir / run_id / "charts" / filename
        if not chart_path.exists():
            return "Not found", 404
        return send_file(str(chart_path), mimetype="image/png")

    # ------------------------------------------------------------------
    # Compare (placeholder page)
    # ------------------------------------------------------------------

    @app.route("/compare")
    def compare_page() -> str:
        r_dir: Path = app.config["RESULTS_DIR"]
        runs = [d.name for d in _run_dirs(r_dir)]
        return render_template("compare.html", runs=runs)

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


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


@click.command("dashboard")
@click.option("--port", default=5000, show_default=True, help="Port to listen on")
@click.option(
    "--cases-dir",
    default="cases",
    show_default=True,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Directory containing case YAML files",
)
@click.option(
    "--results-dir",
    default="results",
    show_default=True,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Root directory for run outputs",
)
@click.option("--debug", is_flag=True, default=False, help="Enable Flask debug mode")
def dashboard_cmd(
    port: int, cases_dir: str, results_dir: str, debug: bool
) -> None:
    """Launch the local review dashboard."""
    app = create_app(Path(cases_dir), Path(results_dir))
    click.echo(f"Dashboard -> http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=debug)
