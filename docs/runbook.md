# Runbook

## Quick Start

Full pipeline from dataset construction through analysis:

```bash
# 1. Dataset construction
bugeval mine --repo ProvableHQ/leo --output-dir cases/leo
bugeval blame --cases-dir cases/leo --repo-dir ./repos/leo
bugeval ground-truth --cases-dir cases/leo --repo-dir ./repos/leo
bugeval validate --cases-dir cases/leo --repo-dir ./repos/leo
bugeval clean-cases --repo ProvableHQ/leo --cases-dir cases/leo

# 2. Evaluation (PR tools need --org for forks)
bugeval evaluate --tool copilot --cases-dir cases/leo --run-dir results/run-001 --repo-dir ./repos/leo --org bug-finder-eval
bugeval evaluate --tool greptile --cases-dir cases/leo --run-dir results/run-001 --repo-dir ./repos/leo --org bug-finder-eval
bugeval evaluate --tool coderabbit --cases-dir cases/leo --run-dir results/run-001 --repo-dir ./repos/leo --org bug-finder-eval
bugeval evaluate --tool agent --cases-dir cases/leo --run-dir results/run-001 --repo-dir ./repos/leo --context diff+repo

# 3. Scoring and analysis
bugeval score --run-dir results/run-001 --cases-dir cases/leo
bugeval analyze --run-dir results/run-001 --cases-dir cases/leo
```

## GitHub Org Setup

The evaluation org is `bug-finder-eval`. For each target repo (leo, snarkOS, snarkVM, sdk), forks are created per tool under this org.

- `ensure_fork()` creates forks lazily during evaluation -- no manual setup needed
- Each PR tool (copilot, greptile, coderabbit) needs its GitHub App installed on the org's forks
- `--org bug-finder-eval` must be passed to `bugeval evaluate` for all PR-based tools

Fork naming follows GitHub defaults: `bug-finder-eval/leo`, `bug-finder-eval/snarkOS`, etc.

## Dashboard

`bugeval dashboard` launches a local Flask web UI for experiment management and dataset review.

| Page | URL | Purpose |
|------|-----|---------|
| Home | `/` | Dataset stats: by repo, by kind (bug/clean), by blame confidence tier, by validation status |
| Cases | `/cases` | Filterable/sortable case browser with v2 fields (kind, blame_confidence, validation) |
| Case Detail | `/cases/<id>` | Ground truth (buggy_lines), validation verdicts, PR relations, introducing PR metadata |
| Runs | `/runs` | Run list with result/score counts; experiment grouping |
| Run Detail | `/runs/<id>` | Per-run notes, links to metrics/scores |
| Golden Set | `/golden` | Case confirmation workflow: confirm/dispute with coverage stats |
| Metrics | `/metrics/<run>` | Catch rate, false alarm rate, SNR, contamination warnings, tool comparison table, charts |
| Compare | `/compare` | Run comparison |

State is stored in sidecar files (no database): experiments in `results/experiments.yaml`, run notes in `.notes.json`, golden set in `cases/.golden_set.json`, human scores in `run_dir/human_scores/`.

## CLI Reference

| Command | Purpose |
|---------|---------|
| `bugeval mine` | Scrape fix PRs from GitHub, build initial TestCase YAMLs |
| `bugeval blame` | Find introducing commits via git blame |
| `bugeval ground-truth` | Build ground truth via diff intersection |
| `bugeval validate` | Cross-model validation (Claude + Gemini) |
| `bugeval clean-cases` | Generate negative control cases (clean PRs) |
| `bugeval evaluate` | Run a tool against test cases (`--org` for PR tool forks) |
| `bugeval score` | Mechanical catch rate + LLM quality judge |
| `bugeval analyze` | Statistics, comparison tables, charts |
| `bugeval dashboard` | Local Flask web UI for experiment management |
| `bugeval add-case` | Manually add a case from a fix PR URL (mines, blames, builds ground truth) |
