# Experiment Runbook

Step-by-step guide to running the bug-tools-eval experiment from dataset construction through final analysis. See `docs/experiment-design.md` for the full design rationale.

---

## Prerequisites

```bash
uv sync
cp .env.example .env   # then fill in API keys
uv run bugeval validate-env --cases-dir cases/
```

Required env vars:
- `ANTHROPIC_API_KEY` — for judging and in-house agent
- `GITHUB_TOKEN` — for fork management and PR scraping
- `GREPTILE_API_KEY` — for Greptile API tool

---

## Phase 1 — Dataset Construction

### 1a. Mine candidates from local repos

```bash
uv run bugeval mine-candidates \
  --repo-dir /path/to/aleo-lang \
  --repo-name aleo-lang \
  --min-confidence 0.4 \
  --output-dir candidates/
```

### 1b. Scrape GitHub for PR-based bugs

```bash
uv run bugeval scrape-github \
  --repo provable-org/aleo-lang \
  --output-dir candidates/
```

### 1c. Curate candidates into test cases

```bash
uv run bugeval curate \
  --candidates-file candidates/aleo-lang.yaml \
  --output-dir cases/
```

### 1d. Validate cases

```bash
uv run bugeval validate-cases \
  --cases-dir cases/ \
  --repo-dir /path/to/aleo-lang
```

### 1e. Extract patches

```bash
uv run bugeval extract-patch \
  --cases-dir cases/ \
  --repo-dir /path/to/aleo-lang \
  --output-dir patches/
```

### 1f. Tag the dataset

```bash
git add cases/ patches/
git commit -m "dataset: add test cases"
git tag dataset-v1
```

---

## Phase 2 — Environment Setup

### Set up GitHub forks for PR tools

```bash
uv run bugeval manage-forks create \
  --config config/config.yaml \
  --org <eval-org>
```

Edit `config/config.yaml` to fill in:
- `github.eval_org`
- `repos` section (name → org/repo)
- Per-tool `org` fields

---

## Phase 3 — Evaluation Runs

### Run PR-mode tools

```bash
uv run bugeval run-pr-eval \
  --config config/config.yaml \
  --cases-dir cases/ \
  --patches-dir patches/ \
  --run-dir results/run-$(date +%Y-%m-%d)-pr
```

### Run API-mode tools (e.g. Greptile)

```bash
uv run bugeval run-api-eval \
  --config config/config.yaml \
  --cases-dir cases/ \
  --patches-dir patches/ \
  --context-level diff-only \
  --run-dir results/run-$(date +%Y-%m-%d)-api
```

### Run in-house agents (3 context levels)

```bash
for level in diff-only diff+repo diff+repo+domain; do
  uv run bugeval run-agent-eval \
    --config config/config.yaml \
    --cases-dir cases/ \
    --patches-dir patches/ \
    --context-level $level \
    --run-dir results/run-$(date +%Y-%m-%d)-agent-$level
done
```

Check progress at any time:
```bash
uv run bugeval status --run-dir results/run-<date>
```

---

## Phase 4 — Post-Processing

### Run the full pipeline (normalize → judge → analyze) in one shot

```bash
uv run bugeval pipeline \
  --run-dir results/run-<date> \
  --cases-dir cases/ \
  --config config/config.yaml
```

Or run stages individually:

```bash
uv run bugeval normalize --run-dir results/run-<date>
uv run bugeval judge --run-dir results/run-<date> --cases-dir cases/
uv run bugeval analyze --run-dir results/run-<date> --cases-dir cases/
```

Results in: `results/run-<date>/analysis/report.md`

---

## Phase 5 — Human Judge Calibration

```bash
# Export blinded sample for human raters
uv run bugeval human-judge export \
  --run-dir results/run-<date> \
  --output human_judge_sample.csv

# After raters fill in human_score column:
uv run bugeval human-judge import-scores \
  --run-dir results/run-<date> \
  --input human_judge_sample_filled.csv

# Check kappa — must be >= 0.85 to proceed
uv run bugeval human-judge kappa \
  --run-dir results/run-<date>
```

If kappa < 0.85: adjust `config/judge_prompt.md`, re-run judging, re-calibrate.

---

## Isolation Guarantees

- `cases/` and `patches/` are immutable during a run — never edit mid-run
- `results/` is gitignored — never commit outputs
- PR tool forks are independent per tool
- Each run has its own `results/run-<date>/` directory
- Runs resume from `checkpoint.yaml` if interrupted

---

## Known Gaps

- **Docker isolation for agent mode:** The agent currently clones to a local temp dir (no container). Use `--require-docker` to enforce Docker availability or run in a CI environment with Docker.
- **PR tool cost tracking:** Commercial PR tool costs are not captured automatically (no API to query). Record manually in a separate cost log.
- **Kappa threshold configurability:** The 0.85 kappa threshold is hardcoded in `human_judge.py`. Adjust it there if the experiment design changes.
