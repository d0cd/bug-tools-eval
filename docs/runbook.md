# Experiment Runbook

Step-by-step guide to running the bug-tools-eval experiment from dataset construction through final analysis. See `docs/experiment-design.md` for the full design rationale.

---

## Prerequisites

```bash
uv sync
cp .env.example .env   # fill in API keys (see below)
uv run bugeval validate-env --cases-dir cases/final
```

Required env vars:
- `ANTHROPIC_API_KEY` — judging + Claude CLI/API tools
- `GITHUB_TOKEN` — fork management and PR scraping
- `GREPTILE_API_KEY` — Greptile API tool
- `GEMINI_API_KEY` — Gemini CLI and Google API tools
- `OPENAI_API_KEY` — Codex CLI and OpenAI API tools

**Current dataset:** 827 cases across 6 repos in `cases/final/`
(leo, sdk, snarkOS, snarkVM, grafana, keycloak)

---

## Phase 0 — GitHub Org Setup (one-time, manual)

These steps are done once and don't repeat between runs.

### 0a. Create the GitHub org

Create `bug-tools-eval` at github.com/organizations/new.

### 0b. Fork all evaluation repos

```bash
uv run bugeval manage-forks --action create --dry-run   # preview
uv run bugeval manage-forks --action create              # execute
```

This creates per-tool forks in the `bug-tools-eval` org:
```
bug-tools-eval/snarkVM-coderabbit
bug-tools-eval/snarkVM-bugbot
bug-tools-eval/snarkVM-augment-code
...
```

### 0c. Install GitHub Apps on the org

Install each tool's GitHub App on `bug-tools-eval`, scoped to only that tool's repos:

| Tool | App slug | Enabled repos |
|------|----------|---------------|
| CodeRabbit | `coderabbit-ai` | `*-coderabbit` |
| BugBot | `linear-bugbot` | `*-bugbot` |
| Augment Code | `augment-code` | `*-augment-code` |
| DeepSource | `deepsource-io` | `*-deepsource` |
| Graphite Diamond | `graphite-app` | `*-graphite-diamond` |

### 0d. Install CLI tools locally

```bash
# Google Gemini CLI
npm install -g @google/gemini-cli   # or via brew

# OpenAI Codex CLI
npm install -g @openai/codex        # or via brew
```

Verify:
```bash
gemini --version
codex --version
```

### 0e. Build the Docker image (for Claude agent runs)

```bash
docker build -t bugeval-agent .
```

---

## Phase 1 — Dataset Construction (already complete for v1)

The `cases/final/` directory has 827 cases. Skip to Phase 2 unless adding new cases.

### 1a. Mine candidates from local repos

```bash
uv run bugeval mine-candidates \
  --repo-dir /path/to/snarkVM \
  --repo-name snarkVM \
  --min-confidence 0.4 \
  --output-dir candidates/
```

### 1b. Scrape GitHub for PR-based bugs

```bash
uv run bugeval scrape-github \
  --repo ProvableHQ/snarkVM \
  --output-dir candidates/
```

### 1c. Curate candidates into test cases

```bash
uv run bugeval curate \
  --candidates-file candidates/snarkVM.yaml \
  --output-dir cases/
```

### 1d. Validate cases

```bash
uv run bugeval validate-cases --cases-dir cases/ --dry-run
```

### 1e. Extract patches

```bash
uv run bugeval extract-patch \
  --cases-dir cases/ \
  --repo-dir /path/to/snarkVM \
  --output-dir patches/
```

### 1f. Tag the dataset

```bash
git add cases/ patches/
git commit -m "dataset: v2"
git tag dataset-v2
```

---

## Phase 2 — Pilot Run (recommended before full run)

Run ~20 cases with 1 PR tool + 1 agent tool to verify the full pipeline end-to-end.

```bash
RUN=results/run-$(date +%Y-%m-%d)-pilot

# Pick 20 cases (e.g. from snarkVM)
uv run bugeval run-pr-eval \
  --cases-dir cases/final \
  --tools coderabbit \
  --limit 20 \
  --run-dir $RUN/pr

uv run bugeval run-agent-eval \
  --cases-dir cases/final \
  --tools claude-cli-sonnet \
  --context-level diff-only \
  --limit 20 \
  --run-dir $RUN/agent

uv run bugeval pipeline \
  --run-dir $RUN/pr \
  --cases-dir cases/final

uv run bugeval pipeline \
  --run-dir $RUN/agent \
  --cases-dir cases/final
```

Review results in the dashboard:
```bash
uv run bugeval dashboard --run-dir $RUN/agent
# Opens at http://localhost:5000
```

---

## Phase 3 — Full Evaluation Runs

### 3a. PR tools (commercial)

```bash
uv run bugeval run-pr-eval \
  --cases-dir cases/final \
  --patches-dir patches/ \
  --run-dir results/run-$(date +%Y-%m-%d)-pr
```

Check progress:
```bash
uv run bugeval status --run-dir results/run-<date>-pr
```

### 3b. API tools (Greptile)

```bash
uv run bugeval run-api-eval \
  --cases-dir cases/final \
  --patches-dir patches/ \
  --context-level diff-only \
  --run-dir results/run-$(date +%Y-%m-%d)-api
```

### 3c. Agent tools — all tiers × 3 context levels

```bash
TOOLS="claude-cli-haiku,claude-cli-sonnet,gemini-cli-flash-lite,gemini-cli-flash,codex-cli-mini,codex-cli-o4,google-api-flash-lite,google-api-flash,openai-api-mini,openai-api-o4"

for level in diff-only diff+repo diff+repo+domain; do
  uv run bugeval run-agent-eval \
    --cases-dir cases/final \
    --patches-dir patches/ \
    --context-level $level \
    --tools $TOOLS \
    --use-docker \
    --docker-image bugeval-agent \
    --require-docker \
    --run-dir results/run-$(date +%Y-%m-%d)-agent-$level
done
```

Runs resume automatically from `checkpoint.yaml` if interrupted.

---

## Phase 4 — Post-Processing

### Run the full pipeline in one shot

```bash
uv run bugeval pipeline \
  --run-dir results/run-<date> \
  --cases-dir cases/final
```

Or run stages individually:

```bash
uv run bugeval normalize --run-dir results/run-<date>
uv run bugeval judge --run-dir results/run-<date> --cases-dir cases/final
uv run bugeval analyze --run-dir results/run-<date> --cases-dir cases/final
```

Results appear in: `results/run-<date>/analysis/report.md`

---

## Phase 5 — Human Calibration

Target: Cohen's κ ≥ 0.85 on a 25% random sample (~207 cases).

```bash
# Export blinded sample
uv run bugeval human-judge export \
  --run-dir results/run-<date> \
  --output human_judge_sample.csv

# After raters fill in human_score column:
uv run bugeval human-judge import-scores \
  --run-dir results/run-<date> \
  --input human_judge_sample_filled.csv

# Compute kappa — must be >= 0.85
uv run bugeval human-judge kappa \
  --run-dir results/run-<date>
```

If κ < 0.85: revise `config/judge_prompt.md`, re-run judging, re-calibrate.

View calibration status in the dashboard at `/metrics/<run>`.

---

## Phase 6 — DX Assessment (optional)

Score each tool on actionability, false-positive burden, integration friction, and latency.

```bash
uv run bugeval dashboard --run-dir results/run-<date>
# Navigate to /dx?run=<run-name> and enter scores
```

---

## Tool Reference

| Tool name | Type | Model / endpoint |
|-----------|------|-----------------|
| `coderabbit` | PR | coderabbit-ai app |
| `bugbot` | PR | linear-bugbot app |
| `augment-code` | PR | augment-code app |
| `deepsource` | PR | deepsource-io app |
| `graphite-diamond` | PR | graphite-app |
| `greptile` | API | greptile.com API |
| `claude-cli-haiku` | agent | claude-haiku-4-5 |
| `claude-cli-sonnet` | agent | claude-sonnet-4-6 |
| `gemini-cli-flash-lite` | agent | gemini-2.5-flash-lite |
| `gemini-cli-flash` | agent | gemini-2.5-flash |
| `codex-cli-mini` | agent | gpt-4.1-mini |
| `codex-cli-o4` | agent | o4-mini |
| `google-api-flash-lite` | agent | gemini-2.5-flash-lite (SDK) |
| `google-api-flash` | agent | gemini-2.5-flash (SDK) |
| `openai-api-mini` | agent | gpt-4.1-mini (SDK) |
| `openai-api-o4` | agent | o4-mini (SDK) |

---

## Isolation Guarantees

- `cases/` and `patches/` are immutable during a run — never edit mid-run
- `results/` is gitignored — never commit outputs
- PR tool forks are independent per tool and per repo
- Each run has its own `results/run-<date>/` directory
- Runs resume automatically from `checkpoint.yaml` if interrupted
- Docker isolation for Claude CLI agent (`--use-docker --require-docker`)

---

## Known Gaps

- **PR tool cost tracking:** Commercial tool costs are not captured automatically. Record manually in a cost log alongside each run.
- **Kappa threshold:** The 0.85 threshold is hardcoded in `human_judge.py`. Adjust there if the experiment design changes.
- **Gemini/Codex CLI flags:** Verify CLI flag syntax with `gemini --help` and `codex --help` before the first run — flag names may shift between CLI versions.
