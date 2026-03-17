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
- `ANTHROPIC_API_KEY` — judging + Claude CLI/API/SDK tools
- `GITHUB_TOKEN` — fork management and PR scraping
- `GREPTILE_API_KEY` — Greptile API tool (when running commercial eval)
- `GEMINI_API_KEY` — Gemini CLI and Google API tools (when running Gemini eval)
- `OPENAI_API_KEY` — Codex CLI and OpenAI API tools (when running OpenAI eval)

**Current dataset:** 1,271 cases across 9 repos in `cases/final/`

| Repo | Cases | Language |
|------|-------|----------|
| leo | 253 | Rust |
| sentry | 191 | Python |
| snarkOS | 187 | Rust |
| snarkVM | 186 | Rust |
| grafana | 122 | Go |
| discourse | 119 | Ruby |
| keycloak | 80 | Java |
| cal.com | 77 | TypeScript |
| sdk | 56 | Rust |
| **Total** | **1,271** | |

> **Patch extraction for public repos** (sentry, cal.com, discourse, grafana, keycloak) requires bare clones.
> See Phase 1e below.

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

## Phase 1 — Dataset Construction (already complete for v2)

The `cases/final/` directory has 1,271 cases across 9 repos. Skip to Phase 2 unless adding new cases.

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

LLM-assisted enrichment (claude-opus-4-6 + adaptive thinking). Resumes automatically from checkpoint on re-run.

```bash
uv run bugeval curate \
  --candidates candidates/snarkVM.yaml \
  --output-dir cases/final/snarkVM/
```

Useful flags:
- `--limit N` — process only N candidates per run (safe batching for large repos)
- `--fail-after N` — abort after N consecutive errors (default: 5)
- `--shard K/N` — split work across parallel processes (use separate `--output-dir` per shard)
- `--dry-run` — preview prompts without calling the API
- `--no-checkpoint` — re-process all candidates, ignoring prior checkpoint

**For public repos** (sentry, cal.com, discourse, grafana, keycloak), candidates are scraped directly from GitHub (step 1b). These repos don't require a local checkout for curation.

### 1d. Validate cases

```bash
uv run bugeval validate-cases --cases-dir cases/ --dry-run
```

### 1e. Extract patches

**Private repos** (leo, snarkVM, snarkOS, sdk) — use a full or bare clone:

```bash
uv run bugeval extract-patch --all \
  --cases-dir cases/final/snarkVM/ \
  --repo-dir /path/to/snarkVM
```

**Public repos** (sentry, cal.com, discourse, grafana, keycloak) — create bare clones first (faster, no working tree):

```bash
# One-time bare clone (only needed if not already cloned)
git clone --bare https://github.com/getsentry/sentry /tmp/sentry-bare
git clone --bare https://github.com/calcom/cal.com  /tmp/calcom-bare
git clone --bare https://github.com/discourse/discourse /tmp/discourse-bare
git clone --bare https://github.com/grafana/grafana /tmp/grafana-bare
git clone --bare https://github.com/keycloak/keycloak /tmp/keycloak-bare

# Extract patches using the bare clones
uv run bugeval extract-patch --all --repo-dir /tmp/sentry-bare   --cases-dir cases/final/sentry/
uv run bugeval extract-patch --all --repo-dir /tmp/calcom-bare   --cases-dir cases/final/cal.com/
uv run bugeval extract-patch --all --repo-dir /tmp/discourse-bare --cases-dir cases/final/discourse/
uv run bugeval extract-patch --all --repo-dir /tmp/grafana-bare  --cases-dir cases/final/grafana/
uv run bugeval extract-patch --all --repo-dir /tmp/keycloak-bare --cases-dir cases/final/keycloak/
```

Commits not present in the bare clone will be skipped (non-fatal).

### 1f. Verify dataset quality (optional)

Check that expected_findings actually exist in the pre-fix diffs:

```bash
uv run bugeval groundedness-check \
  --cases-dir cases/final \
  --patches-dir patches/ \
  --workers 8 \
  --dry-run                     # preview first
```

Cases that fail verification are flagged with `quality_flags: ["groundedness-failed"]` and `needs_manual_review: true`. Re-run without `--dry-run` to update case files.

### 1g. Tag the dataset

```bash
git add cases/ patches/
git commit -m "dataset: v2"
git tag dataset-v2
```

---

## Phase 2 — Pilot Run (recommended before full run)

Run ~20 cases with 1–2 agent tools to verify the full pipeline end-to-end.

```bash
RUN=results/run-$(date +%Y-%m-%d)-pilot

uv run bugeval run-agent-eval \
  --cases-dir cases/final \
  --tools claude-cli-sonnet \
  --context-level diff-only \
  --limit 20 \
  --run-dir $RUN

uv run bugeval pipeline \
  --run-dir $RUN \
  --cases-dir cases/final
```

Review results in the dashboard:
```bash
uv run bugeval dashboard --run-dir $RUN
# Opens at http://localhost:5000
```

---

## Phase 3 — Full Evaluation Runs

### 3a. Agent tools — Claude tiers (current prototyping focus)

```bash
uv run bugeval run-agent-eval \
  --cases-dir cases/final \
  --tools claude-cli-haiku,claude-cli-sonnet,claude-cli-opus \
  --context-level diff-only \
  --max-concurrent 2 \
  --run-dir results/run-$(date +%Y-%m-%d)-agent-diff-only
```

Repeat for `diff+repo` and `diff+repo+domain` context levels as needed.

### 3b. Agent tools — all tiers (full evaluation)

```bash
TOOLS="claude-cli-haiku,claude-cli-sonnet,claude-cli-opus,gemini-cli-flash-lite,gemini-cli-flash,codex-cli-mini,codex-cli-o4,google-api-flash-lite,google-api-flash,openai-api-mini,openai-api-o4"

for level in diff-only diff+repo diff+repo+domain; do
  uv run bugeval run-agent-eval \
    --cases-dir cases/final \
    --patches-dir patches/ \
    --context-level $level \
    --tools $TOOLS \
    --use-docker \
    --docker-image bugeval-agent \
    --require-docker \
    --max-concurrent 2 \
    --run-dir results/run-$(date +%Y-%m-%d)-agent-$level
done
```

### 3c. PR tools (commercial)

```bash
uv run bugeval run-pr-eval \
  --cases-dir cases/final \
  --patches-dir patches/ \
  --max-concurrent 3 \
  --run-dir results/run-$(date +%Y-%m-%d)-pr
```

### 3d. API tools (Greptile)

```bash
uv run bugeval run-api-eval \
  --cases-dir cases/final \
  --patches-dir patches/ \
  --context-level diff-only \
  --max-concurrent 4 \
  --run-dir results/run-$(date +%Y-%m-%d)-api
```

### Rate limiting

All eval commands support `--max-concurrent` to cap simultaneous API calls. A `cooldown_seconds` between requests can be configured per tool in `config/config.yaml`.

Recommended starting values:
- PR tools: `--max-concurrent 3` (webhook-driven, low API pressure)
- API tools (Greptile): `--max-concurrent 4`
- Agent tools (Claude, Gemini, OpenAI): `--max-concurrent 2` (avoid rate limits)
- Use `--fail-after 5` (default) to abort a tool after 5 consecutive errors

Check progress:
```bash
uv run bugeval status --run-dir results/run-<date>
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

The pipeline transforms data through three stages:
1. **Normalize**: raw tool output → `NormalizedResult` YAML (common schema across all tool types)
2. **Judge**: LLM-as-judge scores each (case, tool) pair on the 0–3 rubric (3 votes, majority wins)
3. **Analyze**: aggregates scores into `analysis/report.md`, `scores.csv`, and charts

Results appear in: `results/run-<date>/analysis/report.md`

---

## Phase 5 — Human Calibration

Target: Cohen's kappa >= 0.85 on a 25% random sample.

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

If kappa < 0.85: revise `config/judge_prompt.md`, re-run judging, re-calibrate.

View calibration status in the dashboard at `/metrics/<run>`.

---

## Phase 6 — DX Assessment (optional)

Score each tool on actionability, false-positive burden, integration friction, and latency.

```bash
uv run bugeval dashboard --run-dir results/run-<date>
# Navigate to /dx?run=<run-name> and enter scores
```

---

## CLI Reference

### Dataset commands

| Command | Purpose |
|---------|---------|
| `scrape-github` | Scrape PR-based bug candidates from GitHub repos |
| `scrape-benchmark` | Scrape benchmark cases (variant of scrape-github) |
| `mine-candidates` | Mine bug candidates from local git repos |
| `curate` | LLM-assisted curation of candidates into test cases |
| `validate-cases` | Validate case YAML files against Pydantic schema |
| `extract-patch` | Generate `.patch` files from case commits |
| `groundedness-check` | Verify expected_findings exist in pre-fix diffs (LLM-based QA) |

### Evaluation commands

| Command | Purpose |
|---------|---------|
| `run-pr-eval` | Run PR-based commercial tool evaluations |
| `run-api-eval` | Run API-based tool evaluations (Greptile) |
| `run-agent-eval` | Run agent-based evaluations (Claude, Gemini, OpenAI) |

### Post-processing commands

| Command | Purpose |
|---------|---------|
| `normalize` | Convert raw tool output to NormalizedResult YAML |
| `judge` | LLM-as-judge scoring (0–3 scale, ensemble voting) |
| `analyze` | Generate analysis report, CSV, and charts |
| `pipeline` | Run normalize → judge → analyze in sequence |

### Inspection commands

| Command | Purpose |
|---------|---------|
| `status` | Show run progress (checkpoint state, result counts) |
| `dashboard` | Flask web UI for browsing cases, scores, and DX assessment |
| `validate-env` | Check API keys, repos, and cases directory |

### Calibration and export

| Command | Purpose |
|---------|---------|
| `human-judge export` | Export blinded sample for human rating |
| `human-judge import-scores` | Import human scores from filled CSV |
| `human-judge kappa` | Compute Cohen's kappa (LLM vs. human agreement) |
| `export-predictions` | Export NormalizedResult YAMLs to JSONL (external benchmarking) |
| `import-predictions` | Import JSONL predictions into run directory for scoring |

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
| `claude-cli-opus` | agent | claude-opus-4-6 |
| `anthropic-api-sonnet` | agent | claude-sonnet-4-6 (API) |
| `anthropic-api-opus` | agent | claude-opus-4-6 (API) |
| `claude-agent-sdk-sonnet` | agent | claude-sonnet-4-6 (Agent SDK) |
| `claude-agent-sdk-opus` | agent | claude-opus-4-6 (Agent SDK) |
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
- Docker isolation for CLI agents (`--use-docker --require-docker`)

---

## Known Gaps

- **PR tool cost tracking:** Commercial tool costs are not captured automatically. Record manually in a cost log alongside each run.
- **Kappa threshold:** The 0.85 threshold is hardcoded in `config/config.yaml` (`judging.calibration_threshold`). Adjust there if the experiment design changes.
- **Gemini/Codex CLI flags:** Verify CLI flag syntax with `gemini --help` and `codex --help` before the first run — flag names may shift between CLI versions.
- **Human calibration not yet run:** LLM judge scores are provisional until the kappa gate is closed.
