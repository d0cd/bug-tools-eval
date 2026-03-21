# Pilot Plan: Incremental Validation

Validate the full pipeline cheaply before scaling up. Each step builds on the last. Early steps use the Anthropic Agent SDK runner (no API key needed — uses your Claude Code subscription).

## Prerequisites

```bash
uv sync
export GITHUB_TOKEN=...        # for mining + PR tools (gh CLI)
# ANTHROPIC_API_KEY only needed from Step 6 onward
```

Local repo clone:
```bash
mkdir -p repos
git clone https://github.com/ProvableHQ/leo.git repos/leo
```

One-time GitHub org setup (needed for Step 3):
1. Create org `bug-tools-eval` at github.com/organizations/new
2. Install GitHub Copilot, Greptile, and CodeRabbit Apps on the org

---

## Step 0: Mine + Build Ground Truth (no cost)

Mine fix PRs, find introducing commits, compute buggy lines.

```bash
# Fetch up to 20 fix PRs (after filtering, expect ~5-15 usable cases)
uv run bugeval mine --repo ProvableHQ/leo --limit 20 --output-dir cases --since 2024-01-01

# Find introducing commits via git blame
uv run bugeval blame --cases-dir cases/leo --repo-dir repos/leo

# Compute buggy lines via diff intersection
uv run bugeval ground-truth --cases-dir cases/leo --repo-dir repos/leo
```

**Check:** Open a few case YAMLs and verify:
- `truth.buggy_lines` populated with file, line, content
- `truth.blame_confidence` is A, B, C, or D
- `introducing_pr_title` and `introducing_pr_body` present
- `base_commit` set (parent of introducing commit)

```bash
# Count usable cases (have buggy_lines)
grep -rl "buggy_lines" cases/leo/*.yaml | wc -l
```

**Cost: $0. Time: ~5 min.**

---

## Step 1: Agent SDK on Initial Cases (diff-only)

Validate evaluate → score → analyze. Runs on ALL cases from Step 0 (expected 5-15).

```bash
uv run bugeval evaluate \
  --tool agent-sdk \
  --cases-dir cases/leo \
  --run-dir results/pilot-01-sdk-diffonly \
  --repo-dir repos/leo \
  --context diff-only \
  --timeout 120

# Mechanical scoring only (no LLM, no API key)
uv run bugeval score --run-dir results/pilot-01-sdk-diffonly --cases-dir cases/leo --dry-run
uv run bugeval analyze --run-dir results/pilot-01-sdk-diffonly --cases-dir cases/leo
```

**Check:**
- `results/pilot-01-sdk-diffonly/run_metadata.json` exists with tool, context, model
- `results/` subdir has YAML result files
- `scores/` subdir has YAML score files with `caught` field
- `transcripts/` has SDK transcript JSON files
- `comparison.csv` generated
- Catch rate shown in stdout

**Cost: ~$0.10-0.30 (depends on case count). Time: ~5 min.**

---

## Step 2: Agent SDK — diff+repo

Validate workspace-as-fixture pattern and tool use.

```bash
uv run bugeval evaluate \
  --tool agent-sdk \
  --cases-dir cases/leo \
  --run-dir results/pilot-02-sdk-repo \
  --repo-dir repos/leo \
  --context diff+repo \
  --timeout 300

uv run bugeval score --run-dir results/pilot-02-sdk-repo --cases-dir cases/leo --dry-run
uv run bugeval analyze --run-dir results/pilot-02-sdk-repo --cases-dir cases/leo
```

**Check:**
- Transcripts show the agent using `Read`, `Glob`, `Grep`, `WebSearch` tools
- `.pr/description.md` and `diff.patch` exist in workspace
- Catch rate hopefully higher than diff-only

**Cost: ~$0.50-1.00. Time: ~10 min.**

---

## Step 3: PR Tools on 1 Case — Copilot, Greptile, CodeRabbit

Validate fork → PR → poll → scrape lifecycle. Create a temp directory with just 1 case to test.

```bash
# Copy 1 case to a temp dir for isolated testing
mkdir -p cases/leo-pilot1
cp cases/leo/leo-001.yaml cases/leo-pilot1/

for tool in copilot greptile coderabbit; do
  uv run bugeval evaluate \
    --tool $tool \
    --cases-dir cases/leo-pilot1 \
    --run-dir results/pilot-03-$tool \
    --repo-dir repos/leo \
    --org bug-tools-eval \
    --concurrency 1 \
    --timeout 600
done
```

**Check for each tool:**
- Fork created at `bug-tools-eval/leo`
- PR opened with scrubbed title (no fix/bug keywords)
- Tool review appears (check transcript for author)
- Comments scraped and filtered correctly
- PR closed and branch deleted after

**Cost: $0 (free on public repos). Time: ~15 min.**

---

## Step 4: Scale to 30 Cases + Clean Cases

Mine more cases (checkpoint resumes from Step 0) and generate negative controls.

```bash
# Wider date range to get more PRs (checkpoint skips already-mined PRs)
uv run bugeval mine --repo ProvableHQ/leo --limit 100 --output-dir cases --since 2023-01-01
uv run bugeval blame --cases-dir cases/leo --repo-dir repos/leo
uv run bugeval ground-truth --cases-dir cases/leo --repo-dir repos/leo

# Generate 10 clean cases for false alarm testing
uv run bugeval clean-cases --repo ProvableHQ/leo --count 10 --cases-dir cases --since 2023-01-01

# Count total cases
echo "Bug cases:" && ls cases/leo/leo-*.yaml | wc -l
echo "Clean cases:" && ls cases/leo/leo-clean-*.yaml 2>/dev/null | wc -l
```

**Run SDK on all cases:**
```bash
uv run bugeval evaluate \
  --tool agent-sdk \
  --cases-dir cases/leo \
  --run-dir results/pilot-04-sdk-30 \
  --repo-dir repos/leo \
  --context diff+repo \
  --timeout 300 \
  --concurrency 3

uv run bugeval score --run-dir results/pilot-04-sdk-30 --cases-dir cases/leo --dry-run
uv run bugeval analyze --run-dir results/pilot-04-sdk-30 --cases-dir cases/leo
```

**Check via dashboard:**
```bash
uv run bugeval dashboard --cases-dir cases --results-dir results --debug
# Visit http://localhost:5000
# Check: blame confidence distribution, clean cases, catch rate >10%
```

**Cost: ~$2-5. Time: ~15 min.**

---

## Step 5: CLI Runner on 30 Cases

Test Claude Code CLI subprocess runner.

```bash
uv run bugeval evaluate \
  --tool agent-cli-claude \
  --cases-dir cases/leo \
  --run-dir results/pilot-05-cli \
  --repo-dir repos/leo \
  --context diff+repo \
  --timeout 600 \
  --concurrency 1
```

**Check:** Transcripts have stdout/stderr from `claude` CLI. Results comparable to SDK.

**Cost: ~$3. Time: ~30 min.**

---

## Step 6: LLM Judge + Validation (needs ANTHROPIC_API_KEY)

```bash
export ANTHROPIC_API_KEY=...

# Cross-validate ground truth (Claude confirms/disputes each case)
uv run bugeval validate --cases-dir cases/leo --repo-dir repos/leo

# Re-score Step 4 results WITH LLM judge (adds detection_score, review_quality)
uv run bugeval score --run-dir results/pilot-04-sdk-30 --cases-dir cases/leo
uv run bugeval analyze --run-dir results/pilot-04-sdk-30 --cases-dir cases/leo
```

**Check:**
- Validation verdicts in case YAMLs (claude_verdict = confirmed/disputed)
- Scores now have `detection_score` (0-3), `review_quality` (0-4)
- Comment verdicts: TP, FP, low-value, TP-novel
- `reasoning` field has judge explanation

**Cost: ~$1. Time: ~10 min.**

---

## Step 7: PR Tools on 30 Cases

Scale PR tools to the full leo dataset. All 3 tools write to one run-dir for comparison.

```bash
for tool in copilot greptile coderabbit; do
  uv run bugeval evaluate \
    --tool $tool \
    --cases-dir cases/leo \
    --run-dir results/pilot-07-pr-tools \
    --repo-dir repos/leo \
    --org bug-tools-eval \
    --concurrency 1 \
    --timeout 600
done

uv run bugeval score --run-dir results/pilot-07-pr-tools --cases-dir cases/leo
uv run bugeval analyze --run-dir results/pilot-07-pr-tools --cases-dir cases/leo
```

**Check:** Comparison table shows copilot vs greptile vs coderabbit catch rates.

**Cost: $0 (free) + ~$0.50 judge. Time: ~1 hour.**

---

## Step 8: Multi-Repo Dataset

Mine all 4 target repos.

```bash
for repo in ProvableHQ/snarkVM ProvableHQ/snarkOS AleoNet/sdk; do
  slug=$(echo $repo | cut -d/ -f2)
  git clone https://github.com/$repo.git repos/$slug
  uv run bugeval mine --repo $repo --limit 100 --output-dir cases --since 2023-01-01
  uv run bugeval blame --cases-dir cases/$slug --repo-dir repos/$slug
  uv run bugeval ground-truth --cases-dir cases/$slug --repo-dir repos/$slug
  uv run bugeval clean-cases --repo $repo --count 10 --cases-dir cases --since 2023-01-01
done
```

**Cost: $0 (mining only). Time: ~30 min.**

---

## Step 9: Full Model Comparison (needs all API keys)

Compare Anthropic, Google, and OpenAI API runners across all repos. Uses the `agent` (API) runners, not SDK/CLI.

```bash
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...

# Run each API runner across all repos
for slug in leo snarkVM snarkOS sdk; do
  for tool_model in "agent claude-sonnet-4-6" "agent-gemini gemini-2.5-flash" "agent-openai o4-mini"; do
    tool=$(echo $tool_model | cut -d' ' -f1)
    model=$(echo $tool_model | cut -d' ' -f2)
    uv run bugeval evaluate \
      --tool $tool --model $model \
      --cases-dir cases/$slug \
      --run-dir results/pilot-09-models \
      --repo-dir repos/$slug \
      --context diff+repo \
      --timeout 600 --concurrency 3
  done
done

uv run bugeval score --run-dir results/pilot-09-models --cases-dir cases
uv run bugeval analyze --run-dir results/pilot-09-models --cases-dir cases
```

**Check:** Model quality ladder visible (Sonnet > Flash > o4-mini expected). Compare in dashboard.

**Cost: ~$15-30. Time: ~1-2 hours.**

---

## Decision Points

| Step | Gate | If fails |
|------|------|----------|
| 0 | Cases have buggy_lines | Fix blame heuristics or mine with wider date range |
| 1 | Results + scores generated | Debug evaluate/score pipeline |
| 2 | Agent uses Read/Glob/Grep tools | Fix workspace materialization |
| 3 | PR lifecycle completes for all 3 tools | Fix fork creation / app install / timeout |
| 4 | >10% catch rate on 30 cases | Review dataset quality, tune agent prompt |
| 5 | CLI produces results | Fix subprocess invocation |
| 6 | Judge scores are reasonable | Tune judge prompt |
| 7 | PR tools produce results at scale | Fix rate limits / timeouts |
| 8 | Multi-repo mining works | Fix repo-specific blame issues |
| 9 | Model quality ladder visible | Experiment design validated |

---

## Cost Summary

| Step | What | API Key? | Cases | Cost |
|------|------|----------|-------|------|
| 0 | Mine + blame + ground truth | No | 5-15 | $0 |
| 1 | Agent SDK diff-only | No | 5-15 | ~$0.20 |
| 2 | Agent SDK diff+repo | No | 5-15 | ~$0.75 |
| 3 | PR tools × 1 case each | No | 3 | $0 |
| 4 | Agent SDK 30+ cases + clean | No | ~40 | ~$3 |
| 5 | Claude CLI 30 cases | No | 30 | ~$3 |
| 6 | LLM judge + validation | ANTHROPIC | 30 | ~$1 |
| 7 | PR tools × 30 cases + judge | ANTHROPIC | 90 | ~$0.50 |
| 8 | Mine 3 more repos | No | ~120 | $0 |
| 9 | API runners × all repos | ALL | ~400 | ~$20 |
| **Total pilot** | | | | **~$28** |

---

## After the Pilot

Once Step 9 validates the full experiment design:

1. **Golden set curation** — Use the dashboard to confirm/dispute cases, build a vetted golden set
2. **Full-scale evaluation** — Run all 10 tools × all repos × all context levels (~200+ cases × 10 tools)
3. **Analysis report** — Generate comparison tables, charts, statistical tests
4. **Build-vs-buy recommendation** — Catch rate, cost per bug, false alarm rate across commercial vs in-house tools
