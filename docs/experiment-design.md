# Experiment Design

## 1. Objective

Evaluate whether a custom Claude-based agent can match or exceed commercial AI code review tools at finding bugs in ProvableHQ Rust/ZK repositories.

The tools under test are: GitHub Copilot (PR-based), Greptile (API), and several variants of a custom Claude agent (API, CLI, SDK). Each tool reviews the **introducing PR** -- the pull request that introduced the bug -- not the fix PR. This is the key design choice: the tool sees exactly what a human reviewer would have seen at review time, and the fix PR is held back as ground truth.

Target repos: `ProvableHQ/snarkVM`, `ProvableHQ/snarkOS`, `ProvableHQ/leo`, `AleoNet/sdk`.

## 2. Tools Under Evaluation

### Commercial (PR-based)

| Tool | Type | Runner | Notes |
|------|------|--------|-------|
| `copilot` | PR | `copilot_runner.py` | Fork repo, open PR with introducing changes, poll for Copilot review, scrape comments, close PR |
| `greptile` | PR | `greptile_runner.py` | Same PR lifecycle as Copilot; reuses fork infrastructure from `copilot_runner.py` |
| `coderabbit` | PR | `coderabbit_runner.py` | Same PR lifecycle as Copilot; reuses fork infrastructure from `copilot_runner.py` |

### In-house agents (API — model capability comparison)

All three API runners share identical tools (`read_file`, `list_directory`, `search_text`), identical prompts, identical tool execution, and identical anti-contamination measures. Differences in scores reflect **model capability**, not tooling.

| Tool | Runner | Default Model | Cost (per MTok in/out) |
|------|--------|---------------|----------------------|
| `agent` | `agent_runner.run_anthropic_api` | `claude-sonnet-4-6` | $3.00 / $15.00 |
| `agent-gemini` | `agent_runner.run_google_api` | `gemini-2.5-flash` | $0.15 / $0.60 |
| `agent-openai` | `agent_runner.run_openai_api` | `o4-mini` | $1.10 / $4.40 |

### In-house agents (CLI — product capability comparison)

Each vendor's CLI brings its own system prompt, tool set, and agent loop. Scores measure **product capability** — how well the product works out of the box.

| Tool | Binary | Notes |
|------|--------|-------|
| `agent-cli-claude` | `claude` | `--output-format json`, `--allowedTools Read,Glob,Grep` |
| `agent-cli-gemini` | `gemini` | `--yolo` for repo access |
| `agent-cli-codex` | `codex` | `--sandbox read-only` or `workspace-write` |

### In-house agents (SDK)

| Tool | Runner | Default Model |
|------|--------|---------------|
| `agent-sdk` | `agent_runner.run_agent_sdk` | `claude-sonnet-4-6` |

All models are configurable via `--model` flag. API runners support extended thinking via `--thinking-budget`. A per-case cost ceiling of `$2.00` is enforced for all API runners.

### Interpretation guide

| Question | Use these runners |
|----------|------------------|
| Which **model** is best at finding bugs? | API runners (`agent`, `agent-gemini`, `agent-openai`) |
| Which **product** catches the most bugs? | CLI runners (`agent-cli-*`) |
| Is it worth building in-house vs buying? | Compare all runners against commercial PR tools |
| Does repo context help? | Compare `diff-only` vs `diff+repo` within a single runner |

## 3. Dataset Construction Pipeline

The pipeline runs five sequential stages, each with its own CLI command and checkpoint file:

### 3.1 Mine (`bugeval mine`)

**Module:** `mine.py`

Scrapes merged PRs from a GitHub repo via `gh pr list`, filters to those with fix signals (keywords like "fix", "bug", "revert" in title/body, or bug-related labels). Filters out PRs that are too small (<3 lines), too large (>1000 lines), or documentation-only.

For each fix PR, fetches rich metadata via GraphQL batch queries (commits, reviews, review threads, discussion comments, closing issues) and builds a `TestCase` YAML with the fix PR data populated. Also detects cross-references between PRs and revert chains to build the `related_prs` relationship graph.

**Outputs:** `cases/<repo>/<repo>-NNN.yaml` with `kind: bug`, fix PR fields populated, ground truth empty.
**Checkpoint:** `.mine_checkpoint.json` (keyed by PR number).

### 3.2 Blame (`bugeval blame`)

**Module:** `blame.py`

Finds the commit that introduced the bug by running `git blame -C -C -C` on lines deleted by the fix commit. Uses majority vote across blamed SHAs to select the introducing commit, with confidence tiers:

- **Tier A:** >60% of blamed lines point to the same commit
- **Tier B:** 40-60% agreement
- **Tier C:** <40% agreement, or file-level fallback (most recent commit touching the same files)
- **Tier D:** Omission bug -- no deleted lines, blamed via enclosing function signature

Merge commits are resolved to the feature branch (second parent). Initial commits are excluded.

After identifying the introducing commit, resolves it to its parent PR via the GitHub API (`repos/{owner}/{name}/commits/{sha}/pulls`) and populates the `introducing_pr_*` fields on the TestCase. Also computes `bug_latency_days` (time between introducing and fix merges) and sets `base_commit` to the parent of the introducing commit.

**Outputs:** Updates existing case YAMLs with `truth.introducing_commit`, `truth.blame_confidence`, `introducing_pr_*` fields, `base_commit`.
**Checkpoint:** `.blame_checkpoint.json` (keyed by case ID).

### 3.3 Ground Truth (`bugeval ground-truth`)

**Module:** `ground_truth.py`

Computes `buggy_lines` via diff intersection: lines **added** by the introducing commit that were **deleted** (or modified) by the fix commit, with a 3-line drift tolerance. This identifies the exact lines where the bug was introduced.

Also extracts bug descriptions (priority: issue body > PR body > PR title > commit message), computes metadata (latency, same-author fix), and heuristically classifies bugs by category (concurrency, runtime, memory, logic, security, type, other), difficulty (easy/medium/hard based on fix size), and severity (critical/high/medium/low from labels and keywords).

**Outputs:** Updates case YAMLs with `truth.buggy_lines`, `bug_description`, `category`, `difficulty`, `severity`, `bug_latency_days`, `same_author_fix`.
**Checkpoint:** `.ground_truth_checkpoint.json` (keyed by case ID).

### 3.4 Validate (`bugeval validate`)

**Module:** `validate.py`

Cross-model validation: sends the introducing diff and bug description to Claude (Haiku) and optionally Gemini, asking each to independently judge whether the diff introduces the described bug. Parses verdicts as `confirmed`, `disputed`, or `ambiguous`.

Agreement requires both models to return the same verdict. A case is `test_validated` only if both agree on `confirmed`. Gemini support is currently a placeholder that returns `ambiguous`.

**Outputs:** Updates case YAMLs with `validation.claude_verdict`, `validation.gemini_verdict`, `validation.agreement`, `validation.test_validated`.
**Checkpoint:** `.validate_checkpoint.json` (keyed by case ID).

### 3.5 Clean Cases (`bugeval clean-cases`)

**Module:** `clean_cases.py`

Generates negative control cases (`kind: clean`) by fetching merged PRs that lack any fix signal. Filters identically to mine (size bounds, code-only). Additionally checks that no subsequent PR references the candidate (no later fix), to reduce the chance of including a PR that actually introduced a bug.

Clean cases use the PR itself as the introducing PR (the tool reviews this PR as if it were a normal code review), with `truth: null`.

**Outputs:** `cases/<repo>/<repo>-clean-NNN.yaml`.
**Checkpoint:** `.clean_checkpoint.json` (keyed by PR number).

## 4. Test Case Schema

The `TestCase` model (defined in `models.py`) separates tool-visible data from ground truth:

### Tool-Visible Fields (Introducing PR)
- `introducing_pr_number`, `introducing_pr_title`, `introducing_pr_body` -- the PR the tool reviews
- `introducing_pr_commit_messages`, `introducing_pr_commit_shas` -- commits in the introducing PR
- `introducing_pr_author`, `introducing_pr_merge_date`
- `introducing_pr_review_comments` -- original human review comments
- `introducing_pr_ci_status` -- CI state at merge time
- `base_commit` -- parent of introducing commit (workspace checkout target)

### Ground Truth Fields (Fix PR, Hidden from Tools)
- `fix_commit`, `fix_pr_number`, `fix_pr_title`, `fix_pr_body`
- `fix_pr_commit_messages`, `fix_pr_commit_shas`, `fix_pr_merge_date`
- `fix_pr_review_comments`, `fix_pr_discussion_comments`
- `fix_pr_merge_method`, `fix_pr_ci_status`
- `truth: GroundTruth` -- introducing_commit, blame_confidence, buggy_lines, fix_summary, fix_pr_numbers
- `validation: Validation` -- claude_verdict, gemini_verdict, agreement, test_validated

### Classification
- `kind: CaseKind` -- `bug` or `clean`
- `category` -- concurrency, runtime, memory, logic, security, type, other
- `difficulty` -- easy (<10 lines), medium (10-50), hard (50+)
- `severity` -- critical, high, medium, low
- `pr_size` -- tiny (<10), small (10-50), medium (50-200), large (200-500), xl (500+)
- `language` -- detected from file extensions (rust, python, typescript, etc.)

### Relationship Graph
- `related_prs: list[PRRelation]` -- role is one of: introducing, partial_fix, full_fix, revert, regression, related
- `linked_issues`, `issue_bodies`, `issue_labels`, `referenced_issues`

### Derived Metadata
- `bug_latency_days` -- days between introducing merge and fix merge
- `same_author_fix` -- whether the same person wrote both PRs
- `stats: CaseStats` -- lines_added, lines_deleted, files_changed

## 5. Ground Truth Construction

Buggy lines are identified via **diff intersection**:

1. Parse the introducing commit's diff to extract **added lines** (file, line number, content).
2. Parse the fix commit's diff (or multiple fix diffs from `truth.fix_pr_numbers`) to extract **deleted lines**.
3. For each added line in the introducing diff, check if any deleted line in the fix diff matches within a **3-line tolerance** (`_LINE_DRIFT_TOLERANCE = 3`) on the same file.
4. Matching lines become `BuggyLine(file, line, content)`.

The 3-line tolerance accounts for minor line drift caused by intermediate commits between the introducing and fix commits.

### Blame Confidence Tiers

Confidence reflects how reliably the introducing commit was identified:

| Tier | Criteria | Reliability |
|------|----------|-------------|
| A | >60% of blamed lines converge on one commit | High -- strong blame signal |
| B | 40-60% convergence | Moderate -- some noise |
| C | <40% convergence, or file-level fallback | Low -- weak signal |
| D | Omission bug, enclosing-function blame | Lowest -- heuristic guess |
| excluded | No fix lines to blame, initial commit, or git error | Case excluded from analysis |

The analysis module reports results separately for high-confidence cases (tier A/B) vs all cases.

## 6. Context Levels

Agent tools (`agent`, `agent-cli-*`, `agent-sdk`) support three context levels, controlled by `--context`:

| Level | What the tool sees | Tools available |
|-------|-------------------|----------------|
| `diff-only` | Sanitized diff + PR metadata only | None (CLI tools have file access disabled) |
| `diff+repo` | Diff + full repo checkout at `base_commit` | `read_file`, `list_directory`, `search_text` (API); `Read`, `Glob`, `Grep` (CLI/SDK) |
| `diff+repo+domain` | Above + ZK/Rust domain hints in system prompt | Same as `diff+repo` |

Domain hints (for `diff+repo+domain`) include: cryptographic correctness, consensus safety, serialization round-trip fidelity, resource exhaustion / DoS vectors, and unsafe blocks / FFI boundaries.

Non-agent tools have fixed context: Copilot, Greptile, and CodeRabbit always operate at `diff+repo` (they see the full PR on a fork).

### Workspace-as-Fixture Pattern

Agent tools use a **workspace-as-fixture** pattern. Instead of passing context through prompt text, `materialize_workspace` writes structured files into the workspace:

- `diff.patch` -- the sanitized unified diff
- `.pr/description.md` -- scrubbed PR title and body
- `.pr/commits.txt` -- scrubbed commit messages
- `.pr/domain.md` -- domain hints (only for `diff+repo+domain`)

The agent reads `diff.patch` and `.pr/description.md` from the workspace, not from the prompt. This keeps prompts consistent across runners and context levels.

Workspace setup (`setup_workspace`) clones the repo at `base_commit` via `clone_at_sha` so the tool cannot see post-fix code.

## 7. Scoring

Scoring has two layers, implemented in `score.py` and `score_models.py`.

### Layer 1: Mechanical Catch Rate (Primary Metric)

No LLM required. For each tool comment with `line > 0`, checks if `file` matches a ground truth buggy file and `abs(comment.line - buggy_line.line) <= 10` (tolerance). File matching handles partial paths (`a/b/c.rs` matches `b/c.rs`).

- `caught: bool` -- at least one comment within tolerance
- `localization_distance: int | None` -- distance of best match

Comments with `line == 0` are classified as `low-value` and excluded from mechanical catch matching (but still counted in comment totals). Comments shorter than 20 chars or matching generic bodies ("lgtm", "looks good", "+1", etc.) are also classified as `low-value`.

For clean cases (`kind: clean`), any non-empty comment set is a `false_alarm`. All comments on clean cases are classified as FP (since `truth` is None).

In `--dry-run` mode (no LLM), a mechanical heuristic assigns detection scores: `caught` + `suggested_fix` present → score 3, `caught` only → score 2, not caught → score 0.

### Layer 2: LLM Quality Judge

Uses `claude-haiku-4-5` (temperature=0) to evaluate review quality. The judge sees the known bug description, buggy lines, and the tool's comments, then returns:

- **Detection Score (0-3):** 0=missed, 1=wrong-area, 2=correct-id, 3=correct-id-and-fix
- **Review Quality (0-4):** 0=useless, 1=shallow, 2=adequate, 3=strong, 4=exceptional
- **Comment Verdicts:** per-comment classification as TP, TP-novel, FP, or low-value

The judge prompt explicitly instructs independent scoring of detection vs quality -- a review can have high quality even if it missed the specific known bug.

LLM verdicts override mechanical comment classifications, and TP/FP/novel counts are recounted after override. **TP-novel is an LLM-only verdict** -- the mechanical classifier cannot assign it (it can only distinguish TP, FP, and low-value).

Judge failures (API errors, parse failures) are tracked via `judge_failed`. Failed cases are excluded from quality-dependent metrics (review_quality, comment verdicts) but **included** in catch rate (the mechanical metric is unaffected).

When `--dry-run` is passed to `bugeval score`, only mechanical scoring runs (no LLM calls).

### Contamination Detection

Before scoring, `detect_contamination` checks if tool comments overlap suspiciously with fix PR text (title, body, commit messages, review comments). If >50% of a comment's tokens appear in fix PR text, the result is flagged as `potentially_contaminated`.

## 8. Evaluation Architecture

### Dispatch (`evaluate.py`)

The `evaluate_tool` orchestrator:

1. Loads all cases from `cases_dir`
2. Creates `run_dir` with `metadata.json` (tool, context_level, start time)
3. Filters to pending cases via checkpoint (`checkpoint.json`, keyed by `{case_id}::{tool}::{context}`)
4. For each case, calls `process_case` which dispatches to the appropriate runner:
   - `copilot` -> `copilot_runner.run_copilot`
   - `greptile` -> `greptile_runner.run_greptile`
   - `coderabbit` -> `coderabbit_runner.run_coderabbit`
   - `agent` -> `agent_runner.run_anthropic_api`
   - `agent-gemini` -> `agent_runner.run_google_api`
   - `agent-openai` -> `agent_runner.run_openai_api`
   - `agent-cli-*` -> `agent_runner.run_agent_cli` (dispatches to claude/gemini/codex)
   - `agent-sdk` -> `agent_runner.run_agent_sdk`

5. Saves `ToolResult` YAML to `run_dir/results/{case_id}--{tool}--{context}.yaml`
6. Updates checkpoint after each case

### Concurrency

`ThreadPoolExecutor` with configurable `--concurrency`. Checkpoint writes are protected by a threading lock.

### Runner Details

**Copilot:** Full lifecycle -- `ensure_fork` -> `create_eval_branch` (checkout `introducing~1`, apply patch, push) -> `_isolate_fork` (force-push default branch to `introducing~1` so PR diff only shows introducing changes) -> `open_eval_pr` -> `poll_for_review` (15s intervals) -> `scrape_pr_comments` (filter to Copilot-authored) -> `close_eval_pr`.

**Greptile:** Same PR lifecycle as Copilot; reuses fork infrastructure from `copilot_runner.py`. Polls for `greptile`-authored reviews.

**CodeRabbit:** Same PR lifecycle as Copilot; reuses fork infrastructure from `copilot_runner.py`. Polls for `coderabbitai`-authored reviews.

**Agent API:** Multi-turn conversation loop (up to 10 turns). Tools (`read_file`, `list_directory`, `search_text`) execute locally with path traversal guards. Thinking blocks are captured in transcripts. Cost ceiling ($2.00) and timeout enforced per turn.

**Agent CLI:** Shared `_run_cli_tool` dispatcher builds command, pipes prompt via stdin, parses stdout. Claude CLI uses `--system-prompt` flag and `--output-format json`. Gemini uses `--yolo` for repo access. Codex uses `--sandbox read-only` or `workspace-write`.

**Agent SDK:** Async iteration over `query()` messages. Captures `AssistantMessage` and `ResultMessage` for transcript and cost.

### Anti-Contamination

- `sanitize_diff` strips commit SHAs, author/date headers, From: lines
- `_scrub_fix_references` removes lines containing fix/bug keywords and issue references from PR metadata shown to tools
- Copilot fork isolation resets default branch so PR diff only contains introducing changes

### Transcript Storage

All runners save conversation transcripts to `run_dir/transcripts/` as JSON for debugging and audit.

### Result Storage

```
run_dir/
  run_metadata.json       # reproducibility: tool, context, model, code_commit, config_sha256, python_version
  checkpoint.json         # completed case::tool::context keys
  results/
    {case_id}--{tool}--{context}.yaml
  scores/
    {case_id}__{tool}.yaml
    checkpoint.json
  transcripts/
    {case_id}.json              # agent API/SDK: full conversation history
    {case_id}-{cli_tool}.json   # agent CLI: prompt + stdout/stderr
    {case_id}-copilot.json      # copilot: PR metadata, scrubbed title/body, raw comments, diff
    {case_id}-greptile.json     # greptile: same as copilot
    {case_id}-coderabbit.json   # coderabbit: same as copilot
  comparison.csv
  charts/
    catch_rate.png
    detection_dist.png
```

`run_metadata.json` is written automatically by `evaluate_tool()` on the first invocation. It captures the git commit SHA, config file hash, tool name, context level, model, thinking budget, timeout, case count, and Python version for reproducibility.

## 9. Analysis

### Metrics (`analyze.py`)

| Metric | Definition |
|--------|-----------|
| Catch rate | Fraction of bug cases where `caught=True` |
| Severity-weighted catch rate | Catch rate weighted by severity (critical=4, high=3, medium=2, low=1) |
| Median localization distance | Median line distance for caught cases |
| Mean review quality | Average `review_quality` score (0-4), excluding judge failures |
| False alarm rate | Fraction of clean cases with any bug comments |
| Precision | TP / (TP + FP) across all comments |
| Signal-to-noise | (TP + TP-novel) / total comments |
| Cost per bug | Total cost / number of catches |
| Mean time | Average `time_seconds` per case |
| Mean time seconds | Average wall-clock time per evaluation |

### Statistical Methods

- **Bootstrap CI:** 10,000 resamples with seed 42, 95% confidence interval on catch rate
- **Permutation test:** 10,000 permutations, two-sided test for difference in means between tool pairs
- **Benjamini-Hochberg FDR:** Applied to pairwise p-values at alpha=0.05

### Slicing Dimensions

Analysis reports catch rate broken down by: `repo`, `category`, `difficulty`, `severity`, `pr_size`, `blame_confidence`, `context_level`, `issue_linked`.

A separate high-confidence analysis (tier A/B only) is reported for each tool.

### Output

- `comparison.csv` -- full comparison table
- Stdout: tab-separated table, per-dimension slices, high-confidence subset, pairwise comparisons with significance markers
- `charts/catch_rate.png` -- bar chart of catch rates
- `charts/detection_dist.png` -- detection score histograms per tool

## 10. Pitfalls and Mitigations

### Contamination

**Risk:** Tools may have seen the fix PR during training or in their indexed codebase.

**Mitigations:**
- Tools review the *introducing* PR, not the fix -- the fix PR text is never shown
- `sanitize_diff` strips identifying metadata (SHAs, author, dates) from diffs
- `_scrub_fix_references` removes fix-related keywords and issue references from PR metadata
- Copilot fork isolation resets the fork's default branch so the PR diff only contains introducing changes
- Greptile and CodeRabbit fork isolation resets the fork's default branch, same as Copilot
- Post-hoc `detect_contamination` flags results with >50% token overlap with fix PR text
- Contaminated results are reported separately and can be excluded from analysis

### Self-Evaluation Bias

**Risk:** Using Claude as both the agent under test and the LLM judge.

**Mitigations:**
- Primary metric (mechanical catch rate) requires zero LLM involvement
- LLM judge uses a different model (Haiku) than the agent (Sonnet)
- Judge scores (detection, quality) are secondary metrics
- Judge failures are tracked and excluded from quality analysis
- The judge prompt scores detection and quality independently

### Context Asymmetry

**Risk:** Tools receive different amounts of context, making comparison unfair.

**Mitigations:**
- Copilot, Greptile, and CodeRabbit always operate at `diff+repo` (inherent to their design)
- Agent tools are tested at all three context levels (`diff-only`, `diff+repo`, `diff+repo+domain`)
- Analysis slices by `context_level` so same-context comparisons are straightforward
- Workspace checkout is at `base_commit` (parent of introducing commit) -- no tool sees post-fix code

### Ground Truth Quality

**Risk:** Buggy lines computed via diff intersection may be incomplete or noisy.

**Mitigations:**
- Blame confidence tiers (A/B/C/D) quantify reliability; analysis reports high-confidence subset separately
- Cross-model validation (Claude + Gemini) checks whether the introducing diff actually introduces the described bug
- 3-line drift tolerance in ground truth construction accounts for intermediate commits
- Mechanical catch rate uses a generous 10-line tolerance
- Omission bugs (pure additions) are handled via enclosing-function blame (tier D) and flagged as lower confidence
- Initial commits are excluded entirely

### Clean Case Validity

**Risk:** "Clean" PRs might actually contain bugs.

**Mitigations:**
- Clean PRs are filtered to exclude any fix-signal keywords or labels
- `check_not_subsequently_fixed` searches for later PRs that reference the candidate, rejecting any that have fix signals
- False alarm rate on clean cases is reported as a metric

### Cost and Rate Limits

**Risk:** Evaluation runs are expensive and may hit API rate limits.

**Mitigations:**
- Per-case cost ceiling ($2.00) on API agent
- Checkpoint resume on all pipeline stages (mine, blame, ground-truth, validate, evaluate, score)
- `--dry-run` flag on evaluate and score commands for validation without API calls
- Configurable `--concurrency` and `--timeout` per evaluation run

