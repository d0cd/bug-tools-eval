# bug-tools-eval

Build-vs-buy evaluation: commercial AI code review tools vs. an in-house Claude Code agent,
tested against a curated dataset of real bug-fix PRs from Provable and open-source repos.

## Quick Start

```bash
git clone <repo-url> && cd bug-tools-eval
uv sync
cp .env.example .env          # fill in ANTHROPIC_API_KEY at minimum
uv run bugeval validate-env --cases-dir cases/final
uv run bugeval run-agent-eval --tools claude-agent-sdk-sonnet --limit 1 --dry-run
uv run bugeval pipeline --run-dir results/run-YYYY-MM-DD
```

## What's Being Evaluated

**Tools:** agents (Claude SDK, Claude CLI, Gemini CLI, Codex CLI), API tools (Gemini, OpenAI, Greptile), and PR-review tools (CodeRabbit, BugBot). Full list in `config/config.yaml`.

**Dataset:** 1,271 curated bug-fix cases across 9 repos (leo, snarkVM, snarkOS, sdk, sentry, cal.com, grafana, keycloak, discourse). Cases in `cases/final/`, patches in `patches/`.

**Scoring:** 0-3 scale — 0 missed, 1 wrong area, 2 correct ID, 3 correct ID + fix.

**Context levels:** `diff-only`, `diff+repo`, `diff+repo+domain`.

## Pipeline

```
run-{pr,api,agent}-eval → normalize → judge → analyze
                                                 ↓
                                          report.md, scores.csv, charts
```

Each stage can be run independently, or chained via `bugeval pipeline`.

## Docs

| Document | Contents |
|----------|----------|
| [`docs/experiment-design.md`](docs/experiment-design.md) | Research objectives, methodology, validity threats |
| [`docs/runbook.md`](docs/runbook.md) | Step-by-step execution from setup to analysis |
| [`docs/results-schema.md`](docs/results-schema.md) | Output file formats and field definitions |
| [`.claude/CLAUDE.md`](.claude/CLAUDE.md) | Code conventions, scoring scale, project rules |

## Development

```bash
uv run pytest
uv run ruff check src/ tests/
uv run pyright src/
```
