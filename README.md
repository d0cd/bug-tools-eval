# bug-tools-eval

Evaluation framework for AI code review tools. Compares GitHub Copilot, Greptile, and a custom Claude agent on bug detection in ProvableHQ's Rust/ZK repositories.

## Quick Start

```bash
uv sync
uv run bugeval --help
```

## Pipeline

```
mine → blame → ground-truth → validate → clean-cases → evaluate → score → analyze
```

See `CLAUDE.md` for full documentation.

## Tools Evaluated

| Tool | Mode | Command |
|------|------|---------|
| GitHub Copilot | PR review | `bugeval evaluate --tool copilot` |
| Greptile | API | `bugeval evaluate --tool greptile` |
| Claude Agent (API) | Multi-turn API | `bugeval evaluate --tool agent` |
| Claude Code CLI | CLI subprocess | `bugeval evaluate --tool agent-cli-claude` |
| Gemini CLI | CLI subprocess | `bugeval evaluate --tool agent-cli-gemini` |
| Codex CLI | CLI subprocess | `bugeval evaluate --tool agent-cli-codex` |
| Claude Agent SDK | SDK | `bugeval evaluate --tool agent-sdk` |

## Development

```bash
uv run pytest
uv run ruff check src/ tests/
uv run pyright src/
```
