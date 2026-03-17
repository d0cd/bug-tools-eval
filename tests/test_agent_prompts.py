"""Tests for agent_prompts."""

from pathlib import Path

from bugeval.agent_prompts import _DEFAULT_SYSTEM_PROMPT, build_user_prompt, load_agent_prompt
from bugeval.models import Category, Difficulty, PRSize, Severity, TestCase


def _make_case() -> TestCase:
    return TestCase(
        id="aleo-lang-001",
        repo="provable-org/aleo-lang",
        base_commit="abc123",
        head_commit="def456",
        fix_commit="def456",
        category=Category.logic,
        difficulty=Difficulty.medium,
        severity=Severity.high,
        language="rust",
        pr_size=PRSize.small,
        description="Off-by-one in loop bound",
        expected_findings=[],
    )


def test_load_agent_prompt_from_file(tmp_path: Path) -> None:
    prompt_file = tmp_path / "agent_prompt.md"
    prompt_file.write_text("Custom system prompt")
    result = load_agent_prompt(prompt_file)
    assert result == "Custom system prompt"


def test_load_agent_prompt_falls_back_to_default(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.md"
    result = load_agent_prompt(missing)
    assert result == _DEFAULT_SYSTEM_PROMPT


def test_build_user_prompt_diff_only() -> None:
    case = _make_case()
    prompt = build_user_prompt(case, "- old\n+ new\n", "diff-only")
    assert "- old" in prompt
    assert "+ new" in prompt
    assert "repository" not in prompt.lower()
    assert "domain" not in prompt.lower()


def test_build_user_prompt_diff_plus_repo() -> None:
    case = _make_case()
    prompt = build_user_prompt(case, "patch content", "diff+repo")
    assert "patch content" in prompt
    assert "working directory" in prompt.lower()
    assert "Domain Context" not in prompt


def test_build_user_prompt_diff_plus_repo_directs_tool_use() -> None:
    """diff+repo prompt must explicitly instruct the agent to use Read/Grep/Glob."""
    case = _make_case()
    prompt = build_user_prompt(case, "patch content", "diff+repo")
    lower = prompt.lower()
    assert "read" in lower
    assert "grep" in lower
    assert "step 1" in lower or "understand" in lower


def test_build_user_prompt_diff_plus_repo_plus_domain() -> None:
    case = _make_case()
    prompt = build_user_prompt(case, "patch content", "diff+repo+domain")
    assert "patch content" in prompt
    assert "working directory" in prompt.lower()
    assert "Domain Context" in prompt
    assert "logic" in prompt
    assert "high" in prompt
    assert "rust" in prompt
    assert "Off-by-one" in prompt


# ---------------------------------------------------------------------------
# Language-aware prompt loading
# ---------------------------------------------------------------------------


def test_load_agent_prompt_language_specific_file_used(tmp_path: Path) -> None:
    """When a language-specific prompt file exists in the config dir, it is used."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agent_prompt.md").write_text("generic prompt")
    (config_dir / "agent_prompt_rust.md").write_text("rust-specific prompt")

    result = load_agent_prompt(config_dir=config_dir, language="rust")
    assert result == "rust-specific prompt"


def test_load_agent_prompt_falls_back_to_generic_when_no_language_specific(tmp_path: Path) -> None:
    """Falls back to generic agent_prompt.md when no language-specific file exists."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agent_prompt.md").write_text("generic prompt")

    result = load_agent_prompt(config_dir=config_dir, language="go")
    assert result == "generic prompt"


def test_load_agent_prompt_no_language_uses_generic(tmp_path: Path) -> None:
    """When language is not given, uses generic agent_prompt.md."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agent_prompt.md").write_text("generic prompt")

    result = load_agent_prompt(config_dir=config_dir)
    assert result == "generic prompt"


def test_load_agent_prompt_explicit_path_still_works(tmp_path: Path) -> None:
    """Explicit path= argument still overrides config_dir lookup."""
    prompt_file = tmp_path / "custom.md"
    prompt_file.write_text("custom prompt")
    result = load_agent_prompt(path=prompt_file)
    assert result == "custom prompt"


# ---------------------------------------------------------------------------
# Context-level-aware prompt loading
# ---------------------------------------------------------------------------


def test_load_agent_prompt_context_level_specific_file_used(tmp_path: Path) -> None:
    """When a context-level-specific prompt file exists it takes priority over generic."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agent_prompt.md").write_text("generic prompt")
    (config_dir / "agent_prompt_diff+repo.md").write_text("diff+repo prompt")

    result = load_agent_prompt(config_dir=config_dir, context_level="diff+repo")
    assert result == "diff+repo prompt"


def test_load_agent_prompt_context_level_falls_back_to_generic(tmp_path: Path) -> None:
    """Falls back to generic agent_prompt.md when no context-level file exists."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agent_prompt.md").write_text("generic prompt")

    result = load_agent_prompt(config_dir=config_dir, context_level="diff+repo")
    assert result == "generic prompt"


def test_load_agent_prompt_context_level_priority_over_language(tmp_path: Path) -> None:
    """Context-level file takes priority over language-specific file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "agent_prompt.md").write_text("generic")
    (config_dir / "agent_prompt_rust.md").write_text("rust prompt")
    (config_dir / "agent_prompt_diff+repo.md").write_text("diff+repo prompt")

    result = load_agent_prompt(config_dir=config_dir, context_level="diff+repo", language="rust")
    assert result == "diff+repo prompt"


def test_build_user_prompt_diff_plus_repo_allows_reasoning() -> None:
    """diff+repo closing instruction should encourage reasoning and require a code fence."""
    case = _make_case()
    prompt = build_user_prompt(case, "patch content", "diff+repo")
    lower = prompt.lower()
    # Should NOT say "only" (i.e. "return ONLY the JSON") in the closing
    assert "return only" not in lower
    # Should encourage reasoning or explanation before the JSON array
    assert any(word in lower for word in ("reasoning", "explain", "analysis", "walk"))
    # Should instruct use of a code block (ensures fence-based extraction)
    assert "code block" in lower or "```" in prompt


def test_build_user_prompt_diff_plus_repo_plus_domain_allows_reasoning() -> None:
    """diff+repo+domain closing instruction should also encourage reasoning."""
    case = _make_case()
    prompt = build_user_prompt(case, "patch content", "diff+repo+domain")
    lower = prompt.lower()
    assert "return only" not in lower
    assert any(word in lower for word in ("reasoning", "explain", "analysis", "walk"))
    assert "code block" in lower or "```" in prompt


def test_introducing_commit_not_in_any_prompt() -> None:
    """introducing_commit must never leak into any agent prompt (analysis-only field)."""
    case = _make_case()
    case = case.model_copy(update={"introducing_commit": "deadbeef1234567890"})
    for level in ("diff-only", "diff+repo", "diff+repo+domain"):
        prompt = build_user_prompt(case, "--- a/foo\n+++ b/foo\n", level)
        assert "deadbeef" not in prompt, f"introducing_commit leaked into {level} prompt"
        assert "introducing_commit" not in prompt, (
            f"introducing_commit field name leaked into {level} prompt"
        )
