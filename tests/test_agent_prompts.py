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
