"""Tests for agent_runner module."""

from __future__ import annotations

import json as json_mod
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from bugeval.agent_runner import (
    _execute_tool,
    _save_transcript,
    _scrub_fix_references,
    build_system_prompt,
    build_user_prompt,
    materialize_workspace,
    parse_agent_findings,
    run_agent_cli,
    run_agent_sdk,
    run_anthropic_api,
    run_google_api,
    run_openai_api,
    sanitize_diff,
)
from bugeval.models import CaseKind, TestCase


def _make_case(**overrides: object) -> TestCase:
    defaults: dict[str, object] = {
        "id": "leo-001",
        "repo": "AleoNet/leo",
        "kind": CaseKind.bug,
        "base_commit": "abc123",
        "fix_commit": "def456",
        "introducing_pr_title": "Add new parser",
        "introducing_pr_body": "Implements expression parsing.",
        "introducing_pr_commit_messages": ["feat: add parser"],
    }
    defaults.update(overrides)
    return TestCase(**defaults)  # type: ignore[arg-type]


SAMPLE_DIFF = (
    "--- a/foo.rs\n+++ b/foo.rs\n@@ -1,3 +1,3 @@\n-old\n+new\n"
)


class TestBuildSystemPrompt:
    def test_diff_only_mentions_workspace_files(self) -> None:
        prompt = build_system_prompt("diff-only")
        assert "diff.patch" in prompt
        assert ".pr/description.md" in prompt
        assert "JSON" in prompt
        # diff-only should NOT mention repo tools
        assert "full repository" not in prompt

    def test_diff_repo_mentions_tools(self) -> None:
        prompt = build_system_prompt("diff+repo")
        assert "full repository" in prompt
        assert "tools" in prompt.lower()
        assert "diff.patch" in prompt

    def test_diff_repo_domain_has_zk_context(self) -> None:
        prompt = build_system_prompt("diff+repo+domain")
        assert "zero-knowledge" in prompt.lower()
        assert ".pr/domain.md" in prompt
        assert "full repository" in prompt


class TestBuildUserPrompt:
    def test_workspace_references(self) -> None:
        case = _make_case()
        prompt = build_user_prompt(case, SAMPLE_DIFF, "diff-only")
        assert "diff.patch" in prompt
        assert ".pr/description.md" in prompt
        # No inline diff by default
        assert "```diff" not in prompt

    def test_inline_diff_when_requested(self) -> None:
        case = _make_case()
        prompt = build_user_prompt(
            case, SAMPLE_DIFF, "diff-only", inline_diff=True,
        )
        assert "```diff" in prompt
        assert "foo.rs" in prompt

    def test_repo_context_mentions_tools(self) -> None:
        case = _make_case()
        prompt = build_user_prompt(case, SAMPLE_DIFF, "diff+repo")
        assert "repository tools" in prompt.lower()

    def test_domain_context_mentions_domain_md(self) -> None:
        case = _make_case()
        prompt = build_user_prompt(case, SAMPLE_DIFF, "diff+repo+domain")
        assert ".pr/domain.md" in prompt


class TestSanitizeDiff:
    def test_strips_index_lines(self) -> None:
        diff = (
            "diff --git a/f.rs b/f.rs\n"
            "index abc1234..def5678 100644\n"
            "--- a/f.rs\n"
            "+++ b/f.rs\n"
            "@@ -1,3 +1,3 @@\n"
            "-old\n"
            "+new\n"
        )
        result = sanitize_diff(diff)
        assert "index " not in result
        assert "--- a/f.rs" in result
        assert "+new" in result

    def test_strips_author_date(self) -> None:
        diff = "Author: alice\nDate: 2024-01-01\n--- a/f.rs\n+++ b/f.rs\n"
        result = sanitize_diff(diff)
        assert "Author:" not in result
        assert "Date:" not in result
        assert "--- a/f.rs" in result

    def test_strips_from_header(self) -> None:
        diff = "From: alice@example.com\n--- a/f.rs\n+++ b/f.rs\n"
        result = sanitize_diff(diff)
        assert "From:" not in result
        assert "--- a/f.rs" in result

    def test_strips_from_sha_line(self) -> None:
        diff = (
            "From abc1234def5678901234567890abcdef12345678 Mon Sep 17\n"
            "--- a/f.rs\n+++ b/f.rs\n"
        )
        result = sanitize_diff(diff)
        assert "From abc1234" not in result
        assert "--- a/f.rs" in result


class TestParseAgentFindings:
    def test_json_array(self) -> None:
        response = '[{"file":"f.rs","line":10,"description":"bug here"}]'
        comments = parse_agent_findings(response)
        assert len(comments) == 1
        assert comments[0].file == "f.rs"
        assert comments[0].line == 10
        assert comments[0].body == "bug here"

    def test_json_with_surrounding_text(self) -> None:
        response = (
            'Here are my findings:\n'
            '[{"file":"a.rs","line":5,"description":"issue"}]\n'
            'That is all.'
        )
        comments = parse_agent_findings(response)
        assert len(comments) == 1
        assert comments[0].file == "a.rs"

    def test_malformed_returns_empty(self) -> None:
        assert parse_agent_findings("no json here") == []
        assert parse_agent_findings("{not an array}") == []
        assert parse_agent_findings("") == []

    def test_empty_array(self) -> None:
        assert parse_agent_findings("[]") == []

    def test_with_suggested_fix(self) -> None:
        response = (
            '[{"file":"x.rs","line":1,"description":"d","suggested_fix":"f"}]'
        )
        comments = parse_agent_findings(response)
        assert comments[0].suggested_fix == "f"


class TestRunAgentApiDiffOnly:
    def test_mocked_anthropic_returns_result(self) -> None:
        case = _make_case()

        # Build mock response
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '[{"file":"f.rs","line":1,"description":"bug"}]'

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(
            input_tokens=100, output_tokens=50
        )

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("bugeval.agent_runner.anthropic.Anthropic", return_value=mock_client):
            result = run_anthropic_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
            )

        assert result.case_id == "leo-001"
        assert result.tool == "agent"
        assert result.context_level == "diff-only"
        assert len(result.comments) == 1
        assert result.comments[0].file == "f.rs"
        assert result.error == ""
        assert result.cost_usd > 0

    def test_api_error_captured(self) -> None:
        case = _make_case()

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API down")

        with patch("bugeval.agent_runner.anthropic.Anthropic", return_value=mock_client):
            result = run_anthropic_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
            )

        assert result.case_id == "leo-001"
        assert "API down" in result.error


class TestScrubFixReferences:
    def test_removes_fix_lines(self) -> None:
        body = "Add parser\nThis fixes the crash\nGood change"
        result = _scrub_fix_references(body)
        assert "fixes" not in result.lower()
        assert "Add parser" in result
        assert "Good change" in result

    def test_removes_issue_references(self) -> None:
        body = "Implement feature\nCloses #123\nDetails here"
        result = _scrub_fix_references(body)
        assert "#123" not in result
        assert "Implement feature" in result

    def test_removes_bug_lines(self) -> None:
        body = "Update parser\nThis is a bug fix\nEnd"
        result = _scrub_fix_references(body)
        assert "bug" not in result.lower()

    def test_preserves_clean_body(self) -> None:
        body = "Add new feature\nImprove performance"
        result = _scrub_fix_references(body)
        assert "Add new feature" in result
        assert "Improve performance" in result


class TestMaterializeWorkspaceScrubsTitle:
    def test_title_with_fix_keyword_scrubbed(self, tmp_path: Path) -> None:
        case = _make_case(
            introducing_pr_title="Fix crash in parser",
        )
        ws = tmp_path / "ws"
        ws.mkdir()
        materialize_workspace(case, "diff", ws, "diff+repo")
        desc = (ws / ".pr" / "description.md").read_text()
        assert "Fix crash" not in desc

    def test_title_without_fix_keyword_preserved(self, tmp_path: Path) -> None:
        case = _make_case(
            introducing_pr_title="Add new parser",
        )
        ws = tmp_path / "ws"
        ws.mkdir()
        materialize_workspace(case, "diff", ws, "diff+repo")
        desc = (ws / ".pr" / "description.md").read_text()
        assert "Add new parser" in desc


class TestMaterializeWorkspaceScrubsCommitMessages:
    def test_commit_messages_with_fix_keyword_scrubbed(
        self, tmp_path: Path,
    ) -> None:
        case = _make_case(
            introducing_pr_commit_messages=[
                "fix: handle edge case",
                "feat: add parser",
            ],
        )
        ws = tmp_path / "ws"
        ws.mkdir()
        materialize_workspace(case, "diff", ws, "diff+repo")
        commits = (ws / ".pr" / "commits.txt").read_text()
        assert "fix: handle" not in commits.lower()
        assert "feat: add parser" in commits

    def test_all_messages_scrubbed_uses_placeholder(
        self, tmp_path: Path,
    ) -> None:
        case = _make_case(
            introducing_pr_commit_messages=["fix: patch bug #42"],
        )
        ws = tmp_path / "ws"
        ws.mkdir()
        materialize_workspace(case, "diff", ws, "diff+repo")
        commits = (ws / ".pr" / "commits.txt").read_text()
        assert commits == "(no commits)"


class TestMaterializeWorkspaceAntiContamination:
    def test_scrubs_fix_references_from_body(self, tmp_path: Path) -> None:
        case = _make_case(
            introducing_pr_body="Add parser\nThis fixes the crash\nGood code",
        )
        ws = tmp_path / "ws"
        ws.mkdir()
        materialize_workspace(case, "diff", ws, "diff+repo")
        desc = (ws / ".pr" / "description.md").read_text()
        assert "fixes" not in desc.lower()
        assert "Add parser" in desc

    def test_body_entirely_fix_references_omitted(
        self, tmp_path: Path,
    ) -> None:
        case = _make_case(
            introducing_pr_title="",
            introducing_pr_body="Fixes #42",
        )
        ws = tmp_path / "ws"
        ws.mkdir()
        materialize_workspace(case, "diff", ws, "diff+repo")
        desc = (ws / ".pr" / "description.md").read_text()
        assert desc == "(no description)"


class TestMaterializeWorkspace:
    def test_creates_all_files(self, tmp_path: Path) -> None:
        case = _make_case()
        ws = tmp_path / "ws"
        ws.mkdir()
        result = materialize_workspace(case, "the diff", ws, "diff+repo")
        assert result == ws
        assert (ws / "diff.patch").read_text() == "the diff"
        assert (ws / ".pr" / "description.md").exists()
        assert (ws / ".pr" / "commits.txt").exists()
        # No domain.md for diff+repo
        assert not (ws / ".pr" / "domain.md").exists()

    def test_diff_only_creates_temp_dir(self, tmp_path: Path) -> None:
        case = _make_case()
        ws = tmp_path / "ws"
        ws.mkdir()
        result = materialize_workspace(case, "diff", ws, "diff-only")
        # Returns a NEW temp dir, not the original ws
        assert result != ws
        assert (result / "diff.patch").read_text() == "diff"
        assert (result / ".pr" / "description.md").exists()

    def test_domain_context_creates_domain_md(self, tmp_path: Path) -> None:
        case = _make_case()
        ws = tmp_path / "ws"
        ws.mkdir()
        materialize_workspace(case, "diff", ws, "diff+repo+domain")
        domain = (ws / ".pr" / "domain.md").read_text()
        assert "zero-knowledge" in domain.lower()

    def test_diff_repo_no_domain_md(self, tmp_path: Path) -> None:
        case = _make_case()
        ws = tmp_path / "ws"
        ws.mkdir()
        materialize_workspace(case, "diff", ws, "diff+repo")
        assert not (ws / ".pr" / "domain.md").exists()


class TestExecuteToolReadFile:
    def test_read_file_success(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "hello.txt").write_text("hello world")
        result = _execute_tool("read_file", {"path": "hello.txt"}, repo)
        assert result == "hello world"

    def test_read_file_path_traversal(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        # Create a file outside repo
        (tmp_path / "secret.txt").write_text("secret")
        result = _execute_tool(
            "read_file", {"path": "../../etc/passwd"}, repo
        )
        assert "path outside workspace" in result.lower()

    def test_read_file_path_traversal_prefix_trick(self, tmp_path: Path) -> None:
        # This is the specific case the old string check missed:
        # /tmp/repo-evil starts with /tmp/repo
        repo = tmp_path / "repo"
        repo.mkdir()
        evil = tmp_path / "repo-evil"
        evil.mkdir()
        (evil / "data.txt").write_text("evil")
        # ../repo-evil/data.txt resolves outside repo
        result = _execute_tool(
            "read_file", {"path": "../repo-evil/data.txt"}, repo
        )
        assert "path outside workspace" in result.lower()

    def test_read_file_not_found(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        result = _execute_tool("read_file", {"path": "nope.txt"}, repo)
        assert "file not found" in result.lower()

    def test_read_file_git_dir_blocked(self, tmp_path: Path) -> None:
        """Agents must not read .git internals (prevents history-based cheating)."""
        repo = tmp_path / "repo"
        git_dir = repo / ".git" / "logs"
        git_dir.mkdir(parents=True)
        (git_dir / "HEAD").write_text("ref: refs/heads/main")
        result = _execute_tool("read_file", {"path": ".git/logs/HEAD"}, repo)
        assert "version control" in result.lower()

    def test_list_directory_git_dir_blocked(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        (repo / ".git" / "refs").mkdir(parents=True)
        result = _execute_tool("list_directory", {"path": ".git/refs"}, repo)
        assert "version control" in result.lower()

    def test_search_text_git_dir_blocked(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        result = _execute_tool("search_text", {"pattern": "HEAD", "path": ".git"}, repo)
        assert "version control" in result.lower()


class TestExecuteToolListDirectory:
    def test_list_directory_success(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.txt").write_text("a")
        (repo / "b.txt").write_text("b")
        result = _execute_tool("list_directory", {"path": "."}, repo)
        assert "a.txt" in result
        assert "b.txt" in result

    def test_list_directory_path_traversal(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        result = _execute_tool("list_directory", {"path": ".."}, repo)
        assert "path outside workspace" in result.lower()


class TestExecuteToolSearchText:
    def test_search_text_success(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="foo.rs:10:match here\n", returncode=0
            )
            result = _execute_tool(
                "search_text", {"pattern": "match", "path": "."}, repo
            )
        assert "match here" in result

    def test_search_text_path_traversal(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        result = _execute_tool(
            "search_text", {"pattern": "x", "path": "../.."}, repo
        )
        assert "path outside workspace" in result.lower()


class TestExecuteToolUnknown:
    def test_unknown_tool(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        result = _execute_tool("delete_everything", {}, repo)
        assert "unknown tool" in result.lower()


class TestRunAgentApiMultiTurn:
    def test_tool_use_then_text(self) -> None:
        case = _make_case()

        # First response: tool_use
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "read_file"
        tool_block.input = {"path": "foo.rs"}
        tool_block.id = "tool_1"

        resp1 = MagicMock()
        resp1.stop_reason = "tool_use"
        resp1.content = [tool_block]
        resp1.usage = MagicMock(input_tokens=50, output_tokens=20)

        # Second response: final text
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '[{"file":"foo.rs","line":1,"description":"bug"}]'

        resp2 = MagicMock()
        resp2.stop_reason = "end_turn"
        resp2.content = [text_block]
        resp2.usage = MagicMock(input_tokens=100, output_tokens=50)

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [resp1, resp2]

        with patch(
            "bugeval.agent_runner.anthropic.Anthropic",
            return_value=mock_client,
        ):
            result = run_anthropic_api(
                case, SAMPLE_DIFF, None, "diff+repo",
                max_turns=5, timeout=300,
            )

        assert len(result.comments) == 1
        assert result.comments[0].file == "foo.rs"
        assert result.error == ""
        # Should have been called twice
        assert mock_client.messages.create.call_count == 2

    def test_cost_ceiling_breached(self) -> None:
        case = _make_case()

        # Response with enormous usage to blow past ceiling
        text_block = MagicMock()
        text_block.type = "tool_use"
        text_block.name = "read_file"
        text_block.input = {"path": "x"}
        text_block.id = "t1"

        resp = MagicMock()
        resp.stop_reason = "tool_use"
        resp.content = [text_block]
        # $3/MTok * 1M = $3, which exceeds $2 ceiling
        resp.usage = MagicMock(input_tokens=1_000_000, output_tokens=0)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = resp

        with patch(
            "bugeval.agent_runner.anthropic.Anthropic",
            return_value=mock_client,
        ):
            result = run_anthropic_api(
                case, SAMPLE_DIFF, None, "diff+repo",
                max_turns=10, timeout=300,
            )

        assert "cost ceiling" in result.error.lower()


class TestRunAgentApiTimeout:
    def test_timeout_exceeded(self) -> None:
        case = _make_case()

        mock_client = MagicMock()
        # Make monotonic return increasing values
        call_count = 0

        def fake_monotonic() -> float:
            nonlocal call_count
            call_count += 1
            # First call (start): 0, second call (check): 400
            return 0.0 if call_count <= 1 else 400.0

        with (
            patch(
                "bugeval.agent_runner.anthropic.Anthropic",
                return_value=mock_client,
            ),
            patch("bugeval.agent_runner.time.monotonic", side_effect=fake_monotonic),
        ):
            result = run_anthropic_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
            )

        assert "timeout" in result.error.lower()
        mock_client.messages.create.assert_not_called()


class TestRunAgentApiTranscript:
    def test_transcript_saved(self, tmp_path: Path) -> None:
        case = _make_case()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "[]"

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        transcript_dir = tmp_path / "transcripts"
        with patch(
            "bugeval.agent_runner.anthropic.Anthropic",
            return_value=mock_client,
        ):
            result = run_anthropic_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
                transcript_dir=transcript_dir,
            )

        assert result.transcript_path != ""
        assert Path(result.transcript_path).exists()
        import json

        data = json.loads(Path(result.transcript_path).read_text())
        assert isinstance(data, list)
        assert data[0]["role"] == "user"


class TestCliRunnerUsesStdin:
    @patch("bugeval.agent_runner.subprocess.run")
    def test_prompt_piped_via_stdin(self, mock_run: MagicMock) -> None:
        """Verify the CLI runner passes prompt via stdin, not as an argument."""
        import json as json_mod
        import subprocess as sp

        output = {
            "result": '[{"file":"f.rs","line":1,"description":"bug"}]',
            "cost": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_run.return_value = sp.CompletedProcess(
            args=["claude"],
            returncode=0,
            stdout=json_mod.dumps(output),
            stderr="",
        )
        case = _make_case()
        result = run_agent_cli(
            case, SAMPLE_DIFF, None, "diff-only",
            cli_tool="claude", timeout=60,
        )
        assert result.error == ""
        assert len(result.comments) == 1
        # Check that subprocess.run was called with input= (stdin)
        call_kwargs = mock_run.call_args
        cmd_list = call_kwargs[0][0]
        assert cmd_list[0] == "claude"
        assert "-p" in cmd_list
        # Prompt should be piped via input keyword arg
        assert call_kwargs.kwargs.get("input") is not None


class TestRunAgentApiWithThinking:
    def test_run_anthropic_api_with_thinking(self, tmp_path: Path) -> None:
        """Thinking block appears in transcript but not in findings."""
        case = _make_case()

        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Let me analyze this diff carefully..."

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = '[{"file":"f.rs","line":1,"description":"bug"}]'

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [thinking_block, text_block]
        mock_response.usage = MagicMock(
            input_tokens=100, output_tokens=200
        )

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        transcript_dir = tmp_path / "transcripts"
        with patch(
            "bugeval.agent_runner.anthropic.Anthropic",
            return_value=mock_client,
        ):
            result = run_anthropic_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
                transcript_dir=transcript_dir,
                thinking_budget=8000,
            )

        assert result.error == ""
        assert len(result.comments) == 1
        assert result.comments[0].file == "f.rs"
        # Verify thinking is in transcript
        import json

        data = json.loads(Path(result.transcript_path).read_text())
        # The assistant message has the response content
        assistant_msgs = [m for m in data if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        content = assistant_msgs[0]["content"]
        thinking_items = [
            c for c in content
            if isinstance(c, dict) and c.get("type") == "thinking"
        ]
        assert len(thinking_items) == 1
        assert thinking_items[0]["thinking"] == "Let me analyze this diff carefully..."

    def test_thinking_budget_in_kwargs(self) -> None:
        """Verify thinking config is passed to the API when budget > 0."""
        case = _make_case()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "[]"

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch(
            "bugeval.agent_runner.anthropic.Anthropic",
            return_value=mock_client,
        ):
            run_anthropic_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
                thinking_budget=8000,
            )

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs.get("thinking") == {
            "type": "enabled",
            "budget_tokens": 8000,
        }
        # max_tokens must be > budget_tokens
        assert call_kwargs.kwargs["max_tokens"] >= 8000 + 4096

    def test_thinking_not_enabled_when_zero(self) -> None:
        """Verify thinking config is NOT passed when budget is 0."""
        case = _make_case()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "[]"

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch(
            "bugeval.agent_runner.anthropic.Anthropic",
            return_value=mock_client,
        ):
            run_anthropic_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
                thinking_budget=0,
            )

        call_kwargs = mock_client.messages.create.call_args
        assert "thinking" not in call_kwargs.kwargs

    def test_thinking_not_in_findings(self) -> None:
        """Thinking text should not be parsed as findings."""
        case = _make_case()

        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = (
            '[{"file":"fake.rs","line":99,"description":"from thinking"}]'
        )

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "[]"

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [thinking_block, text_block]
        mock_response.usage = MagicMock(input_tokens=50, output_tokens=100)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch(
            "bugeval.agent_runner.anthropic.Anthropic",
            return_value=mock_client,
        ):
            result = run_anthropic_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
                thinking_budget=8000,
            )

        # Thinking block contains JSON that looks like findings,
        # but it should NOT be parsed — only text blocks are parsed
        assert len(result.comments) == 0


class TestSaveTranscriptThinkingBlocks:
    def test_thinking_blocks_serialized(self, tmp_path: Path) -> None:
        """Verify thinking blocks are serialized correctly in transcripts."""
        thinking_block = MagicMock()
        thinking_block.type = "thinking"
        thinking_block.thinking = "Deep analysis of the code..."

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Here are my findings."

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "read_file"
        tool_block.input = {"path": "foo.rs"}
        tool_block.id = "tool_123"

        messages: list[dict[str, object]] = [
            {"role": "user", "content": "Review this diff"},
            {
                "role": "assistant",
                "content": [thinking_block, text_block, tool_block],
            },
        ]

        import json

        path = _save_transcript(
            messages, tmp_path, "test-001"  # type: ignore[arg-type]
        )
        data = json.loads(Path(path).read_text())

        assert data[0]["role"] == "user"
        assert data[0]["content"] == "Review this diff"

        assistant_content = data[1]["content"]
        assert len(assistant_content) == 3

        assert assistant_content[0] == {
            "type": "thinking",
            "thinking": "Deep analysis of the code...",
        }
        assert assistant_content[1] == {
            "type": "text",
            "text": "Here are my findings.",
        }
        assert assistant_content[2] == {
            "type": "tool_use",
            "name": "read_file",
            "input": {"path": "foo.rs"},
            "id": "tool_123",
        }


# ---------------------------------------------------------------------------
# Agent SDK tests
# ---------------------------------------------------------------------------

def _make_sdk_mocks() -> types.ModuleType:
    """Create a fake claude_agent_sdk module with mock classes."""
    mod = types.ModuleType("claude_agent_sdk")

    class ClaudeAgentOptions:
        def __init__(self, **kwargs: object) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class AssistantMessage:
        def __init__(self, content: list[object]) -> None:
            self.content = content

    class ResultMessage:
        def __init__(
            self, total_cost_usd: float = 0.0, session_id: str = "",
        ) -> None:
            self.total_cost_usd = total_cost_usd
            self.session_id = session_id

    mod.ClaudeAgentOptions = ClaudeAgentOptions  # type: ignore[attr-defined]
    mod.AssistantMessage = AssistantMessage  # type: ignore[attr-defined]
    mod.ResultMessage = ResultMessage  # type: ignore[attr-defined]
    return mod


def _sdk_text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


class TestRunAgentSdkSuccess:
    def test_mocked_sdk_returns_result(self, tmp_path: Path) -> None:
        sdk_mod = _make_sdk_mocks()
        AssistantMessage = sdk_mod.AssistantMessage  # type: ignore[attr-defined]
        ResultMessage = sdk_mod.ResultMessage  # type: ignore[attr-defined]

        text_block = _sdk_text_block(
            '[{"file":"f.rs","line":1,"description":"bug"}]',
        )

        async def fake_query(**kwargs: object):  # type: ignore[no-untyped-def]
            yield AssistantMessage(content=[text_block])
            yield ResultMessage(total_cost_usd=0.05, session_id="sess-123")

        sdk_mod.query = fake_query  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"claude_agent_sdk": sdk_mod}):
            case = _make_case()
            result = run_agent_sdk(
                case, SAMPLE_DIFF, None, "diff-only",
                timeout=300, transcript_dir=tmp_path / "transcripts",
            )

        assert result.case_id == "leo-001"
        assert result.tool == "agent-sdk"
        assert result.context_level == "diff-only"
        assert len(result.comments) == 1
        assert result.comments[0].file == "f.rs"
        assert result.error == ""
        assert result.cost_usd == 0.05
        assert result.time_seconds >= 0


class TestRunAgentSdkImportError:
    def test_import_error_returns_error_result(self) -> None:
        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            case = _make_case()
            result = run_agent_sdk(
                case, SAMPLE_DIFF, None, "diff-only", timeout=60,
            )

        assert result.tool == "agent-sdk"
        assert "claude-agent-sdk not installed" in result.error


class TestRunAgentSdkTimeout:
    def test_timeout_returns_error(self) -> None:
        sdk_mod = _make_sdk_mocks()
        AssistantMessage = sdk_mod.AssistantMessage  # type: ignore[attr-defined]

        async def slow_query(**kwargs: object):  # type: ignore[no-untyped-def]
            # Yield enough messages that the timeout check triggers
            for _ in range(20):
                yield AssistantMessage(content=[_sdk_text_block("partial")])

        sdk_mod.query = slow_query  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"claude_agent_sdk": sdk_mod}):
            case = _make_case()
            # Use timeout=0 so the first check (monotonic - start > 0) triggers
            result = run_agent_sdk(
                case, SAMPLE_DIFF, None, "diff-only", timeout=0,
            )

        assert "timeout" in result.error.lower()
        assert result.tool == "agent-sdk"


class TestRunAgentSdkTranscriptSaved:
    def test_transcript_file_created(self, tmp_path: Path) -> None:
        sdk_mod = _make_sdk_mocks()
        AssistantMessage = sdk_mod.AssistantMessage  # type: ignore[attr-defined]
        ResultMessage = sdk_mod.ResultMessage  # type: ignore[attr-defined]

        async def fake_query(**kwargs: object):  # type: ignore[no-untyped-def]
            yield AssistantMessage(content=[_sdk_text_block("[]")])
            yield ResultMessage(total_cost_usd=0.01, session_id="sess-t")

        sdk_mod.query = fake_query  # type: ignore[attr-defined]

        transcript_dir = tmp_path / "transcripts"
        with patch.dict(sys.modules, {"claude_agent_sdk": sdk_mod}):
            case = _make_case()
            result = run_agent_sdk(
                case, SAMPLE_DIFF, None, "diff-only",
                timeout=300, transcript_dir=transcript_dir,
            )

        assert result.transcript_path != ""
        t_path = Path(result.transcript_path)
        assert t_path.exists()
        data = json_mod.loads(t_path.read_text())
        assert data["session_id"] == "sess-t"
        assert isinstance(data["messages"], list)
        assert data["cost_usd"] == 0.01


class TestRunAgentSdkCostTracking:
    def test_cost_from_result_message(self) -> None:
        sdk_mod = _make_sdk_mocks()
        AssistantMessage = sdk_mod.AssistantMessage  # type: ignore[attr-defined]
        ResultMessage = sdk_mod.ResultMessage  # type: ignore[attr-defined]

        async def fake_query(**kwargs: object):  # type: ignore[no-untyped-def]
            yield AssistantMessage(content=[_sdk_text_block("[]")])
            yield ResultMessage(total_cost_usd=1.23, session_id="s1")

        sdk_mod.query = fake_query  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"claude_agent_sdk": sdk_mod}):
            case = _make_case()
            result = run_agent_sdk(
                case, SAMPLE_DIFF, None, "diff-only", timeout=300,
            )

        assert result.cost_usd == 1.23


class TestRunAgentSdkDiffOnlyNoTools:
    def test_allowed_tools_websearch_only_for_diff_only(self) -> None:
        sdk_mod = _make_sdk_mocks()
        AssistantMessage = sdk_mod.AssistantMessage  # type: ignore[attr-defined]
        ResultMessage = sdk_mod.ResultMessage  # type: ignore[attr-defined]

        captured_options: list[dict[str, object]] = []
        original_init = sdk_mod.ClaudeAgentOptions.__init__  # type: ignore[attr-defined]

        class TrackingOptions(sdk_mod.ClaudeAgentOptions):  # type: ignore[misc]
            def __init__(self, **kwargs: object) -> None:
                original_init(self, **kwargs)
                captured_options.append(kwargs)

        sdk_mod.ClaudeAgentOptions = TrackingOptions  # type: ignore[attr-defined]

        async def fake_query(**kwargs: object):  # type: ignore[no-untyped-def]
            yield AssistantMessage(content=[_sdk_text_block("[]")])
            yield ResultMessage(total_cost_usd=0.0)

        sdk_mod.query = fake_query  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"claude_agent_sdk": sdk_mod}):
            case = _make_case()
            run_agent_sdk(case, SAMPLE_DIFF, None, "diff-only", timeout=300)

        assert len(captured_options) == 1
        assert captured_options[0]["allowed_tools"] == ["WebSearch"]


class TestRunAgentSdkContextLevels:
    def test_tools_set_for_diff_repo(self) -> None:
        sdk_mod = _make_sdk_mocks()
        AssistantMessage = sdk_mod.AssistantMessage  # type: ignore[attr-defined]
        ResultMessage = sdk_mod.ResultMessage  # type: ignore[attr-defined]

        captured_options: list[dict[str, object]] = []
        original_init = sdk_mod.ClaudeAgentOptions.__init__  # type: ignore[attr-defined]

        class TrackingOptions(sdk_mod.ClaudeAgentOptions):  # type: ignore[misc]
            def __init__(self, **kwargs: object) -> None:
                original_init(self, **kwargs)
                captured_options.append(kwargs)

        sdk_mod.ClaudeAgentOptions = TrackingOptions  # type: ignore[attr-defined]

        async def fake_query(**kwargs: object):  # type: ignore[no-untyped-def]
            yield AssistantMessage(content=[_sdk_text_block("[]")])
            yield ResultMessage(total_cost_usd=0.0)

        sdk_mod.query = fake_query  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"claude_agent_sdk": sdk_mod}):
            case = _make_case()
            run_agent_sdk(case, SAMPLE_DIFF, None, "diff+repo", timeout=300)

        assert len(captured_options) == 1
        assert captured_options[0]["allowed_tools"] == ["Read", "Glob", "Grep", "WebSearch"]

    def test_tools_set_for_diff_repo_domain(self) -> None:
        sdk_mod = _make_sdk_mocks()
        AssistantMessage = sdk_mod.AssistantMessage  # type: ignore[attr-defined]
        ResultMessage = sdk_mod.ResultMessage  # type: ignore[attr-defined]

        captured_options: list[dict[str, object]] = []
        original_init = sdk_mod.ClaudeAgentOptions.__init__  # type: ignore[attr-defined]

        class TrackingOptions(sdk_mod.ClaudeAgentOptions):  # type: ignore[misc]
            def __init__(self, **kwargs: object) -> None:
                original_init(self, **kwargs)
                captured_options.append(kwargs)

        sdk_mod.ClaudeAgentOptions = TrackingOptions  # type: ignore[attr-defined]

        async def fake_query(**kwargs: object):  # type: ignore[no-untyped-def]
            yield AssistantMessage(content=[_sdk_text_block("[]")])
            yield ResultMessage(total_cost_usd=0.0)

        sdk_mod.query = fake_query  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"claude_agent_sdk": sdk_mod}):
            case = _make_case()
            run_agent_sdk(
                case, SAMPLE_DIFF, None, "diff+repo+domain", timeout=300,
            )

        assert len(captured_options) == 1
        assert captured_options[0]["allowed_tools"] == ["Read", "Glob", "Grep", "WebSearch"]


# ---------------------------------------------------------------------------
# Model override tests
# ---------------------------------------------------------------------------


class TestRunAgentApiModelOverride:
    def test_model_override_passed_to_api(self) -> None:
        """When model is set, it should override the default MODEL constant."""
        case = _make_case()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "[]"

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch(
            "bugeval.agent_runner.anthropic.Anthropic",
            return_value=mock_client,
        ):
            run_anthropic_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
                model="claude-opus-4-6",
            )

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-opus-4-6"

    def test_empty_model_uses_default(self) -> None:
        """When model is empty, the default MODEL constant is used."""
        case = _make_case()

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "[]"

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch(
            "bugeval.agent_runner.anthropic.Anthropic",
            return_value=mock_client,
        ):
            run_anthropic_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
                model="",
            )

        from bugeval.agent_runner import MODEL

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == MODEL


class TestRunAgentSdkModelOverride:
    def test_model_override_passed_to_sdk(self) -> None:
        sdk_mod = _make_sdk_mocks()
        AssistantMessage = sdk_mod.AssistantMessage  # type: ignore[attr-defined]
        ResultMessage = sdk_mod.ResultMessage  # type: ignore[attr-defined]

        captured_options: list[dict[str, object]] = []
        original_init = sdk_mod.ClaudeAgentOptions.__init__  # type: ignore[attr-defined]

        class TrackingOptions(sdk_mod.ClaudeAgentOptions):  # type: ignore[misc]
            def __init__(self, **kwargs: object) -> None:
                original_init(self, **kwargs)
                captured_options.append(kwargs)

        sdk_mod.ClaudeAgentOptions = TrackingOptions  # type: ignore[attr-defined]

        async def fake_query(**kwargs: object):  # type: ignore[no-untyped-def]
            yield AssistantMessage(content=[_sdk_text_block("[]")])
            yield ResultMessage(total_cost_usd=0.0)

        sdk_mod.query = fake_query  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"claude_agent_sdk": sdk_mod}):
            case = _make_case()
            run_agent_sdk(
                case, SAMPLE_DIFF, None, "diff-only",
                timeout=300, model="claude-opus-4-6",
            )

        assert len(captured_options) == 1
        assert captured_options[0]["model"] == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Google Gemini API tests
# ---------------------------------------------------------------------------


def _make_google_mocks() -> tuple[
    types.ModuleType, types.ModuleType, types.ModuleType,
]:
    """Create fake google, google.genai, and google.genai.types modules."""
    google_mod = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    types_mod = types.ModuleType("google.genai.types")

    class FunctionDeclaration:
        def __init__(
            self, name: str = "", description: str = "",
            parameters: object = None,
        ) -> None:
            self.name = name
            self.description = description
            self.parameters = parameters

    class GoogleSearch:
        pass

    class Tool:
        def __init__(
            self, function_declarations: list[object] | None = None,
            google_search: object | None = None,
        ) -> None:
            self.function_declarations = function_declarations
            self.google_search = google_search

    class Content:
        def __init__(
            self, role: str = "", parts: list[object] | None = None,
        ) -> None:
            self.role = role
            self.parts = parts or []

    class Part:
        def __init__(self, text: str = "") -> None:
            self.text = text
            self.function_call = None

        @staticmethod
        def from_text(text: str = "") -> Part:
            return Part(text=text)

        @staticmethod
        def from_function_response(
            name: str = "", response: object = None,
        ) -> Part:
            p = Part()
            p.text = ""
            return p

    class FunctionCall:
        def __init__(
            self, name: str = "",
            args: dict[str, object] | None = None,
        ) -> None:
            self.name = name
            self.args = args or {}

    class GenerateContentConfig:
        def __init__(self, **kwargs: object) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    types_mod.FunctionDeclaration = FunctionDeclaration  # type: ignore[attr-defined]
    types_mod.GoogleSearch = GoogleSearch  # type: ignore[attr-defined]
    types_mod.Tool = Tool  # type: ignore[attr-defined]
    types_mod.Content = Content  # type: ignore[attr-defined]
    types_mod.Part = Part  # type: ignore[attr-defined]
    types_mod.GenerateContentConfig = GenerateContentConfig  # type: ignore[attr-defined]
    types_mod.FunctionCall = FunctionCall  # type: ignore[attr-defined]

    # Wire up the module hierarchy
    google_mod.genai = genai_mod  # type: ignore[attr-defined]
    genai_mod.types = types_mod  # type: ignore[attr-defined]

    return google_mod, genai_mod, types_mod


def _make_google_text_response(
    text: str, inp_tokens: int = 100, out_tokens: int = 50,
) -> MagicMock:
    """Build a mock Google generate_content response with text only."""
    part = MagicMock()
    part.text = text
    part.function_call = None

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    usage = MagicMock()
    usage.prompt_token_count = inp_tokens
    usage.candidates_token_count = out_tokens

    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata = usage
    return response


def _make_google_tool_response(
    fn_name: str, fn_args: dict[str, object],
    inp_tokens: int = 50, out_tokens: int = 20,
) -> MagicMock:
    """Build a mock Google response with a function call."""
    fc = MagicMock()
    fc.name = fn_name
    fc.args = fn_args

    part = MagicMock()
    part.text = None
    part.function_call = fc

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    usage = MagicMock()
    usage.prompt_token_count = inp_tokens
    usage.candidates_token_count = out_tokens

    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata = usage
    return response


class TestRunGoogleApiDiffOnly:
    def test_mocked_google_returns_result(self) -> None:
        case = _make_case()
        google_mod, genai_mod, types_mod = _make_google_mocks()

        text_resp = _make_google_text_response(
            '[{"file":"f.rs","line":1,"description":"bug"}]'
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = text_resp

        genai_mod.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {
            "google": google_mod,
            "google.genai": genai_mod,
            "google.genai.types": types_mod,
        }):
            result = run_google_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
            )

        assert result.case_id == "leo-001"
        assert result.tool == "agent-gemini"
        assert result.context_level == "diff-only"
        assert len(result.comments) == 1
        assert result.comments[0].file == "f.rs"
        assert result.error == ""
        assert result.cost_usd > 0

    def test_import_error_returns_error_result(self) -> None:
        with patch.dict(sys.modules, {
            "google": None,
            "google.genai": None,
        }):
            case = _make_case()
            result = run_google_api(
                case, SAMPLE_DIFF, None, "diff-only", timeout=60,
            )

        assert result.tool == "agent-gemini"
        assert "google-genai not installed" in result.error


class TestRunGoogleApiMultiTurn:
    def test_tool_use_then_text(self) -> None:
        case = _make_case()
        google_mod, genai_mod, types_mod = _make_google_mocks()

        tool_resp = _make_google_tool_response(
            "read_file", {"path": "foo.rs"},
        )
        text_resp = _make_google_text_response(
            '[{"file":"foo.rs","line":1,"description":"bug"}]'
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            tool_resp, text_resp,
        ]

        genai_mod.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {
            "google": google_mod,
            "google.genai": genai_mod,
            "google.genai.types": types_mod,
        }):
            result = run_google_api(
                case, SAMPLE_DIFF, None, "diff+repo",
                max_turns=5, timeout=300,
            )

        assert len(result.comments) == 1
        assert result.comments[0].file == "foo.rs"
        assert result.error == ""
        assert mock_client.models.generate_content.call_count == 2


class TestRunGoogleApiCost:
    def test_cost_tracked(self) -> None:
        case = _make_case()
        google_mod, genai_mod, types_mod = _make_google_mocks()

        text_resp = _make_google_text_response(
            "[]", inp_tokens=1000, out_tokens=500,
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = text_resp

        genai_mod.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {
            "google": google_mod,
            "google.genai": genai_mod,
            "google.genai.types": types_mod,
        }):
            result = run_google_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
            )

        # 1000 * 0.15/1M + 500 * 0.60/1M
        assert result.cost_usd > 0
        assert result.cost_usd < 0.01


# ---------------------------------------------------------------------------
# OpenAI API tests
# ---------------------------------------------------------------------------


def _make_openai_text_response(
    text: str, inp_tokens: int = 100, out_tokens: int = 50,
) -> MagicMock:
    """Build a mock OpenAI chat completion response with text only."""
    message = MagicMock()
    message.content = text
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    usage = MagicMock()
    usage.prompt_tokens = inp_tokens
    usage.completion_tokens = out_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_openai_tool_response(
    fn_name: str, fn_args: str, tool_call_id: str = "call_1",
    inp_tokens: int = 50, out_tokens: int = 20,
) -> MagicMock:
    """Build a mock OpenAI response with tool calls."""
    fn = MagicMock()
    fn.name = fn_name
    fn.arguments = fn_args

    tc = MagicMock()
    tc.id = tool_call_id
    tc.function = fn

    message = MagicMock()
    message.content = None
    message.tool_calls = [tc]

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls"

    usage = MagicMock()
    usage.prompt_tokens = inp_tokens
    usage.completion_tokens = out_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


class TestRunOpenaiApiDiffOnly:
    def test_mocked_openai_returns_result(self) -> None:
        case = _make_case()

        text_resp = _make_openai_text_response(
            '[{"file":"f.rs","line":1,"description":"bug"}]'
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = text_resp

        openai_mod = types.ModuleType("openai")
        openai_mod.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"openai": openai_mod}):
            result = run_openai_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
            )

        assert result.case_id == "leo-001"
        assert result.tool == "agent-openai"
        assert result.context_level == "diff-only"
        assert len(result.comments) == 1
        assert result.comments[0].file == "f.rs"
        assert result.error == ""
        assert result.cost_usd > 0

    def test_import_error_returns_error_result(self) -> None:
        with patch.dict(sys.modules, {"openai": None}):
            case = _make_case()
            result = run_openai_api(
                case, SAMPLE_DIFF, None, "diff-only", timeout=60,
            )

        assert result.tool == "agent-openai"
        assert "openai not installed" in result.error


class TestRunOpenaiApiMultiTurn:
    def test_tool_use_then_text(self) -> None:
        case = _make_case()

        tool_resp = _make_openai_tool_response(
            "read_file", '{"path": "foo.rs"}', "call_1",
        )
        text_resp = _make_openai_text_response(
            '[{"file":"foo.rs","line":1,"description":"bug"}]'
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [tool_resp, text_resp]

        openai_mod = types.ModuleType("openai")
        openai_mod.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"openai": openai_mod}):
            result = run_openai_api(
                case, SAMPLE_DIFF, None, "diff+repo",
                max_turns=5, timeout=300,
            )

        assert len(result.comments) == 1
        assert result.comments[0].file == "foo.rs"
        assert result.error == ""
        assert mock_client.chat.completions.create.call_count == 2


class TestRunOpenaiApiCost:
    def test_cost_tracked(self) -> None:
        case = _make_case()

        text_resp = _make_openai_text_response("[]", inp_tokens=1000, out_tokens=500)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = text_resp

        openai_mod = types.ModuleType("openai")
        openai_mod.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"openai": openai_mod}):
            result = run_openai_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
            )

        # 1000 * 1.10/1M + 500 * 4.40/1M = 0.0011 + 0.0022 = 0.0033
        assert result.cost_usd > 0
        assert result.cost_usd < 0.01


# ---------------------------------------------------------------------------
# Google Search grounding tests
# ---------------------------------------------------------------------------


class TestRunGoogleApiSearchGrounding:
    def test_google_search_tool_included_diff_only(self) -> None:
        """Google Search grounding is added even for diff-only (no func tools)."""
        case = _make_case()
        google_mod, genai_mod, types_mod = _make_google_mocks()

        text_resp = _make_google_text_response("[]")
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = text_resp
        genai_mod.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {
            "google": google_mod,
            "google.genai": genai_mod,
            "google.genai.types": types_mod,
        }):
            run_google_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
            )

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config")
        tools = getattr(config, "tools", None)
        assert tools is not None
        # Should have exactly one Tool: google_search (no function decls)
        has_search = any(
            getattr(t, "google_search", None) is not None for t in tools
        )
        assert has_search, "Google Search grounding tool not found"
        has_func = any(
            getattr(t, "function_declarations", None) is not None
            for t in tools
        )
        assert not has_func, "diff-only should not have function tools"

    def test_google_search_tool_alongside_function_tools(self) -> None:
        """Google Search grounding coexists with function declaration tools."""
        case = _make_case()
        google_mod, genai_mod, types_mod = _make_google_mocks()

        text_resp = _make_google_text_response("[]")
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = text_resp
        genai_mod.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {
            "google": google_mod,
            "google.genai": genai_mod,
            "google.genai.types": types_mod,
        }):
            run_google_api(
                case, SAMPLE_DIFF, None, "diff+repo",
                max_turns=5, timeout=300,
            )

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config")
        tools = getattr(config, "tools", None)
        assert tools is not None
        has_search = any(
            getattr(t, "google_search", None) is not None for t in tools
        )
        has_func = any(
            getattr(t, "function_declarations", None) is not None
            for t in tools
        )
        assert has_search, "Google Search grounding tool not found"
        assert has_func, "Function declaration tools not found"

    def test_graceful_fallback_when_google_search_missing(self) -> None:
        """If GoogleSearch is absent from SDK, runner still works."""
        case = _make_case()
        google_mod, genai_mod, types_mod = _make_google_mocks()

        # Remove GoogleSearch to simulate old SDK
        del types_mod.GoogleSearch  # type: ignore[attr-defined]

        text_resp = _make_google_text_response("[]")
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = text_resp
        genai_mod.Client = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {
            "google": google_mod,
            "google.genai": genai_mod,
            "google.genai.types": types_mod,
        }):
            result = run_google_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
            )

        # Should succeed without error (no google_search, but still works)
        assert result.error == ""


# ---------------------------------------------------------------------------
# OpenAI web search tool tests
# ---------------------------------------------------------------------------


class TestRunOpenaiApiWebSearch:
    def test_web_search_preview_included_diff_only(self) -> None:
        """web_search_preview is present even for diff-only (no func tools)."""
        case = _make_case()

        text_resp = _make_openai_text_response("[]")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = text_resp

        openai_mod = types.ModuleType("openai")
        openai_mod.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"openai": openai_mod}):
            run_openai_api(
                case, SAMPLE_DIFF, None, "diff-only",
                max_turns=5, timeout=300,
            )

        call_kwargs = mock_client.chat.completions.create.call_args
        tools = call_kwargs.kwargs.get("tools")
        assert tools is not None
        web_tools = [t for t in tools if t.get("type") == "web_search_preview"]
        assert len(web_tools) == 1
        # diff-only: no function tools
        func_tools = [t for t in tools if t.get("type") == "function"]
        assert len(func_tools) == 0

    def test_web_search_preview_alongside_function_tools(self) -> None:
        """web_search_preview coexists with function tools for diff+repo."""
        case = _make_case()

        text_resp = _make_openai_text_response("[]")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = text_resp

        openai_mod = types.ModuleType("openai")
        openai_mod.OpenAI = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"openai": openai_mod}):
            run_openai_api(
                case, SAMPLE_DIFF, None, "diff+repo",
                max_turns=5, timeout=300,
            )

        call_kwargs = mock_client.chat.completions.create.call_args
        tools = call_kwargs.kwargs.get("tools")
        assert tools is not None
        web_tools = [t for t in tools if t.get("type") == "web_search_preview"]
        func_tools = [t for t in tools if t.get("type") == "function"]
        assert len(web_tools) == 1
        assert len(func_tools) == 3  # read_file, list_directory, search_text
