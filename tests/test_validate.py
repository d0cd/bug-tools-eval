"""Tests for cross-model validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from bugeval.models import (
    BuggyLine,
    CaseKind,
    GroundTruth,
    TestCase,
    Validation,
)
from bugeval.validate import (
    build_validation_prompt,
    parse_verdict,
    validate_case,
    validate_cases,
)


def _make_case(
    case_id: str = "test-001",
    *,
    truth: GroundTruth | None = None,
    validation: Validation | None = None,
) -> TestCase:
    return TestCase(
        id=case_id,
        repo="ProvableHQ/snarkVM",
        kind=CaseKind.bug,
        base_commit="abc123",
        fix_commit="def456",
        bug_description="Off-by-one in loop counter",
        truth=truth,
        validation=validation,
    )


def _default_truth() -> GroundTruth:
    return GroundTruth(
        introducing_commit="abc123",
        blame_confidence="A",
        buggy_lines=[
            BuggyLine(file="src/lib.rs", line=42, content="for i in 0..n {"),
        ],
        fix_summary="Changed loop bound to 0..=n",
    )


SAMPLE_DIFF = """\
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -40,3 +40,3 @@
-    for i in 0..=n {
+    for i in 0..n {
"""


class TestBuildValidationPrompt:
    def test_includes_diff(self) -> None:
        case = _make_case(truth=_default_truth())
        prompt = build_validation_prompt(case, SAMPLE_DIFF)
        assert "for i in 0..n {" in prompt

    def test_includes_bug_description(self) -> None:
        case = _make_case(truth=_default_truth())
        prompt = build_validation_prompt(case, SAMPLE_DIFF)
        assert "Off-by-one in loop counter" in prompt

    def test_includes_buggy_lines(self) -> None:
        case = _make_case(truth=_default_truth())
        prompt = build_validation_prompt(case, SAMPLE_DIFF)
        assert "src/lib.rs" in prompt
        assert "42" in prompt

    def test_requests_json_response(self) -> None:
        case = _make_case(truth=_default_truth())
        prompt = build_validation_prompt(case, SAMPLE_DIFF)
        assert "verdict" in prompt
        assert "confirmed" in prompt


class TestParseVerdict:
    def test_valid_json(self) -> None:
        response = json.dumps({"verdict": "confirmed", "reasoning": "clear"})
        assert parse_verdict(response) == "confirmed"

    def test_malformed(self) -> None:
        assert parse_verdict("not json at all") == "ambiguous"

    def test_fenced_json(self) -> None:
        response = '```json\n{"verdict": "disputed", "reasoning": "nope"}\n```'
        assert parse_verdict(response) == "disputed"

    def test_missing_verdict_key(self) -> None:
        response = json.dumps({"answer": "yes"})
        assert parse_verdict(response) == "ambiguous"

    def test_invalid_verdict_value(self) -> None:
        response = json.dumps({"verdict": "maybe"})
        assert parse_verdict(response) == "ambiguous"


class TestValidateCase:
    @patch("bugeval.validate.call_claude")
    @patch("bugeval.validate.call_gemini")
    def test_agreement(
        self, mock_gemini: MagicMock, mock_claude: MagicMock
    ) -> None:
        mock_claude.return_value = json.dumps({"verdict": "confirmed"})
        mock_gemini.return_value = json.dumps({"verdict": "confirmed"})
        case = _make_case(truth=_default_truth())
        result = validate_case(case, SAMPLE_DIFF, ["claude", "gemini"])
        assert result.agreement is True
        assert result.claude_verdict == "confirmed"
        assert result.gemini_verdict == "confirmed"

    @patch("bugeval.validate.call_claude")
    @patch("bugeval.validate.call_gemini")
    def test_disagreement(
        self, mock_gemini: MagicMock, mock_claude: MagicMock
    ) -> None:
        mock_claude.return_value = json.dumps({"verdict": "confirmed"})
        mock_gemini.return_value = json.dumps({"verdict": "disputed"})
        case = _make_case(truth=_default_truth())
        result = validate_case(case, SAMPLE_DIFF, ["claude", "gemini"])
        assert result.agreement is False
        assert result.claude_verdict == "confirmed"
        assert result.gemini_verdict == "disputed"

    @patch("bugeval.validate.call_claude")
    @patch("bugeval.validate.call_gemini")
    def test_single_model_claude_only(
        self, mock_gemini: MagicMock, mock_claude: MagicMock
    ) -> None:
        mock_claude.return_value = json.dumps({"verdict": "confirmed"})
        case = _make_case(truth=_default_truth())
        result = validate_case(case, SAMPLE_DIFF, ["claude"])
        assert result.claude_verdict == "confirmed"
        assert result.gemini_verdict == ""
        # Single model: agreement is vacuously true
        assert result.agreement is True
        mock_gemini.assert_not_called()


class TestValidateCases:
    @patch("bugeval.validate.call_claude")
    @patch("bugeval.validate.call_gemini")
    def test_dry_run_skips_llm_calls(
        self,
        mock_gemini: MagicMock,
        mock_claude: MagicMock,
        tmp_path: Path,
    ) -> None:
        from bugeval.io import save_case

        case = _make_case(truth=_default_truth())
        save_case(case, tmp_path / "cases" / "test-001.yaml")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        validate_cases(
            tmp_path / "cases", repo_dir, ["claude", "gemini"], 1, dry_run=True
        )
        mock_claude.assert_not_called()
        mock_gemini.assert_not_called()

    @patch("bugeval.validate.call_claude")
    @patch("bugeval.validate.call_gemini")
    def test_skips_already_validated(
        self,
        mock_gemini: MagicMock,
        mock_claude: MagicMock,
        tmp_path: Path,
    ) -> None:
        from bugeval.io import save_case

        case = _make_case(
            truth=_default_truth(),
            validation=Validation(
                claude_verdict="confirmed",
                gemini_verdict="confirmed",
                agreement=True,
                test_validated=True,
            ),
        )
        save_case(case, tmp_path / "cases" / "test-001.yaml")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        validate_cases(
            tmp_path / "cases", repo_dir, ["claude", "gemini"], 1, dry_run=False
        )
        mock_claude.assert_not_called()
        mock_gemini.assert_not_called()

    @patch("bugeval.validate.call_claude")
    @patch("bugeval.validate.call_gemini")
    def test_skips_no_truth(
        self,
        mock_gemini: MagicMock,
        mock_claude: MagicMock,
        tmp_path: Path,
    ) -> None:
        from bugeval.io import save_case

        case = _make_case(truth=None)
        save_case(case, tmp_path / "cases" / "test-001.yaml")
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        validate_cases(
            tmp_path / "cases", repo_dir, ["claude", "gemini"], 1, dry_run=False
        )
        mock_claude.assert_not_called()
        mock_gemini.assert_not_called()

    @patch("bugeval.validate._get_introducing_diff", return_value=SAMPLE_DIFF)
    @patch("bugeval.validate.call_claude")
    @patch("bugeval.validate.call_gemini")
    def test_checkpoint_resume(
        self,
        mock_gemini: MagicMock,
        mock_claude: MagicMock,
        mock_diff: MagicMock,
        tmp_path: Path,
    ) -> None:
        from bugeval.io import save_case, save_checkpoint

        # Create two cases with ground truth
        case1 = _make_case("test-001", truth=_default_truth())
        case2 = _make_case("test-002", truth=_default_truth())
        cases_dir = tmp_path / "cases"
        save_case(case1, cases_dir / "test-001.yaml")
        save_case(case2, cases_dir / "test-002.yaml")

        # Mark case1 as already done in checkpoint
        save_checkpoint({"test-001"}, cases_dir / ".validate_checkpoint.json")

        mock_claude.return_value = json.dumps({"verdict": "confirmed"})
        mock_gemini.return_value = json.dumps({"verdict": "confirmed"})

        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        validate_cases(
            cases_dir, repo_dir, ["claude", "gemini"], 1, dry_run=False
        )
        # Only case2 should have been processed
        assert mock_claude.call_count == 1
