"""Tests for result models."""

from __future__ import annotations

from bugeval.result_models import Comment, ToolResult


class TestComment:
    def test_defaults(self) -> None:
        c = Comment()
        assert c.file == ""
        assert c.line == 0
        assert c.body == ""
        assert c.suggested_fix == ""

    def test_with_data(self) -> None:
        c = Comment(file="x.rs", line=10, body="Bug here", suggested_fix="fix it")
        assert c.file == "x.rs"


class TestToolResult:
    def test_minimal(self) -> None:
        r = ToolResult(case_id="t-001", tool="copilot")
        assert r.comments == []
        assert r.error == ""
        assert not r.potentially_contaminated

    def test_full(self, sample_result: ToolResult) -> None:
        assert sample_result.case_id == "snarkVM-001"
        assert len(sample_result.comments) == 2
        assert sample_result.time_seconds == 45.2
