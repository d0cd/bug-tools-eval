"""Tests for score models."""

from __future__ import annotations

from bugeval.score_models import CaseScore, CommentScore, CommentVerdict


class TestCommentVerdict:
    def test_values(self) -> None:
        assert CommentVerdict.tp == "TP"
        assert CommentVerdict.tp_novel == "TP-novel"
        assert CommentVerdict.fp == "FP"
        assert CommentVerdict.low_value == "low-value"


class TestCommentScore:
    def test_tp(self) -> None:
        cs = CommentScore(comment_index=0, verdict=CommentVerdict.tp, matched_buggy_line=0)
        assert cs.matched_buggy_line == 0

    def test_fp(self) -> None:
        cs = CommentScore(comment_index=1, verdict=CommentVerdict.fp)
        assert cs.matched_buggy_line is None


class TestCaseScore:
    def test_defaults(self) -> None:
        s = CaseScore(case_id="t-001", tool="copilot")
        assert not s.caught
        assert s.detection_score == 0
        assert s.review_quality == 0
        assert not s.false_alarm

    def test_full(self, sample_score: CaseScore) -> None:
        assert sample_score.caught is True
        assert sample_score.detection_score == 3
        assert len(sample_score.comment_scores) == 2
