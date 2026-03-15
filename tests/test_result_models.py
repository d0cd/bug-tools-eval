"""Tests for result_models."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bugeval.result_models import (
    Comment,
    CommentType,
    DxAssessment,
    NormalizedResult,
    ResultMetadata,
)


def test_comment_defaults() -> None:
    c = Comment(body="looks off")
    assert c.file == ""
    assert c.line == 0
    assert c.type == CommentType.inline


def test_normalized_result_yaml_round_trip(tmp_path: Path) -> None:
    r = NormalizedResult(
        test_case_id="case-001",
        tool="coderabbit",
        context_level="diff-only",
        comments=[Comment(body="bug here", file="a.rs", line=10)],
        metadata=ResultMetadata(tokens=100, cost_usd=0.01, time_seconds=5.0),
    )
    path = tmp_path / "result.yaml"
    path.write_text(yaml.safe_dump(r.model_dump(mode="json"), sort_keys=False))
    loaded = NormalizedResult(**yaml.safe_load(path.read_text()))
    assert loaded.test_case_id == "case-001"
    assert loaded.comments[0].file == "a.rs"
    assert loaded.comments[0].type == CommentType.inline
    assert loaded.metadata.tokens == 100


def test_comment_pr_level_type_yaml_round_trip(tmp_path: Path) -> None:
    import yaml

    c = Comment(body="general review", type=CommentType.pr_level)
    data = c.model_dump(mode="json")
    path = tmp_path / "comment.yaml"
    path.write_text(yaml.safe_dump(data))
    loaded = Comment(**yaml.safe_load(path.read_text()))
    assert loaded.type == CommentType.pr_level


def test_normalized_result_defaults() -> None:
    r = NormalizedResult(test_case_id="x", tool="y")
    assert r.context_level == ""
    assert r.comments == []
    assert r.metadata.tokens == 0
    assert r.metadata.cost_usd == 0.0
    assert r.metadata.time_seconds == 0.0


class TestDxAssessment:
    def test_dx_assessment_valid(self) -> None:
        dx = DxAssessment(
            actionability=4, false_positive_burden=2, integration_friction=3, response_latency=5
        )
        assert dx.actionability == 4
        assert dx.response_latency == 5

    def test_dx_assessment_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            DxAssessment(actionability=0)
        with pytest.raises(ValidationError):
            DxAssessment(actionability=6)

    def test_dx_assessment_defaults(self) -> None:
        dx = DxAssessment()
        assert dx.actionability == 3
        assert dx.false_positive_burden == 3
        assert dx.integration_friction == 3
        assert dx.response_latency == 3
        assert dx.notes == ""

    def test_normalized_result_dx_none_default(self) -> None:
        r = NormalizedResult(test_case_id="x", tool="y")
        assert r.dx is None


def test_comment_enriched_fields_defaults() -> None:
    c = Comment(body="test")
    assert c.confidence is None
    assert c.severity is None
    assert c.category is None
    assert c.suggested_fix is None
    assert c.reasoning is None


def test_comment_enriched_fields_round_trip(tmp_path: Path) -> None:
    import yaml

    c = Comment(
        body="bug found",
        file="src/lib.rs",
        line=10,
        confidence=0.9,
        severity="high",
        category="memory-safety",
        suggested_fix="Change X to Y",
        reasoning="Because Z",
    )
    data = c.model_dump(mode="json")
    path = tmp_path / "comment.yaml"
    path.write_text(yaml.safe_dump(data))
    loaded = Comment(**yaml.safe_load(path.read_text()))
    assert loaded.confidence == 0.9
    assert loaded.severity == "high"
    assert loaded.category == "memory-safety"
    assert loaded.suggested_fix == "Change X to Y"
    assert loaded.reasoning == "Because Z"


def test_comment_backward_compat_no_enriched_fields() -> None:
    """Old YAMLs without enriched fields should load fine."""
    data = {"body": "legacy comment", "file": "a.rs", "line": 5}
    c = Comment(**data)
    assert c.confidence is None
    assert c.suggested_fix is None
