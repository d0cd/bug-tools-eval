"""Tests for result_models."""

from pathlib import Path

import yaml

from bugeval.result_models import Comment, CommentType, NormalizedResult, ResultMetadata


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
