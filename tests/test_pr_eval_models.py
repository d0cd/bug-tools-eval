"""Tests for pr_eval_models."""

from pathlib import Path

import pytest
import yaml

from bugeval.pr_eval_models import (
    CaseToolState,
    CaseToolStatus,
    EvalConfig,
    JudgingConfig,
    RunState,
    ScoringConfig,
    ToolDef,
    ToolType,
    default_judging,
    default_scoring,
    load_eval_config,
)


def test_case_tool_state_defaults() -> None:
    state = CaseToolState(case_id="case-001", tool="coderabbit")
    assert state.status == CaseToolStatus.pending
    assert state.pr_number is None
    assert state.branch_name is None
    assert state.error is None


def test_run_state_get_missing_returns_pending() -> None:
    rs = RunState()
    state = rs.get("case-001", "coderabbit")
    assert state.status == CaseToolStatus.pending
    assert state.case_id == "case-001"
    assert state.tool == "coderabbit"


def test_run_state_set_and_get() -> None:
    rs = RunState()
    state = CaseToolState(case_id="case-001", tool="coderabbit", status=CaseToolStatus.done)
    rs.set(state)
    retrieved = rs.get("case-001", "coderabbit")
    assert retrieved.status == CaseToolStatus.done


def test_run_state_yaml_round_trip(tmp_path: Path) -> None:
    rs = RunState()
    s1 = CaseToolState(
        case_id="case-001",
        tool="coderabbit",
        status=CaseToolStatus.failed,
        pr_number=42,
        branch_name="bugeval/case-001-coderabbit",
        error="something went wrong",
        started_at="2026-03-03T12:00:00+00:00",
        completed_at="2026-03-03T12:05:00+00:00",
    )
    rs.set(s1)

    checkpoint = tmp_path / "checkpoint.yaml"
    rs.save(checkpoint)

    loaded = RunState.load(checkpoint)
    retrieved = loaded.get("case-001", "coderabbit")
    assert retrieved.status == CaseToolStatus.failed
    assert retrieved.pr_number == 42
    assert retrieved.branch_name == "bugeval/case-001-coderabbit"
    assert retrieved.error == "something went wrong"
    assert retrieved.started_at == "2026-03-03T12:00:00+00:00"
    assert retrieved.completed_at == "2026-03-03T12:05:00+00:00"


def test_run_state_load_nonexistent(tmp_path: Path) -> None:
    rs = RunState.load(tmp_path / "nope.yaml")
    assert rs.pairs == {}


def test_tool_def_is_pr_tool() -> None:
    pr_tool = ToolDef(name="coderabbit", type=ToolType.pr)
    api_tool = ToolDef(name="greptile", type=ToolType.api)
    agent_tool = ToolDef(name="claude-code-cli", type=ToolType.agent)
    assert pr_tool.is_pr_tool is True
    assert api_tool.is_pr_tool is False
    assert agent_tool.is_pr_tool is False


def test_tool_def_is_api_tool() -> None:
    pr_tool = ToolDef(name="coderabbit", type=ToolType.pr)
    api_tool = ToolDef(name="greptile", type=ToolType.api)
    agent_tool = ToolDef(name="claude-code-cli", type=ToolType.agent)
    assert pr_tool.is_api_tool is False
    assert api_tool.is_api_tool is True
    assert agent_tool.is_api_tool is False


def test_tool_def_api_fields() -> None:
    tool = ToolDef(
        name="greptile",
        type=ToolType.api,
        api_endpoint="https://api.greptile.com/v2/review",
        api_key_env="GREPTILE_API_KEY",
    )
    assert tool.api_endpoint == "https://api.greptile.com/v2/review"
    assert tool.api_key_env == "GREPTILE_API_KEY"


def test_tool_def_api_fields_optional() -> None:
    tool = ToolDef(name="greptile", type=ToolType.api)
    assert tool.api_endpoint is None
    assert tool.api_key_env is None


def test_eval_config_api_tools_filtering() -> None:
    tools = [
        ToolDef(name="coderabbit", type=ToolType.pr),
        ToolDef(name="greptile", type=ToolType.api),
        ToolDef(name="anthropic-api", type=ToolType.api),
    ]
    config = EvalConfig(eval_org="my-org", tools=tools)
    api_tools = config.api_tools
    assert len(api_tools) == 2
    assert {t.name for t in api_tools} == {"greptile", "anthropic-api"}


def test_case_tool_status_new_values() -> None:
    assert CaseToolStatus.submitting == "submitting"
    assert CaseToolStatus.collecting == "collecting"


def test_case_tool_status_yaml_round_trip(tmp_path: Path) -> None:
    rs = RunState()
    state = CaseToolState(case_id="case-001", tool="greptile", status=CaseToolStatus.submitting)
    rs.set(state)
    checkpoint = tmp_path / "checkpoint.yaml"
    rs.save(checkpoint)
    loaded = RunState.load(checkpoint)
    retrieved = loaded.get("case-001", "greptile")
    assert retrieved.status == CaseToolStatus.submitting


def test_load_eval_config_with_api_fields(tmp_path: Path) -> None:
    config_data = {
        "github": {"eval_org": "provable-eval"},
        "tools": [
            {
                "name": "greptile",
                "type": "api",
                "api_endpoint": "https://api.greptile.com/v2/review",
                "api_key_env": "GREPTILE_API_KEY",
                "cooldown_seconds": 30,
            },
        ],
        "repos": {},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_eval_config(config_path)
    assert len(config.api_tools) == 1
    tool = config.api_tools[0]
    assert tool.api_endpoint == "https://api.greptile.com/v2/review"
    assert tool.api_key_env == "GREPTILE_API_KEY"


def test_tool_type_enum_values() -> None:
    assert ToolType.pr == "pr"
    assert ToolType.api == "api"
    assert ToolType.agent == "agent"


def test_case_tool_status_agent_values() -> None:
    assert CaseToolStatus.cloning == "cloning"
    assert CaseToolStatus.running == "running"


def test_tool_def_is_agent_tool() -> None:
    agent_tool = ToolDef(name="claude-code-cli", type=ToolType.agent)
    pr_tool = ToolDef(name="coderabbit", type=ToolType.pr)
    assert agent_tool.is_agent_tool is True
    assert pr_tool.is_agent_tool is False


def test_eval_config_agent_tools_filtering() -> None:
    tools = [
        ToolDef(name="coderabbit", type=ToolType.pr),
        ToolDef(name="greptile", type=ToolType.api),
        ToolDef(name="claude-code-cli", type=ToolType.agent),
        ToolDef(name="anthropic-api", type=ToolType.agent),
    ]
    config = EvalConfig(eval_org="my-org", tools=tools)
    agent_tools = config.agent_tools
    assert len(agent_tools) == 2
    assert {t.name for t in agent_tools} == {"claude-code-cli", "anthropic-api"}


def test_tool_def_rejects_invalid_type() -> None:
    with pytest.raises(Exception):
        ToolDef(name="bad", type="unknown")  # type: ignore[arg-type]


def test_eval_config_pr_tools_filtering() -> None:
    tools = [
        ToolDef(name="coderabbit", type=ToolType.pr),
        ToolDef(name="greptile", type=ToolType.api),
        ToolDef(name="bugbot", type=ToolType.pr),
    ]
    config = EvalConfig(eval_org="my-org", tools=tools)
    pr_tools = config.pr_tools
    assert len(pr_tools) == 2
    assert {t.name for t in pr_tools} == {"coderabbit", "bugbot"}


def test_run_state_states_method() -> None:
    rs = RunState()
    rs.set(CaseToolState(case_id="c1", tool="t1"))
    rs.set(CaseToolState(case_id="c2", tool="t1"))
    assert len(rs.states()) == 2
    assert all(isinstance(s, CaseToolState) for s in rs.states())


def test_scoring_config_defaults() -> None:
    sc = ScoringConfig()
    assert sc.scale == [0, 1, 2, 3]
    assert sc.catch_threshold == 2
    assert sc.labels[0] == "missed"
    assert sc.labels[3] == "correct-id-and-fix"


def test_default_scoring_returns_scoring_config() -> None:
    sc = default_scoring()
    assert isinstance(sc, ScoringConfig)
    assert sc.catch_threshold == 2
    assert sc.scale == [0, 1, 2, 3]


def test_scoring_config_custom() -> None:
    sc = ScoringConfig(scale=[0, 1, 2], labels={0: "no", 1: "partial", 2: "yes"}, catch_threshold=1)
    assert sc.scale == [0, 1, 2]
    assert sc.catch_threshold == 1
    assert sc.labels[2] == "yes"


def test_load_eval_config_parses_scoring(tmp_path: Path) -> None:
    config_data = {
        "github": {"eval_org": "test-org"},
        "tools": [],
        "repos": {},
        "scoring": {
            "scale": [0, 1, 2, 3],
            "labels": {0: "missed", 1: "wrong-area", 2: "correct-id", 3: "correct-id-and-fix"},
            "catch_threshold": 2,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_eval_config(config_path)
    assert config.scoring.catch_threshold == 2
    assert config.scoring.scale == [0, 1, 2, 3]
    assert config.scoring.labels[0] == "missed"


def test_load_eval_config_scoring_defaults_when_missing(tmp_path: Path) -> None:
    config_data = {
        "github": {"eval_org": "test-org"},
        "tools": [],
        "repos": {},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_eval_config(config_path)
    assert config.scoring.catch_threshold == 2
    assert config.scoring.scale == [0, 1, 2, 3]


def test_load_eval_config(tmp_path: Path) -> None:
    config_data = {
        "github": {"eval_org": "provable-eval"},
        "tools": [
            {
                "name": "coderabbit",
                "type": "pr",
                "github_app": "coderabbit-ai",
                "cooldown_seconds": 30,
            },
            {
                "name": "greptile",
                "type": "api",
                "github_app": "greptile",
                "cooldown_seconds": 30,
            },
        ],
        "repos": {"aleo-lang": "provable-org/aleo-lang"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_eval_config(config_path)
    assert config.eval_org == "provable-eval"
    assert len(config.tools) == 2
    assert config.repos == {"aleo-lang": "provable-org/aleo-lang"}
    assert len(config.pr_tools) == 1
    assert config.pr_tools[0].name == "coderabbit"


def test_judging_config_defaults() -> None:
    jc = JudgingConfig()
    assert jc.llm_calls == 3
    assert jc.human_sample_rate == 0.25
    assert jc.calibration_threshold == 0.85
    assert jc.model == "claude-opus-4-6"


def test_default_judging_returns_judging_config() -> None:
    jc = default_judging()
    assert isinstance(jc, JudgingConfig)
    assert jc.llm_calls == 3
    assert jc.model == "claude-opus-4-6"


def test_load_eval_config_parses_judging(tmp_path: Path) -> None:
    config_data = {
        "github": {"eval_org": "test-org"},
        "tools": [],
        "repos": {},
        "judging": {
            "llm_calls": 5,
            "human_sample_rate": 0.1,
            "calibration_threshold": 0.9,
            "model": "claude-sonnet-4-6",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_eval_config(config_path)
    assert config.judging.llm_calls == 5
    assert config.judging.human_sample_rate == 0.1
    assert config.judging.calibration_threshold == 0.9
    assert config.judging.model == "claude-sonnet-4-6"


def test_load_eval_config_judging_defaults_when_missing(tmp_path: Path) -> None:
    config_data = {
        "github": {"eval_org": "test-org"},
        "tools": [],
        "repos": {},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_eval_config(config_path)
    assert config.judging.llm_calls == 3
    assert config.judging.human_sample_rate == 0.25
    assert config.judging.calibration_threshold == 0.85
    assert config.judging.model == "claude-opus-4-6"


def test_pricing_config_estimate_cost() -> None:
    from bugeval.pr_eval_models import PricingConfig

    pc = PricingConfig(rates={"claude-sonnet-4-6": (3.0, 15.0)})
    cost = pc.estimate_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
    assert cost == pytest.approx(18.0)


def test_pricing_config_unknown_model_returns_zero() -> None:
    from bugeval.pr_eval_models import PricingConfig

    pc = PricingConfig(rates={"claude-sonnet-4-6": (3.0, 15.0)})
    cost = pc.estimate_cost("unknown-model-xyz", 1_000_000, 1_000_000)
    assert cost == 0.0


def test_load_eval_config_parses_pricing(tmp_path: Path) -> None:
    from bugeval.pr_eval_models import PricingConfig

    config_data = {
        "github": {"eval_org": "test-org"},
        "tools": [],
        "repos": {},
        "pricing": {
            "claude-sonnet-4-6": [3.0, 15.0],
            "gpt-4.1-mini": [0.40, 1.60],
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data))

    config = load_eval_config(config_path)
    assert isinstance(config.pricing, PricingConfig)
    cost = config.pricing.estimate_cost("claude-sonnet-4-6", 1_000_000, 0)
    assert cost == pytest.approx(3.0)
