# tests/test_judge_models.py
from bugeval.judge_models import CommentClassification, CommentJudgment, JudgeScore, NoiseStats


def test_judge_score_defaults() -> None:
    s = JudgeScore(test_case_id="x", tool="y", score=2, votes=[2, 2, 3], reasoning="ok")
    assert s.comment_judgments == []
    assert s.noise.total_comments == 0
    assert s.noise.snr == 0.0


def test_judge_score_model_dump_round_trip() -> None:
    s = JudgeScore(
        test_case_id="case-001",
        tool="greptile",
        score=3,
        votes=[3, 3, 2],
        reasoning="Correct ID and fix",
        comment_judgments=[
            CommentJudgment(id=0, classification=CommentClassification.tp, relevance="direct")
        ],
        noise=NoiseStats(total_comments=4, true_positives=1, snr=0.25),
    )
    data = s.model_dump(mode="json")
    restored = JudgeScore(**data)
    assert restored.score == 3
    assert restored.noise.snr == 0.25
    assert restored.comment_judgments[0].classification == "TP"


def test_majority_vote() -> None:
    from bugeval.judge_models import majority_vote

    assert majority_vote([2, 2, 3]) == 2
    assert majority_vote([3, 3, 3]) == 3
    assert majority_vote([0, 1, 2]) == 1  # fallback: median (middle value sorted)
