import pytest

from api.guardrails.coverage_gate import coverage_verdict
from api.harness.stage import StageShortCircuit


def test_empty_candidate_set_refused_out_of_scope():
    with pytest.raises(StageShortCircuit) as exc:
        coverage_verdict([])
    assert exc.value.refusal_code == "OUT_OF_SCOPE"


def test_low_top1_refused_out_of_scope():
    with pytest.raises(StageShortCircuit) as exc:
        coverage_verdict([0.01])
    assert exc.value.refusal_code == "OUT_OF_SCOPE"


def test_low_mean5_refused_out_of_scope():
    scores = [0.02, 0.001, 0.001, 0.001, 0.001]
    with pytest.raises(StageShortCircuit) as exc:
        coverage_verdict(scores)
    assert exc.value.refusal_code == "OUT_OF_SCOPE"


def test_flat_distribution_refused_low_confidence():
    scores = [0.02] * 10
    with pytest.raises(StageShortCircuit) as exc:
        coverage_verdict(scores)
    assert exc.value.refusal_code == "LOW_CONFIDENCE"


def test_healthy_distribution_returns_stats():
    scores = [0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.004, 0.003, 0.002, 0.001]
    stats = coverage_verdict(scores)
    assert stats.top1 == pytest.approx(0.05)
    assert stats.mean5 == pytest.approx(0.03)
    assert stats.margin == pytest.approx(0.049)
