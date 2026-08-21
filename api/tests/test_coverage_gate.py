import pytest

from api.guardrails.coverage_gate import TAU_ABSOLUTE, TAU_MEAN, coverage_verdict
from api.harness.stage import StageShortCircuit


def test_empty_candidate_set_refused_out_of_scope():
    with pytest.raises(StageShortCircuit) as exc:
        coverage_verdict([])
    assert exc.value.refusal_code == "OUT_OF_SCOPE"


def test_low_top1_refused_out_of_scope():
    with pytest.raises(StageShortCircuit) as exc:
        coverage_verdict([0.5, 0.4, 0.3])
    assert exc.value.refusal_code == "OUT_OF_SCOPE"


def test_low_mean5_refused_out_of_scope():
    scores = [0.9] + [0.2] * 4 + [0.1] * 5
    with pytest.raises(StageShortCircuit) as exc:
        coverage_verdict(scores)
    assert exc.value.refusal_code == "OUT_OF_SCOPE"


def test_calibrated_thresholds_reject_out_of_domain_and_accept_in_domain():
    # Measured populations from bench/run_guardrails_calibration.py.
    out_of_domain_top1 = [0.6000, 0.5306, 0.4968, 0.4951, 0.4787]
    in_domain_top1 = [0.8172, 0.8674, 0.9696, 0.9972]
    for s in out_of_domain_top1:
        with pytest.raises(StageShortCircuit) as exc:
            coverage_verdict([s] * 20)
        assert exc.value.refusal_code == "OUT_OF_SCOPE"
    for s in in_domain_top1:
        stats = coverage_verdict([s] * 20)
        assert stats.top1 == pytest.approx(s)


def test_thresholds_match_calibration_midpoints():
    assert TAU_ABSOLUTE == pytest.approx(0.70)
    assert TAU_MEAN == pytest.approx(0.62)


def test_healthy_distribution_returns_stats():
    scores = [0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86]
    stats = coverage_verdict(scores)
    assert stats.top1 == pytest.approx(0.95)
    assert stats.mean5 == pytest.approx(0.93)
    assert stats.margin == pytest.approx(0.09)
