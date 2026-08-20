import time

from api.harness.deadline import Deadline


def test_remaining_ms_decays_over_time():
    d = Deadline(budget_ms=100)
    assert d.remaining_ms <= 100
    time.sleep(0.02)
    assert d.remaining_ms < 100 - 15


def test_affords_true_when_plenty_of_budget():
    d = Deadline(budget_ms=1000)
    assert d.affords(10) is True


def test_affords_false_when_budget_exhausted():
    d = Deadline(budget_ms=1)
    time.sleep(0.02)
    assert d.affords(1000) is False


def test_child_caps_at_remaining_budget():
    d = Deadline(budget_ms=10)
    time.sleep(0.02)  # remaining_ms now negative
    child = d.child(5000)
    assert child.budget_ms == 0.0


def test_child_caps_at_requested_slice_when_smaller_than_remaining():
    d = Deadline(budget_ms=1000)
    child = d.child(50)
    assert child.budget_ms == 50
