import pytest

from nhl_engine.betting.strategy import evaluate_bet, fair_odd, is_validated_total_strategy, settle_total_bet


def test_fair_odd_accounts_for_push_probability():
    assert fair_odd(0.45, push_probability=0.10) == pytest.approx(2.0)


def test_bet_requires_full_five_percent_edge():
    threshold = fair_odd(0.50) * 1.05

    below = evaluate_bet(0.50, threshold - 0.001, bankroll=100)
    exact = evaluate_bet(0.50, threshold, bankroll=100)

    assert below.qualifies is False
    assert exact.qualifies is True
    assert exact.edge == pytest.approx(0.05)


def test_fractional_kelly_is_capped_at_five_percent_of_bankroll():
    decision = evaluate_bet(0.80, 3.0, bankroll=100, kelly_multiplier=0.50)

    assert decision.qualifies is True
    assert decision.stake == pytest.approx(5.0)


def test_invalid_market_odd_never_qualifies():
    decision = evaluate_bet(0.60, 1.0, bankroll=100)

    assert decision.qualifies is False
    assert decision.stake == 0.0


@pytest.mark.parametrize(
    ("side", "line", "expected"),
    [
        ("Over", 5.5, True),
        ("Over", 6.0, False),
        ("Under", 6.5, False),
        ("Under", 7.5, True),
    ],
)
def test_total_strategy_only_allows_robust_out_of_sample_segments(side, line, expected):
    assert is_validated_total_strategy(side, line) is expected


@pytest.mark.parametrize(
    ("side", "line", "total", "expected"),
    [
        ("Over", 5.5, 6, "Green"),
        ("Over", 5.5, 5, "Red"),
        ("Under", 5.5, 5, "Green"),
        ("Under", 5.5, 6, "Red"),
        ("Over", 6.0, 6, "Push"),
        ("Under", 6.0, 6, "Push"),
    ],
)
def test_settle_total_bet(side, line, total, expected):
    assert settle_total_bet(side, line, total) == expected
