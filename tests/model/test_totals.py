import pytest

from nhl_engine.model.totals import total_probabilities


def test_half_line_probabilities_are_coherent_without_push():
    probabilities = total_probabilities(expected_total=6.0, line=5.5)

    assert probabilities.push == 0.0
    assert probabilities.over + probabilities.under == pytest.approx(1.0)
    assert probabilities.over > probabilities.under


def test_whole_line_probabilities_include_push():
    probabilities = total_probabilities(expected_total=6.0, line=6.0)

    assert probabilities.push > 0
    assert probabilities.over + probabilities.under + probabilities.push == pytest.approx(1.0)
