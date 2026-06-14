import pytest

from nhl_engine.model.totals import total_probabilities, total_probabilities_from_distribution


def test_half_line_probabilities_are_coherent_without_push():
    probabilities = total_probabilities(expected_total=6.0, line=5.5)

    assert probabilities.push == 0.0
    assert probabilities.over + probabilities.under == pytest.approx(1.0)
    assert probabilities.over > probabilities.under


def test_whole_line_probabilities_include_push():
    probabilities = total_probabilities(expected_total=6.0, line=6.0)

    assert probabilities.push > 0
    assert probabilities.over + probabilities.under + probabilities.push == pytest.approx(1.0)


def test_classifier_distribution_supports_configurable_lines():
    probabilities = total_probabilities_from_distribution([4, 5, 6, 7], [0.1, 0.2, 0.3, 0.4], line=5.5)

    assert probabilities.under == pytest.approx(0.3)
    assert probabilities.over == pytest.approx(0.7)
    assert probabilities.push == 0
