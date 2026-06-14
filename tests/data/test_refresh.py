import pandas as pd

from nhl_engine.data.refresh import refresh_all_data


def test_combined_refresh_invokes_both_collectors():
    calls = []

    result = refresh_all_data(
        game_refresher=lambda: calls.append("NHL") or pd.DataFrame([{"game_id": 1}]),
        nst_refresher=lambda: calls.append("NST") or pd.DataFrame([{"team": "BOS"}]),
    )

    assert calls == ["NHL", "NST"]
    assert [item.success for item in result] == [True, True]


def test_combined_refresh_reports_partial_failure_and_continues():
    def fail_games():
        raise RuntimeError("API indisponivel")

    result = refresh_all_data(
        game_refresher=fail_games,
        nst_refresher=lambda: pd.DataFrame([{"team": "BOS"}]),
    )

    assert result[0].success is False
    assert "API indisponivel" in result[0].message
    assert result[1].success is True
