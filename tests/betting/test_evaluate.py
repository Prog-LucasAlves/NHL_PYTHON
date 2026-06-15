import pandas as pd

from nhl_engine.betting.evaluate import SCENARIO_WARNING, scenario_report, walk_forward_season_splits


def test_walk_forward_splits_never_train_on_validation_or_future_seasons():
    data = pd.DataFrame({"season": ["20202021", "20212022", "20222023", "20232024"]})

    splits = walk_forward_season_splits(data, validation_seasons=2)

    assert len(splits) == 2
    for train_index, validation_index in splits:
        assert data.loc[train_index, "season"].max() < data.loc[validation_index, "season"].min()


def test_scenario_report_carries_warning_and_breakdowns():
    bets = pd.DataFrame(
        [
            {"market": "Moneyline", "game_type": 2, "season": "20252026", "stake": 1.0, "pl": 1.0, "result": "Green"},
            {"market": "Total", "game_type": 3, "season": "20252026", "stake": 1.0, "pl": -1.0, "result": "Red"},
        ],
    )

    report = scenario_report(bets)

    assert report["warning"] == SCENARIO_WARNING
    assert set(report["by_market"]["market"]) == {"Moneyline", "Total"}
    assert set(report["by_game_type"]["game_type"]) == {2, 3}
