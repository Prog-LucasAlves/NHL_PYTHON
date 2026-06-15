import pandas as pd

from nhl_engine.model.pregame import build_pregame_features


def test_pregame_features_use_only_games_completed_before_target_game():
    games = pd.DataFrame(
        [
            {
                "game_id": 1,
                "date": "2026-01-01",
                "season": "20252026",
                "game_type": 2,
                "home_team": "BOS",
                "home_score": 2,
                "away_team": "TOR",
                "away_score": 1,
            },
            {
                "game_id": 2,
                "date": "2026-01-03",
                "season": "20252026",
                "game_type": 2,
                "home_team": "BOS",
                "home_score": 9,
                "away_team": "TOR",
                "away_score": 8,
            },
        ],
    )

    result = build_pregame_features(games, min_games=0)
    second = result.loc[result["game_id"] == 2].iloc[0]

    assert second["home_games_played"] == 1
    assert second["away_games_played"] == 1
    assert second["home_recent_gf"] == 2
    assert second["home_recent_ga"] == 1
    assert second["away_recent_gf"] == 1
    assert second["away_recent_ga"] == 2


def test_pregame_features_keep_playoff_context():
    games = pd.DataFrame(
        [
            {
                "game_id": 1,
                "date": "2026-05-01",
                "season": "20252026",
                "game_type": 3,
                "home_team": "CAR",
                "home_score": 4,
                "away_team": "VGK",
                "away_score": 2,
            },
        ],
    )

    result = build_pregame_features(games, min_games=0)

    assert result.loc[0, "game_type"] == 3
    assert result.loc[0, "total_goals"] == 6
