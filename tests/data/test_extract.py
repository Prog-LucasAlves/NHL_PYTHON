from pathlib import Path

import pandas as pd

from nhl_engine.data.extract import parse_completed_games, refresh_games_file


def _game(game_id: int, game_type: int, date: str, completed: bool = True) -> dict:
    return {
        "id": game_id,
        "gameType": game_type,
        "gameDate": date,
        "homeTeam": {"abbrev": "BOS", "score": 4 if completed else None},
        "awayTeam": {"abbrev": "TOR", "score": 2 if completed else None},
    }


def test_parse_completed_games_includes_regular_season_and_playoffs_only():
    games = [
        _game(1, 1, "2026-09-20"),
        _game(2, 2, "2026-10-10"),
        _game(3, 3, "2027-05-10"),
        _game(4, 2, "2026-10-11", completed=False),
    ]

    result = parse_completed_games(games, "20262027")

    assert result["game_id"].tolist() == [2, 3]
    assert result["game_type"].tolist() == [2, 3]
    assert result["season"].tolist() == ["20262027", "20262027"]


def test_refresh_games_file_merges_history_and_deduplicates(tmp_path: Path):
    data_path = tmp_path / "games.csv"
    pd.DataFrame(
        [
            {
                "game_id": 1,
                "date": "2025-10-01",
                "season": "20252026",
                "game_type": 2,
                "home_team": "BOS",
                "home_score": 3,
                "away_team": "TOR",
                "away_score": 1,
            },
        ],
    ).to_csv(data_path, index=False)

    fetched = pd.DataFrame(
        [
            {
                "game_id": 1,
                "date": "2025-10-01",
                "season": "20252026",
                "game_type": 2,
                "home_team": "BOS",
                "home_score": 3,
                "away_team": "TOR",
                "away_score": 1,
            },
            {
                "game_id": 2,
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

    result = refresh_games_file(["20252026"], data_path=data_path, fetcher=lambda _: fetched)

    assert result["game_id"].tolist() == [1, 2]
    assert pd.read_csv(data_path)["game_type"].tolist() == [2, 3]


def test_refresh_games_file_does_not_overwrite_history_with_empty_fetch(tmp_path: Path):
    data_path = tmp_path / "games.csv"
    original = pd.DataFrame(
        [
            {
                "game_id": 1,
                "date": "2025-10-01",
                "season": "20252026",
                "game_type": 2,
                "home_team": "BOS",
                "home_score": 3,
                "away_team": "TOR",
                "away_score": 1,
            },
        ],
    )
    original.to_csv(data_path, index=False)

    result = refresh_games_file(["20252026"], data_path=data_path, fetcher=lambda _: pd.DataFrame())

    assert result.astype({"season": str}).to_dict("records") == original.to_dict("records")
    assert pd.read_csv(data_path).astype({"season": str}).to_dict("records") == original.to_dict("records")
