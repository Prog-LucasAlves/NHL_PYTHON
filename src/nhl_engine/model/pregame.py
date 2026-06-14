from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

DEFAULT_GOALS = 3.0
DEFAULT_WIN_RATE = 0.5
DEFAULT_REST_DAYS = 7.0

PREGAME_FEATURE_COLUMNS = [
    "game_type",
    "home_games_played",
    "home_recent_gf",
    "home_recent_ga",
    "home_season_gf",
    "home_season_ga",
    "home_win_rate",
    "home_rest_days",
    "away_games_played",
    "away_recent_gf",
    "away_recent_ga",
    "away_season_gf",
    "away_season_ga",
    "away_win_rate",
    "away_rest_days",
]


@dataclass
class TeamState:
    games: int = 0
    goals_for: int = 0
    goals_against: int = 0
    wins: int = 0
    recent_goals_for: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_goals_against: deque = field(default_factory=lambda: deque(maxlen=10))
    last_date: pd.Timestamp | None = None

    def snapshot(self, prefix: str, game_date: pd.Timestamp) -> dict[str, float]:
        recent_gf = sum(self.recent_goals_for) / len(self.recent_goals_for) if self.recent_goals_for else DEFAULT_GOALS
        recent_ga = sum(self.recent_goals_against) / len(self.recent_goals_against) if self.recent_goals_against else DEFAULT_GOALS
        season_gf = self.goals_for / self.games if self.games else DEFAULT_GOALS
        season_ga = self.goals_against / self.games if self.games else DEFAULT_GOALS
        rest_days = min((game_date - self.last_date).days, 14) if self.last_date is not None else DEFAULT_REST_DAYS
        return {
            f"{prefix}_games_played": float(self.games),
            f"{prefix}_recent_gf": recent_gf,
            f"{prefix}_recent_ga": recent_ga,
            f"{prefix}_season_gf": season_gf,
            f"{prefix}_season_ga": season_ga,
            f"{prefix}_win_rate": self.wins / self.games if self.games else DEFAULT_WIN_RATE,
            f"{prefix}_rest_days": float(max(rest_days, 0)),
        }

    def update(self, goals_for: int, goals_against: int, game_date: pd.Timestamp) -> None:
        self.games += 1
        self.goals_for += goals_for
        self.goals_against += goals_against
        self.wins += int(goals_for > goals_against)
        self.recent_goals_for.append(goals_for)
        self.recent_goals_against.append(goals_against)
        self.last_date = game_date


def _prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    prepared = games.copy()
    if "game_type" not in prepared.columns:
        prepared["game_type"] = 2
    prepared["date"] = pd.to_datetime(prepared["date"])
    return prepared.sort_values(["date", "game_id"]).reset_index(drop=True)


def _state(states: dict[tuple[str, str], TeamState], season: str, team: str) -> TeamState:
    return states.setdefault((season, team), TeamState())


def build_pregame_features(games: pd.DataFrame, min_games: int = 5) -> pd.DataFrame:
    """Cria features usando somente o estado existente antes de cada partida."""
    prepared = _prepare_games(games)
    states: dict[tuple[str, str], TeamState] = {}
    rows = []

    for row in prepared.itertuples(index=False):
        season = str(row.season)
        home = _state(states, season, row.home_team)
        away = _state(states, season, row.away_team)
        record = {
            "game_id": row.game_id,
            "date": row.date,
            "season": season,
            "game_type": int(row.game_type),
            "home_team": row.home_team,
            "away_team": row.away_team,
            "home_score": int(row.home_score),
            "away_score": int(row.away_score),
            "total_goals": int(row.home_score + row.away_score),
            "target_home_win": int(row.home_score > row.away_score),
            **home.snapshot("home", row.date),
            **away.snapshot("away", row.date),
        }
        if home.games >= min_games and away.games >= min_games:
            rows.append(record)

        home.update(int(row.home_score), int(row.away_score), row.date)
        away.update(int(row.away_score), int(row.home_score), row.date)

    return pd.DataFrame(rows)


def latest_matchup_features(
    games: pd.DataFrame,
    home_team: str,
    away_team: str,
    game_type: int = 2,
    game_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Monta uma linha de features para uma partida futura."""
    prepared = _prepare_games(games)
    states: dict[tuple[str, str], TeamState] = {}
    for row in prepared.itertuples(index=False):
        season = str(row.season)
        _state(states, season, row.home_team).update(int(row.home_score), int(row.away_score), row.date)
        _state(states, season, row.away_team).update(int(row.away_score), int(row.home_score), row.date)

    latest_season = str(prepared["season"].iloc[-1])
    prediction_date = game_date or (prepared["date"].max() + timedelta(days=1))
    home = _state(states, latest_season, home_team)
    away = _state(states, latest_season, away_team)
    record = {
        "game_type": game_type,
        **home.snapshot("home", prediction_date),
        **away.snapshot("away", prediction_date),
    }
    return pd.DataFrame([record], columns=PREGAME_FEATURE_COLUMNS)
