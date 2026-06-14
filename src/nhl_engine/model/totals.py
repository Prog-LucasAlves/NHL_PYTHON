import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from nhl_engine.config import DATA_PATH, TOTALS_AWAY_MODEL_PATH, TOTALS_DISTRIBUTION_MODEL_PATH, TOTALS_HOME_MODEL_PATH
from nhl_engine.model.pregame import PREGAME_FEATURE_COLUMNS, latest_matchup_features


@dataclass(frozen=True)
class TotalProbabilities:
    over: float
    under: float
    push: float


@dataclass(frozen=True)
class TotalsPrediction:
    expected_home: float
    expected_away: float
    expected_total: float
    probabilities: TotalProbabilities


def total_probabilities(expected_total: float, line: float) -> TotalProbabilities:
    """Calcula probabilidades Over/Under/Push a partir de uma Poisson total."""
    expected = max(0.01, float(expected_total))
    max_goals = max(30, math.ceil(line) + 20)
    probabilities = [math.exp(-expected)]
    for goals in range(1, max_goals + 1):
        probabilities.append(probabilities[-1] * expected / goals)

    under = sum(probability for goals, probability in enumerate(probabilities) if goals < line)
    push = sum(probability for goals, probability in enumerate(probabilities) if goals == line)
    over = sum(probability for goals, probability in enumerate(probabilities) if goals > line)
    over += max(0.0, 1.0 - under - push - over)
    return TotalProbabilities(over=over, under=under, push=push)


def total_probabilities_from_distribution(classes, probabilities, line: float) -> TotalProbabilities:
    """Converte uma distribuição discreta calibrada em probabilidades de mercado."""
    under = sum(float(probability) for goals, probability in zip(classes, probabilities, strict=True) if float(goals) < line)
    push = sum(float(probability) for goals, probability in zip(classes, probabilities, strict=True) if float(goals) == line)
    over = sum(float(probability) for goals, probability in zip(classes, probabilities, strict=True) if float(goals) > line)
    return TotalProbabilities(over=over, under=under, push=push)


class NHLTotalsPredictor:
    """Prediz gols esperados e preços justos para uma linha configurável."""

    def __init__(
        self,
        data_path: str | Path = DATA_PATH,
        home_model_path: str | Path = TOTALS_HOME_MODEL_PATH,
        away_model_path: str | Path = TOTALS_AWAY_MODEL_PATH,
        distribution_model_path: str | Path = TOTALS_DISTRIBUTION_MODEL_PATH,
    ):
        self.data_path = Path(data_path)
        self.home_model_path = Path(home_model_path)
        self.away_model_path = Path(away_model_path)
        self.distribution_model_path = Path(distribution_model_path)
        self.games = pd.DataFrame()
        self.home_model: CatBoostRegressor | None = None
        self.away_model: CatBoostRegressor | None = None
        self.distribution_model: CatBoostClassifier | None = None

    def _initialize(self) -> None:
        self.games = pd.read_csv(self.data_path)
        if self.home_model_path.exists() and self.away_model_path.exists():
            self.home_model = CatBoostRegressor()
            self.away_model = CatBoostRegressor()
            self.home_model.load_model(self.home_model_path)
            self.away_model.load_model(self.away_model_path)
        if self.distribution_model_path.exists():
            self.distribution_model = CatBoostClassifier()
            self.distribution_model.load_model(self.distribution_model_path)

    def predict(self, home_team: str, away_team: str, line: float, game_type: int = 2) -> TotalsPrediction:
        if self.games.empty:
            self._initialize()
        features = latest_matchup_features(self.games, home_team, away_team, game_type)
        if self.home_model is not None and self.away_model is not None:
            expected_home = float(self.home_model.predict(features[PREGAME_FEATURE_COLUMNS])[0])
            expected_away = float(self.away_model.predict(features[PREGAME_FEATURE_COLUMNS])[0])
        else:
            row = features.iloc[0]
            expected_home = (row["home_recent_gf"] + row["away_recent_ga"]) / 2
            expected_away = (row["away_recent_gf"] + row["home_recent_ga"]) / 2
            if game_type == 3:
                expected_home *= 0.97
                expected_away *= 0.97

        expected_home = min(max(expected_home, 0.5), 6.0)
        expected_away = min(max(expected_away, 0.5), 6.0)
        expected_total = expected_home + expected_away
        probabilities = total_probabilities(expected_total, line)
        if self.distribution_model is not None:
            distribution = self.distribution_model.predict_proba(features[PREGAME_FEATURE_COLUMNS])[0]
            probabilities = total_probabilities_from_distribution(self.distribution_model.classes_, distribution, line)
        return TotalsPrediction(
            expected_home=expected_home,
            expected_away=expected_away,
            expected_total=expected_total,
            probabilities=probabilities,
        )
