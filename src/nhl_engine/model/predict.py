import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from nhl_engine.config import (
    DATA_PATH,
    MODEL_PATH,
    NST_STATS_PATH,
    PREGAME_MONEYLINE_MODEL_PATH,
)
from nhl_engine.model.features import (
    FEATURE_COLUMNS,
    NST_DIFF_FEATURES,
    NST_FEATURE_BASE,
)
from nhl_engine.model.pregame import PREGAME_FEATURE_COLUMNS, latest_matchup_features


class NHLPredictorV2:
    """Carrega o modelo treinado e calcula estados dos times para predição via estatísticas do NST."""

    def __init__(self, model_path: str | None = None, nst_stats_path: str | None = None):
        self.model_path = model_path or str(MODEL_PATH)
        self.nst_stats_path = nst_stats_path or str(NST_STATS_PATH)
        self.model: CatBoostClassifier | None = None
        self.pregame_model: CatBoostClassifier | None = None
        self.games_df: pd.DataFrame = pd.DataFrame()
        self.team_states: dict[str, dict] = {}
        self.nst_df: pd.DataFrame = pd.DataFrame()
        self.latest_nst_season: str | None = None

    def _initialize(self):
        """Carrega o modelo e calcula o estado atual de todos os times usando a temporada mais recente do NST."""
        self.model = CatBoostClassifier()
        self.model.load_model(self.model_path)
        if PREGAME_MONEYLINE_MODEL_PATH.exists() and DATA_PATH.exists():
            self.pregame_model = CatBoostClassifier()
            self.pregame_model.load_model(PREGAME_MONEYLINE_MODEL_PATH)
            self.games_df = pd.read_csv(DATA_PATH)

        # Carrega e limpa as stats do NST
        try:
            self.nst_df = pd.read_csv(self.nst_stats_path)

            # Remove a coluna original 'Team' para evitar duplicatas de chaves minúsculas
            if "Team" in self.nst_df.columns and "team" in self.nst_df.columns:
                self.nst_df = self.nst_df.drop(columns=["Team"])

            new_cols = {}
            for col in self.nst_df.columns:
                clean_col = col.lower().replace("%", "_pct").replace(" ", "_").replace("-", "_")
                while "__" in clean_col:
                    clean_col = clean_col.replace("__", "_")
                clean_col = clean_col.strip("_")
                if clean_col == "point_pct":
                    clean_col = "points_pct"
                new_cols[col] = clean_col
            self.nst_df = self.nst_df.rename(columns=new_cols)

            self.nst_df["season"] = self.nst_df["season"].astype(str)
            self.nst_df["team"] = self.nst_df["team"].astype(str)
            self.latest_nst_season = self.nst_df["season"].max()

            # Inicializa os estados dos times para a temporada mais recente
            all_teams = self.nst_df["team"].unique()
            for team in all_teams:
                t_row = self.nst_df[(self.nst_df["team"] == team) & (self.nst_df["season"] == self.latest_nst_season)]
                if len(t_row) > 0:
                    self.team_states[team] = {
                        "points_pct": float(t_row["points_pct"].values[0]),
                        "cf_pct": float(t_row["cf_pct"].values[0]),
                        "xgf_pct": float(t_row["xgf_pct"].values[0]),
                        "gf": float(t_row["gf"].values[0]) if "gf" in t_row.columns else 0.0,
                        "ga": float(t_row["ga"].values[0]) if "ga" in t_row.columns else 0.0,
                    }
                else:
                    self.team_states[team] = {
                        "points_pct": 0.500,
                        "cf_pct": 50.0,
                        "xgf_pct": 50.0,
                        "gf": 0.0,
                        "ga": 0.0,
                    }
        except Exception as e:
            print(f"[AVISO] Erro ao carregar stats NST em {self.nst_stats_path}: {e}")
            self.nst_df = pd.DataFrame()
            self.latest_nst_season = None

    def predict(self, home_team: str, away_team: str, game_type: int = 2) -> tuple[float, float]:
        if not self.model:
            self._initialize()

        # Verifica se ambos os times estão cadastrados nas chaves de estado
        if home_team not in self.team_states:
            self.team_states[home_team] = {"points_pct": 0.500, "cf_pct": 50.0, "xgf_pct": 50.0, "gf": 0.0, "ga": 0.0}
        if away_team not in self.team_states:
            self.team_states[away_team] = {"points_pct": 0.500, "cf_pct": 50.0, "xgf_pct": 50.0, "gf": 0.0, "ga": 0.0}

        if self.pregame_model is not None and not self.games_df.empty:
            features = latest_matchup_features(self.games_df, home_team, away_team, game_type)
            prob_home = float(self.pregame_model.predict_proba(features[PREGAME_FEATURE_COLUMNS])[0][1])
            return prob_home, 1 - prob_home

        # Busca estatísticas avançadas do NST
        h_nst_stats: dict[str, float] = {}
        a_nst_stats: dict[str, float] = {}

        if not self.nst_df.empty and self.latest_nst_season:
            h_row = self.nst_df[(self.nst_df["team"] == home_team) & (self.nst_df["season"] == self.latest_nst_season)]
            a_row = self.nst_df[(self.nst_df["team"] == away_team) & (self.nst_df["season"] == self.latest_nst_season)]

            for col in NST_FEATURE_BASE:
                # Time de casa
                if len(h_row) > 0 and col in h_row.columns and not pd.isna(h_row[col].values[0]):
                    h_nst_stats[col] = float(h_row[col].values[0])
                else:
                    mean_val = self.nst_df[self.nst_df["season"] == self.latest_nst_season][col].mean()
                    h_nst_stats[col] = float(mean_val) if not pd.isna(mean_val) else (0.0 if "pct" in col else 1.0)

                # Time de fora
                if len(a_row) > 0 and col in a_row.columns and not pd.isna(a_row[col].values[0]):
                    a_nst_stats[col] = float(a_row[col].values[0])
                else:
                    mean_val = self.nst_df[self.nst_df["season"] == self.latest_nst_season][col].mean()
                    a_nst_stats[col] = float(mean_val) if not pd.isna(mean_val) else (0.0 if "pct" in col else 1.0)
        else:
            for col in NST_FEATURE_BASE:
                h_nst_stats[col] = np.nan
                a_nst_stats[col] = np.nan

        # Monta dicionário de features compatível com FEATURE_COLUMNS
        feature_dict: dict[str, str | float] = {
            "home_team": home_team,
            "away_team": away_team,
        }

        # Adiciona colunas do NST
        for col in NST_FEATURE_BASE:
            feature_dict[f"home_{col}"] = h_nst_stats[col]
            feature_dict[f"away_{col}"] = a_nst_stats[col]

        # Adiciona diferenças do NST
        for col in NST_DIFF_FEATURES:
            h_val = h_nst_stats[col]
            a_val = a_nst_stats[col]
            feature_dict[f"{col}_diff"] = h_val - a_val if not (pd.isna(h_val) or pd.isna(a_val)) else 0.0

        # Ordena as colunas exatamente no mesmo formato do FEATURE_COLUMNS
        features = pd.DataFrame([feature_dict])
        features = features[FEATURE_COLUMNS]

        assert self.model is not None
        prob_home = self.model.predict_proba(features)[0][1]
        prob_away = 1 - prob_home

        return prob_home, prob_away
