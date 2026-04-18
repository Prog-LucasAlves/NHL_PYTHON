import os

import pandas as pd
from catboost import CatBoostClassifier


class NHLPredictorV2:
    def __init__(self, model_path="nhl_model.cbm", data_path="nhl_games_all_seasons.csv"):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.team_states = {}

    def _initialize(self):
        """Carrega o modelo e calcula o estado atual de todos os times."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Modelo não encontrado. Treine primeiro com model_pipeline.py")

        self.model = CatBoostClassifier()
        self.model.load_model(self.model_path)

        # Carrega dados históricos para calcular ELO e Rolling atuais
        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Recalcula estados (ELO e Rolling)
        elo_ratings = {team: 1500 for team in pd.concat([df["home_team"], df["away_team"]]).unique()}
        team_history = {team: [] for team in elo_ratings.keys()}

        K = 20
        for _, row in df.iterrows():
            h_team, a_team = row["home_team"], row["away_team"]
            h_score, a_score = row["home_score"], row["away_score"]

            # ELO
            h_elo_pre, a_elo_pre = elo_ratings[h_team], elo_ratings[a_team]
            exp_h = 1 / (1 + 10 ** ((a_elo_pre - h_elo_pre - 50) / 400))
            actual_h = 1 if h_score > a_score else 0
            elo_ratings[h_team] += K * (actual_h - exp_h)
            elo_ratings[a_team] -= K * (actual_h - exp_h)

            # Histórico para Rolling (Wins, GF, GA)
            team_history[h_team].append({"won": actual_h, "gf": h_score, "ga": a_score})
            team_history[a_team].append({"won": 1 - actual_h, "gf": a_score, "ga": h_score})

        # Salva o estado final
        for team in elo_ratings.keys():
            hist = pd.DataFrame(team_history[team]).tail(10)
            self.team_states[team] = {"elo": elo_ratings[team], "rolling_wins": hist["won"].mean(), "rolling_gf": hist["gf"].mean(), "rolling_ga": hist["ga"].mean()}

    def predict(self, home_team, away_team):
        if not self.model:
            self._initialize()

        if home_team not in self.team_states or away_team not in self.team_states:
            return "Erro: Time não encontrado no histórico."

        h_state = self.team_states[home_team]
        a_state = self.team_states[away_team]

        # Constrói o vetor de features conforme o modelo espera
        features = pd.DataFrame(
            [
                {
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_elo": h_state["elo"],
                    "away_elo": a_state["elo"],
                    "elo_diff": h_state["elo"] - a_state["elo"],
                    "home_rolling_wins": h_state["rolling_wins"],
                    "away_rolling_wins": a_state["rolling_wins"],
                    "home_rolling_gf": h_state["rolling_gf"],
                    "away_rolling_gf": a_state["rolling_gf"],
                    "home_rolling_ga": h_state["rolling_ga"],
                    "away_rolling_ga": a_state["rolling_ga"],
                },
            ],
        )

        prob_home = self.model.predict_proba(features)[0][1]
        prob_away = 1 - prob_home

        # Cálculo de ODD Justa
        fair_odd_home = 1 / prob_home if prob_home > 0 else 999
        fair_odd_away = 1 / prob_away if prob_away > 0 else 999

        print("\n" + "=" * 40)
        print(f"PREVISÃO: {away_team} vs {home_team}")
        print("=" * 40)
        print(f"{home_team} (Casa): {prob_home:.1%} | Odd Justa: {fair_odd_home:.2f}")
        print(f"{away_team} (Fora): {prob_away:.1%} | Odd Justa: {fair_odd_away:.2f}")
        print("=" * 40)
        print("DICA: Se a odd na casa de apostas for MAIOR que a Odd Justa, há VALOR.")

        return prob_home, prob_away


if __name__ == "__main__":
    import sys

    predictor = NHLPredictorV2()
    # Pega times via linha de comando ou usa default
    h = sys.argv[1] if len(sys.argv) > 1 else "BOS"
    a = sys.argv[2] if len(sys.argv) > 2 else "TOR"
    predictor.predict(h, a)
