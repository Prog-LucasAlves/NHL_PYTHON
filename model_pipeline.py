import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss


class NHLPredictiveModel:
    def __init__(self, data_path="nhl_games_all_seasons.csv"):
        self.data_path = data_path
        self.model = None
        self.df = None

    def load_and_preprocess(self):
        """Carrega e prepara os dados básicos."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Arquivo {self.data_path} não encontrado. Execute extract_games.py primeiro.")

        df = pd.read_csv(self.data_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Alvo: 1 se o time da casa vencer, 0 se o visitante vencer
        df["target"] = (df["home_score"] > df["away_score"]).astype(int)

        self.df = df
        return df

    def feature_engineering(self):
        """Gera atributos complexos evitando Data Leakage."""
        df = self.df.copy()

        # 1. ELO Rating System
        elo_ratings = {team: 1500 for team in pd.concat([df["home_team"], df["away_team"]]).unique()}
        home_elo = []
        away_elo = []

        K = 20  # Fator de ajuste ELO

        for idx, row in df.iterrows():
            h_team = row["home_team"]
            a_team = row["away_team"]

            # Pega o ELO antes do jogo (Prevenção de Leakage)
            h_elo_pre = elo_ratings[h_team]
            a_elo_pre = elo_ratings[a_team]
            home_elo.append(h_elo_pre)
            away_elo.append(a_elo_pre)

            # Cálculo de probabilidade esperada
            exp_h = 1 / (1 + 10 ** ((a_elo_pre - h_elo_pre - 50) / 400))  # +50 para vantagem em casa

            # Resultado real (1 casa vence, 0 fora vence)
            actual_h = 1 if row["home_score"] > row["away_score"] else 0

            # Atualiza ELO
            update = K * (actual_h - exp_h)
            elo_ratings[h_team] += update
            elo_ratings[a_team] -= update

        df["home_elo"] = home_elo
        df["away_elo"] = away_elo
        df["elo_diff"] = df["home_elo"] - df["away_elo"]

        # 2. Médias Móveis (Rolling Stats) por Time
        # Precisamos reorganizar os dados por time
        for team in elo_ratings.keys():
            team_games = df[(df["home_team"] == team) | (df["away_team"] == team)].copy()
            team_games["is_home"] = team_games["home_team"] == team
            team_games["team_goals"] = np.where(team_games["is_home"], team_games["home_score"], team_games["away_score"])
            team_games["opp_goals"] = np.where(team_games["is_home"], team_games["away_score"], team_games["home_score"])
            team_games["won"] = (team_games["team_goals"] > team_games["opp_goals"]).astype(int)

            # Médias dos últimos 10 jogos ANTES do jogo atual (shift 1)
            team_games["rolling_wins_10"] = team_games["won"].shift(1).rolling(10).mean()
            team_games["rolling_gf_10"] = team_games["team_goals"].shift(1).rolling(10).mean()
            team_games["rolling_ga_10"] = team_games["opp_goals"].shift(1).rolling(10).mean()

            for idx, row in team_games.iterrows():
                prefix = "home" if row["is_home"] else "away"
                df.at[idx, f"{prefix}_rolling_wins"] = row["rolling_wins_10"]
                df.at[idx, f"{prefix}_rolling_gf"] = row["rolling_gf_10"]
                df.at[idx, f"{prefix}_rolling_ga"] = row["rolling_ga_10"]

        # Limpa NaNs (primeiros jogos da história onde não há média móvel)
        self.df = df.dropna()
        return self.df

    def train(self):
        """Realiza o treinamento com CatBoost e Time-Series Split."""
        df = self.df

        # Seleção de Features
        features = ["home_team", "away_team", "home_elo", "away_elo", "elo_diff", "home_rolling_wins", "away_rolling_wins", "home_rolling_gf", "away_rolling_gf", "home_rolling_ga", "away_rolling_ga"]
        cat_features = ["home_team", "away_team"]

        # Split Cronológico: Treina em tudo exceto na última temporada coletada
        last_season = df["season"].max()
        train_df = df[df["season"] != last_season]
        test_df = df[df["season"] == last_season]

        X_train, y_train = train_df[features], train_df["target"]
        X_test, y_test = test_df[features], test_df["target"]

        print(f"Treinando em {len(train_df)} jogos. Testando em {len(test_df)} jogos ({last_season}).")

        self.model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, loss_function="Logloss", eval_metric="Accuracy", random_seed=42, verbose=100)

        self.model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), early_stopping_rounds=50)

        # Avaliação
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]

        print("\n" + "=" * 50)
        print(f"Acurácia no Teste: {accuracy_score(y_test, preds):.4f}")
        print(f"Log Loss: {log_loss(y_test, probs):.4f}")
        print("=" * 50)

        # Salva o modelo
        self.model.save_model("nhl_model.cbm")
        return self.model


def main():
    pipeline = NHLPredictiveModel()
    pipeline.load_and_preprocess()
    print("Engenharia de atributos em progresso...")
    pipeline.feature_engineering()
    pipeline.train()


if __name__ == "__main__":
    main()
