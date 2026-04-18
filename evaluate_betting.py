import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


def evaluate_betting_performance():
    # 1. Carregar dados e modelo
    model = CatBoostClassifier()
    model.load_model("nhl_model.cbm")

    df = pd.read_csv("nhl_games_all_seasons.csv")
    df["date"] = pd.to_datetime(df["date"])

    # Pegar apenas a temporada de teste (a última)
    last_season = df["season"].max()

    # Precisamos das mesmas features do pipeline
    # Para simplificar na avaliação, vamos assumir que o model_pipeline.py
    # já salvou as predições ou vamos calculá-las rapidamente aqui.

    # Re-executar a engenharia de atributos mínima para o teste
    # Nota: Em um sistema real, o pipeline de treino já retornaria isso.
    # Vou importar a lógica do model_pipeline para garantir consistência.

    print(f"Avaliando desempenho para a temporada {last_season}...")

    # features e cat_features (conforme definido no model_pipeline.py)
    features = ["home_team", "away_team", "home_elo", "away_elo", "elo_diff", "home_rolling_wins", "away_rolling_wins", "home_rolling_gf", "away_rolling_gf", "home_rolling_ga", "away_rolling_ga"]

    # Nota: Para rodar este script standalone, precisaríamos das features calculadas.
    # Vou ler o dataframe final que o model_pipeline gerou se possível,
    # ou vou apenas simular a análise baseada nos logs.

    # COMO NÃO SALVAMOS O DF COM FEATURES, VOU RODAR O TREINO NOVAMENTE
    # MAS RETORNANDO O TEST_DF COM PREDIÇÕES
    from model_pipeline import NHLPredictiveModel

    pipeline = NHLPredictiveModel()
    pipeline.load_and_preprocess()
    pipeline.feature_engineering()
    model = pipeline.train()

    # Pegando o df com features e predições
    df_with_features = pipeline.df
    last_season = df_with_features["season"].max()
    test_results = df_with_features[df_with_features["season"] == last_season].copy()

    X_test = test_results[features]
    test_results["prob_home"] = model.predict_proba(X_test)[:, 1]
    test_results["pred_home"] = (test_results["prob_home"] > 0.5).astype(int)

    # 2. Análise por Faixa de Confiança
    print("\nANÁLISE POR CONFIANÇA DO MODELO:")
    conf_bins = [0.5, 0.55, 0.60, 0.65, 1.0]
    for i in range(len(conf_bins) - 1):
        low = conf_bins[i]
        high = conf_bins[i + 1]

        # Jogos onde o modelo está confiante na vitória da CASA
        mask = (test_results["prob_home"] >= low) & (test_results["prob_home"] < high)
        subset = test_results[mask]
        if len(subset) > 0:
            acc = accuracy_score(subset["target"], [1] * len(subset))
            print(f"Confiança [{low:.2f} - {high:.2f}]: {len(subset)} jogos | Precisão: {acc:.2%}")

        # Jogos onde o modelo está confiante na vitória do VISITANTE
        mask_away = (test_results["prob_home"] <= (1 - low)) & (test_results["prob_home"] > (1 - high))
        subset_away = test_results[mask_away]
        if len(subset_away) > 0:
            acc_away = accuracy_score(subset_away["target"], [0] * len(subset_away))
            print(f"Confiança [{low:.2f} - {high:.2f}] (Visitante): {len(subset_away)} jogos | Precisão: {acc_away:.2%}")

    # 3. Simulação de ROI Simples
    # Assumindo odds médias de 1.90 para ambos os lados (mercado equilibrado)
    odd = 1.90
    test_results["win_amount"] = np.where(test_results["pred_home"] == test_results["target"], odd - 1, -1)
    total_roi = test_results["win_amount"].sum() / len(test_results)

    print("\nSIMULAÇÃO DE APOSTAS (Odds Fixas 1.90):")
    print(f"Total de Apostas: {len(test_results)}")
    print(f"Retorno Total: {test_results['win_amount'].sum():.2f} unidades")
    print(f"ROI Estimado: {total_roi:.2%}")

    if total_roi > 0:
        print("\nCONCLUSÃO: O modelo demonstra potencial lucrativo nesta temporada!")
    else:
        print("\nCONCLUSÃO: O modelo precisa de mais refinamento (features de boxscore) para superar as taxas da casa.")


if __name__ == "__main__":
    evaluate_betting_performance()
