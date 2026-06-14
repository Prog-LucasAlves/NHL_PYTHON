import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, log_loss

from nhl_engine.config import (
    DATA_PATH,
    MODEL_PATH,
    PREGAME_MONEYLINE_MODEL_PATH,
    TOTALS_AWAY_MODEL_PATH,
    TOTALS_DISTRIBUTION_MODEL_PATH,
    TOTALS_HOME_MODEL_PATH,
)
from nhl_engine.model.features import (
    CAT_FEATURES,
    FEATURE_COLUMNS,
    build_features,
)
from nhl_engine.model.pregame import PREGAME_FEATURE_COLUMNS, build_pregame_features


def brier_score_loss(y_true, y_prob):
    """Calcula o Brier Score Loss para calibração de probabilidade."""
    return float(np.mean((y_true - y_prob) ** 2))


def train_model(data_path: str | None = None) -> CatBoostClassifier:
    """Treina o modelo com otimização hiperparamétrica L2 e validação cruzada temporal."""
    df = build_features(data_path)

    # Identifica as temporadas em ordem cronológica
    seasons = sorted(df["season"].unique())
    print(f"Temporadas detectadas: {seasons}")

    # As últimas 3 temporadas serão as dobras de validação temporal
    validation_seasons = seasons[-3:]
    print(f"Temporadas de validação cruzada: {validation_seasons}")

    # Hiperparâmetros a avaliar (grade focada em calibração e generalização)
    param_grid = [
        {"depth": 4, "l2_leaf_reg": 3},
        {"depth": 4, "l2_leaf_reg": 10},
        {"depth": 6, "l2_leaf_reg": 3},
        {"depth": 6, "l2_leaf_reg": 10},
    ]

    best_params: dict[str, int] = param_grid[0]
    best_log_loss = float("inf")

    # Otimização hiperparamétrica via Validação Temporal
    print("\nIniciando busca por grade hiperparamétrica...")
    for params in param_grid:
        fold_losses = []
        fold_accs = []
        fold_briers = []

        for val_season in validation_seasons:
            train_df = df[df["season"] < val_season]
            val_df = df[df["season"] == val_season]

            if len(train_df) == 0 or len(val_df) == 0:
                continue

            x_train, y_train = train_df[FEATURE_COLUMNS], train_df["target"]
            x_val, y_val = val_df[FEATURE_COLUMNS], val_df["target"]

            # Treinamento rápido para validação de dobra
            fold_model = CatBoostClassifier(
                iterations=400,
                learning_rate=0.05,
                depth=params["depth"],
                l2_leaf_reg=params["l2_leaf_reg"],
                loss_function="Logloss",
                random_seed=42,
                verbose=0,
                allow_writing_files=False,
            )
            fold_model.fit(x_train, y_train, cat_features=CAT_FEATURES, eval_set=(x_val, y_val), early_stopping_rounds=30)

            val_probs = fold_model.predict_proba(x_val)[:, 1]
            val_preds = fold_model.predict(x_val)

            fold_losses.append(log_loss(y_val, val_probs))
            fold_accs.append(accuracy_score(y_val, val_preds))
            fold_briers.append(brier_score_loss(y_val, val_probs))

        mean_loss = np.mean(fold_losses)
        mean_acc = np.mean(fold_accs)
        mean_brier = np.mean(fold_briers)

        print(f"Parâmetros: {params} | Log Loss Médio: {mean_loss:.4f} | Acurácia Média: {mean_acc:.2%} | Brier Score Médio: {mean_brier:.4f}")

        if mean_loss < best_log_loss:
            best_log_loss = mean_loss
            best_params = params

    print(f"\nMelhor configuração encontrada: {best_params} (Log Loss: {best_log_loss:.4f})")

    # Treinamento do modelo final usando a melhor configuração
    last_season = seasons[-1]
    train_df = df[df["season"] != last_season]
    test_df = df[df["season"] == last_season]

    x_train, y_train = train_df[FEATURE_COLUMNS], train_df["target"]
    x_test, y_test = test_df[FEATURE_COLUMNS], test_df["target"]

    print("\nTreinando modelo final com os melhores parâmetros...")
    print(f"Dados de Treino: {len(train_df)} jogos. Dados de Validação: {len(test_df)} jogos ({last_season}).")

    final_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=best_params["depth"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=100,
        allow_writing_files=False,
    )

    final_model.fit(
        x_train,
        y_train,
        cat_features=CAT_FEATURES,
        eval_set=(x_test, y_test),
        early_stopping_rounds=50,
        use_best_model=True,
    )

    final_preds = final_model.predict(x_test)
    final_probs = final_model.predict_proba(x_test)[:, 1]

    final_acc = accuracy_score(y_test, final_preds)
    final_loss = log_loss(y_test, final_probs)
    final_brier = brier_score_loss(y_test, final_probs)

    print("\n" + "=" * 50)
    print("MÉTRICAS DO MODELO FINAL (TEMPORADA DE TESTE):")
    print(f"Acurácia de Teste: {final_acc:.4f}")
    print(f"Log Loss de Teste: {final_loss:.4f}")
    print(f"Brier Score de Teste: {final_brier:.4f}")
    print("=" * 50)

    final_model.save_model(str(MODEL_PATH))
    print(f"Modelo salvo em {MODEL_PATH}")
    return final_model


def train_pregame_models(data_path: str | None = None) -> tuple[CatBoostClassifier, CatBoostRegressor, CatBoostRegressor, CatBoostClassifier]:
    """Treina modelos de moneyline e gols com features disponíveis antes do jogo."""
    import pandas as pd

    games = pd.read_csv(data_path or DATA_PATH)
    features = build_pregame_features(games, min_games=5)
    x_train = features[PREGAME_FEATURE_COLUMNS]

    moneyline = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.04, loss_function="Logloss", verbose=100, random_seed=42, allow_writing_files=False)
    home_goals = CatBoostRegressor(iterations=500, depth=5, learning_rate=0.04, loss_function="RMSE", verbose=100, random_seed=42, allow_writing_files=False)
    away_goals = CatBoostRegressor(iterations=500, depth=5, learning_rate=0.04, loss_function="RMSE", verbose=100, random_seed=42, allow_writing_files=False)
    total_distribution = CatBoostClassifier(iterations=500, depth=5, learning_rate=0.04, loss_function="MultiClass", verbose=100, random_seed=42, allow_writing_files=False)

    moneyline.fit(x_train, features["target_home_win"])
    home_goals.fit(x_train, features["home_score"])
    away_goals.fit(x_train, features["away_score"])
    total_distribution.fit(x_train, features["total_goals"].clip(upper=12))

    moneyline.save_model(str(PREGAME_MONEYLINE_MODEL_PATH))
    home_goals.save_model(str(TOTALS_HOME_MODEL_PATH))
    away_goals.save_model(str(TOTALS_AWAY_MODEL_PATH))
    total_distribution.save_model(str(TOTALS_DISTRIBUTION_MODEL_PATH))
    print("Modelos pregame salvos com features temporais sem vazamento.")
    return moneyline, home_goals, away_goals, total_distribution


def main():
    train_pregame_models()


if __name__ == "__main__":
    main()
