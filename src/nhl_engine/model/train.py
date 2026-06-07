from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss

from nhl_engine.config import MODEL_PATH
from nhl_engine.model.features import (
    CAT_FEATURES,
    FEATURE_COLUMNS,
    build_features,
)


def train_model(data_path: str | None = None) -> CatBoostClassifier:
    """Treina o CatBoost com time-series split e salva o modelo."""
    df = build_features(data_path)

    last_season = df["season"].max()
    train_df = df[df["season"] != last_season]
    test_df = df[df["season"] == last_season]

    x_train, y_train = train_df[FEATURE_COLUMNS], train_df["target"]
    x_test, y_test = test_df[FEATURE_COLUMNS], test_df["target"]

    print(f"Treinando em {len(train_df)} jogos. Testando em {len(test_df)} jogos ({last_season}).")

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=100,
    )

    model.fit(x_train, y_train, cat_features=CAT_FEATURES, eval_set=(x_test, y_test), early_stopping_rounds=50)

    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]

    print("\n" + "=" * 50)
    print(f"Acurácia no Teste: {accuracy_score(y_test, preds):.4f}")
    print(f"Log Loss: {log_loss(y_test, probs):.4f}")
    print("=" * 50)

    model.save_model(str(MODEL_PATH))
    print(f"Modelo salvo em {MODEL_PATH}")
    return model


def main():
    train_model()


if __name__ == "__main__":
    main()
