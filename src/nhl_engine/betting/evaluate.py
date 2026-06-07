import numpy as np
from sklearn.metrics import accuracy_score

from nhl_engine.model.features import FEATURE_COLUMNS, build_features
from nhl_engine.model.train import train_model


def evaluate_betting_performance():
    """Avalia o desempenho do modelo para apostas com análise por faixa de confiança."""
    df = build_features()
    model = train_model()

    last_season = df["season"].max()
    test_results = df[df["season"] == last_season].copy()

    print(f"Avaliando desempenho para a temporada {last_season}...")

    x_test = test_results[FEATURE_COLUMNS]
    test_results["prob_home"] = model.predict_proba(x_test)[:, 1]
    test_results["pred_home"] = (test_results["prob_home"] > 0.5).astype(int)

    # Análise por faixa de confiança
    print("\nANÁLISE POR CONFIANÇA DO MODELO:")
    conf_bins = [0.5, 0.55, 0.60, 0.65, 1.0]
    for i in range(len(conf_bins) - 1):
        low, high = conf_bins[i], conf_bins[i + 1]

        mask = (test_results["prob_home"] >= low) & (test_results["prob_home"] < high)
        subset = test_results[mask]
        if len(subset) > 0:
            acc = accuracy_score(subset["target"], [1] * len(subset))
            print(f"Confiança [{low:.2f} - {high:.2f}]: {len(subset)} jogos | Precisão: {acc:.2%}")

        mask_away = (test_results["prob_home"] <= (1 - low)) & (test_results["prob_home"] > (1 - high))
        subset_away = test_results[mask_away]
        if len(subset_away) > 0:
            acc_away = accuracy_score(subset_away["target"], [0] * len(subset_away))
            print(f"Confiança [{low:.2f} - {high:.2f}] (Visitante): {len(subset_away)} jogos | Precisão: {acc_away:.2%}")

    # Simulação de ROI
    odd = 1.90
    test_results["win_amount"] = np.where(test_results["pred_home"] == test_results["target"], odd - 1, -1)
    total_roi = test_results["win_amount"].sum() / len(test_results)

    print("\nSIMULAÇÃO DE APOSTAS (Odds Fixas 1.90):")
    print(f"Total de Apostas: {len(test_results)}")
    print(f"Retorno Total: {test_results['win_amount'].sum():.2f} unidades")
    print(f"ROI Estimado: {total_roi:.2%}")

    verdict = "potencial lucrativo" if total_roi > 0 else "precisa de mais refinamento"
    print(f"\nCONCLUSÃO: O modelo demonstra {verdict} nesta temporada!")


def main():
    evaluate_betting_performance()


if __name__ == "__main__":
    main()
