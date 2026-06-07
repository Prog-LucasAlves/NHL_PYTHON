import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss

from nhl_engine.config import MODEL_PATH
from nhl_engine.model.features import FEATURE_COLUMNS, build_features


def brier_score_loss(y_true, y_prob):
    """Calcula o Brier Score Loss para calibração de probabilidade."""
    return float(np.mean((y_true - y_prob) ** 2))


def evaluate_betting_performance():
    """Avalia o desempenho do modelo em produção usando Backtest financeiro e Kelly."""
    df = build_features()

    model = CatBoostClassifier()
    try:
        model.load_model(str(MODEL_PATH))
        print(f"Modelo carregado com sucesso a partir de: {MODEL_PATH}")
    except Exception:
        print("Modelo local não encontrado. Iniciando treinamento do zero...")
        from nhl_engine.model.train import train_model

        model = train_model()

    last_season = df["season"].max()
    test_results = df[df["season"] == last_season].copy()

    print(f"\nAvaliando desempenho na temporada de teste: {last_season} ({len(test_results)} jogos)...")

    x_test = test_results[FEATURE_COLUMNS]
    y_test = test_results["target"]

    test_results["prob_home"] = model.predict_proba(x_test)[:, 1]
    test_results["pred_home"] = (test_results["prob_home"] > 0.5).astype(int)

    # 1. Métricas Estatísticas Tradicionais
    acc = accuracy_score(y_test, test_results["pred_home"])
    loss = log_loss(y_test, test_results["prob_home"])
    brier = brier_score_loss(y_test, test_results["prob_home"])

    print("\n" + "=" * 50)
    print("MÉTRICAS ESTATÍSTICAS DA TEMPORADA:")
    print(f"Acurácia: {acc:.2%}")
    print(f"Log Loss: {loss:.4f}")
    print(f"Brier Score (Calibração): {brier:.4f}")
    print("=" * 50)

    # 2. Backtest de Apostas de Valor com Critério de Kelly
    # Odd de mercado simulada: 1.91 (Vig conservador de 4.5% para ambos os lados)
    market_odd = 1.91
    implied_prob = 1 / market_odd  # ~52.36%

    initial_bankroll = 100.0
    kelly_fraction = 0.25  # 1/4 Kelly para controle de risco profissional

    bets = []

    for idx, row in test_results.iterrows():
        prob_h = row["prob_home"]
        target = row["target"]

        # Aposta no Mandante (Home) se houver EV+
        if prob_h > implied_prob:
            kelly_f = (prob_h * market_odd - 1) / (market_odd - 1)
            stake = kelly_f * kelly_fraction * initial_bankroll
            stake = min(stake, 10.0)  # Limite de stake máxima por aposta (10% da banca)

            pl = stake * (market_odd - 1) if target == 1 else -stake
            bets.append({"tipo": "HOME", "prob": prob_h, "stake": stake, "pl": pl, "win": int(target == 1)})

        # Aposta no Visitante (Away) se houver EV+
        elif (1 - prob_h) > implied_prob:
            prob_a = 1 - prob_h
            kelly_f = (prob_a * market_odd - 1) / (market_odd - 1)
            stake = kelly_f * kelly_fraction * initial_bankroll
            stake = min(stake, 10.0)

            pl = stake * (market_odd - 1) if target == 0 else -stake
            bets.append({"tipo": "AWAY", "prob": prob_a, "stake": stake, "pl": pl, "win": int(target == 0)})

    # Métricas de Backtesting
    if len(bets) > 0:
        df_bets = pd.DataFrame(bets)
        total_bets = len(df_bets)
        win_bets = df_bets["win"].sum()
        win_rate = win_bets / total_bets
        total_staked = df_bets["stake"].sum()
        total_pl = df_bets["pl"].sum()
        yield_pct = (total_pl / total_staked * 100) if total_staked > 0 else 0.0

        # Cálculo do Drawdown Máximo
        saldo = initial_bankroll + df_bets["pl"].cumsum()
        peak = initial_bankroll
        max_drawdown = 0.0
        for s in saldo:
            if s > peak:
                peak = s
            dd = (peak - s) / peak * 100
            if dd > max_drawdown:
                max_drawdown = dd

        print("\nSIMULAÇÃO FINANCEIRA DE BACKTEST (+EV com 1/4 Kelly):")
        print(f"Total de Oportunidades (+EV): {total_bets} jogos")
        print(f"Taxa de Acerto nas Apostas: {win_rate:.2%}")
        print(f"Volume Total Apostado: {total_staked:.2f} unidades")
        print(f"Lucro Líquido Acumulado: {total_pl:+.2f} unidades")
        print(f"Yield do Backtest: {yield_pct:+.2f}%")
        print(f"Drawdown Máximo Estimado: {max_drawdown:.2f}%")
    else:
        print("\nNenhuma aposta de valor (+EV) foi identificada no backtest.")
        total_pl = 0.0
        yield_pct = 0.0

    # 3. Auditoria de Aptidão para Produção
    print("\n" + "=" * 50)
    print("AUDITORIA DE PRODUÇÃO (VEREDICTO TÉCNICO):")
    print("=" * 50)

    cond_acc = acc >= 0.595
    cond_brier = brier <= 0.243
    cond_yield = yield_pct > 0

    print(f"1. Acurácia de Validação (>= 59.5%): {'PASS ✅' if cond_acc else 'FAIL ❌'} ({acc:.2%})")
    print(f"2. Brier Score de Calibração (<= 0.243): {'PASS ✅' if cond_brier else 'FAIL ❌'} ({brier:.4f})")
    print(f"3. Yield de Backtest Financeiro (> 0.0%): {'PASS ✅' if cond_yield else 'FAIL ❌'} ({yield_pct:+.2f}%)")

    is_ready = cond_acc and cond_brier and cond_yield

    if is_ready:
        print("\n🏆 VEREDICTO FINAL: APTO PARA PRODUÇÃO! 🎉")
        print("O modelo demonstra estabilidade preditiva, calibração confiável e lucratividade histórica consistente.")
    else:
        print("\n⚠️ VEREDICTO FINAL: REJEITADO PARA PRODUÇÃO.")
        print("O modelo falhou em uma ou mais métricas críticas de auditoria. Refine as features ou colete mais dados.")
    print("=" * 50)


def main():
    evaluate_betting_performance()


if __name__ == "__main__":
    main()
