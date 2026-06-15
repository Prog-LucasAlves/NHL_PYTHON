import pandas as pd
from catboost import CatBoostClassifier

from nhl_engine.betting.strategy import evaluate_bet, fair_odd, is_validated_total_strategy, settle_total_bet
from nhl_engine.config import DATA_PATH
from nhl_engine.model.pregame import PREGAME_FEATURE_COLUMNS, build_pregame_features
from nhl_engine.model.totals import total_probabilities_from_distribution

SCENARIO_WARNING = "Cenario teorico: pressupoe que a odd minima exigida esteve disponivel."
SCENARIO_EDGE = 0.05
SCENARIO_TOTAL_LINES = (5.5, 7.5)


def walk_forward_season_splits(data: pd.DataFrame, validation_seasons: int = 3) -> list[tuple[pd.Index, pd.Index]]:
    """Cria dobras em que toda temporada de treino antecede a validacao."""
    seasons = sorted(data["season"].astype(str).unique())
    splits = []
    for season in seasons[-validation_seasons:]:
        train_index = data.index[data["season"].astype(str) < season]
        validation_index = data.index[data["season"].astype(str) == season]
        if len(train_index) and len(validation_index):
            splits.append((train_index, validation_index))
    return splits


def _summary(data: pd.DataFrame) -> dict[str, float | int]:
    if data.empty:
        return {"bets": 0, "staked": 0.0, "pl": 0.0, "yield_pct": 0.0, "wins": 0, "losses": 0, "pushes": 0, "max_drawdown_pct": 0.0}
    staked = float(data["stake"].sum())
    pl = float(data["pl"].sum())
    max_drawdown = 0.0
    if "bankroll" in data.columns:
        equity = data["bankroll"].astype(float)
        drawdowns = (equity.cummax() - equity) / equity.cummax().clip(lower=0.01) * 100
        max_drawdown = float(drawdowns.max())
    return {
        "bets": len(data),
        "staked": staked,
        "pl": pl,
        "yield_pct": (pl / staked * 100) if staked else 0.0,
        "wins": int((data["result"] == "Green").sum()),
        "losses": int((data["result"] == "Red").sum()),
        "pushes": int((data["result"] == "Push").sum()),
        "max_drawdown_pct": max_drawdown,
    }


def _breakdown(data: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = [{column: value, **_summary(group)} for value, group in data.groupby(column, sort=True)]
    return pd.DataFrame(rows)


def scenario_report(bets: pd.DataFrame) -> dict:
    """Resume o cenario assumido, sempre preservando o alerta de limitacao."""
    totals = bets[bets["market"] == "Total"]
    return {
        "warning": SCENARIO_WARNING,
        "overall": _summary(bets),
        "by_market": _breakdown(bets, "market"),
        "by_game_type": _breakdown(bets, "game_type"),
        "by_season": _breakdown(bets, "season"),
        "by_line": _breakdown(totals, "line") if "line" in totals.columns else pd.DataFrame(),
    }


def _fit_fold_models(train: pd.DataFrame) -> tuple[CatBoostClassifier, CatBoostClassifier]:
    classifier = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.05, loss_function="Logloss", verbose=0, random_seed=42, allow_writing_files=False)
    total_distribution = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05, loss_function="MultiClass", verbose=0, random_seed=42, allow_writing_files=False)
    x_train = train[PREGAME_FEATURE_COLUMNS]
    classifier.fit(x_train, train["target_home_win"])
    total_distribution.fit(x_train, train["total_goals"].clip(upper=12))
    return classifier, total_distribution


def build_walk_forward_predictions(
    features: pd.DataFrame,
    validation_seasons: int = 3,
    total_lines: tuple[float, ...] = SCENARIO_TOTAL_LINES,
) -> pd.DataFrame:
    """Treina em temporadas passadas e gera uma selecao por mercado/jogo."""
    predictions = []
    for train_index, validation_index in walk_forward_season_splits(features, validation_seasons):
        train = features.loc[train_index]
        validation = features.loc[validation_index]
        classifier, total_distribution = _fit_fold_models(train)
        x_validation = validation[PREGAME_FEATURE_COLUMNS]
        home_probabilities = classifier.predict_proba(x_validation)[:, 1]
        total_distributions = total_distribution.predict_proba(x_validation)

        for position, (_, row) in enumerate(validation.iterrows()):
            prob_home = float(home_probabilities[position])
            moneyline_side = "Home" if prob_home >= 0.5 else "Away"
            moneyline_probability = prob_home if moneyline_side == "Home" else 1 - prob_home
            moneyline_result = "Green" if (row["target_home_win"] == 1) == (moneyline_side == "Home") else "Red"
            base = {"date": row["date"], "season": row["season"], "game_type": int(row["game_type"])}
            predictions.append(
                {
                    **base,
                    "market": "Moneyline",
                    "side": moneyline_side,
                    "win_probability": moneyline_probability,
                    "push_probability": 0.0,
                    "result": moneyline_result,
                },
            )

            for total_line in total_lines:
                probabilities = total_probabilities_from_distribution(total_distribution.classes_, total_distributions[position], total_line)
                total_side = "Over" if probabilities.over >= probabilities.under else "Under"
                if not is_validated_total_strategy(total_side, total_line):
                    continue
                total_probability = probabilities.over if total_side == "Over" else probabilities.under
                total_result = settle_total_bet(total_side, total_line, int(row["total_goals"]))
                predictions.append(
                    {
                        **base,
                        "market": "Total",
                        "line": total_line,
                        "side": f"{total_side} {total_line:g}",
                        "win_probability": total_probability,
                        "push_probability": probabilities.push,
                        "result": total_result,
                    },
                )
    return pd.DataFrame(predictions).sort_values(["market", "date"]).reset_index(drop=True)


def simulate_scenario(predictions: pd.DataFrame, initial_bankroll: float = 100.0) -> pd.DataFrame:
    """Simula uma unica selecao por mercado assumindo a odd minima exigida."""
    bankrolls: dict[str, float] = {}
    bets = []
    for row in predictions.itertuples(index=False):
        bankroll = bankrolls.setdefault(row.market, initial_bankroll)
        assumed_odd = fair_odd(row.win_probability, row.push_probability) * (1 + SCENARIO_EDGE)
        decision = evaluate_bet(
            row.win_probability,
            assumed_odd,
            bankroll=bankroll,
            kelly_multiplier=0.25,
            push_probability=row.push_probability,
        )
        if not decision.qualifies or decision.stake <= 0:
            continue
        if row.result == "Green":
            pl = decision.stake * (assumed_odd - 1)
        elif row.result == "Red":
            pl = -decision.stake
        else:
            pl = 0.0
        bankrolls[row.market] = bankroll + pl
        bets.append(
            {
                **row._asdict(),
                "odd": assumed_odd,
                "stake": decision.stake,
                "pl": pl,
                "bankroll": bankrolls[row.market],
            },
        )
    return pd.DataFrame(bets)


def _print_breakdown(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    if frame.empty:
        print("Sem apostas qualificadas.")
        return
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.2f}"))


def evaluate_betting_performance(validation_seasons: int = 3) -> dict:
    """Executa avaliacao walk-forward e cenario de precos minimos."""
    games = pd.read_csv(DATA_PATH)
    features = build_pregame_features(games, min_games=5)
    predictions = build_walk_forward_predictions(features, validation_seasons=validation_seasons)
    bets = simulate_scenario(predictions)
    report = scenario_report(bets)

    print("=" * 72)
    print("AVALIACAO WALK-FORWARD DE MONEYLINE E TOTAL DE GOLS")
    print(SCENARIO_WARNING)
    print("Resultados positivos nao comprovam que essas odds existiram no mercado.")
    print("=" * 72)
    _print_breakdown("POR MERCADO", report["by_market"])
    _print_breakdown("POR TIPO DE JOGO (2=REGULAR, 3=PLAYOFFS)", report["by_game_type"])
    _print_breakdown("POR TEMPORADA", report["by_season"])
    _print_breakdown("TOTAIS POR LINHA VALIDADA", report["by_line"])
    return report


def main() -> None:
    evaluate_betting_performance()


if __name__ == "__main__":
    main()
