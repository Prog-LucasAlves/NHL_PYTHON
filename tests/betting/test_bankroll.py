from pathlib import Path

import pandas as pd

from nhl_engine.betting.bankroll import load_history, log_bet


def test_old_moneyline_rows_remain_readable(tmp_path: Path):
    path = tmp_path / "bets.csv"
    pd.DataFrame(
        [
            {
                "Data": "2026-06-01",
                "Mandante": "CAR",
                "Visitante": "VGK",
                "Entrada": "CAR",
                "Odd": 1.9,
                "Resultado": "Green",
                "Stake": 1.0,
                "PL": 0.9,
            },
        ],
    ).to_csv(path, index=False)

    result = load_history(path)

    assert result is not None
    assert result.loc[0, "Mercado"] == "Moneyline"
    assert pd.isna(result.loc[0, "Linha"])


def test_total_bet_persists_market_and_line(tmp_path: Path):
    path = tmp_path / "bets.csv"

    result = log_bet(
        "2026-06-01",
        "CAR",
        "VGK",
        "Over 5.5",
        1.95,
        "Pendente",
        1.0,
        market="Total",
        line=5.5,
        log_path=path,
    )

    assert result.loc[0, "Mercado"] == "Total"
    assert result.loc[0, "Linha"] == 5.5
