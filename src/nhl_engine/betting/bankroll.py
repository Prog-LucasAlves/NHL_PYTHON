import json
from pathlib import Path

import pandas as pd

from nhl_engine.config import BANKROLL_CONFIG_PATH, BETS_LOG_PATH

_BANKROLL_DEFAULTS = {"bankroll": 100.0, "unit_value": 10.0, "kelly_fraction": 0.50}


def load_bankroll_config() -> dict:
    """Carrega configurações de banca do arquivo JSON. Retorna defaults se não existir."""
    if BANKROLL_CONFIG_PATH.exists():
        try:
            with open(BANKROLL_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {**_BANKROLL_DEFAULTS, **data}
        except Exception:
            pass
    return dict(_BANKROLL_DEFAULTS)


def save_bankroll_config(bankroll: float, unit_value: float, kelly_fraction: float, unit_pct: float = 10.0) -> None:
    """Persiste configurações de banca no arquivo JSON."""
    BANKROLL_CONFIG_PATH.parent.mkdir(exist_ok=True)
    with open(BANKROLL_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "bankroll": bankroll,
                "unit_value": unit_value,
                "unit_pct": unit_pct,
                "kelly_fraction": kelly_fraction,
            },
            f,
            indent=2,
        )


def log_bet(
    date: str,
    home: str,
    away: str,
    entry: str,
    odd: float,
    result: str,
    stake: float = 1.0,
    market: str = "Moneyline",
    line: float | None = None,
    log_path: str | Path = BETS_LOG_PATH,
) -> pd.DataFrame:
    """Registra uma aposta no CSV de histórico."""
    if result == "Green":
        pl = stake * (odd - 1)
    elif result == "Red":
        pl = -stake
    else:  # Pendente
        pl = 0.0
    new_bet = pd.DataFrame(
        [
            {
                "Data": date,
                "Mandante": home,
                "Visitante": away,
                "Entrada": entry,
                "Odd": odd,
                "Resultado": result,
                "Stake": round(stake, 2),
                "PL": round(pl, 2),
                "Mercado": market,
                "Linha": line,
            },
        ],
    )

    path = Path(log_path)
    if path.exists():
        df_log = pd.read_csv(path)
        if "Stake" not in df_log.columns:
            df_log["Stake"] = 1.0
        if "Mercado" not in df_log.columns:
            df_log["Mercado"] = "Moneyline"
        if "Linha" not in df_log.columns:
            df_log["Linha"] = None
        df_log = pd.concat([df_log, new_bet], ignore_index=True)
    else:
        df_log = new_bet

    path.parent.mkdir(parents=True, exist_ok=True)
    df_log.to_csv(path, index=False)
    return df_log


def load_history(log_path: str | Path = BETS_LOG_PATH) -> pd.DataFrame | None:
    """Carrega o histórico de apostas. Retorna None se não existir."""
    path = Path(log_path)
    if not path.exists():
        return None
    df_log = pd.read_csv(path)
    if "Stake" not in df_log.columns:
        df_log["Stake"] = 1.0
    if "Mercado" not in df_log.columns:
        df_log["Mercado"] = "Moneyline"
    if "Linha" not in df_log.columns:
        df_log["Linha"] = None
    return df_log
