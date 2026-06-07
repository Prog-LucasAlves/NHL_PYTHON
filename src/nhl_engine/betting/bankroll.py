import pandas as pd

from nhl_engine.config import BETS_LOG_PATH


def log_bet(date: str, home: str, away: str, entry: str, odd: float, result: str, stake: float = 1.0) -> pd.DataFrame:
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
            },
        ],
    )

    if BETS_LOG_PATH.exists():
        df_log = pd.read_csv(BETS_LOG_PATH)
        if "Stake" not in df_log.columns:
            df_log["Stake"] = 1.0
        df_log = pd.concat([df_log, new_bet], ignore_index=True)
    else:
        df_log = new_bet

    df_log.to_csv(BETS_LOG_PATH, index=False)
    return df_log


def load_history() -> pd.DataFrame | None:
    """Carrega o histórico de apostas. Retorna None se não existir."""
    if not BETS_LOG_PATH.exists():
        return None
    df_log = pd.read_csv(BETS_LOG_PATH)
    if "Stake" not in df_log.columns:
        df_log["Stake"] = 1.0
    return df_log
