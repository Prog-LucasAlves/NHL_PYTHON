import pandas as pd

from nhl_engine.config import BETS_LOG_PATH


def log_bet(date: str, home: str, away: str, entry: str, odd: float, result: str) -> pd.DataFrame:
    """Registra uma aposta no CSV de histórico."""
    pl = odd - 1 if result == "Green" else -1
    new_bet = pd.DataFrame(
        [
            {
                "Data": date,
                "Mandante": home,
                "Visitante": away,
                "Entrada": entry,
                "Odd": odd,
                "Resultado": result,
                "PL": round(pl, 2),
            },
        ],
    )

    if BETS_LOG_PATH.exists():
        df_log = pd.read_csv(BETS_LOG_PATH)
        df_log = pd.concat([df_log, new_bet], ignore_index=True)
    else:
        df_log = new_bet

    df_log.to_csv(BETS_LOG_PATH, index=False)
    return df_log


def load_history() -> pd.DataFrame | None:
    """Carrega o histórico de apostas. Retorna None se não existir."""
    if not BETS_LOG_PATH.exists():
        return None
    return pd.read_csv(BETS_LOG_PATH)
