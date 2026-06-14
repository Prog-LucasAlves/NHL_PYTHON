from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from nhl_engine.data.extract import refresh_games_file
from nhl_engine.data.scraper import scrape_all_seasons


@dataclass(frozen=True)
class RefreshResult:
    source: str
    success: bool
    rows: int
    message: str


def _run_source(source: str, refresher: Callable[[], pd.DataFrame]) -> RefreshResult:
    try:
        data = refresher()
        return RefreshResult(source, True, len(data), f"{source}: {len(data)} registros disponíveis.")
    except Exception as exc:
        return RefreshResult(source, False, 0, f"{source}: falha ao atualizar: {exc}")


def refresh_all_data(
    game_refresher: Callable[[], pd.DataFrame] = refresh_games_file,
    nst_refresher: Callable[[], pd.DataFrame] = scrape_all_seasons,
) -> list[RefreshResult]:
    """Atualiza jogos NHL e estatísticas NST, preservando resultados parciais."""
    return [
        _run_source("NHL Games", game_refresher),
        _run_source("Natural Stat Trick", nst_refresher),
    ]
