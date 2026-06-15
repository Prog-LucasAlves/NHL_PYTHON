import time
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from nhl_engine.config import DATA_PATH, TEAMS_LIST

GAME_COLUMNS = [
    "game_id",
    "date",
    "season",
    "game_type",
    "home_team",
    "home_score",
    "away_team",
    "away_score",
]
SUPPORTED_GAME_TYPES = {2, 3}


def parse_completed_games(games: list[dict], season: str) -> pd.DataFrame:
    """Converte jogos concluídos de temporada regular e playoffs em registros."""
    rows = []
    for game in games:
        home = game.get("homeTeam", {})
        away = game.get("awayTeam", {})
        if game.get("gameType") not in SUPPORTED_GAME_TYPES:
            continue
        if home.get("score") is None or away.get("score") is None:
            continue
        rows.append(
            {
                "game_id": game.get("id"),
                "date": game.get("gameDate"),
                "season": season,
                "game_type": game.get("gameType"),
                "home_team": home.get("abbrev"),
                "home_score": home.get("score"),
                "away_team": away.get("abbrev"),
                "away_score": away.get("score"),
            },
        )
    return pd.DataFrame(rows, columns=GAME_COLUMNS)


def default_seasons(start_year: int = 2015, today: date | None = None) -> list[str]:
    """Retorna temporadas desde start_year até a temporada NHL corrente."""
    current = today or date.today()
    current_start_year = current.year if current.month >= 7 else current.year - 1
    return [f"{year}{year + 1}" for year in range(start_year, current_start_year + 1)]


def fetch_all_games(seasons: list[str]) -> pd.DataFrame:
    """Busca jogos concluídos de temporada regular e playoffs na API oficial."""
    all_games: list[dict] = []
    processed_game_ids: set[int] = set()

    for season in seasons:
        print(f"Buscando jogos da temporada {season}...")
        for team in TEAMS_LIST:
            url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
            try:
                response = requests.get(url, timeout=30)
                if response.status_code != 200:
                    continue

                data = response.json()
                parsed = parse_completed_games(data.get("games", []), season)
                for game_data in parsed.to_dict("records"):
                    game_id = game_data["game_id"]
                    if game_id not in processed_game_ids:
                        all_games.append(game_data)
                        processed_game_ids.add(game_id)

                time.sleep(0.05)
            except Exception as e:
                print(f"Erro ao buscar {team} na temporada {season}: {e}")

    return pd.DataFrame(all_games, columns=GAME_COLUMNS)


def refresh_games_file(
    seasons: list[str] | None = None,
    data_path: str | Path = DATA_PATH,
    fetcher: Callable[[list[str]], pd.DataFrame] = fetch_all_games,
) -> pd.DataFrame:
    """Atualiza o CSV incrementalmente sem apagar histórico em falhas vazias."""
    path = Path(data_path)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=GAME_COLUMNS)
    if not existing.empty and "game_type" not in existing.columns:
        existing["game_type"] = 2
    existing = existing.reindex(columns=GAME_COLUMNS)

    fetched = fetcher(seasons or default_seasons())
    if fetched.empty:
        return existing

    fetched = fetched.reindex(columns=GAME_COLUMNS)
    result = pd.concat([existing, fetched], ignore_index=True)
    result = result.drop_duplicates(subset=["game_id"], keep="last").sort_values(["date", "game_id"]).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    result.to_csv(temp_path, index=False)
    temp_path.replace(path)
    return result


def main():
    df = refresh_games_file()
    print(f"Sucesso! {len(df)} jogos salvos em {DATA_PATH}")


if __name__ == "__main__":
    main()
