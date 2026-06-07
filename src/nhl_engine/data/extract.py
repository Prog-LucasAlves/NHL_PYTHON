import time

import pandas as pd
import requests

from nhl_engine.config import DATA_PATH, TEAMS_LIST


def fetch_all_games(seasons: list[str]) -> pd.DataFrame:
    """Busca jogos da temporada regular via API oficial da NHL."""
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
                games = data.get("games", [])

                for g in games:
                    if g.get("gameType") == 2 and g.get("homeTeam", {}).get("score") is not None:
                        game_id = g.get("id")
                        if game_id not in processed_game_ids:
                            game_data = {
                                "game_id": game_id,
                                "date": g.get("gameDate"),
                                "season": season,
                                "home_team": g.get("homeTeam", {}).get("abbrev"),
                                "home_score": g.get("homeTeam", {}).get("score"),
                                "away_team": g.get("awayTeam", {}).get("abbrev"),
                                "away_score": g.get("awayTeam", {}).get("score"),
                            }
                            all_games.append(game_data)
                            processed_game_ids.add(game_id)

                time.sleep(0.05)
            except Exception as e:
                print(f"Erro ao buscar {team} na temporada {season}: {e}")

    return pd.DataFrame(all_games)


def main():
    seasons = [f"{year}{year + 1}" for year in range(2015, 2026)]

    df = fetch_all_games(seasons)

    if not df.empty:
        df = df.sort_values("date")
        df.to_csv(DATA_PATH, index=False)
        print(f"Sucesso! {len(df)} jogos salvos em {DATA_PATH}")
    else:
        print("Nenhum jogo encontrado.")


if __name__ == "__main__":
    main()
