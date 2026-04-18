import time

import pandas as pd
import requests


def fetch_all_games(seasons):
    teams = [
        "ANA",
        "BOS",
        "BUF",
        "CGY",
        "CAR",
        "CHI",
        "COL",
        "CBJ",
        "DAL",
        "DET",
        "EDM",
        "FLA",
        "LAK",
        "MIN",
        "MTL",
        "NSH",
        "NJD",
        "NYI",
        "NYR",
        "OTT",
        "PHI",
        "PIT",
        "SJS",
        "SEA",
        "STL",
        "TBL",
        "TOR",
        "VAN",
        "VGK",
        "WSH",
        "WPG",
        "ARI",
        "UTA",
    ]

    all_games = []
    processed_game_ids = set()

    for season in seasons:
        print(f"Buscando jogos da temporada {season}...")
        for team in teams:
            url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    continue

                data = response.json()
                games = data.get("games", [])

                count_added = 0
                for g in games:
                    # Apenas temporada regular (gameType=2) e jogos com placar
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
                            count_added += 1

                if count_added > 0:
                    # print(f"  {team}: {count_added} jogos") # Removido para evitar spam no terminal
                    pass

                time.sleep(0.05)
            except Exception as e:
                print(f"Erro ao buscar {team} na temporada {season}: {e}")

    return pd.DataFrame(all_games)


def main():
    # Coletar apenas 5 temporadas para ser mais rápido e evitar timeouts, mas garantindo volume
    current_year = 2025
    seasons = [f"{year}{year + 1}" for year in range(current_year, current_year - 5, -1)]

    df = fetch_all_games(seasons)

    if not df.empty:
        df = df.sort_values("date")
        output_file = "nhl_games_all_seasons.csv"
        df.to_csv(output_file, index=False)
        print(f"Sucesso! {len(df)} jogos salvos em {output_file}")
    else:
        print("Nenhum jogo encontrado.")


if __name__ == "__main__":
    main()
