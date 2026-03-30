import time
from datetime import datetime

import pandas as pd
import requests

NOW = datetime.now()
dates = ["2022-05-01", "2023-04-14", "2024-04-18", "2025-04-17"]
dates.append(NOW.strftime("%Y-%m-%d"))


def fetch_nhl_seasons():
    for date in dates:
        all_data = []
        url = f"https://api-web.nhle.com/v1/standings/{date}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                json_response = response.json()  # Mudamos o nome para não confundir com o DataFrame

                standings = json_response.get("standings", [])
                if not standings:
                    print(f"Sem dados para a data {date}")
                    continue

                for team in standings:
                    team_info = {
                        # Info dos times
                        "teamLogo": team["teamLogo"],
                        "teamName": team["teamName"]["default"],
                        "teamAbbrev": team["teamAbbrev"]["default"],
                        # Dados Gerais
                        "gamesPlayed": team.get("gamesPlayed"),
                        "points": team.get("points"),
                        "wins": team.get("wins"),
                        "losses": team.get("losses"),
                        "otLosses": team.get("otLosses"),
                        "ties": team.get("ties"),
                        "goalFor": team.get("goalFor"),
                        "goalAgainst": team.get("goalAgainst"),
                        "goalsForPctg": team.get("goalsForPctg"),
                        # Dados Home
                        "homeGamesPlayed": team.get("homeGamesPlayed"),
                        "homePoints": team.get("homePoints"),
                        "homeWins": team.get("homeWins"),
                        "homeLosses": team.get("homeLosses"),
                        "homeOtLosses": team.get("homeOtLosses"),
                        "homeTies": team.get("homeTies"),
                        "homeGoalsFor": team.get("homeGoalsFor"),
                        "homeGoalsAgainst": team.get("homeGoalsAgainst"),
                        # Dados Away
                        "roadGamesPlayed": team.get("roadGamesPlayed"),
                        "roadPoints": team.get("roadPoints"),
                        "roadWins": team.get("roadWins"),
                        "roadLosses": team.get("roadLosses"),
                        "roadOtLosses": team.get("roadOtLosses"),
                        "roadTies": team.get("roadTies"),
                        "roadGoalsFor": team.get("roadGoalsFor"),
                        "roadGoalsAgainst": team.get("roadGoalsAgainst"),
                    }

                    # colocar um espaço no final do texto da coluna 'teamLogo' para evitar problemas de leitura no Power BI
                    team_info["teamLogo"] += " "

                    all_data.append(team_info)

                # Definimos o SEASON com base no primeiro item da lista
                SEASON = standings[0]["seasonId"]

                # Criamos o DataFrame e salvamos FORA do loop dos times
                df_teams = pd.DataFrame(all_data)
                file_path = f"../NHL_PYTHON/team/nhl_{SEASON}.csv"
                df_teams.to_csv(file_path, index=False, sep=";")

                print(f"Sucesso: {file_path} gerado ({len(all_data)} times).")

            else:
                print(f"Erro ao buscar dados para a data {date}: {response.status_code}")

            time.sleep(1)  # Respeita o limite da API

        except Exception as e:
            print(f"Erro ao processar a data {date}: {e}")


if __name__ == "__main__":
    fetch_nhl_seasons()
