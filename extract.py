import time

import pandas as pd
import requests


def fetch_nhl_team_stats(season_id):
    """
    Coleta estatísticas de resumo dos times da NHL para uma temporada específica.
    """
    base_url = "https://api.nhle.com/stats/rest/en/team/summary"

    # Parâmetros da API (baseados na observação do site nhl.com/stats)
    # gameTypeId=2 refere-se à temporada regular
    params = {
        "isAggregate": "false",
        "isGame": "false",
        "sort": '[{"property":"points","direction":"DESC"},{"property":"wins","direction":"DESC"}]',
        "start": 0,
        "limit": 100,  # Aumentado para garantir que pegamos todos os times (atualmente 32)
        "factCayce": "count",
        "cayenneExp": f"gameTypeId=2 and seasonId={season_id}",
    }

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

    print(f"Coletando dados da temporada {season_id}...")

    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        if "data" in data and data["data"]:
            df = pd.DataFrame(data["data"])
            # Adiciona a temporada como uma coluna para facilitar a análise posterior
            df["season"] = f"{str(season_id)[:4]}-{str(season_id)[4:]}"
            return df
        else:
            print(f"Nenhum dado encontrado para a temporada {season_id}.")
            return None

    except Exception as e:
        print(f"Erro ao coletar dados da temporada {season_id}: {e}")
        return None


def main():
    # Define as últimas 10 temporadas (ex: 20252026, 20242025, ..., 20162017)
    # Nota: A temporada é representada como YYYY(Y+1)
    current_year = 2025
    seasons = [f"{year}{year + 1}" for year in range(current_year, current_year - 10, -1)]

    all_seasons_data = []

    for season in seasons:
        df = fetch_nhl_team_stats(season)
        if df is not None:
            all_seasons_data.append(df)

        # Pequeno delay para ser amigável com a API
        time.sleep(1)

    if all_seasons_data:
        # Combina todos os DataFrames
        final_df = pd.concat(all_seasons_data, ignore_index=True)

        # Salva em CSV
        output_file = "nhl_team_stats_last_10_seasons.csv"
        final_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 50)
        print(f"Sucesso! Dados coletados e salvos em: {output_file}")
        print(f"Total de registros: {len(final_df)}")
        print(f"Temporadas processadas: {final_df['season'].nunique()}")
        print("=" * 50)
    else:
        print("Nenhum dado foi coletado.")


if __name__ == "__main__":
    main()
