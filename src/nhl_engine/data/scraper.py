import sys
import time

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from nhl_engine.config import NST_NAME_TO_ABBR, NST_STATS_PATH

# Garante suporte a UTF-8 no console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

NST_BASE_URL = "https://www.naturalstattrick.com/teamtable.php"

# Temporadas no formato do NST: "20152016", "20162017", ...
DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2026
REQUEST_DELAY = 5  # segundos entre requests


def _build_url(season: str) -> str:
    """Constrói a URL do NST para uma temporada específica."""
    return f"{NST_BASE_URL}?fromseason={season}&thruseason={season}&stype=2&sit=all&score=all&rate=n&team=all&loc=B&gpf=410&fd=&td="


def _create_driver(version_main=None) -> uc.Chrome:
    """Cria uma instância do Chrome com undetected-chromedriver."""

    def get_options():
        opts = uc.ChromeOptions()
        # Desativa o modo headless para evitar o bloqueio Turnstile do Cloudflare no Windows
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1920,1080")
        return opts

    try:
        if version_main:
            return uc.Chrome(options=get_options(), version_main=version_main)
        return uc.Chrome(options=get_options())
    except Exception as e:
        import re

        err_msg = str(e)
        match = re.search(r"Current browser version is (\d+)\.", err_msg)
        if match and not version_main:
            detected_version = int(match.group(1))
            print(f"  [Auto-Detect] Versao do Chrome detectada: {detected_version}. Tentando novamente com version_main={detected_version}...")
            return uc.Chrome(options=get_options(), version_main=detected_version)
        raise e


def _parse_table(html: str) -> pd.DataFrame:
    """Parseia a tabela HTML do NST e retorna um DataFrame."""
    soup = BeautifulSoup(html, "html.parser")

    # A tabela principal tem id "teams" ou é a primeira tabela grande
    table = soup.find("table", {"id": "teams"})
    if table is None:
        # Fallback: procura a maior tabela na página
        tables = soup.find_all("table")
        if not tables:
            return pd.DataFrame()
        table = max(tables, key=lambda t: len(t.find_all("tr")))

    # Extrai headers
    headers = []
    header_row = table.find("thead")
    if header_row:
        for th in header_row.find_all("th"):
            headers.append(th.get_text(strip=True))

    # Extrai dados
    rows = []
    tbody = table.find("tbody")
    if tbody:
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(cells)

    if not headers or not rows:
        # Tenta usar pd.read_html como fallback
        dfs = pd.read_html(str(table))
        if dfs:
            return dfs[0]
        return pd.DataFrame()

    # Ajusta tamanho se headers e rows não batem
    max_cols = max(len(headers), max(len(r) for r in rows))
    headers = headers + [f"col_{i}" for i in range(len(headers), max_cols)]
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    return pd.DataFrame(rows, columns=headers[:max_cols])


def scrape_team_stats(driver: uc.Chrome, season: str) -> pd.DataFrame:
    """Scrapes team stats de uma temporada específica do NST."""
    url = _build_url(season)
    print(f"  Acessando {season}...")

    driver.get(url)

    # Espera a tabela carregar (até 30s para Cloudflare + rendering)
    try:
        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        # Delay extra para garantir que a tabela renderizou completamente
        time.sleep(2)
    except Exception:
        print(f"  [TIMEOUT] Timeout ao carregar {season}")
        return pd.DataFrame()

    html = driver.page_source
    df = _parse_table(html)

    if df.empty:
        print(f"  [AVISO] Nenhum dado encontrado para {season}")
        return df

    df["season"] = season
    print(f"  [OK] {len(df)} times coletados para {season}")
    return df


def scrape_all_seasons(start_year: int = DEFAULT_START_YEAR, end_year: int = DEFAULT_END_YEAR) -> pd.DataFrame:
    """Coleta stats de todas as temporadas entre start_year e end_year de forma incremental."""
    seasons = [f"{y}{y + 1}" for y in range(start_year, end_year)]
    print(f"Coletando dados do Natural Stat Trick: {len(seasons)} temporadas")

    # Carrega dados existentes para evitar re-scraping
    existing_df = pd.DataFrame()
    scraped_seasons = set()
    if NST_STATS_PATH.exists():
        try:
            existing_df = pd.read_csv(NST_STATS_PATH)
            if "season" in existing_df.columns:
                existing_df["season"] = existing_df["season"].astype(str)
                scraped_seasons = set(existing_df["season"].unique())
                print(f"  [INFO] Encontradas temporadas ja coletadas localmente: {scraped_seasons}")
        except Exception as e:
            print(f"  [AVISO] Erro ao ler arquivo local existente: {e}")

    # Filtra as temporadas a coletar
    seasons_to_scrape = [s for s in seasons if s not in scraped_seasons]

    if not seasons_to_scrape:
        print("[OK] Todas as temporadas ja estao coletadas localmente!")
        # Atualiza o mapeamento dos times no arquivo existente por garantia
        if "Team" in existing_df.columns:
            existing_df["team"] = existing_df["Team"].map(NST_NAME_TO_ABBR).fillna(existing_df["Team"])
            existing_df.to_csv(NST_STATS_PATH, index=False)
        return existing_df

    print(f"  [INFO] Temporadas restantes para coletar: {seasons_to_scrape}")
    driver = _create_driver()
    new_data: list[pd.DataFrame] = []

    try:
        for i, season in enumerate(seasons_to_scrape):
            df = scrape_team_stats(driver, season)
            if not df.empty:
                new_data.append(df)

            # Rate limiting (exceto último)
            if i < len(seasons_to_scrape) - 1:
                print(f"  [WAIT] Aguardando {REQUEST_DELAY}s...")
                time.sleep(REQUEST_DELAY)
    finally:
        driver.quit()

    if not new_data and existing_df.empty:
        print("[ERRO] Nenhum dado novo ou antigo disponivel!")
        return pd.DataFrame()

    # Junta o antigo com o novo
    dfs_to_concat = []
    if not existing_df.empty:
        dfs_to_concat.append(existing_df)
    dfs_to_concat.extend(new_data)
    result = pd.concat(dfs_to_concat, ignore_index=True)

    # Converte colunas numéricas
    skip_cols = {"Team", "team", "season"}
    for col in result.columns:
        if col not in skip_cols:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # Mapeia/atualiza os nomes dos times para siglas (ex: Boston Bruins -> BOS)
    if "Team" in result.columns:
        unmapped = set(result["Team"].unique()) - set(NST_NAME_TO_ABBR.keys())
        if unmapped:
            print(f"  [ALERTA] Times nao mapeados no NST_NAME_TO_ABBR: {unmapped}")
        result["team"] = result["Team"].map(NST_NAME_TO_ABBR)
        result["team"] = result["team"].fillna(result["Team"])

    # Salva
    result.to_csv(NST_STATS_PATH, index=False)
    print(f"\n[SUCESSO] {len(result)} registros totais salvos em {NST_STATS_PATH}")
    return result


def main():
    scrape_all_seasons()


if __name__ == "__main__":
    main()
