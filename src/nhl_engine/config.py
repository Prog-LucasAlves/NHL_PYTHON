from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

# Arquivos
MODEL_PATH = DATA_DIR / "nhl_model.cbm"
TOTALS_HOME_MODEL_PATH = DATA_DIR / "nhl_totals_home.cbm"
TOTALS_AWAY_MODEL_PATH = DATA_DIR / "nhl_totals_away.cbm"
DATA_PATH = DATA_DIR / "nhl_games_all_seasons.csv"
NST_STATS_PATH = DATA_DIR / "nst_team_stats.csv"
BETS_LOG_PATH = LOGS_DIR / "bets_log.csv"
BANKROLL_CONFIG_PATH = LOGS_DIR / "bankroll_config.json"

# Garante que os diretórios existam
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Mapeamento único de siglas para nomes completos
TEAM_MAPPING: dict[str, str] = {
    "ANA": "Anaheim Ducks",
    "ARI": "Arizona Coyotes",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montréal Canadiens",
    "NSH": "Nashville Predators",
    "NJD": "New Jersey Devils",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SJS": "San Jose Sharks",
    "SEA": "Seattle Kraken",
    "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals",
    "WPG": "Winnipeg Jets",
    "UTA": "Utah Hockey Club",
}

# Mapeamento reverso: nome completo NST → sigla NHL
# O NST usa nomes completos na tabela, precisamos converter para siglas
NST_NAME_TO_ABBR: dict[str, str] = {v: k for k, v in TEAM_MAPPING.items()}
# Variações de nomes que o NST pode usar
NST_NAME_TO_ABBR.update(
    {
        "Montreal Canadiens": "MTL",
        "Montréal Canadiens": "MTL",
        "St Louis Blues": "STL",
        "St. Louis Blues": "STL",
        "Vegas Golden Knights": "VGK",
        "Utah Hockey Club": "UTA",
        "Utah Mammoth": "UTA",
        "Arizona Coyotes": "ARI",
        "Phoenix Coyotes": "ARI",
        "Atlanta Thrashers": "WPG",
    },
)

TEAMS_LIST: list[str] = sorted(TEAM_MAPPING.keys())

# Constantes do modelo
ELO_K_FACTOR = 20
ELO_HOME_ADVANTAGE = 50
ELO_DEFAULT_RATING = 1500
ROLLING_WINDOW = 10
