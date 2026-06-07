import numpy as np
import pandas as pd

from nhl_engine.config import (
    DATA_PATH,
    NST_STATS_PATH,
)

# Estatísticas base coletadas do Natural Stat Trick para features
NST_FEATURE_BASE = ["points_pct", "cf_pct", "ff_pct", "sf_pct", "gf_pct", "xgf_pct", "scf_pct", "hdcf_pct", "hdgf_pct", "sh_pct", "sv_pct", "pdo"]

NST_DIFF_FEATURES = ["points_pct", "cf_pct", "xgf_pct", "hdcf_pct", "pdo"]

NST_FEATURE_COLUMNS = []
for col in NST_FEATURE_BASE:
    NST_FEATURE_COLUMNS.extend([f"home_{col}", f"away_{col}"])

FEATURE_COLUMNS = (
    [
        "home_team",
        "away_team",
    ]
    + NST_FEATURE_COLUMNS
    + [f"{col}_diff" for col in NST_DIFF_FEATURES]
)

CAT_FEATURES = ["home_team", "away_team"]


def load_and_preprocess(data_path: str | None = None) -> pd.DataFrame:
    """Carrega o CSV dos jogos e cria a coluna target."""
    path = data_path or str(DATA_PATH)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["target"] = (df["home_score"] > df["away_score"]).astype(int)
    return df


def merge_nst_stats(df: pd.DataFrame, nst_df: pd.DataFrame) -> pd.DataFrame:
    """Une as estatísticas avançadas do NST para os times de casa e visitante por temporada."""
    nst_df = nst_df.copy()

    # Remove a coluna Team original para evitar duplicação pós conversão de nomes para minúsculas
    if "Team" in nst_df.columns and "team" in nst_df.columns:
        nst_df = nst_df.drop(columns=["Team"])

    new_cols = {}
    for col in nst_df.columns:
        clean_col = col.lower().replace("%", "_pct").replace(" ", "_").replace("-", "_")
        while "__" in clean_col:
            clean_col = clean_col.replace("__", "_")
        clean_col = clean_col.strip("_")
        if clean_col == "point_pct":
            clean_col = "points_pct"
        new_cols[col] = clean_col
    nst_df = nst_df.rename(columns=new_cols)

    nst_df["season"] = nst_df["season"].astype(str)
    nst_df["team"] = nst_df["team"].astype(str)

    df["season"] = df["season"].astype(str)
    df["home_team"] = df["home_team"].astype(str)
    df["away_team"] = df["away_team"].astype(str)

    cols_to_keep = ["team", "season"] + NST_FEATURE_BASE
    nst_subset = nst_df[[c for c in cols_to_keep if c in nst_df.columns]].copy()

    # Merge para time de casa (home)
    home_nst = nst_subset.rename(columns={col: f"home_{col}" for col in NST_FEATURE_BASE if col in nst_subset.columns})
    home_nst = home_nst.rename(columns={"team": "home_team"})

    # Merge para time visitante (away)
    away_nst = nst_subset.rename(columns={col: f"away_{col}" for col in NST_FEATURE_BASE if col in nst_subset.columns})
    away_nst = away_nst.rename(columns={"team": "away_team"})

    df = df.merge(home_nst, on=["home_team", "season"], how="left")
    df = df.merge(away_nst, on=["away_team", "season"], how="left")

    # Calcula diferenças de performance relativa
    for col in NST_DIFF_FEATURES:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col in df.columns and away_col in df.columns:
            df[f"{col}_diff"] = df[home_col] - df[away_col]

    # Preenche valores nulos com a média da temporada por segurança
    for col in NST_FEATURE_BASE:
        for prefix in ["home", "away"]:
            col_name = f"{prefix}_{col}"
            if col_name in df.columns:
                df[col_name] = df.groupby("season")[col_name].transform(lambda x: x.fillna(x.mean()))
                df[col_name] = df[col_name].fillna(0.0 if "pct" in col else 1.0)

    for col in NST_DIFF_FEATURES:
        diff_col = f"{col}_diff"
        if diff_col in df.columns:
            df[diff_col] = df[diff_col].fillna(0.0)

    return df


def build_features(data_path: str | None = None, nst_stats_path: str | None = None) -> pd.DataFrame:
    """Pipeline completo de features: carregar jogos e mesclar estatísticas do NST."""
    df = load_and_preprocess(data_path)

    nst_path = nst_stats_path or str(NST_STATS_PATH)
    try:
        nst_df = pd.read_csv(nst_path)
        df = merge_nst_stats(df, nst_df)
    except Exception as e:
        print(f"[AVISO] Erro ao carregar/processar stats NST em {nst_path}: {e}. Continuando com NaNs.")
        for col in FEATURE_COLUMNS:
            if col not in df.columns and col not in ["home_team", "away_team"]:
                df[col] = np.nan

    return df
