import os
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st

st.set_page_config(page_title="NHL Super App Apostas (+EV)", layout="wide", page_icon="🏒")

file_map = {
    "2025-2026": "team/nhl_20252026.csv",
    "2024-2025": "team/nhl_20242025.csv",
    "2023-2024": "team/nhl_20232024.csv",
    "2022-2023": "team/nhl_20222023.csv",
    "2021-2022": "team/nhl_20212022.csv",
}

st.sidebar.title("🏒 NHL Super App Apostas")
st.sidebar.markdown("---")
season_options = list(file_map.keys())
season = st.sidebar.selectbox("Selecione a Temporada", season_options)
file_path = file_map[season]


def normalize_name(name):
    if pd.isna(name):
        return ""
    name = unicodedata.normalize("NFKD", str(name)).encode("ASCII", "ignore").decode("utf-8")
    name = name.replace(".", "").strip().lower()
    return name


@st.cache_data
def load_data(path, season_name):
    try:
        df = pd.read_csv(path, sep=";")

        stats_file = f"team_stats/nhl_{season_name.replace('-', '')}_stats.csv"
        try:
            df_stats = pd.read_csv(stats_file)
            df["mergeName"] = df["teamName"].apply(normalize_name)
            df_stats["mergeName"] = df_stats["Team"].apply(normalize_name)
            df = pd.merge(df, df_stats, on="mergeName", how="left")
        except Exception:
            st.sidebar.warning(f"Dados avançados não encontrados para {season_name}.")
            for col in ["xGF", "xGA", "CF%", "GP_y"]:
                if col not in df.columns:
                    df[col] = np.nan

        if "homeGamesPlayed" not in df.columns or df["homeGamesPlayed"].sum() == 0:
            df["homeGamesPlayed"] = 1
        if "roadGamesPlayed" not in df.columns or df["roadGamesPlayed"].sum() == 0:
            df["roadGamesPlayed"] = 1

        df["HomeWinPct"] = df["homeWins"] / df["homeGamesPlayed"]
        df["RoadWinPct"] = df["roadWins"] / df["roadGamesPlayed"]
        df["HomeGoalsForPG"] = df["homeGoalsFor"] / df["homeGamesPlayed"]
        df["HomeGoalsAgainstPG"] = df["homeGoalsAgainst"] / df["homeGamesPlayed"]
        df["RoadGoalsForPG"] = df["roadGoalsFor"] / df["roadGamesPlayed"]
        df["RoadGoalsAgainstPG"] = df["roadGoalsAgainst"] / df["roadGamesPlayed"]
        df["TotalGoalsPG"] = (df["goalFor"] + df["goalAgainst"]) / df["gamesPlayed"]

        gp_col = "GP_y" if "GP_y" in df.columns else "gamesPlayed"
        df["xGF_PG"] = df["xGF"] / df[gp_col]
        df["xGA_PG"] = df["xGA"] / df[gp_col]

        df["xGF_PG"] = df["xGF_PG"].fillna(df["goalFor"] / df["gamesPlayed"])
        df["xGA_PG"] = df["xGA_PG"].fillna(df["goalAgainst"] / df["gamesPlayed"])
        df["CF%"] = df.get("CF%", pd.Series(50.0, index=df.index)).fillna(50.0)

        if "teamLogo" in df.columns:
            df["teamLogo"] = df["teamLogo"].str.strip()

        df["League_xGF_PG"] = df["xGF_PG"].mean()
        df["League_xGA_PG"] = df["xGA_PG"].mean()

        return df
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()


@st.cache_data
def load_game_logs(season_name):
    path = f"team_stats_game/nhl_{season_name.replace('-', '')}_stats_game.csv"
    try:
        df_g = pd.read_csv(path)
        df_g["DateStr"] = df_g["Game"].str.extract(r"(\d{4}-\d{2}-\d{2})")
        df_g["Date"] = pd.to_datetime(df_g["DateStr"])
        df_g["TeamNorm"] = df_g["Team"].apply(normalize_name)
        for col in ["xGF", "xGA", "GF", "GA"]:
            if col in df_g.columns:
                df_g[col] = pd.to_numeric(df_g[col], errors="coerce")
        return df_g
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_goalie_stats(season_name):
    # Season name format might need to be '20252026'
    path = f"player_goalies_stats/nhl_{season_name.replace('-', '')}_goalies_stats.csv"
    try:
        df_goalies = pd.read_csv(path)
        df_goalies["PlayerNorm"] = df_goalies["Player"].apply(normalize_name)
        df_goalies["TeamNorm"] = df_goalies["Team"].apply(normalize_name)
        for col in ["GSAA", "SV%", "xG Against", "HD Saves", "HD Shots Against"]:
            if col in df_goalies.columns:
                df_goalies[col] = pd.to_numeric(df_goalies[col].astype(str).str.replace("-", ""), errors="coerce")
        return df_goalies
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_individual_stats(season_name):
    path = f"player_individual_stats/nhl_{season_name.replace('-', '')}_individual_stats.csv"
    try:
        df_players = pd.read_csv(path)
        df_players["PlayerNorm"] = df_players["Player"].apply(normalize_name)
        df_players["TeamNorm"] = df_players["Team"].apply(normalize_name)
        for col in ["ixG", "iHDCF", "Shots", "Faceoffs %"]:
            if col in df_players.columns:
                df_players[col] = pd.to_numeric(df_players[col].astype(str).str.replace("-", ""), errors="coerce")
        return df_players
    except Exception:
        return pd.DataFrame()


def get_team_form(df_g, team_norm, sim_date_pd, fallback_xgf, fallback_xga):
    if df_g.empty:
        return fallback_xgf, fallback_xga, False, 0, "N/A"

    team_hist = df_g[(df_g["TeamNorm"] == team_norm) & (df_g["Date"].dt.date < sim_date_pd)]

    if team_hist.empty:
        return fallback_xgf, fallback_xga, False, 0, "N/A"

    team_hist = team_hist.sort_values(by="Date", ascending=False)
    team_15 = team_hist.head(15)

    last_game_date = team_15.iloc[0]["Date"].date()
    days_rest = (sim_date_pd - last_game_date).days

    b2b = days_rest <= 1

    xgf_mean = team_15["xGF"].mean()
    xga_mean = team_15["xGA"].mean()
    games_found = len(team_15)

    if pd.isna(xgf_mean):
        xgf_mean = fallback_xgf
    if pd.isna(xga_mean):
        xga_mean = fallback_xga

    # Parse last 10 Form String
    last_10 = team_hist.head(10).iloc[::-1]  # Oldest to Newest
    form_chars = []
    for _, row in last_10.iterrows():
        try:
            if row["GF"] > row["GA"]:
                form_chars.append("✅")
            elif row["GF"] < row["GA"]:
                form_chars.append("❌")
            else:
                form_chars.append("➖")
        except:
            form_chars.append("?")

    form_str = "".join(form_chars)

    return xgf_mean, xga_mean, b2b, games_found, form_str


df = load_data(file_path, season)
df_game = load_game_logs(season)
df_goalies = load_goalie_stats(season)
df_players = load_individual_stats(season)

if df.empty:
    st.warning("Nenhum dado encontrado.")
    st.stop()

# Adiciona uma nova Tab para Mercados de Jogadores
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Classificação", "⚔️ Simulador (+EV)", "📊 Métricas", "💰 Gestão", "🏒 Props (Jogadores)"])

# ====== TAB 1: CLASSIFICAÇÃO ======
with tab1:
    st.header(f"Classificação e Estatísticas Avancadas - {season}")
    view_cols = ["teamName", "gamesPlayed", "points", "wins", "losses", "goalFor", "goalAgainst", "CF%", "xGF", "xGA"]
    st.dataframe(
        df[["teamLogo"] + [c for c in view_cols if c in df.columns]].sort_values(by="points", ascending=False),
        column_config={
            "teamLogo": st.column_config.ImageColumn("Logo"),
            "teamName": "Time",
            "gamesPlayed": "J",
            "points": "Pts",
            "wins": "V",
            "losses": "D",
            "goalFor": "GF",
            "goalAgainst": "GA",
            "CF%": "Corsi For %",
            "xGF": "xG Ataque",
            "xGA": "xG Defesa",
        },
        use_container_width=True,
        hide_index=True,
    )

# ====== TAB 2: SIMULADOR DE CONFRONTO HÍBRIDO ======
with tab2:
    st.header("Simulador de Confronto: Automação Total (Poisson)")
    st.markdown("O modelo extrai o Histórico dos **Últimos 15 Jogos** no banco de dados e calcula a Fadiga e Média de Força exata de ambos os times no período.")

    sim_date = st.date_input("📅 Data da Simulação (Corta o histórico de jogos *antes* deste dia)", value=datetime.today().date())
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Selecione o Mandante (Home)", df["teamName"].sort_values().tolist(), index=0)
    with col2:
        away_team = st.selectbox("Selecione o Visitante (Away)", df["teamName"].sort_values().tolist(), index=1)

    if home_team == away_team:
        st.warning("Selecione times diferentes.")
    else:
        h_data = df[df["teamName"] == home_team].iloc[0]
        a_data = df[df["teamName"] == away_team].iloc[0]

        h_norm = normalize_name(home_team)
        a_norm = normalize_name(away_team)

        h_xgf_15, h_xga_15, h_b2b, h_games, h_form = get_team_form(df_game, h_norm, sim_date, h_data["xGF_PG"], h_data["xGA_PG"])
        a_xgf_15, a_xga_15, a_b2b, a_games, a_form = get_team_form(df_game, a_norm, sim_date, a_data["xGF_PG"], a_data["xGA_PG"])

        st.markdown("### ⚙️ Detecção de Forma Algorítmica (Últimos Jogos)")

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.info(
                f"**🏠 {home_team}** | Histórico Filtrado (Nº Jogos): {h_games}\n"
                + (
                    "⚠️ **ALERTA DE FADIGA:** O Mandante jogou ONTEM e entrará em **Back-to-Back**! Penalty de Defesa/Ataque será aplicado."
                    if h_b2b
                    else "💤 *Team descansado (Jogou há mais de 1 dia)*"
                ),
            )
        with col_m2:
            st.info(
                f"**✈️ {away_team}** | Histórico Filtrado (Nº Jogos): {a_games}\n"
                + (
                    "⚠️ **ALERTA DE FADIGA:** O Visitante jogou ONTEM e entrará em **Back-to-Back**! Penalty de Defesa/Ataque será aplicado."
                    if a_b2b
                    else "💤 *Team descansado (Jogou há mais de 1 dia)*"
                ),
            )

        st.markdown("---")
        h_col, a_col = st.columns(2)
        with h_col:
            st.image(h_data["teamLogo"], width=80)
            st.subheader(f"🏠 {home_team}")

            # GOALIE SELECTOR
            h_goalies = df_goalies[df_goalies["TeamNorm"] == h_norm]
            if not h_goalies.empty:
                h_goalie_name = st.selectbox("Goleiro Titular (Mandante):", h_goalies["Player"].tolist(), key="h_goalie")
                h_g_data = h_goalies[h_goalies["Player"] == h_goalie_name].iloc[0]
                h_gsaa_pg = (h_g_data["GSAA"] / h_g_data["GP"]) if h_g_data["GP"] > 0 else 0
                st.caption(f"🛡️ GSAA/J: **{h_gsaa_pg:+.2f}** | SV%: {h_g_data['SV%']}")
                # Adjust Team xGA based on this specific goalie's performance vs average
                h_xga_15 -= h_gsaa_pg
            else:
                st.caption("Sem dados de goleiro.")

            st.markdown(f"**Forma (Últimos 10):** {h_form}")
            st.write(f"**Ataque (xGF/J):** {h_xgf_15:.2f}")
            st.write(f"**Defesa Ajustada (xGA/J):** {h_xga_15:.2f}")

        with a_col:
            st.image(a_data["teamLogo"], width=80)
            st.subheader(f"✈️ {away_team}")

            # GOALIE SELECTOR
            a_goalies = df_goalies[df_goalies["TeamNorm"] == a_norm]
            if not a_goalies.empty:
                a_goalie_name = st.selectbox("Goleiro Titular (Visitante):", a_goalies["Player"].tolist(), key="a_goalie")
                a_g_data = a_goalies[a_goalies["Player"] == a_goalie_name].iloc[0]
                a_gsaa_pg = (a_g_data["GSAA"] / a_g_data["GP"]) if a_g_data["GP"] > 0 else 0
                st.caption(f"🛡️ GSAA/J: **{a_gsaa_pg:+.2f}** | SV%: {a_g_data['SV%']}")
                # Adjust Team xGA based on this specific goalie's performance vs average
                a_xga_15 -= a_gsaa_pg
            else:
                st.caption("Sem dados de goleiro.")

            st.markdown(f"**Forma (Últimos 10):** {a_form}")
            st.write(f"**Ataque (xGF/J):** {a_xgf_15:.2f}")
            st.write(f"**Defesa Ajustada (xGA/J):** {a_xga_15:.2f}")

        # REGRESSÃO ESTATÍSTICA (xG Form + Fatores)
        league_xg = h_data["League_xGF_PG"]

        # Ensure we don't drop below a minimum threshold for Poisson lambda
        h_xga_15 = max(0.5, h_xga_15)
        a_xga_15 = max(0.5, a_xga_15)

        h_att_raw = h_xgf_15 / league_xg
        h_def_raw = h_xga_15 / league_xg
        a_att_raw = a_xgf_15 / league_xg
        a_def_raw = a_xga_15 / league_xg

        home_adv_atk = 1.05
        home_adv_def = 0.95

        h_att_adj = h_att_raw * home_adv_atk
        h_def_adj = h_def_raw * home_adv_def
        a_att_adj = a_att_raw * (1 / home_adv_atk)
        a_def_adj = a_def_raw * (1 / home_adv_def)

        if h_b2b:
            h_att_adj *= 0.90
            h_def_adj *= 1.10
        if a_b2b:
            a_att_adj *= 0.90
            a_def_adj *= 1.10

        lam_home = h_att_adj * a_def_adj * league_xg
        lam_away = a_att_adj * h_def_adj * league_xg

        max_g = 10
        prob_matrix = np.zeros((max_g, max_g))
        for i in range(max_g):
            for j in range(max_g):
                prob_matrix[i, j] = stats.poisson.pmf(i, lam_home) * stats.poisson.pmf(j, lam_away)

        prob_home_win = np.tril(prob_matrix, -1).sum()
        prob_away_win = np.triu(prob_matrix, 1).sum()
        prob_draw = np.trace(prob_matrix)
        total_p = prob_home_win + prob_away_win + prob_draw

        prob_home_win /= total_p
        prob_away_win /= total_p
        prob_draw /= total_p

        prob_under_55 = sum(prob_matrix[i, j] for i in range(max_g) for j in range(max_g) if i + j <= 5)
        prob_under_55 /= total_p
        prob_over_55 = 1 - prob_under_55

        fair_odd_home = 1 / prob_home_win if prob_home_win > 0 else 0
        fair_odd_away = 1 / prob_away_win if prob_away_win > 0 else 0
        fair_odd_draw = 1 / prob_draw if prob_draw > 0 else 0
        fair_odd_over = 1 / prob_over_55 if prob_over_55 > 0 else 0
        fair_odd_under = 1 / prob_under_55 if prob_under_55 > 0 else 0

        st.markdown("### 🎲 Precificação Estrutural Pós-Modificadores")
        st.write(f"*O algoritmo calculou: **{lam_home:.2f} Gols Esperados** para o Mandante e **{lam_away:.2f} Gols Esperados** para o Visitante baseando-se na forma mais recente da NHL.*")

        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric(f"Prob. Atualizada {home_team}", f"{prob_home_win:.1%}", f"Odd Justa: {fair_odd_home:.2f}")
        p_col2.metric("Prob. Empate (Tempo Reg.)", f"{prob_draw:.1%}", f"Odd Justa: {fair_odd_draw:.2f}")
        p_col3.metric(f"Prob. Atualizada {away_team}", f"{prob_away_win:.1%}", f"Odd Justa: {fair_odd_away:.2f}")

        o_col1, o_col2 = st.columns(2)
        o_col1.metric("Prob. Over 5.5 Gols", f"{prob_over_55:.1%}", f"Odd Justa: {fair_odd_over:.2f}")
        o_col2.metric("Prob. Under 5.5 Gols", f"{prob_under_55:.1%}", f"Odd Justa: {fair_odd_under:.2f}")

        st.markdown("---")
        st.markdown("### 🎯 Avaliador de Apostas de Valor (+EV)")
        st.info("Só insira seu dinheiro no mercado (Back ou Lay) se a Casa de Apostas tiver **errado a linha** e estiver pagando mais do que a Odd Justa Calculada pelo nosso robô estatístico.")

        ev_c1, ev_c2 = st.columns(2)
        with ev_c1:
            st.success(f"📈 **Apoie Financeiramente (Back) o {home_team}** SOMENTE SE o mercado pagar uma **Odd > {fair_odd_home:.2f}**.")
            st.success(f"📈 **Apoie Financeiramente (Back) o {away_team}** SOMENTE SE o mercado pagar uma **Odd > {fair_odd_away:.2f}**.")
        with ev_c2:
            st.warning(f"⚽ **Faça Entrada no OVER 5.5 GOH** SOMENTE SE o mercado pagar uma **Odd > {fair_odd_over:.2f}**.")
            st.warning(f"⚽ **Faça Entrada no UNDER 5.5 GOH** SOMENTE SE o mercado pagar uma **Odd > {fair_odd_under:.2f}**.")

# ====== TAB 3: MÉTRICAS DE VALOR ======
with tab3:
    st.header("Metrificador de Apostas Avançado (> 15 Indicadores)")
    st.markdown("O sistema calcula as principais métricas cruciais de apostas baseadas na temporada inteira. Útil para identificar anomalias, times over/under e domínios absolutos em casa e fora.")

    with st.expander("📚 **DICIONÁRIO DE ESTATÍSTICAS E INTERPRETAÇÃO (Clique para ver o significado de cada tabela)**", expanded=False):
        st.markdown("""
        **🔥 Formação e Placar**
        * **Melhores Mandantes (Vitórias %):** Mostra a taxa real de vitórias do time jogando dentro de casa. *Aposta Ideal:* Base primária para confiar no 'Back' do mandante quando a odd do mercado estiver justa.
        * **Melhores Visitantes (Vitórias %):** Taxa real de vitórias como visitante. *Aposta Ideal:* Encontrar times com alto índice aqui para pegar 'Back Zebras' (Odds 2.50+) contra mandantes instáveis.
        * **Máquinas de Gol (GF/J):** Gols reais marcados por jogo. *Aposta Ideal:* Mercados de 'Over Gols da Equipe', confiando na consistência do ataque.
        * **Muralhas Intransponíveis (GA/J):** Gols reais sofridos por jogo. Média baixa indica defesa impenetrável. *Aposta Ideal:* Excelente para mercados de 'Under 5.5 Gols' na partida toda.
        * **Jogos OVER / UNDER (Gols Totais P/Jogo):** A soma somada (Gols Próprios + Gols Sofridos). *Aposta Ideal:* Se um perfil for alto (ex: 6.40), jogue sempre a favor da bagunça no 'Over 5.5 Gols' ou até 6.5.

        **🏒 Controle e Posse**
        * **Mestres da Posse (CF%):** O Corsi For % mede quem dita o ritmo (chute certo, na trave, bloqueado ou fora). *Aposta Ideal:* Índice constante acima de 50.0% indica um time dominante no longo prazo ('Back').
        * **Reis do xG (xGF%):** "Expected Goals %" une volume MAIS qualidade de chances criadas no gelo contra o que sofre. *Aposta Ideal:* A métrica central do Hóquei. Índice > 50% é ouro para 'Apostar a Favor (Back)'.
        * **Maior Força Bruta (xGF/J):** O xG projetado de ataque por jogo. Remove sorte. *Aposta Ideal:* Quando o time domina essa métrica, aposte no Time Over Gols.
        * **Peneira Defensiva (xGA/J):** Gols Esperados de sofrer. Se a Peneira for muito alta, a defesa permite finalizações fáceis demais da Slot. *Aposta Ideal:* 'Lay' nesse time (apostar contra) ou 'Over Gols' do adversário.
        * **Ametralhadoras (SF):** Tiros ao gol absolutos. *Aposta Ideal:* Times que chutam muito (volume bruto) provocam cansaço nos goleiros rivais o que ajuda na virada.
        * **Maior Sorte / Superação (PDO):** PDO mede uma linha tênue somando (Eficácia de Gol + Qualidade de Goleiro). A média da NHl é fixa em 1.000 (100%). *Aposta Ideal:*
            - Acima de **1.020+**, o time está em maré de muita Sorte, conseguindo gols impossíveis. Logo a fase quebrará: aposte em **Lay neles!**
            - Abaixo de **0.980-**, o time está super Azarado e injustiçado, perdendo de bobeira. Fase logo vira: ótimo para **Back de Valor Alto** neles.

        **⚡ Qualidade e Perigo**
        * **Chances de Alto Perigo (HDCF%):** Semelhante ao xG, mas limita as contas **SOMENTE à boca do gol ("Slot")**. O miolo da zaga. *Aposta Ideal:* Times com domínio do slot ignoram chutes lixos de longe. Muito valioso.
        * **Matadores Profissionais (HDGF%):** Porcentagem de conversão de Alto Perigo *realizada*. *Aposta Ideal:* Analisa se a estrela/centroavante da franquia é letal quando tem chance de matar o jogo.
        * **Eficácia de Chute (SH%):** Porcentagem das tacadas que viram gol de fato. *Aposta Ideal:* Times que tenham CF% (Posse Pobre), mas explodem de muito alto SH% são ninjas retranqueiros de **Contra-Ataque**.
        """)

    # Criar DataFrame seguro evitando erros se faltarem colunas
    metric_df = df[["teamName", "teamLogo"]].copy()
    metric_df["Vitórias Casa %"] = (df["HomeWinPct"] * 100) if "HomeWinPct" in df else 0
    metric_df["Vitórias Fórum %"] = (df["RoadWinPct"] * 100) if "RoadWinPct" in df else 0
    metric_df["Gols a Favor (GF/J)"] = df["goalFor"] / df["gamesPlayed"] if "goalFor" in df else 0
    metric_df["Gols Contra (GA/J)"] = df["goalAgainst"] / df["gamesPlayed"] if "goalAgainst" in df else 0
    metric_df["xG Index (xGF%)"] = df["xGF%"] if "xGF%" in df else 0
    metric_df["Controle Puck (CF%)"] = df["CF%"] if "CF%" in df else 0
    metric_df["Força de Ataque (xGF/J)"] = df["xGF_PG"] if "xGF_PG" in df else 0
    metric_df["Vazamento de Defesa (xGA/J)"] = df["xGA_PG"] if "xGA_PG" in df else 0
    metric_df["Chances Extremas (HDCF%)"] = df["HDCF%"] if "HDCF%" in df else 0
    metric_df["Gols em HD (HDGF%)"] = df["HDGF%"] if "HDGF%" in df else 0
    metric_df["Finalizações a Favor (SF)"] = df["SF"] if "SF" in df else 0
    metric_df["Eficácia Chute (SH%)"] = df["SH%"] if "SH%" in df else 0
    metric_df["Paredões Goleiro (SV%)"] = df["SV%"] if "SV%" in df else 0
    metric_df["Fator Sorte/Azar (PDO)"] = df["PDO"] if "PDO" in df else 0
    metric_df["Perfil Over (Tot Gols/J)"] = df["TotalGoalsPG"] if "TotalGoalsPG" in df else 0
    metric_df["Perfil Under (Tot Gols/J)"] = df["TotalGoalsPG"] if "TotalGoalsPG" in df else 0

    cat_tabs = st.tabs(["🔥 Formação e Placar", "🏒 Controle e Posse", "⚡ Qualidade e Perigo"])

    def render_top_5(col_obj, title, column, is_ascending, format_str="%.2f"):
        temp = metric_df.dropna(subset=[column]).sort_values(by=column, ascending=is_ascending).head(5)
        with col_obj:
            st.markdown(f"**{title}**")
            st.dataframe(
                temp[["teamLogo", "teamName", column]],
                column_config={"teamLogo": st.column_config.ImageColumn(""), "teamName": "Time", column: st.column_config.NumberColumn("Stat", format=format_str)},
                hide_index=True,
                use_container_width=True,
            )

    with cat_tabs[0]:
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        render_top_5(r1_c1, "Melhores Mandantes (Vitórias %)", "Vitórias Casa %", False, "%.1f %%")
        render_top_5(r1_c2, "Melhores Visitantes (Vitórias %)", "Vitórias Fórum %", False, "%.1f %%")
        render_top_5(r1_c3, "Máquinas de Gol (GF/J)", "Gols a Favor (GF/J)", False, "%.2f")

        r2_c1, r2_c2, r2_c3 = st.columns(3)
        render_top_5(r2_c1, "Muralhas Intransponíveis (GA/J)", "Gols Contra (GA/J)", True, "%.2f")
        render_top_5(r2_c2, "Jogos OVER (Gols Totais P/Jogo)", "Perfil Over (Tot Gols/J)", False, "%.2f")
        render_top_5(r2_c3, "Jogos UNDER (Gols Totais P/Jogo)", "Perfil Under (Tot Gols/J)", True, "%.2f")

    with cat_tabs[1]:
        r3_c1, r3_c2, r3_c3 = st.columns(3)
        render_top_5(r3_c1, "Mestres da Posse (CF%)", "Controle Puck (CF%)", False, "%.1f %%")
        render_top_5(r3_c2, "Reis do xG (xGF%)", "xG Index (xGF%)", False, "%.1f %%")
        render_top_5(r3_c3, "Maior Força Bruta (xGF/J)", "Força de Ataque (xGF/J)", False, "%.2f")

        r4_c1, r4_c2, r4_c3 = st.columns(3)
        render_top_5(r4_c1, "Peneira Defensiva (xGA/J)", "Vazamento de Defesa (xGA/J)", False, "%.2f")
        render_top_5(r4_c2, "Ametralhadoras (SF)", "Finalizações a Favor (SF)", False, "%.0f")
        render_top_5(r4_c3, "Maior Sorte / Superação (PDO)", "Fator Sorte/Azar (PDO)", False, "%.3f")

    with cat_tabs[2]:
        r5_c1, r5_c2, r5_c3 = st.columns(3)
        render_top_5(r5_c1, "Chances de Alto Perigo (HDCF%)", "Chances Extremas (HDCF%)", False, "%.1f %%")
        render_top_5(r5_c2, "Matadores Profissionais (HDGF%)", "Gols em HD (HDGF%)", False, "%.1f %%")
        render_top_5(r5_c3, "Eficácia de Chute (SH%)", "Eficácia Chute (SH%)", False, "%.2f %%")

        st.markdown("---")
        st.write(
            "Dica: Times com alto PDO e baixo xGF% tendem a ter regressão (cair de rendimento em breve). Aposte de lay! E times com alta Eficácia de Chute (SH%) mas baixo CF% aproveitam contra-ataques.",
        )

# ====== TAB 4: Gestão de Banca ======
with tab4:
    st.header("Gestão de Banca e Planilha de Apostas 💰")
    st.markdown("Registre suas entradas no mercado para acompanhar seu lucro (ROI) e sua taxa de acerto no longo prazo.")

    bet_file = "bets_log.csv"
    if not os.path.exists(bet_file):
        df_bets = pd.DataFrame(columns=["Data", "Mandante", "Visitante", "Mercado", "Odd", "Stake (R$)", "Status", "Retorno Líquido (R$)"])
        df_bets.to_csv(bet_file, index=False)

    df_bets = pd.read_csv(bet_file)
    df_bets["Data"] = pd.to_datetime(df_bets["Data"], errors="coerce")

    if "Mandante" not in df_bets.columns:
        df_bets["Mandante"] = ""
    if "Visitante" not in df_bets.columns:
        df_bets["Visitante"] = ""
    if "Partida" in df_bets.columns:
        df_bets = df_bets.drop(columns=["Partida"])

    # Ordem fixa
    col_order = ["Data", "Mandante", "Visitante", "Mercado", "Odd", "Stake (R$)", "Status", "Retorno Líquido (R$)"]
    df_bets = df_bets[col_order]

    # Prevenção de Bug Crítico: Coerção Silenciosa de Float (NaN) vs String (Dropdown) no Streamlit
    for col in ["Mandante", "Visitante", "Mercado", "Status"]:
        df_bets[col] = df_bets[col].fillna("").astype(str).replace({"nan": "", "None": ""})

    for col in ["Odd", "Stake (R$)", "Retorno Líquido (R$)"]:
        df_bets[col] = pd.to_numeric(df_bets[col], errors="coerce").fillna(0.0)

    def auto_calc(row):
        status = str(row.get("Status", ""))
        try:
            stake = float(row["Stake (R$)"])
        except:
            stake = 0.0
        try:
            odd = float(row["Odd"])
        except:
            odd = 1.0

        if status == "Green ✅":
            return stake * (odd - 1.0)
        elif status == "Red ❌":
            return -stake
        elif status == "Reembolso 🔄":
            return 0.0
        else:  # Pendente ⏳ or other
            return 0.0

    df_bets["Retorno Líquido (R$)"] = df_bets.apply(auto_calc, axis=1)

    st.markdown("### 📝 Registrar Entradas")

    team_list = df["teamName"].sort_values().tolist()

    config = {
        "Data": st.column_config.DateColumn("Data da Aposta", format="DD/MM/YYYY"),
        "Mandante": st.column_config.SelectboxColumn("Mandante (Casa)", options=team_list, required=True),
        "Visitante": st.column_config.SelectboxColumn("Visitante (Fórum)", options=team_list, required=True),
        "Mercado": st.column_config.SelectboxColumn("Mercado", options=["Back Mandante", "Back Visitante", "Empate", "Over 5.5", "Under 5.5", "Aposta Cíclica", "Outro"], required=True),
        "Odd": st.column_config.NumberColumn("Odd Pega", min_value=1.01, format="%.2f", required=True),
        "Stake (R$)": st.column_config.NumberColumn("Unidade (Stake R$)", min_value=0.0, format="%.2f", required=True),
        "Status": st.column_config.SelectboxColumn("Status", options=["Pendente ⏳", "Green ✅", "Red ❌", "Reembolso 🔄"], required=True),
        "Retorno Líquido (R$)": st.column_config.NumberColumn("Retorno Líquido (+/- R$)", format="%.2f", disabled=True, help="Calculado automaticamente via Sistema."),
    }

    edited_df = st.data_editor(df_bets, num_rows="dynamic", use_container_width=True, column_config=config, key="bet_editor_tabela")

    edited_df["Retorno Líquido (R$)"] = edited_df.apply(auto_calc, axis=1)

    has_changes = False
    try:
        if not df_bets.equals(edited_df):
            has_changes = True
    except Exception:
        # If there's an error comparing (e.g., new rows with NaNs), assume changes
        has_changes = True

    if has_changes:
        edited_df.to_csv(bet_file, index=False)
        st.rerun()

    st.markdown("---")
    st.markdown("### 📈 Dashboard de Desempenho (Analytics)")

    if len(edited_df) > 0:
        resolved_bets = edited_df[edited_df["Status"].isin(["Green ✅", "Red ❌"])]

        edited_df["Stake (R$)"] = pd.to_numeric(edited_df["Stake (R$)"], errors="coerce").fillna(0)
        edited_df["Retorno Líquido (R$)"] = pd.to_numeric(edited_df["Retorno Líquido (R$)"], errors="coerce").fillna(0)

        total_staked = edited_df[edited_df["Status"] != "Pendente ⏳"]["Stake (R$)"].sum()
        total_profit = edited_df["Retorno Líquido (R$)"].sum()

        greens = len(edited_df[edited_df["Status"] == "Green ✅"])
        reds = len(edited_df[edited_df["Status"] == "Red ❌"])
        reembolsos = len(edited_df[edited_df["Status"] == "Reembolso 🔄"])

        total_resolved = greens + reds
        win_rate = (greens / total_resolved * 100) if total_resolved > 0 else 0
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Capital Gasto Analisado", f"R$ {total_staked:.2f}")
        c2.metric("Lucro LÍQUIDO", f"R$ {total_profit:.2f}", "Banca Final" if total_profit >= 0 else "Prejuízo Acumulado")
        c3.metric("Taxa de Conversão (Win Rate)", f"{win_rate:.1f}%", f"{greens} Greens / {reds} Reds")
        c4.metric("Desempenho (ROI %)", f"{roi:.2f}%", "Rendimento Relativo")
    else:
        st.info("Adicione sua primeira aposta clicando na tabela acima para liberar os KPIs da sua Banca.")

# ====== TAB 5: MERCADOS DE JOGADORES (PROPS) ======
with tab5:
    st.header("🏒 Mercados de Jogadores (Player Props)")
    st.markdown("Analise o desempenho individual de cada atleta para explorar mercados de apostas como **Anytime Goalscorer**, **Over Chutes ao Gol** e **Faceoffs**.")

    if df_players.empty:
        st.warning("Dados de jogadores (Skater Stats) não encontrados na pasta `player_individual_stats`.")
    else:
        colP1, colP2 = st.columns(2)
        with colP1:
            available_teams = sorted(df_players["Team"].dropna().unique().tolist()) if "Team" in df_players.columns else []
            teams_list = ["Todos"] + available_teams
            selected_team = st.selectbox("🎯 Filtrar por Time:", teams_list, index=0)
        with colP2:
            search_player = st.text_input("🔍 Buscar Jogador:")

        # Filtros
        filtered_players = df_players.copy()

        if selected_team != "Todos":
            filtered_players = filtered_players[filtered_players["Team"] == selected_team]

        if search_player:
            filtered_players = filtered_players[filtered_players["Player"].str.contains(search_player, case=False, na=False)]

        # Selecionar e Traduzir Colunas Importantes
        props_cols = {
            "Player": "Jogador",
            "Team": "Time",
            "Position": "Pos",
            "GP": "Jogos",
            "Goals": "Gols",
            "Total Points": "Pontos",
            "Shots": "Chutes (SOG)",
            "ixG": "Gols Esperados (ixG)",
            "iHDCF": "Chances Alto Perigo",
            "Faceoffs %": "Faceoffs Vencidos %",
            "TOI": "Tempo de Gelo (Min)",
        }

        display_df = filtered_players[[c for c in props_cols.keys() if c in filtered_players.columns]].rename(columns=props_cols)

        st.dataframe(
            display_df.sort_values(by="Gols Esperados (ixG)", ascending=False),
            column_config={
                "Gols Esperados (ixG)": st.column_config.NumberColumn(format="%.2f"),
                "Faceoffs Vencidos %": st.column_config.NumberColumn(format="%.2f%%"),
                "Tempo de Gelo (Min)": st.column_config.NumberColumn(format="%.1f"),
            },
            use_container_width=True,
            hide_index=True,
            height=600,
        )

        # Dictionary for Props
        with st.expander("📖 Inteligência de Mercados (Como usar essas métricas)"):
            st.markdown("""
            - **Gols Esperados (ixG)**: Mede a qualidade dos chutes que o jogador tentou. Se ele tem um ixG alto mas poucos gols marcados, ele está "azarado" e a tendência é que volte a marcar na sua Média Regredida (+EV para *Anytime Goalscorer*).
            - **Chances Alto Perigo (iHDCF)**: Quantas vezes o jogador consegue chutar de posições claríssimas de gol (geralmente no 'Slot'). Excelentes alvos para Gols e Assistências.
            - **Chutes (SOG)**: Total de chutes no alvo. Divida por "Jogos" para saber a média. Excelente para o mercado *Over/Under Shots on Goal*.
            - **Faceoffs Vencidos %**: Crucial para apostas ao vivo. Centers (Position: C) com mais de 55% são nível Elite e garantem posse ofensiva constante para o seu time no 5 contra 5.
            """)
