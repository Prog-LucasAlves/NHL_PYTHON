import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from predict import NHLPredictorV2

# Configuração da página
st.set_page_config(page_title="NHL Predictive Engine 🏒", page_icon="🏒", layout="wide", initial_sidebar_state="expanded")

# Estilização CSS Customizada para o "Wow Factor"
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30333d;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1a1c24 0%, #2d313d 100%);
        padding: 30px;
        border-radius: 15px;
        border-left: 5px solid #00ff88;
        margin-bottom: 20px;
    }
    .value-badge {
        background-color: #00ff88;
        color: #000000;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
    }
    .header-text {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: -webkit-linear-gradient(#00ff88, #00bdff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


TEAM_MAPPING = {
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


@st.cache_resource
def get_predictor():
    predictor = NHLPredictorV2()
    predictor._initialize()
    return predictor


def log_bet(date, home, away, entry, odd, result):
    file = "bets_log.csv"
    pl = odd - 1 if result == "Green" else -1
    new_bet = pd.DataFrame([{"Data": date, "Mandante": home, "Visitante": away, "Entrada": entry, "Odd": odd, "Resultado": result, "PL": round(pl, 2)}])

    if os.path.exists(file):
        df_log = pd.read_csv(file)
        df_log = pd.concat([df_log, new_bet], ignore_index=True)
    else:
        df_log = new_bet

    df_log.to_csv(file, index=False)
    return df_log


def main():
    # Header
    st.markdown('<h1 class="header-text">NHL Predictive Engine</h1>', unsafe_allow_html=True)
    st.markdown("### Inteligência Artificial aplicada ao mercado de Apostas NHL")
    st.divider()

    try:
        predictor = get_predictor()
        # Filtra siglas que temos no mapeamento
        teams = sorted([t for t in predictor.team_states.keys() if t in TEAM_MAPPING])
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.info("Certifique-se de que os arquivos 'nhl_model.cbm' e 'nhl_games_all_seasons.csv' existam.")
        return

    # Tabs
    tab1, tab2 = st.tabs(["🎯 Predição de Partida", "📊 Gestão de Banca"])

    # Sidebar
    with st.sidebar:
        st.image("https://assets.nhle.com/logos/nhl/svg/NHL_light.svg", width=100)
        st.header("Configurações")

        home_team_abbr = st.selectbox("Time da Casa (Home)", teams, index=teams.index("BOS") if "BOS" in teams else 0, format_func=lambda x: TEAM_MAPPING.get(x, x))
        away_team_abbr = st.selectbox("Time de Fora (Away)", teams, index=teams.index("TOR") if "TOR" in teams else 1, format_func=lambda x: TEAM_MAPPING.get(x, x))

        st.divider()
        st.markdown("#### Mercado de Apostas")
        market_odd_home = st.number_input(f"Odd na Casa ({home_team_abbr})", min_value=1.0, value=2.0, step=0.01)
        market_odd_away = st.number_input(f"Odd na Casa ({away_team_abbr})", min_value=1.0, value=2.0, step=0.01)

    with tab1:
        # Cálculo da Predição
        if home_team_abbr == away_team_abbr:
            st.warning("Selecione times diferentes para a predição.")
        else:
            prob_home, prob_away = predictor.predict(home_team_abbr, away_team_abbr)
            fair_odd_home = 1 / prob_home
            fair_odd_away = 1 / prob_away

            # Dashboard Principal
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"### {TEAM_MAPPING[away_team_abbr]} @ {TEAM_MAPPING[home_team_abbr]}")

                # Card de Predição Profissional
                with st.container():
                    st.markdown(
                        f"""
                    <div class="prediction-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="text-align: center; flex: 1;">
                                <h2 style="color: #ffffff; margin-bottom: 0;">{home_team_abbr}</h2>
                                <h1 style="color: #00ff88; font-size: 4rem; margin-top: 0;">{prob_home:.1%}</h1>
                                <p style="color: #888;">Odd Justa: <b>{fair_odd_home:.2f}</b></p>
                            </div>
                            <div style="font-size: 3rem; color: #444; padding: 0 20px;">VS</div>
                            <div style="text-align: center; flex: 1;">
                                <h2 style="color: #ffffff; margin-bottom: 0;">{away_team_abbr}</h2>
                                <h1 style="color: #00bdff; font-size: 4rem; margin-top: 0;">{prob_away:.1%}</h1>
                                <p style="color: #888;">Odd Justa: <b>{fair_odd_away:.2f}</b></p>
                            </div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Verificação de Valor (Value Betting)
                val_col1, val_col2 = st.columns(2)

                with val_col1:
                    if market_odd_home > fair_odd_home:
                        val = (market_odd_home / fair_odd_home - 1) * 100
                        st.success(f"🔥 VALOR ENCONTRADO em {home_team_abbr}!")
                        st.markdown(f"**Vantagem Estimada (EV+):** {val:.2f}%")
                    else:
                        st.info(f"Sem valor em {home_team_abbr}")

                with val_col2:
                    if market_odd_away > fair_odd_away:
                        val = (market_odd_away / fair_odd_away - 1) * 100
                        st.success(f"🔥 VALOR ENCONTRADO em {away_team_abbr}!")
                        st.markdown(f"**Vantagem Estimada (EV+):** {val:.2f}%")
                    else:
                        st.info(f"Sem valor em {away_team_abbr}")

            with col2:
                st.markdown("### Comparativo de Força")

                h_state = predictor.team_states[home_team_abbr]
                a_state = predictor.team_states[away_team_abbr]

                st.metric("ELO Rating", f"{h_state['elo']:.0f}", delta=f"{h_state['elo'] - a_state['elo']:.0f} vs {away_team_abbr}")
                st.metric("Momentum (L10)", f"{h_state['rolling_wins']:.1%}", delta=f"{h_state['rolling_wins'] - a_state['rolling_wins']:.1%}")

                # Gráfico de Gols Recentes
                fig = go.Figure()
                fig.add_trace(go.Bar(name=home_team_abbr, x=["GF (L10)", "GA (L10)"], y=[h_state["rolling_gf"], h_state["rolling_ga"]], marker_color="#00ff88"))
                fig.add_trace(go.Bar(name=away_team_abbr, x=["GF (L10)", "GA (L10)"], y=[a_state["rolling_gf"], a_state["rolling_ga"]], marker_color="#00bdff"))
                fig.update_layout(barmode="group", template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), height=300)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Registrar Nova Aposta")
        reg_col1, reg_col2, reg_col3 = st.columns(3)

        with reg_col1:
            entry_type = st.radio("Entrada", [f"Mandante ({home_team_abbr})", f"Visitante ({away_team_abbr})"])
            entry_abbr = home_team_abbr if "Mandante" in entry_type else away_team_abbr
        with reg_col2:
            bet_odd = st.number_input("Odd da Entrada", min_value=1.0, value=market_odd_home if "Mandante" in entry_type else market_odd_away, step=0.01)
        with reg_col3:
            bet_result = st.selectbox("Resultado", ["Pendente", "Green", "Red"])

        if st.button("Salvar Aposta"):
            if bet_result == "Pendente":
                st.error("Por favor, selecione o resultado (Green ou Red) para registrar.")
            else:
                log_bet(pd.Timestamp.now().strftime("%Y-%m-%d"), home_team_abbr, away_team_abbr, entry_abbr, bet_odd, bet_result)
                st.success("Aposta registrada com sucesso!")
                st.rerun()

        st.divider()
        st.header("Histórico e Performance")

        if os.path.exists("bets_log.csv"):
            df_history = pd.read_csv("bets_log.csv")

            # Métricas
            p_total = df_history["PL"].sum()
            win_rate = (df_history["Resultado"] == "Green").mean()

            met1, met2, met3 = st.columns(3)
            met1.metric("P/L Total", f"{p_total:+.2f} uds", delta=None)
            met2.metric("Win Rate", f"{win_rate:.1%}")
            met3.metric("Total de Bets", len(df_history))

            st.dataframe(df_history.sort_index(ascending=False), use_container_width=True)

            # Gráfico de Evolução
            df_history["Acumulado"] = df_history["PL"].cumsum()
            fig_evol = px.line(df_history, x=df_history.index, y="Acumulado", title="Curva de Lucro/Prejuízo")
            fig_evol.update_layout(template="plotly_dark")
            st.plotly_chart(fig_evol, use_container_width=True)
        else:
            st.info("Nenhuma aposta registrada ainda. Use o formulário acima para começar.")

    # Rodapé Técnico
    st.divider()
    with st.expander("ℹ️ Detalhes Técnicos do Modelo"):
        st.write("""
        Este modelo utiliza o algoritmo **CatBoost** treinado com dados históricos das últimas 5 temporadas da NHL.

        **Fatores considerados:**
        - **ELO Rating**: Força relativa ajustada pela dificuldade dos adversários.
        - **Rolling Stats**: Momentum ofensivo e defensivo dos últimos 10 jogos.
        - **Home Advantage**: Vantagem histórica de jogar em casa.
        """)


if __name__ == "__main__":
    main()
