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


@st.cache_resource
def get_predictor():
    predictor = NHLPredictorV2()
    predictor._initialize()
    return predictor


def main():
    # Header
    st.markdown('<h1 class="header-text">NHL Predictive Engine</h1>', unsafe_allow_html=True)
    st.markdown("### Inteligência Artificial aplicada ao mercado de Apostas NHL")
    st.divider()

    try:
        predictor = get_predictor()
        teams = sorted(list(predictor.team_states.keys()))
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.info("Certifique-se de que os arquivos 'nhl_model.cbm' e 'nhl_games_all_seasons.csv' existam.")
        return

    # Sidebar
    with st.sidebar:
        st.image("https://assets.nhle.com/logos/nhl/svg/NHL_light.svg", width=100)
        st.header("Configurações")

        home_team = st.selectbox("Time da Casa (Home)", teams, index=teams.index("BOS") if "BOS" in teams else 0)
        away_team = st.selectbox("Time de Fora (Away)", teams, index=teams.index("TOR") if "TOR" in teams else 1)

        st.divider()
        st.markdown("#### Mercado de Apostas")
        market_odd_home = st.number_input(f"Odd na Casa ({home_team})", min_value=1.0, value=2.0, step=0.01)
        market_odd_away = st.number_input(f"Odd na Casa ({away_team})", min_value=1.0, value=2.0, step=0.01)

    # Cálculo da Predição
    if home_team == away_team:
        st.warning("Selecione times diferentes para a predição.")
    else:
        prob_home, prob_away = predictor.predict(home_team, away_team)
        fair_odd_home = 1 / prob_home
        fair_odd_away = 1 / prob_away

        # Dashboard Principal
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"### Confronto: {away_team} @ {home_team}")

            # Card de Predição Profissional
            with st.container():
                st.markdown(
                    f"""
                <div class="prediction-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="text-align: center; flex: 1;">
                            <h2 style="color: #ffffff; margin-bottom: 0;">{home_team}</h2>
                            <h1 style="color: #00ff88; font-size: 4rem; margin-top: 0;">{prob_home:.1%}</h1>
                            <p style="color: #888;">Odd Justa: <b>{fair_odd_home:.2f}</b></p>
                        </div>
                        <div style="font-size: 3rem; color: #444; padding: 0 20px;">VS</div>
                        <div style="text-align: center; flex: 1;">
                            <h2 style="color: #ffffff; margin-bottom: 0;">{away_team}</h2>
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
                    st.success(f"🔥 VALOR ENCONTRADO em {home_team}!")
                    st.markdown(f"**Vantagem Estimada (EV+):** {val:.2f}%")
                else:
                    st.info(f"Sem valor em {home_team}")

            with val_col2:
                if market_odd_away > fair_odd_away:
                    val = (market_odd_away / fair_odd_away - 1) * 100
                    st.success(f"🔥 VALOR ENCONTRADO em {away_team}!")
                    st.markdown(f"**Vantagem Estimada (EV+):** {val:.2f}%")
                else:
                    st.info(f"Sem valor em {away_team}")

        with col2:
            st.markdown("### Comparativo de Força")

            h_state = predictor.team_states[home_team]
            a_state = predictor.team_states[away_team]

            # Gráfico de Gols Recentes
            # (ELO é muito maior que os outros, então vamos focar apenas no ELO em um gauge separadamente)
            st.metric("ELO Rating", f"{h_state['elo']:.0f}", delta=f"{h_state['elo'] - a_state['elo']:.0f} vs {away_team}")
            st.metric("Momentum (L10)", f"{h_state['rolling_wins']:.1%}", delta=f"{h_state['rolling_wins'] - a_state['rolling_wins']:.1%}")

            # Gráfico de Gols Recentes
            fig = go.Figure()
            fig.add_trace(go.Bar(name=home_team, x=["GF (L10)", "GA (L10)"], y=[h_state["rolling_gf"], h_state["rolling_ga"]], marker_color="#00ff88"))
            fig.add_trace(go.Bar(name=away_team, x=["GF (L10)", "GA (L10)"], y=[a_state["rolling_gf"], a_state["rolling_ga"]], marker_color="#00bdff"))
            fig.update_layout(barmode="group", template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Rodapé Técnico
    st.divider()
    with st.expander("ℹ️ Detalhes Técnicos do Modelo"):
        st.write("""
        Este modelo utiliza o algoritmo **CatBoost** treinado com dados históricos das últimas 5 temporadas da NHL.

        **Fatores considerados:**
        - **ELO Rating**: Força relativa ajustada pela dificuldade dos adversários.
        - **Rolling Stats**: Momentum ofensivo e defensivo dos últimos 10 jogos.
        - **Home Advantage**: Vantagem histórica de jogar em casa.

        As predições são atualizadas em tempo real com os dados mais recentes do arquivo `nhl_games_all_seasons.csv`.
        """)


if __name__ == "__main__":
    main()
