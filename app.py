import streamlit as st

from nhl_engine.config import TEAM_MAPPING
from nhl_engine.model.predict import NHLPredictorV2
from nhl_engine.ui import tab_bankroll, tab_prediction
from nhl_engine.ui.styles import CUSTOM_CSS

st.set_page_config(page_title="NHL Predictive Engine 🏒", page_icon="🏒", layout="wide", initial_sidebar_state="expanded")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_predictor() -> NHLPredictorV2:
    predictor = NHLPredictorV2()
    predictor._initialize()
    return predictor


def main():
    st.markdown('<h1 class="header-text">NHL Predictive Engine</h1>', unsafe_allow_html=True)
    st.markdown("### Inteligência Artificial aplicada ao mercado de Apostas NHL")
    st.divider()

    try:
        predictor = get_predictor()
        teams = sorted([t for t in predictor.team_states if t in TEAM_MAPPING])
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.info("Certifique-se de que os arquivos 'nhl_model.cbm' e 'nhl_games_all_seasons.csv' existam na pasta data/.")
        return

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
        tab_prediction.render(predictor, home_team_abbr, away_team_abbr, market_odd_home, market_odd_away)

    with tab2:
        tab_bankroll.render()

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
