import streamlit as st

from nhl_engine.betting.bankroll import load_bankroll_config
from nhl_engine.config import TEAM_MAPPING
from nhl_engine.model.predict import NHLPredictorV2
from nhl_engine.ui import tab_bankroll, tab_model, tab_prediction
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

    # Lê configurações de banca persistidas (definidas na aba Gestão de Banca)
    cfg = load_bankroll_config()
    initial_bankroll = float(cfg["bankroll"])
    unit_value = float(cfg["unit_value"])
    kelly_fraction = float(cfg["kelly_fraction"])

    tab1, tab2, tab3 = st.tabs(["🎯 Predição de Partida", "📊 Gestão de Banca", "🤖 Desempenho do Modelo"])

    # Sidebar — apenas seleção de times e odds de mercado
    with st.sidebar:
        st.image("https://assets.nhle.com/logos/nhl/svg/NHL_light.svg", width=100)
        st.header("Configurações")

        home_team_abbr = st.selectbox(
            "Time da Casa (Home)",
            teams,
            index=teams.index("BOS") if "BOS" in teams else 0,
            format_func=lambda x: TEAM_MAPPING.get(x, x),
        )
        away_team_abbr = st.selectbox(
            "Time de Fora (Away)",
            teams,
            index=teams.index("TOR") if "TOR" in teams else 1,
            format_func=lambda x: TEAM_MAPPING.get(x, x),
        )

        st.divider()
        st.markdown("#### Mercado de Apostas")
        market_odd_home = st.number_input(f"Odd na Casa ({home_team_abbr})", min_value=1.0, value=2.0, step=0.01)
        market_odd_away = st.number_input(f"Odd Visitante ({away_team_abbr})", min_value=1.0, value=2.0, step=0.01)

        st.divider()
        st.markdown("#### Banca Ativa")
        st.caption(f"**{initial_bankroll:.0f} uds** | R$ {unit_value:,.2f}/ud | Kelly {kelly_fraction:.0%}")
        st.caption("_Configure na aba 📊 Gestão de Banca_")

        st.divider()
        scraping = st.session_state.get("scraper_running", False)
        if st.button(
            "⏳ Coletando..." if scraping else "🔄 Atualizar Dados NST",
            use_container_width=True,
            disabled=scraping,
            help="Captura os dados mais recentes do Natural Stat Trick.",
            key="sidebar_scraper_btn",
        ):
            import threading

            from nhl_engine.ui.tab_bankroll import _run_scraper_background

            st.session_state["scraper_running"] = True
            st.session_state["scraper_log"] = "⏳ Iniciando coleta..."
            threading.Thread(target=_run_scraper_background, args=("scraper_log",), daemon=True).start()
            st.rerun()

        log = st.session_state.get("scraper_log", "")
        if log:
            st.caption(log[:120])

    with tab1:
        tab_prediction.render(
            predictor,
            home_team_abbr,
            away_team_abbr,
            market_odd_home,
            market_odd_away,
            initial_bankroll,
            unit_value,
            kelly_fraction,
        )

    with tab2:
        # render() agora retorna os valores atualizados caso o usuário salve nova banca
        initial_bankroll, unit_value, kelly_fraction = tab_bankroll.render()

    with tab3:
        tab_model.render(predictor)

    # Rodapé Técnico
    st.divider()
    with st.expander("ℹ️ Detalhes Técnicos do Modelo"):
        st.write("""
        Este modelo utiliza o algoritmo **CatBoost** treinado com estatísticas consolidadas por temporada do **Natural Stat Trick (NST)** abrangendo de 2015 a 2026.

        **Fatores estruturais considerados:**
        - **Aproveitamento (Points %):** Rendimento acumulado das equipes na tabela geral da temporada.
        - **Posse de Disco (Corsi % e Fenwick %):** Proporção de chutes e tentativas de finalização das equipes.
        - **Gols Esperados (xGF% e High Danger Chances):** Eficiência e periculosidade ofensiva/defensiva em zonas de alto risco.
        - **Sorte/Regressão à Média (PDO):** Soma das porcentagens de finalização (Sh%) e defesas (Sv%) para indicar anomalias de desempenho.
        - **Mapeamento de Diferenciais:** O classificador avalia a disparidade de performance relativa entre Mandante e Visitante para calcular as probabilidades exatas de vitória.
        """)


if __name__ == "__main__":
    main()
