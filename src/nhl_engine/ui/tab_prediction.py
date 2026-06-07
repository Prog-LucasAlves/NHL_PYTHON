import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nhl_engine.betting.bankroll import log_bet
from nhl_engine.config import TEAM_MAPPING
from nhl_engine.model.predict import NHLPredictorV2
from nhl_engine.ui.components import bet_register_header_html, prediction_card_html


def render(
    predictor: NHLPredictorV2,
    home_team_abbr: str,
    away_team_abbr: str,
    market_odd_home: float,
    market_odd_away: float,
):
    """Renderiza a tab de predição de partida com registro de aposta."""
    if home_team_abbr == away_team_abbr:
        st.warning("Selecione times diferentes para a predição.")
        return

    prob_home, prob_away = predictor.predict(home_team_abbr, away_team_abbr)
    fair_odd_home = 1 / prob_home
    fair_odd_away = 1 / prob_away

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### {TEAM_MAPPING[away_team_abbr]} @ {TEAM_MAPPING[home_team_abbr]}")

        with st.container():
            st.markdown(
                prediction_card_html(home_team_abbr, away_team_abbr, prob_home, prob_away, fair_odd_home, fair_odd_away),
                unsafe_allow_html=True,
            )

        # Value Betting
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

        st.metric("Aproveitamento (Pts%)", f"{h_state['points_pct']:.3f}", delta=f"{h_state['points_pct'] - a_state['points_pct']:.3f} vs {away_team_abbr}")
        st.metric("Corsi For % (CF%)", f"{h_state['cf_pct']:.1f}%", delta=f"{h_state['cf_pct'] - a_state['cf_pct']:.1f}% vs {away_team_abbr}")
        st.metric("Expected Goals % (xGF%)", f"{h_state['xgf_pct']:.1f}%", delta=f"{h_state['xgf_pct'] - a_state['xgf_pct']:.1f}% vs {away_team_abbr}")

        fig = go.Figure()
        fig.add_trace(go.Bar(name=home_team_abbr, x=["Gols Pró", "Gols Contra"], y=[h_state["gf"], h_state["ga"]], marker_color="#00ff88"))
        fig.add_trace(go.Bar(name=away_team_abbr, x=["Gols Pró", "Gols Contra"], y=[a_state["gf"], a_state["ga"]], marker_color="#00bdff"))
        fig.update_layout(barmode="group", template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Registro de aposta
    st.divider()
    st.markdown(bet_register_header_html(), unsafe_allow_html=True)

    reg_col1, reg_col2, reg_col3 = st.columns(3)

    with reg_col1:
        entry_type = st.radio("Entrada", [f"Mandante ({home_team_abbr})", f"Visitante ({away_team_abbr})"])
        entry_abbr = home_team_abbr if "Mandante" in entry_type else away_team_abbr
    with reg_col2:
        bet_odd = st.number_input("Odd da Entrada", min_value=1.0, value=market_odd_home if "Mandante" in entry_type else market_odd_away, step=0.01)
    with reg_col3:
        bet_result = st.selectbox("Resultado", ["Pendente", "Green", "Red"])

    if st.button("💾 Salvar Aposta", use_container_width=True):
        if bet_result == "Pendente":
            st.error("Selecione o resultado (Green ou Red) para registrar.")
        else:
            log_bet(pd.Timestamp.now().strftime("%Y-%m-%d"), home_team_abbr, away_team_abbr, entry_abbr, bet_odd, bet_result)
            st.success("✅ Aposta registrada com sucesso!")
            st.rerun()
