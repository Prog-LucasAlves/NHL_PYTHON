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
    initial_bankroll: float = 100.0,
    unit_value: float = 100.0,
    kelly_fraction: float = 0.50,
):
    """Renderiza a tab de predição de partida com registro de aposta e calculadora de Kelly."""
    if home_team_abbr == away_team_abbr:
        st.warning("Selecione times diferentes para a predição.")
        return

    prob_home, prob_away = predictor.predict(home_team_abbr, away_team_abbr)
    fair_odd_home = 1 / prob_home
    fair_odd_away = 1 / prob_away

    # Cálculo do Critério de Kelly
    # Formula de Kelly: f* = (p * odd - 1) / (odd - 1)
    kelly_pct_home = 0.0
    if market_odd_home > fair_odd_home:
        kelly_pct_home = (prob_home * market_odd_home - 1) / (market_odd_home - 1)
        kelly_pct_home = max(0.0, kelly_pct_home) * kelly_fraction

    kelly_pct_away = 0.0
    if market_odd_away > fair_odd_away:
        kelly_pct_away = (prob_away * market_odd_away - 1) / (market_odd_away - 1)
        kelly_pct_away = max(0.0, kelly_pct_away) * kelly_fraction

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### {TEAM_MAPPING[away_team_abbr]} @ {TEAM_MAPPING[home_team_abbr]}")

        with st.container():
            st.markdown(
                prediction_card_html(home_team_abbr, away_team_abbr, prob_home, prob_away, fair_odd_home, fair_odd_away),
                unsafe_allow_html=True,
            )

        # Value Betting Badges
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

        # Seção de Recomendação e Gestão de Risco
        st.markdown("### 🛡️ Recomendação e Gestão de Risco")

        # Determinar Stake Sugerida em Unidades
        if kelly_fraction > 0.0:
            rec_home_uds = kelly_pct_home * initial_bankroll
            rec_away_uds = kelly_pct_away * initial_bankroll
        else:
            rec_home_uds = 1.0 if market_odd_home > fair_odd_home else 0.0
            rec_away_uds = 1.0 if market_odd_away > fair_odd_away else 0.0

        risk_col1, risk_col2 = st.columns(2)
        with risk_col1:
            if market_odd_home > fair_odd_home and rec_home_uds > 0:
                st.markdown(
                    f"""
                    <div style="background-color: #12151c; padding: 15px; border: 1px solid #232733; border-top: 3px solid #00ff88; border-radius: 3px;">
                        <span style="color: #888; font-size: 0.8rem; text-transform: uppercase;">Stake Sugerida ({home_team_abbr})</span>
                        <h2 style="color: #00ff88; margin: 5px 0 0 0; font-size: 1.8rem;">{rec_home_uds:.2f} <span style="font-size: 0.9rem; color: #fff;">uds</span></h2>
                        <p style="color: #888; margin: 2px 0 0 0; font-size: 0.85rem;">Equivale a <b>R$ {rec_home_uds * unit_value:.2f}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #12151c; padding: 15px; border: 1px solid #232733; opacity: 0.5; border-radius: 3px;">
                        <span style="color: #888; font-size: 0.8rem; text-transform: uppercase;">Stake Sugerida ({home_team_abbr})</span>
                        <h3 style="color: #ff4b4b; margin: 5px 0 0 0; font-size: 1.4rem;">Sem Entrada</h3>
                        <p style="color: #888; margin: 2px 0 0 0; font-size: 0.85rem;">Sem valor esperado positivo (EV-)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with risk_col2:
            if market_odd_away > fair_odd_away and rec_away_uds > 0:
                st.markdown(
                    f"""
                    <div style="background-color: #12151c; padding: 15px; border: 1px solid #232733; border-top: 3px solid #00bdff; border-radius: 3px;">
                        <span style="color: #888; font-size: 0.8rem; text-transform: uppercase;">Stake Sugerida ({away_team_abbr})</span>
                        <h2 style="color: #00bdff; margin: 5px 0 0 0; font-size: 1.8rem;">{rec_away_uds:.2f} <span style="font-size: 0.9rem; color: #fff;">uds</span></h2>
                        <p style="color: #888; margin: 2px 0 0 0; font-size: 0.85rem;">Equivale a <b>R$ {rec_away_uds * unit_value:.2f}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #12151c; padding: 15px; border: 1px solid #232733; opacity: 0.5; border-radius: 3px;">
                        <span style="color: #888; font-size: 0.8rem; text-transform: uppercase;">Stake Sugerida ({away_team_abbr})</span>
                        <h3 style="color: #ff4b4b; margin: 5px 0 0 0; font-size: 1.4rem;">Sem Entrada</h3>
                        <p style="color: #888; margin: 2px 0 0 0; font-size: 0.85rem;">Sem valor esperado positivo (EV-)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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

    reg_col1, reg_col2, reg_col3, reg_col4 = st.columns(4)

    with reg_col1:
        entry_type = st.radio("Entrada", [f"Mandante ({home_team_abbr})", f"Visitante ({away_team_abbr})"])
        entry_abbr = home_team_abbr if "Mandante" in entry_type else away_team_abbr
    with reg_col2:
        bet_odd = st.number_input("Odd da Entrada", min_value=1.0, value=market_odd_home if "Mandante" in entry_type else market_odd_away, step=0.01)
    with reg_col3:
        suggested_stake = rec_home_uds if "Mandante" in entry_type else rec_away_uds
        if suggested_stake <= 0.0:
            suggested_stake = 1.0
        bet_stake = st.number_input("Stake Usada (uds)", min_value=0.05, value=float(suggested_stake), step=0.05)
    with reg_col4:
        bet_result = st.selectbox("Resultado", ["Pendente", "Green", "Red"])

    if st.button("💾 Salvar Aposta", use_container_width=True):
        log_bet(
            pd.Timestamp.now().strftime("%Y-%m-%d"),
            home_team_abbr,
            away_team_abbr,
            entry_abbr,
            bet_odd,
            bet_result,
            bet_stake,
        )
        label = "⏳ Aposta pendente registrada!" if bet_result == "Pendente" else "✅ Aposta registrada com sucesso!"
        st.success(label)
        st.rerun()
