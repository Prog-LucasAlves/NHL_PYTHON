import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nhl_engine.betting.bankroll import log_bet
from nhl_engine.betting.strategy import BetDecision, evaluate_bet, is_validated_total_strategy
from nhl_engine.config import TEAM_MAPPING
from nhl_engine.model.predict import NHLPredictorV2
from nhl_engine.model.totals import NHLTotalsPredictor
from nhl_engine.ui.components import bet_register_header_html, prediction_card_html


def _stake(decision: BetDecision, kelly_fraction: float, allowed: bool = True) -> float:
    if not decision.qualifies or not allowed:
        return 0.0
    return decision.stake if kelly_fraction > 0 else 1.0


def _render_market_decision(label: str, probability: float, decision: BetDecision, stake: float, unit_value: float, allowed: bool = True) -> None:
    st.metric(label, f"{probability:.1%}", help=f"Odd justa: {decision.fair_odd:.2f}")
    st.caption(f"Odd justa **{decision.fair_odd:.2f}** | Edge **{decision.edge:+.2%}**")
    if not allowed:
        st.warning("Sem entrada: segmento não foi lucrativo de forma consistente no walk-forward.")
    elif decision.qualifies and stake > 0:
        st.success(f"Entrada liberada: {stake:.2f} uds (R$ {stake * unit_value:,.2f})")
    else:
        st.info("Sem entrada: exige pelo menos 5% acima da odd justa.")


def render(
    predictor: NHLPredictorV2,
    home_team_abbr: str,
    away_team_abbr: str,
    game_type: int,
    market_odd_home: float,
    market_odd_away: float,
    totals_predictor: NHLTotalsPredictor,
    total_line: float,
    market_odd_over: float,
    market_odd_under: float,
    initial_bankroll: float = 100.0,
    unit_value: float = 100.0,
    kelly_fraction: float = 0.50,
):
    """Renderiza moneyline e total de gols com uma regra única de valor e risco."""
    if home_team_abbr == away_team_abbr:
        st.warning("Selecione times diferentes para a predição.")
        return

    prob_home, prob_away = predictor.predict(home_team_abbr, away_team_abbr, game_type)
    home_decision = evaluate_bet(prob_home, market_odd_home, initial_bankroll, kelly_fraction)
    away_decision = evaluate_bet(prob_away, market_odd_away, initial_bankroll, kelly_fraction)
    totals = totals_predictor.predict(home_team_abbr, away_team_abbr, total_line, game_type)
    over_decision = evaluate_bet(
        totals.probabilities.over,
        market_odd_over,
        initial_bankroll,
        kelly_fraction,
        push_probability=totals.probabilities.push,
    )
    under_decision = evaluate_bet(
        totals.probabilities.under,
        market_odd_under,
        initial_bankroll,
        kelly_fraction,
        push_probability=totals.probabilities.push,
    )
    over_allowed = is_validated_total_strategy("Over", total_line)
    under_allowed = is_validated_total_strategy("Under", total_line)

    stakes = {
        "home": _stake(home_decision, kelly_fraction),
        "away": _stake(away_decision, kelly_fraction),
        "over": _stake(over_decision, kelly_fraction, over_allowed),
        "under": _stake(under_decision, kelly_fraction, under_allowed),
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### {TEAM_MAPPING[away_team_abbr]} @ {TEAM_MAPPING[home_team_abbr]}")
        st.caption("Entradas são liberadas somente com odd de mercado pelo menos 5% acima da odd justa.")
        st.markdown(
            prediction_card_html(
                home_team_abbr,
                away_team_abbr,
                prob_home,
                prob_away,
                home_decision.fair_odd,
                away_decision.fair_odd,
            ),
            unsafe_allow_html=True,
        )

        st.markdown("#### Moneyline")
        ml_home, ml_away = st.columns(2)
        with ml_home:
            _render_market_decision(home_team_abbr, prob_home, home_decision, stakes["home"], unit_value)
        with ml_away:
            _render_market_decision(away_team_abbr, prob_away, away_decision, stakes["away"], unit_value)

        st.divider()
        st.markdown(f"#### Total de Gols: linha {total_line:g}")
        total_a, total_b, total_c = st.columns(3)
        total_a.metric("Gols esperados", f"{totals.expected_total:.2f}")
        total_b.metric(home_team_abbr, f"{totals.expected_home:.2f}")
        total_c.metric(away_team_abbr, f"{totals.expected_away:.2f}")
        if totals.probabilities.push > 0:
            st.caption(f"Probabilidade estimada de push: {totals.probabilities.push:.1%}")

        over_col, under_col = st.columns(2)
        with over_col:
            _render_market_decision(f"Over {total_line:g}", totals.probabilities.over, over_decision, stakes["over"], unit_value, over_allowed)
        with under_col:
            _render_market_decision(f"Under {total_line:g}", totals.probabilities.under, under_decision, stakes["under"], unit_value, under_allowed)

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

    st.divider()
    st.markdown(bet_register_header_html(), unsafe_allow_html=True)
    choices = {
        f"Mandante ({home_team_abbr})": (home_team_abbr, market_odd_home, stakes["home"], "Moneyline", None),
        f"Visitante ({away_team_abbr})": (away_team_abbr, market_odd_away, stakes["away"], "Moneyline", None),
        f"Over {total_line:g}": (f"Over {total_line:g}", market_odd_over, stakes["over"], "Total", total_line),
        f"Under {total_line:g}": (f"Under {total_line:g}", market_odd_under, stakes["under"], "Total", total_line),
    }

    reg_col1, reg_col2, reg_col3, reg_col4 = st.columns(4)
    with reg_col1:
        entry_type = st.radio("Entrada", list(choices))
        entry_label, default_odd, suggested_stake, market, line = choices[entry_type]
    with reg_col2:
        bet_odd = st.number_input("Odd da Entrada", min_value=1.0, value=float(default_odd), step=0.01)
    with reg_col3:
        bet_stake = st.number_input("Stake Usada (uds)", min_value=0.05, value=float(suggested_stake or 1.0), step=0.05)
    with reg_col4:
        bet_result = st.selectbox("Resultado", ["Pendente", "Green", "Red", "Push"])

    if st.button("💾 Salvar Aposta", use_container_width=True):
        log_bet(
            pd.Timestamp.now().strftime("%Y-%m-%d"),
            home_team_abbr,
            away_team_abbr,
            entry_label,
            bet_odd,
            bet_result,
            bet_stake,
            market=market,
            line=line,
        )
        label = "⏳ Aposta pendente registrada!" if bet_result == "Pendente" else "✅ Aposta registrada com sucesso!"
        st.success(label)
        st.rerun()
