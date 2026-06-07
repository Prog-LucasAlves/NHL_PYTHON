import plotly.express as px
import streamlit as st

from nhl_engine.betting.bankroll import load_history


def render():
    """Renderiza a tab de gestão de banca com histórico e métricas."""
    st.header("📊 Histórico e Performance")

    df_history = load_history()

    if df_history is None:
        st.info("Nenhuma aposta registrada ainda. Vá para a aba **🎯 Predição de Partida** para registrar suas apostas.")
        return

    # Métricas
    p_total = df_history["PL"].sum()
    win_rate = (df_history["Resultado"] == "Green").mean()
    total_bets = len(df_history)
    avg_odd = df_history["Odd"].mean()

    met1, met2, met3, met4 = st.columns(4)
    met1.metric("P/L Total", f"{p_total:+.2f} uds")
    met2.metric("Win Rate", f"{win_rate:.1%}")
    met3.metric("Total de Bets", total_bets)
    met4.metric("Odd Média", f"{avg_odd:.2f}")

    st.divider()

    # Gráfico de evolução
    df_history["Acumulado"] = df_history["PL"].cumsum()
    fig_evol = px.area(df_history, x=df_history.index, y="Acumulado", title="Curva de Lucro/Prejuízo")
    fig_evol.update_traces(fill="tozeroy", line_color="#00ff88", fillcolor="rgba(0,255,136,0.15)")
    fig_evol.update_layout(template="plotly_dark", xaxis_title="Apostas", yaxis_title="P/L Acumulado (uds)")
    st.plotly_chart(fig_evol, use_container_width=True)

    # Tabela de apostas
    st.subheader("📋 Apostas Registradas")
    st.dataframe(
        df_history.sort_index(ascending=False).style.applymap(
            lambda v: "color: #00ff88" if v == "Green" else ("color: #ff4b4b" if v == "Red" else ""),
            subset=["Resultado"],
        ),
        use_container_width=True,
    )
