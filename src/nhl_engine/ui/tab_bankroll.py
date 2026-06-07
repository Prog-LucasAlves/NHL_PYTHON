import pandas as pd
import plotly.express as px
import streamlit as st

from nhl_engine.betting.bankroll import load_history
from nhl_engine.config import BETS_LOG_PATH


def _recalc_pl(row: pd.Series) -> float:
    """Recalcula o P/L em unidades com base no resultado atual da linha."""
    if row["Resultado"] == "Green":
        return round(float(row["Stake_orig"]) * (float(row["Odd"]) - 1), 2)
    if row["Resultado"] == "Red":
        return round(-float(row["Stake_orig"]), 2)
    return 0.0  # Pendente


def render(initial_bankroll: float = 100.0, unit_value: float = 100.0):
    """Renderiza a tab de gestão de banca com histórico editável e métricas em R$."""
    st.header("📊 Gestão de Banca & Performance Profissional")

    df_history = load_history()

    if df_history is None or df_history.empty:
        st.info("Nenhuma aposta registrada ainda. Vá para a aba **🎯 Predição de Partida** para registrar suas apostas.")
        return

    # Garante que colunas obrigatórias existam com valores seguros
    df_history.setdefault("Stake", 1.0)
    if "Resultado" not in df_history.columns:
        df_history["Resultado"] = "Pendente"
    df_history["Resultado"] = df_history["Resultado"].fillna("Pendente")

    # Métricas calculadas apenas sobre apostas resolvidas
    df_resolved = df_history[df_history["Resultado"].isin(["Green", "Red"])].copy()
    total_resolved = len(df_resolved)
    p_total_uds = df_resolved["PL"].sum() if total_resolved > 0 else 0.0
    win_rate = (df_resolved["Resultado"] == "Green").mean() if total_resolved > 0 else 0.0
    avg_odd = df_resolved["Odd"].mean() if total_resolved > 0 else 0.0
    roi_pct = (p_total_uds / initial_bankroll * 100) if initial_bankroll > 0 else 0.0
    pl_cash = p_total_uds * unit_value
    pending_count = len(df_history) - total_resolved

    met1, met2, met3, met4, met5 = st.columns(5)
    met1.metric("ROI (%)", f"{roi_pct:+.2f}%", help="Retorno sobre o investimento calculado apenas sobre apostas resolvidas.")
    met2.metric("Odd Média", f"{avg_odd:.2f}", help="Média das cotações de apostas resolvidas (Green + Red).")
    met3.metric("Taxa de Acerto (WR)", f"{win_rate:.1%}", help="Proporção de apostas vencedoras sobre o total de apostas resolvidas.")
    met4.metric("P/L Acumulado (R$)", f"R$ {pl_cash:,.2f}", delta=f"{p_total_uds:+.2f} uds", help="Lucro líquido acumulado em Reais com base no valor da unidade.")
    met5.metric("⏳ Pendentes", str(pending_count), help="Apostas ainda aguardando resultado.")

    st.divider()

    # Gráfico de evolução da banca (apenas apostas resolvidas)
    if total_resolved > 0:
        saldos = [initial_bankroll] + list(initial_bankroll + df_resolved["PL"].cumsum())
        df_chart = pd.DataFrame({"Apostas": range(len(saldos)), "Saldo (uds)": saldos})
        fig_evol = px.area(df_chart, x="Apostas", y="Saldo (uds)", title="Curva de Crescimento Patrimonial (Apostas Resolvidas)")
        fig_evol.update_traces(fill="tozeroy", line_color="#00ff88", fillcolor="rgba(0,255,136,0.08)")
        fig_evol.update_layout(template="plotly_dark", xaxis_title="Sequência de Entradas", yaxis_title="Banca (uds)", margin=dict(l=20, r=20, t=40, b=20), height=300)
        st.plotly_chart(fig_evol, use_container_width=True)

    # ─── Tabela editável ───────────────────────────────────────────────────────
    st.subheader("📋 Histórico Completo de Entradas")
    st.caption("✏️ Edite a coluna **Resultado** diretamente. Use o ícone 🗑️ para excluir entradas.")

    df_display = df_history.copy().sort_index(ascending=False).reset_index(drop=True)
    # Guarda Stake original (em unidades) antes de converter para exibição R$
    df_display["Stake_orig"] = df_display["Stake"]
    df_display["Stake"] = df_display["Stake"] * unit_value
    df_display["PL"] = df_display["PL"] * unit_value

    cols_ordered = ["Data", "Mandante", "Visitante", "Entrada", "Odd", "Resultado", "Stake", "PL"]
    df_view = df_display[[c for c in cols_ordered if c in df_display.columns]].copy()

    edited_df = st.data_editor(
        df_view,
        column_config={
            "Resultado": st.column_config.SelectboxColumn(
                "Resultado",
                options=["Pendente", "Green", "Red"],
                required=True,
            ),
            "Odd": st.column_config.NumberColumn(
                "Odd",
                format="%.2f",
                disabled=True,
            ),
            "Stake": st.column_config.NumberColumn(
                "Stake",
                format="R$ %.2f",
                disabled=True,
            ),
            "PL": st.column_config.NumberColumn(
                "P/L",
                format="R$ %.2f",
                disabled=True,
            ),
            "Data": st.column_config.Column(disabled=True),
            "Mandante": st.column_config.Column(disabled=True),
            "Visitante": st.column_config.Column(disabled=True),
            "Entrada": st.column_config.Column(disabled=True),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="bets_editor",
    )

    # ─── Detecta alterações e persiste ────────────────────────────────────────
    rows_deleted = len(df_view) != len(edited_df)
    resultado_changed = not rows_deleted and not df_view["Resultado"].reset_index(drop=True).equals(edited_df["Resultado"].reset_index(drop=True))

    if rows_deleted or resultado_changed:
        # Reconstrói o DataFrame em unidades originais para salvar
        # Índice na df_display referencia a mesma ordem de df_history (invertida)
        df_history_sorted = df_history.copy().sort_index(ascending=False).reset_index(drop=True)

        if rows_deleted:
            # Mantém apenas as linhas que ainda existem no editor
            df_history_sorted = df_history_sorted.iloc[edited_df.index]
        else:
            # Atualiza resultado e recalcula PL em unidades
            df_history_sorted["Resultado"] = edited_df["Resultado"].values
            df_history_sorted["Stake_orig"] = df_history_sorted["Stake"]
            df_history_sorted["Odd_col"] = df_history_sorted["Odd"]
            df_history_sorted["PL"] = df_history_sorted.apply(
                lambda r: round(r["Stake"] * (r["Odd"] - 1), 2) if r["Resultado"] == "Green" else (-round(r["Stake"], 2) if r["Resultado"] == "Red" else 0.0),
                axis=1,
            )

        # Remove coluna auxiliar caso exista
        df_to_save = df_history_sorted.drop(columns=["Stake_orig", "Odd_col"], errors="ignore")
        # Restaura a ordem original (mais antigo primeiro)
        df_to_save = df_to_save.sort_index(ascending=False).reset_index(drop=True)
        df_to_save.to_csv(BETS_LOG_PATH, index=False)

        if rows_deleted:
            st.success("🗑️ Entrada removida com sucesso!")
        else:
            st.success("✅ Resultado atualizado! P/L recalculado automaticamente.")
        st.rerun()
