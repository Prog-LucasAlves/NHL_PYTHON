import subprocess
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

from nhl_engine.betting.bankroll import (
    load_bankroll_config,
    load_history,
    save_bankroll_config,
)
from nhl_engine.config import BETS_LOG_PATH

# ─── Scraper helpers ──────────────────────────────────────────────────────────


def _run_scraper_background(log_key: str) -> None:
    """Executa o scraper em thread separada e grava resultado no session_state."""
    st.session_state[log_key] = "⏳ Coletando dados do Natural Stat Trick..."
    try:
        result = subprocess.run(
            [sys.executable, "-m", "nhl_engine.data.scraper"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = result.stdout or result.stderr or "(sem saída)"
        if result.returncode == 0:
            st.session_state[log_key] = f"✅ Dados atualizados!\n\n```\n{output[-1500:]}\n```"
        else:
            st.session_state[log_key] = f"❌ Erro ao coletar dados:\n\n```\n{output[-1500:]}\n```"
    except subprocess.TimeoutExpired:
        st.session_state[log_key] = "⚠️ Timeout: o scraper demorou mais de 5 min."
    except Exception as exc:
        st.session_state[log_key] = f"❌ Exceção: {exc}"
    st.session_state["scraper_running"] = False


# ─── Painel de configuração de banca ─────────────────────────────────────────


def _render_bankroll_setup() -> tuple[float, float, float]:
    """Painel de criação/edição da banca. Retorna (bankroll, unit_value, kelly_fraction)."""
    cfg = load_bankroll_config()

    with st.expander("⚙️ Configurar Banca", expanded=not st.session_state.get("bankroll_set", False)):
        st.markdown("#### 🏦 Criar / Editar Banca")

        col_a, col_b = st.columns(2)
        with col_a:
            bankroll = st.number_input(
                "Tamanho da Banca (uds)",
                min_value=10.0,
                value=float(cfg["bankroll"]),
                step=10.0,
                help="Total da banca expresso em Unidades de Stake.",
                key="setup_bankroll",
            )

            saved_pct = float(cfg.get("unit_pct", 10.0))
            unit_pct = st.slider(
                "Valor da Unidade (% da banca)",
                min_value=1.0,
                max_value=25.0,
                value=saved_pct,
                step=0.5,
                format="%.1f%%",
                help="Define quanto 1 Unidade representa em relação ao total da banca.",
                key="setup_unit_pct",
            )
            unit_value = round(bankroll * unit_pct / 100, 2)
            st.info(f"💡 **{unit_pct:.1f}%** da banca = **R$ {unit_value:,.2f}** por unidade")

        with col_b:
            kelly_fraction = st.selectbox(
                "Fração de Kelly",
                [0.25, 0.50, 1.0, 0.0],
                index=[0.25, 0.50, 1.0, 0.0].index(float(cfg.get("kelly_fraction", 0.50))),
                format_func=lambda x: {
                    1.0: "Kelly Completo (100%)",
                    0.50: "Meio Kelly (50%)",
                    0.25: "Um Quarto de Kelly (25%)",
                    0.0: "Desativado (Stake Fixa 1.0)",
                }.get(x, str(x)),
                key="setup_kelly",
            )

        if st.button("💾 Salvar Banca", use_container_width=True, type="primary"):
            save_bankroll_config(bankroll, unit_value, kelly_fraction, unit_pct)
            st.session_state["bankroll_set"] = True
            st.success(f"✅ Banca salva: **{bankroll:.0f} uds** | Unidade: **{unit_pct:.1f}%** → **R$ {unit_value:,.2f}** | Kelly: **{kelly_fraction:.0%}**")
            st.rerun()

    # Lê a config salva mais recente (pode ter acabado de salvar acima)
    cfg = load_bankroll_config()
    return float(cfg["bankroll"]), float(cfg["unit_value"]), float(cfg["kelly_fraction"])


# ─── Tab principal ────────────────────────────────────────────────────────────


def render() -> tuple[float, float, float]:
    """Renderiza a tab de gestão de banca. Retorna (bankroll, unit_value, kelly_fraction)."""
    st.header("📊 Gestão de Banca & Performance Profissional")

    # 1. Configuração de banca (persistent)
    initial_bankroll, unit_value, kelly_fraction = _render_bankroll_setup()

    st.divider()

    # Histórico de apostas
    df_history = load_history()

    if df_history is None or df_history.empty:
        st.info("Nenhuma aposta registrada ainda. Vá para a aba **🎯 Predição de Partida** para registrar suas apostas.")
        return initial_bankroll, unit_value, kelly_fraction

    if "Stake" not in df_history.columns:
        df_history["Stake"] = 1.0
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
    met1.metric("ROI (%)", f"{roi_pct:+.2f}%", help="Retorno sobre o investimento (apostas resolvidas).")
    met2.metric("Odd Média", f"{avg_odd:.2f}", help="Média das cotações de apostas resolvidas.")
    met3.metric("Taxa de Acerto (WR)", f"{win_rate:.1%}", help="Proporção de apostas vencedoras.")
    met4.metric("P/L Acumulado (R$)", f"R$ {pl_cash:,.2f}", delta=f"{p_total_uds:+.2f} uds")
    met5.metric("⏳ Pendentes", str(pending_count), help="Apostas aguardando resultado.")

    st.divider()

    # Gráfico de evolução
    if total_resolved > 0:
        saldos = [initial_bankroll] + list(initial_bankroll + df_resolved["PL"].cumsum())
        df_chart = pd.DataFrame({"Apostas": range(len(saldos)), "Saldo (uds)": saldos})
        fig_evol = px.area(df_chart, x="Apostas", y="Saldo (uds)", title="Curva de Crescimento Patrimonial")
        fig_evol.update_traces(fill="tozeroy", line_color="#00ff88", fillcolor="rgba(0,255,136,0.08)")
        fig_evol.update_layout(
            template="plotly_dark",
            xaxis_title="Sequência de Entradas",
            yaxis_title="Banca (uds)",
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
        )
        st.plotly_chart(fig_evol, use_container_width=True)

    # ─── Tabela editável ──────────────────────────────────────────────────────
    st.subheader("📋 Histórico Completo de Entradas")
    st.caption("✏️ Edite a coluna **Resultado** diretamente. Use o ícone 🗑️ para excluir entradas.")

    df_display = df_history.copy().sort_index(ascending=False).reset_index(drop=True)
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
            "Odd": st.column_config.NumberColumn("Odd", format="%.2f", disabled=True),
            "Stake": st.column_config.NumberColumn("Stake", format="R$ %.2f", disabled=True),
            "PL": st.column_config.NumberColumn("P/L", format="R$ %.2f", disabled=True),
            "Data": st.column_config.Column(disabled=True),
            "Mandante": st.column_config.Column(disabled=True),
            "Visitante": st.column_config.Column(disabled=True),
            "Entrada": st.column_config.Column(disabled=True),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="bets_editor",
    )

    # Detecta alterações e persiste
    rows_deleted = len(df_view) != len(edited_df)
    resultado_changed = not rows_deleted and not df_view["Resultado"].reset_index(drop=True).equals(edited_df["Resultado"].reset_index(drop=True))

    if rows_deleted or resultado_changed:
        df_hist_sorted = df_history.copy().sort_index(ascending=False).reset_index(drop=True)
        if rows_deleted:
            df_hist_sorted = df_hist_sorted.iloc[edited_df.index]
        else:
            df_hist_sorted["Resultado"] = edited_df["Resultado"].values
            df_hist_sorted["PL"] = df_hist_sorted.apply(
                lambda r: round(r["Stake"] * (r["Odd"] - 1), 2) if r["Resultado"] == "Green" else (-round(r["Stake"], 2) if r["Resultado"] == "Red" else 0.0),
                axis=1,
            )
        df_to_save = df_hist_sorted.sort_index(ascending=False).reset_index(drop=True)
        df_to_save.to_csv(BETS_LOG_PATH, index=False)
        msg = "🗑️ Entrada removida com sucesso!" if rows_deleted else "✅ Resultado atualizado! P/L recalculado."
        st.success(msg)
        st.rerun()

    return initial_bankroll, unit_value, kelly_fraction
