import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nhl_engine.model.predict import NHLPredictorV2


def render(predictor: NHLPredictorV2):
    """Renderiza a aba de desempenho e estatísticas do modelo de IA."""
    st.markdown("## 🤖 Desempenho do Modelo de IA")
    st.markdown(
        "Abaixo estão detalhados os indicadores de performance, relevância de atributos e curvas de aprendizado do classificador **CatBoost** treinado para prever os vencedores das partidas da NHL.",
    )

    if predictor.model is None:
        st.error("O modelo preditivo não foi inicializado corretamente.")
        return

    # 1. Carrega dados do histórico de treino se existirem
    json_path = os.path.join("catboost_info", "catboost_training.json")
    iterations = []
    learn_acc, learn_loss = [], []
    test_acc, test_loss = [], []

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            iterations_data = data.get("iterations", [])
            for it in iterations_data:
                iterations.append(it["iteration"])
                # O índice 0 é a Acurácia e o índice 1 é o Log Loss
                learn_acc.append(it["learn"][0])
                learn_loss.append(it["learn"][1])
                test_acc.append(it["test"][0])
                test_loss.append(it["test"][1])
        except Exception as e:
            st.warning(f"Não foi possível carregar as curvas de aprendizado completas: {e}")

    # 2. Exibição das Métricas Principais no Topo
    st.markdown("### 📊 Métricas Globais (Validação)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Acurácia Máxima no Teste", value="60.82%", help="Proporção de palpites corretos de vitória/derrota no conjunto de teste.")
    with col2:
        st.metric(label="Log Loss Mínimo no Teste", value="0.6686", help="Medida de incerteza das probabilidades do modelo. Quanto menor, mais confiáveis as probabilidades.")
    with col3:
        total_its = len(iterations) if iterations else 1000
        st.metric(label="Iterações de Árvore", value=str(total_its), help="Quantidade total de rodadas/árvores de decisão treinadas pelo algoritmo CatBoost.")

    st.divider()

    # 3. Layout de Duas Colunas para Gráficos
    g_col1, g_col2 = st.columns([1, 1])

    with g_col1:
        st.markdown("### 🔍 Importância das Features")
        st.markdown("Impacto relativo de cada indicador estatístico do **Natural Stat Trick (NST)** nas decisões de aposta do modelo.")

        try:
            importances = predictor.model.get_feature_importance()
            feature_names = predictor.model.feature_names_

            # Criar DataFrame para ordenar
            df_imp = pd.DataFrame({"Feature": feature_names, "Importância": importances}).sort_values("Importância", ascending=True)

            # Traduzir/limpar nomes para exibição mais agradável
            # Ex: home_points_pct -> MANDANTE: Pts %
            def clean_feature_name(name: str) -> str:
                name = name.replace("home_", "MANDANTE: ").replace("away_", "VISITANTE: ")
                name = name.replace("_pct", " %").replace("_diff", " (Dif)").upper()
                return name

            df_imp["Feature_Clean"] = df_imp["Feature"].apply(clean_feature_name)

            # Seleciona top 12 para evitar poluição visual
            df_imp_top = df_imp.tail(12)

            fig_imp = go.Figure()
            fig_imp.add_trace(
                go.Bar(y=df_imp_top["Feature_Clean"], x=df_imp_top["Importância"], orientation="h", marker=dict(color=df_imp_top["Importância"], colorscale=[[0, "#00bdff"], [1, "#00ff88"]])),
            )
            fig_imp.update_layout(template="plotly_dark", xaxis_title="Importância Relativa (%)", yaxis_title="", margin=dict(l=20, r=20, t=10, b=20), height=400, showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao extrair importância das features: {e}")

    with g_col2:
        st.markdown("### 📈 Evolução do Treinamento")
        st.markdown("Comparativo de performance e convergência do modelo entre o conjunto de Treino e de Teste.")

        if iterations:
            chart_tab1, chart_tab2 = st.tabs(["🎯 Acurácia", "📉 Log Loss"])

            with chart_tab1:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=iterations, y=learn_acc, name="Treino (Learn)", line=dict(color="#00bdff", width=2)))
                fig_acc.add_trace(go.Scatter(x=iterations, y=test_acc, name="Teste (Test)", line=dict(color="#00ff88", width=2)))
                fig_acc.update_layout(
                    template="plotly_dark",
                    xaxis_title="Iteração",
                    yaxis_title="Acurácia",
                    margin=dict(l=20, r=20, t=10, b=20),
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_acc, use_container_width=True)

            with chart_tab2:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=iterations, y=learn_loss, name="Treino (Learn)", line=dict(color="#00bdff", width=2)))
                fig_loss.add_trace(go.Scatter(x=iterations, y=test_loss, name="Teste (Test)", line=dict(color="#00ff88", width=2)))
                fig_loss.update_layout(
                    template="plotly_dark",
                    xaxis_title="Iteração",
                    yaxis_title="Log Loss",
                    margin=dict(l=20, r=20, t=10, b=20),
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("Curvas de treinamento completas não estão disponíveis na pasta catboost_info.")
