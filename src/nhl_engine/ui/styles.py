CUSTOM_CSS = """
<style>
/* Fundo Geral */
.main {
    background-color: #0c0e12;
}

/* Painel de Métricas (stMetric) */
div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700;
    color: #ffffff;
}
div[data-testid="stMetric"] {
    background-color: #12151c;
    padding: 15px;
    border-radius: 3px;
    border: 1px solid #232733;
    box-shadow: none !important;
}

/* Card de Predição VS */
.prediction-card {
    background: #12151c;
    padding: 24px;
    border-radius: 3px;
    border: 1px solid #232733;
    border-left: 4px solid #00ff88;
    margin-bottom: 20px;
}

/* Selo de Valor Esperado (EV+) */
.value-badge {
    background-color: #00ff88;
    color: #0a0e14;
    padding: 4px 10px;
    border-radius: 2px;
    font-weight: 700;
    font-size: 0.85rem;
    display: inline-block;
    margin-top: 8px;
    letter-spacing: 0.05em;
}

/* Título Geral */
.header-text {
    font-family: 'Outfit', 'Inter', sans-serif;
    font-weight: 900;
    background: -webkit-linear-gradient(#00ff88, #00bdff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    letter-spacing: -0.02em;
    margin-bottom: 5px;
}

/* Seção de Registro de Aposta */
.bet-register-header {
    background: #12151c;
    padding: 12px 20px;
    border-radius: 3px;
    border: 1px solid #232733;
    border-left: 4px solid #00bdff;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Botões do Streamlit */
div.stButton > button {
    border-radius: 3px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
    border: 1px solid #232733 !important;
    background-color: #12151c !important;
    color: #ffffff !important;
    transition: all 0.2s ease-in-out;
}
div.stButton > button:hover {
    border-color: #00ff88 !important;
    color: #00ff88 !important;
}

/* Sidebar Customizada */
section[data-testid="stSidebar"] {
    background-color: #0a0c10 !important;
    border-right: 1px solid #1a1e26;
}
</style>
"""
