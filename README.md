# 🏒 NHL Predictive Engine

> **Motor de Inteligência Artificial para análise e apostas profissionais na NHL.**
> Combina dados avançados do Natural Stat Trick (NST) com um classificador CatBoost para prever vencedores de partidas e calcular apostas de valor esperado positivo (EV+).

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Arquitetura do Sistema](#-arquitetura-do-sistema)
- [Modelo Preditivo](#-modelo-preditivo)
- [Interface (Streamlit)](#-interface-streamlit)
- [Instalação](#-instalação)
- [Comandos CLI](#-comandos-cli)
- [Fluxo de Trabalho Completo](#-fluxo-de-trabalho-completo)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Gestão de Banca](#-gestão-de-banca)
- [Qualidade de Código](#-qualidade-de-código)

---

## 🎯 Visão Geral

O **NHL Predictive Engine** é uma plataforma de análise quantitativa para o mercado de apostas **Moneyline da NHL**. O sistema:

1. **Coleta** estatísticas avançadas por temporada do [Natural Stat Trick](https://www.naturalstattrick.com/) (2015–2026)
2. **Treina** um classificador CatBoost com validação cruzada temporal e otimização hiperparamétrica
3. **Prediz** a probabilidade de vitória do time da casa (Mandante) para qualquer confronto
4. **Identifica** apostas com valor esperado positivo (EV+) comparando odds justas vs. odds de mercado
5. **Dimensiona** stakes via Critério de Kelly fracionado para gestão de risco profissional
6. **Registra e monitora** o histórico de apostas com métricas de performance em tempo real

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────┐
│                   app.py  (Streamlit)               │
│  ┌──────────────┬──────────────┬─────────────────┐  │
│  │  tab_pred.   │  tab_bank.   │   tab_model     │  │
│  │  Predição    │  Gestão de   │   Desempenho    │  │
│  │  de Partida  │  Banca       │   do Modelo     │  │
│  └──────┬───────┴──────┬───────┴────────┬────────┘  │
│         │              │                │            │
└─────────┼──────────────┼────────────────┼────────────┘
          │              │                │
   ┌──────▼──────┐ ┌─────▼─────┐  ┌──────▼──────┐
   │NHLPredictor │ │ bankroll  │  │  catboost   │
   │     V2      │ │  .py      │  │  _info/     │
   └──────┬──────┘ └─────┬─────┘  └─────────────┘
          │              │
   ┌──────▼──────┐ ┌─────▼─────┐
   │ features.py │ │bets_log   │
   │ train.py    │ │  .csv     │
   └──────┬──────┘ └───────────┘
          │
   ┌──────▼──────────────────────┐
   │  nst_team_stats.csv         │
   │  (Natural Stat Trick data)  │
   └──────┬──────────────────────┘
          │
   ┌──────▼──────┐   ┌─────────────┐
   │ scraper.py  │   │ extract.py  │
   │ (NST/Selen.)│   │ (NHL API)   │
   └─────────────┘   └─────────────┘
```

---

## 🤖 Modelo Preditivo

### Algoritmo

- **Classificador:** [CatBoost](https://catboost.ai/) — Gradient Boosting otimizado para dados categóricos e tabulares
- **Tarefa:** Classificação binária (`target = 1` → vitória do mandante)
- **Validação:** Walk-forward (temporal cross-validation) com as 3 últimas temporadas como dobras

### Features (29 variáveis)

| Categoria | Features | Descrição |
|---|---|---|
| **Aproveitamento** | `home_points_pct`, `away_points_pct` | % de pontos conquistados na tabela |
| **Posse de Disco** | `home_cf_pct`, `away_cf_pct` | Corsi For % (tentativas de finalização) |
| | `home_ff_pct`, `away_ff_pct` | Fenwick For % (excl. bloqueios) |
| | `home_sf_pct`, `away_sf_pct` | Shot For % |
| **Gols Esperados** | `home_xgf_pct`, `away_xgf_pct` | Expected Goals For % |
| | `home_gf_pct`, `away_gf_pct` | Goals For % real |
| **Alto Risco** | `home_hdcf_pct`, `away_hdcf_pct` | High Danger Chances For % |
| | `home_hdgf_pct`, `away_hdgf_pct` | High Danger Goals For % |
| **Eficiência** | `home_sh_pct`, `away_sh_pct` | Shooting % |
| | `home_sv_pct`, `away_sv_pct` | Save % |
| **Sorte/PDO** | `home_pdo`, `away_pdo` | PDO (Sh% + Sv%) — indicador de regressão |
| **Diferenciais** | `points_pct_diff`, `cf_pct_diff`, `xgf_pct_diff`, `hdcf_pct_diff`, `pdo_diff` | Diferença relativa mandante vs. visitante |
| **Categóricas** | `home_team`, `away_team` | Siglas dos times (tratadas nativamente pelo CatBoost) |

### Hiperparâmetros (Grid Search)

| Parâmetro | Valores Testados |
|---|---|
| `depth` | 4, 6 |
| `l2_leaf_reg` | 3, 10 |
| `iterations` | 1.000 (modelo final) |
| `learning_rate` | 0.03 |
| `loss_function` | Logloss |
| `eval_metric` | Accuracy |

### Métricas de Performance

| Métrica | Valor (temporada de teste) |
|---|---|
| **Acurácia** | ~60.8% |
| **Log Loss** | ~0.669 |
| **Brier Score** | calculado via `nhl-evaluate` |

> A acurácia de ~60% supera consistentemente o *break-even* implícito das odds de mercado (~52.4% para odds de -110), criando edge explorável via Kelly.

---

## 🖥️ Interface (Streamlit)

O aplicativo possui **3 abas**:

### 🎯 Predição de Partida
- Selecione mandante e visitante na sidebar
- Insira as odds de mercado
- Visualize: probabilidades do modelo, odds justas, badge EV+, stake sugerida via Kelly
- Registre apostas com resultado: `Pendente` | `Green` | `Red`

### 📊 Gestão de Banca
- **Painel "Criar Banca":** define tamanho da banca, porcentagem da unidade (1–25%) e fração de Kelly — persistido em JSON
- **Botão "Atualizar Dados NST":** executa o scraper em background e atualiza os dados
- **Métricas em tempo real:** ROI, Odd Média, Taxa de Acerto, P/L Acumulado (R$), Pendentes
- **Histórico editável:** edição de resultado inline, Odds com 2 casas decimais, exclusão dinâmica de linhas

### 🤖 Desempenho do Modelo
- Importância das features (Top 12, gráfico horizontal CatBoost)
- Curvas de aprendizado (Acurácia e Log Loss — Treino vs. Teste)
- Métricas globais de validação

### Sidebar
- Seleção de times (mandante / visitante)
- Odds de mercado
- Resumo da banca ativa (lê da config persistida)

---

## ⚙️ Instalação

### Pré-requisitos
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes recomendado)
- Google Chrome (para o scraper Selenium)

### Passos

```bash
# 1. Clonar o repositório
git clone https://github.com/Prog-LucasAlves/NHL_PYTHON.git
cd NHL_PYTHON

# 2. Criar ambiente virtual e instalar dependências
uv sync

# 3. Instalar hooks de qualidade de código
uv run pre-commit install

# 4. Rodar o aplicativo
uv run streamlit run app.py
```

> O app estará disponível em `http://localhost:8501`

---

## 🛠️ Comandos CLI

O projeto expõe 4 scripts de linha de comando via `pyproject.toml`:

```bash
# Buscar jogos históricos da NHL via API oficial
uv run nhl-extract

# Coletar estatísticas avançadas do Natural Stat Trick (requer Chrome)
uv run nhl-scrape-nst

# Treinar o modelo CatBoost com otimização hiperparamétrica
uv run nhl-train

# Avaliar desempenho do modelo (backtest financeiro + Kelly)
uv run nhl-evaluate
```

---

## 🔄 Fluxo de Trabalho Completo

```
1. nhl-extract          → data/nhl_games_all_seasons.csv
        ↓
2. nhl-scrape-nst       → data/nst_team_stats.csv
        ↓
3. nhl-train            → data/nhl_model.cbm
        ↓
4. nhl-evaluate         → Backtest e métricas no terminal
        ↓
5. streamlit run app.py → Interface de apostas em tempo real
```

> **No dia a dia** (temporada ativa): apenas rode **nhl-scrape-nst** (coleta incremental) e abra o app. O modelo usa sempre a temporada mais recente disponível.

---

## 📁 Estrutura do Projeto

```
NHL_PYTHON/
├── app.py                          # Ponto de entrada do Streamlit
├── pyproject.toml                  # Dependências e scripts CLI
│
├── src/nhl_engine/
│   ├── config.py                   # Paths globais e TEAM_MAPPING
│   │
│   ├── data/
│   │   ├── extract.py              # Fetch de jogos via NHL API
│   │   └── scraper.py              # Scraper NST (Selenium + BeautifulSoup)
│   │
│   ├── model/
│   │   ├── features.py             # Pipeline de features (NST merge + diffs)
│   │   ├── train.py                # Treinamento com walk-forward CV
│   │   └── predict.py              # NHLPredictorV2 — inferência em produção
│   │
│   ├── betting/
│   │   ├── bankroll.py             # log_bet(), load/save config de banca
│   │   └── evaluate.py             # Backtest financeiro + Kelly simulator
│   │
│   └── ui/
│       ├── app.py                  # (entry point alternativo)
│       ├── components.py           # HTML components reutilizáveis
│       ├── styles.py               # CSS customizado (tema dark)
│       ├── tab_prediction.py       # Aba de predição de partida
│       ├── tab_bankroll.py         # Aba de gestão de banca
│       └── tab_model.py            # Aba de desempenho do modelo
│
├── data/
│   ├── nhl_model.cbm               # Modelo treinado (CatBoost)
│   ├── nhl_games_all_seasons.csv   # Histórico de jogos NHL
│   └── nst_team_stats.csv          # Stats NST por temporada/time
│
├── logs/
│   ├── bets_log.csv                # Histórico de apostas registradas
│   └── bankroll_config.json        # Configuração de banca persistida
│
├── catboost_info/
│   └── catboost_training.json      # Curvas de treino (acurácia/logloss)
│
└── graphify-out/
    ├── graph.html                  # Grafo interativo de dependências
    ├── graph.json                  # Dados do grafo (80 nós, 89 arestas)
    └── GRAPH_REPORT.md             # Relatório de análise de arquitetura
```

---

## 💰 Gestão de Banca

### Critério de Kelly

A stake sugerida é calculada como:

```
f* = (p × odd - 1) / (odd - 1)
```

Onde `p` é a probabilidade estimada pelo modelo. O sistema suporta:

| Modo | Fração | Uso Recomendado |
|---|---|---|
| Kelly Completo | 100% | Teórico — muito agressivo |
| **Meio Kelly** | **50%** | **Padrão — equilibrado** |
| Quarto de Kelly | 25% | Conservador |
| Stake Fixa | — | Desativado (1.0 ud fixo) |

### Valor da Unidade

Configurável como **porcentagem da banca total** (1% a 25%). O padrão recomendado é **10%** (gestão de risco moderada). A configuração é persistida em `logs/bankroll_config.json`.

### Métricas de Performance

| Métrica | Descrição |
|---|---|
| **ROI (%)** | Retorno sobre o capital inicial |
| **Odd Média** | Média das cotações apostadas |
| **Taxa de Acerto (WR%)** | % de apostas vencedoras |
| **P/L Acumulado (R$)** | Lucro líquido em reais |
| **⏳ Pendentes** | Apostas aguardando resultado |

> Todas as métricas excluem apostas com resultado **Pendente** do cálculo.

---

## 🔍 Qualidade de Código

O projeto usa [`pre-commit`](https://pre-commit.com/) com os seguintes hooks:

| Hook | Função |
|---|---|
| `ruff` | Linting e correção automática (PEP 8, imports) |
| `ruff-format` | Formatação de código |
| `mypy` | Checagem estática de tipos |
| `detect-secrets` | Prevenção de vazamento de credenciais |
| `add-trailing-comma` | Padronização de vírgulas |
| `end-of-file-fixer` | Arquivos com newline final |
| `trailing-whitespace` | Remove espaços desnecessários |

```bash
# Rodar manualmente todos os hooks
uv run pre-commit run --all-files

# Atualizar hooks para versões mais recentes
uv run pre-commit autoupdate
```

---

## 📊 Arquitetura de Dependências

O grafo interativo de dependências do projeto está disponível em:

```
graphify-out/graph.html
```

Gerado com [graphifyy](https://github.com/graphify/graphifyy) — **80 nós · 89 arestas · 10 comunidades**.

| Comunidade | Módulo |
|---|---|
| Bankroll & Gestão de Banca | `betting/bankroll.py`, `ui/tab_bankroll.py` |
| Scraper NST | `data/scraper.py` |
| Modelo Preditivo | `model/predict.py`, `model/train.py` |
| UI de Predição | `ui/tab_prediction.py`, `ui/components.py` |
| Pipeline de Features | `model/features.py` |
| Avaliação & Backtest | `betting/evaluate.py` |
| App Principal | `app.py` |
| Fetch de Jogos | `data/extract.py` |

---

## 📄 Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.

---

<div align="center">
  <sub>Construído com CatBoost · Streamlit · Natural Stat Trick · NHL API</sub>
</div>
