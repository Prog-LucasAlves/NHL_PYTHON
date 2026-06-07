# Graph Report - .  (2026-06-07)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 80 nodes · 89 edges · 10 communities detected
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 13 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]

## God Nodes (most connected - your core abstractions)
1. `NHLPredictorV2` - 8 edges
2. `evaluate_betting_performance()` - 6 edges
3. `build_features()` - 6 edges
4. `train_model()` - 6 edges
5. `scrape_team_stats()` - 5 edges
6. `scrape_all_seasons()` - 5 edges
7. `_render_bankroll_setup()` - 5 edges
8. `render()` - 5 edges
9. `load_bankroll_config()` - 4 edges
10. `render()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `NHLPredictorV2` --uses--> `Renderiza a tab de predição de partida com registro de aposta e calculadora de K`  [INFERRED]
  src\nhl_engine\model\predict.py → src\nhl_engine\ui\tab_prediction.py
- `get_predictor()` --calls--> `NHLPredictorV2`  [INFERRED]
  app.py → src\nhl_engine\model\predict.py
- `load_bankroll_config()` --calls--> `_render_bankroll_setup()`  [INFERRED]
  src\nhl_engine\betting\bankroll.py → src\nhl_engine\ui\tab_bankroll.py
- `log_bet()` --calls--> `render()`  [INFERRED]
  src\nhl_engine\betting\bankroll.py → src\nhl_engine\ui\tab_prediction.py
- `evaluate_betting_performance()` --calls--> `build_features()`  [INFERRED]
  src\nhl_engine\betting\evaluate.py → src\nhl_engine\model\features.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.16
Nodes (12): load_history(), log_bet(), Persiste configurações de banca no arquivo JSON., Registra uma aposta no CSV de histórico., Carrega o histórico de apostas. Retorna None se não existir., save_bankroll_config(), Renderiza a tab de gestão de banca. Retorna (bankroll, unit_value, kelly_fractio, Executa o scraper em thread separada e grava resultado no session_state. (+4 more)

### Community 1 - "Community 1"
Cohesion: 0.24
Nodes (11): _build_url(), _create_driver(), main(), _parse_table(), Scrapes team stats de uma temporada específica do NST., Coleta stats de todas as temporadas entre start_year e end_year de forma increme, Constrói a URL do NST para uma temporada específica., Cria uma instância do Chrome com undetected-chromedriver. (+3 more)

### Community 2 - "Community 2"
Cohesion: 0.22
Nodes (5): NHLPredictorV2, Carrega o modelo treinado e calcula estados dos times para predição via estatíst, Carrega o modelo e calcula o estado atual de todos os times usando a temporada m, Renderiza a aba de desempenho e estatísticas do modelo de IA., render()

### Community 3 - "Community 3"
Cohesion: 0.29
Nodes (6): bet_register_header_html(), prediction_card_html(), Header estilizado da seção de registro de aposta., Gera o HTML do card de predição VS., Renderiza a tab de predição de partida com registro de aposta e calculadora de K, render()

### Community 4 - "Community 4"
Cohesion: 0.38
Nodes (6): build_features(), load_and_preprocess(), merge_nst_stats(), Pipeline completo de features: carregar jogos e mesclar estatísticas do NST., Carrega o CSV dos jogos e cria a coluna target., Une as estatísticas avançadas do NST para os times de casa e visitante por tempo

### Community 5 - "Community 5"
Cohesion: 0.47
Nodes (5): brier_score_loss(), evaluate_betting_performance(), main(), Calcula o Brier Score Loss para calibração de probabilidade., Avalia o desempenho do modelo em produção usando Backtest financeiro e Kelly.

### Community 6 - "Community 6"
Cohesion: 0.47
Nodes (5): brier_score_loss(), main(), Calcula o Brier Score Loss para calibração de probabilidade., Treina o modelo com otimização hiperparamétrica L2 e validação cruzada temporal., train_model()

### Community 7 - "Community 7"
Cohesion: 0.5
Nodes (4): load_bankroll_config(), Carrega configurações de banca do arquivo JSON. Retorna defaults se não existir., get_predictor(), main()

### Community 8 - "Community 8"
Cohesion: 0.67
Nodes (3): fetch_all_games(), main(), Busca jogos da temporada regular via API oficial da NHL.

### Community 9 - "Community 9"
Cohesion: 1.0
Nodes (1): NHL Predictive Engine — AI-powered NHL betting analysis.

## Knowledge Gaps
- **25 isolated node(s):** `NHL Predictive Engine — AI-powered NHL betting analysis.`, `Carrega configurações de banca do arquivo JSON. Retorna defaults se não existir.`, `Persiste configurações de banca no arquivo JSON.`, `Registra uma aposta no CSV de histórico.`, `Carrega o histórico de apostas. Retorna None se não existir.` (+20 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 9`** (2 nodes): `NHL Predictive Engine — AI-powered NHL betting analysis.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `NHLPredictorV2` connect `Community 2` to `Community 3`, `Community 7`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `render()` connect `Community 3` to `Community 0`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `load_bankroll_config()` connect `Community 7` to `Community 0`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `NHLPredictorV2` (e.g. with `get_predictor()` and `Renderiza a aba de desempenho e estatísticas do modelo de IA.`) actually correct?**
  _`NHLPredictorV2` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `evaluate_betting_performance()` (e.g. with `build_features()` and `train_model()`) actually correct?**
  _`evaluate_betting_performance()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `build_features()` (e.g. with `evaluate_betting_performance()` and `train_model()`) actually correct?**
  _`build_features()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `train_model()` (e.g. with `evaluate_betting_performance()` and `build_features()`) actually correct?**
  _`train_model()` has 2 INFERRED edges - model-reasoned connections that need verification._
