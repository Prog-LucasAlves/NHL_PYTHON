# NHL Totals and Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested configurable Over/Under market, leakage-resistant scenario evaluation, and a combined regular-season/playoff data refresh to the NHL Streamlit app.

**Architecture:** Keep market math in a pure betting module, derive pregame rolling features from completed NHL games, and use independent CatBoost models for moneyline and expected goals. The Streamlit layer consumes those tested services and clearly labels historical results as theoretical scenarios because historical market odds are unavailable.

**Tech Stack:** Python 3.13, pandas, NumPy, CatBoost, scikit-learn, Streamlit, pytest

---

### Task 1: Test Harness and NHL Game Refresh

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/nhl_engine/data/extract.py`
- Modify: `src/nhl_engine/config.py`
- Create: `tests/data/test_extract.py`

- [ ] Add `pytest` to the development dependencies and create tests proving that extraction includes completed game types `2` and `3`, excludes preseason/incomplete games, deduplicates by `game_id`, preserves existing history, and writes `game_type`.
- [ ] Run `uv run pytest tests/data/test_extract.py -v` and verify the tests fail because refresh helpers do not exist.
- [ ] Implement pure response parsing plus `refresh_games_file(...)`, using atomic replacement and refusing to overwrite history with an empty fetch.
- [ ] Run `uv run pytest tests/data/test_extract.py -v` and verify all extraction tests pass.

### Task 2: Shared Value, Kelly, and Bet-Log Math

**Files:**
- Create: `src/nhl_engine/betting/strategy.py`
- Modify: `src/nhl_engine/betting/bankroll.py`
- Create: `tests/betting/test_strategy.py`
- Create: `tests/betting/test_bankroll.py`

- [ ] Write tests for fair odds, the exact 5% edge threshold, capped fractional Kelly, invalid odds, and three-way Over/Under outcomes with pushes.
- [ ] Write tests proving old moneyline log rows remain readable and new total bets persist `Mercado` and `Linha`.
- [ ] Run the focused tests and verify they fail because the strategy API and extended log fields are missing.
- [ ] Implement a `BetDecision` value object and pure decision functions, then extend `log_bet` compatibly.
- [ ] Run the focused tests and verify they pass.

### Task 3: Pregame Rolling Features and Totals Model

**Files:**
- Create: `src/nhl_engine/model/pregame.py`
- Create: `src/nhl_engine/model/totals.py`
- Modify: `src/nhl_engine/config.py`
- Create: `tests/model/test_pregame.py`
- Create: `tests/model/test_totals.py`

- [ ] Write tests proving every rolling feature is shifted and uses only games earlier than the target game.
- [ ] Write tests for Over/Under/push probabilities on half and whole lines and for probability coherence.
- [ ] Run the focused tests and verify they fail because the modules do not exist.
- [ ] Implement team-centric shifted rolling features, matchup features, a Poisson total distribution, and `NHLTotalsPredictor`.
- [ ] Run the focused tests and verify they pass.

### Task 4: Train Models and Leakage-Resistant Scenario Evaluation

**Files:**
- Modify: `src/nhl_engine/model/train.py`
- Rewrite: `src/nhl_engine/betting/evaluate.py`
- Create: `tests/betting/test_evaluate.py`

- [ ] Write tests proving walk-forward splits use earlier seasons only and scenario reports include the theoretical-warning label plus market/game-type breakdowns.
- [ ] Run the evaluation tests and verify they fail against the existing fixed-odd backtest.
- [ ] Add time-aware moneyline and totals model training and scenario evaluation at the minimum qualifying odd, with quarter Kelly and a 5% stake cap.
- [ ] Remove the fixed `1.91` historical-odd claim and Windows-incompatible console symbols.
- [ ] Run evaluation tests and a CLI smoke test.

### Task 5: Combined Refresh and Streamlit Totals UI

**Files:**
- Modify: `app.py`
- Modify: `src/nhl_engine/ui/tab_bankroll.py`
- Modify: `src/nhl_engine/ui/tab_prediction.py`
- Create: `src/nhl_engine/data/refresh.py`
- Create: `tests/data/test_refresh.py`

- [ ] Write tests proving combined refresh invokes both NHL and NST collectors and reports partial failures.
- [ ] Run the refresh tests and verify they fail because the combined service does not exist.
- [ ] Implement combined refresh and connect the sidebar button to it.
- [ ] Add configurable line, Over odd, and Under odd inputs; display totals probabilities, fair odds, edge, capped Kelly, and total-bet registration.
- [ ] Run focused tests and start Streamlit for a smoke test.

### Task 6: Documentation, Real Data Refresh, and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `data/nhl_games_all_seasons.csv`

- [ ] Document regular-season/playoff collection, totals inputs, the 5% entry threshold, stake cap, and the theoretical nature of scenario results.
- [ ] Run the real NHL game refresh and verify recent playoff games are present with `game_type == 3`.
- [ ] Run `uv run pytest -v`.
- [ ] Run `uv run pre-commit run --all-files`.
- [ ] Run `uv run nhl-evaluate` and inspect both market reports.
- [ ] Review `git diff --check`, preserve the user's pre-existing `data/nst_team_stats.csv` change, and summarize residual limitations.
