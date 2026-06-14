# NHL Totals and Profitable Strategy Design

## Objective

Extend the NHL betting application with a configurable total-goals market, repair
the game-data refresh flow, and replace misleading profitability checks with
time-aware strategy evaluation.

The application must identify value only when the offered market odd is at least
5% above the model's fair odd. Because historical market odds are unavailable,
historical profitability must be labeled as a scenario analysis rather than
proof that the required prices were available.

## Scope

The implementation covers:

- NHL game collection for regular-season and playoff games.
- A sidebar refresh action that updates both NHL games and NST team statistics.
- Leakage-resistant historical features based only on games completed before the
  predicted game.
- Configurable total-goals line and Over/Under market odds.
- Fair odds, edge, entry decision, and fractional Kelly stakes for moneyline and
  totals.
- Walk-forward evaluation with separate moneyline and total-goals reports.
- Automated tests for collection, strategy math, features, totals prediction,
  bankroll records, and evaluation.

It does not claim that a strategy is guaranteed to make money or that a
scenario backtest represents executable historical returns.

## Data Collection

### NHL Games

`nhl_engine.data.extract` will collect completed games from the official NHL API
for game types:

- `2`: regular season
- `3`: playoffs

The output file `data/nhl_games_all_seasons.csv` will retain the existing fields
and add `game_type`. Rows will be deduplicated by `game_id`, sorted by date, and
written atomically after a successful collection.

The collector will expose a callable refresh function that can merge newly
completed games into the existing file without discarding valid history.

### Sidebar Refresh

The current sidebar action only starts the NST scraper. It will be replaced by a
combined refresh operation:

1. Refresh completed NHL games.
2. Refresh NST team statistics.
3. Report the result of each source independently.
4. Clear Streamlit's cached predictor after successful data changes.

A failure in one source must not hide a success in the other source.

## Time-Aware Features

Historical evaluation must never use statistics calculated from games played
after the game being predicted.

For each team and game, rolling features will be shifted by one game and include:

- Games played before the prediction.
- Recent goals scored and allowed.
- Season-to-date goals scored and allowed.
- Recent win rate.
- Home/away context.
- Rest days.
- Regular-season or playoff context.

Rows without sufficient prior history will be excluded from model fitting and
evaluation. NST season aggregates may continue to support current-match display,
but they cannot be used to claim leakage-free historical performance unless a
dated snapshot exists.

## Prediction Models

### Moneyline

Moneyline evaluation will use a time-aware classifier trained only on games
before each validation period. It will return calibrated home and away win
probabilities.

### Total Goals

A total-goals model will estimate expected home and away goals from time-aware
features. The combined scoring distribution will produce:

- Expected total goals.
- Probability of Over for a configurable line.
- Probability of Under for a configurable line.
- Push probability for whole-number lines.
- Fair Over and Under odds.

The distribution implementation must support common half-lines and whole-number
lines. Whole-number lines must account for pushes when calculating expected
value and stake.

## Entry Rules and Risk

A shared strategy module will calculate decisions for moneyline and totals.

An entry is allowed only when:

```text
market_odd >= fair_odd * 1.05
```

The displayed edge is:

```text
edge = market_odd / fair_odd - 1
```

Stake sizing uses the configured fractional Kelly value and applies a maximum
stake cap. The default evaluation configuration will use quarter Kelly and a 5%
maximum bankroll exposure per bet. The UI may use the user's configured Kelly
fraction, but it must apply the same cap.

If there is no qualifying edge, the recommendation is explicitly "Sem Entrada".

## Scenario Backtest

Without historical market odds, the evaluator cannot calculate real historical
betting ROI. It will therefore run scenario analysis at explicit assumed prices.

For each prediction, the assumed executable odd is derived from the fair odd and
the configured 5% threshold. The report must be labeled:

> Cenario teorico: pressupoe que a odd minima exigida esteve disponivel.

Reports will be separated for moneyline and total goals and include:

- Number of qualifying bets.
- Win, loss, and push counts.
- Win rate excluding pushes.
- Total amount staked.
- Profit/loss.
- Yield on amount staked.
- Ending bankroll.
- Maximum drawdown.
- Results by season.
- Results split between regular season and playoffs.
- Statistical calibration metrics appropriate to each model.

The evaluator must avoid describing positive scenario yield as proven market
profitability.

## Application Interface

The sidebar will contain:

- Home and away moneyline odds.
- Configurable total-goals line.
- Over odd.
- Under odd.
- Combined data refresh action.

The prediction tab will show separate moneyline and total-goals sections. The
totals section will display expected goals, Over/Under probabilities, fair odds,
edge, entry status, and capped Kelly stake.

Bet registration will support:

- Home moneyline.
- Away moneyline.
- Over with line.
- Under with line.

The betting log will add market and line fields while remaining compatible with
existing records.

## Error Handling

- Network requests use finite timeouts and expose actionable failures.
- A failed refresh does not overwrite a valid data file with an empty result.
- Atomic writes prevent partial CSV files.
- Missing or invalid market odds result in no recommendation.
- Probabilities are bounded away from zero before fair-odd calculations.
- Console output remains compatible with Windows terminals.

## Testing and Acceptance

Automated tests must verify:

- NHL extraction includes game types `2` and `3`, excludes incomplete games,
  deduplicates rows, and preserves history during refresh.
- The combined refresh invokes both collectors and reports partial failures.
- Rolling features contain no future information.
- Fair odds, 5% edge gating, Kelly sizing, caps, and push handling are correct.
- Total-goals probabilities are coherent and sum correctly.
- Bet logs preserve existing moneyline rows and store totals rows.
- Walk-forward splits train only on dates before validation dates.
- Backtest reports separate market and game type and carry the scenario warning.

Final verification includes the full automated suite, static checks, model
evaluation, and a manual Streamlit smoke test.

## Success Criteria

The work is accepted when:

- The sidebar refresh updates `nhl_games_all_seasons.csv` with completed regular
  season and playoff games.
- The app presents configurable Over/Under analysis with fair odds and the 5%
  minimum-edge rule.
- Moneyline and totals use the same tested risk and entry rules.
- Historical evaluation is time-aware and clearly labeled as theoretical when
  historical market odds are absent.
- All automated and static checks pass.
