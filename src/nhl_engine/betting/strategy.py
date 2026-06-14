from dataclasses import dataclass

MIN_EDGE = 0.05
MAX_STAKE_FRACTION = 0.05


@dataclass(frozen=True)
class BetDecision:
    fair_odd: float
    market_odd: float
    edge: float
    qualifies: bool
    kelly_fraction: float
    stake: float


def fair_odd(win_probability: float, push_probability: float = 0.0) -> float:
    """Calcula a odd de equilíbrio, considerando devolução em caso de push."""
    if win_probability <= 0 or push_probability < 0 or win_probability + push_probability > 1:
        return float("inf")
    return (1 - push_probability) / win_probability


def evaluate_bet(
    win_probability: float,
    market_odd: float,
    bankroll: float,
    kelly_multiplier: float = 0.25,
    push_probability: float = 0.0,
    min_edge: float = MIN_EDGE,
    max_stake_fraction: float = MAX_STAKE_FRACTION,
) -> BetDecision:
    """Avalia preço e dimensiona stake apenas quando o edge mínimo é atingido."""
    price = fair_odd(win_probability, push_probability)
    if market_odd <= 1 or bankroll <= 0 or price == float("inf"):
        return BetDecision(price, market_odd, 0.0, False, 0.0, 0.0)

    edge = market_odd / price - 1
    qualifies = edge >= min_edge - 1e-12
    loss_probability = max(0.0, 1 - win_probability - push_probability)
    active_probability = win_probability + loss_probability
    profit_per_unit = market_odd - 1
    full_kelly = 0.0
    if active_probability > 0 and profit_per_unit > 0:
        full_kelly = (win_probability * profit_per_unit - loss_probability) / (profit_per_unit * active_probability)
    applied_kelly = max(0.0, full_kelly) * max(0.0, kelly_multiplier) if qualifies else 0.0
    applied_kelly = min(applied_kelly, max_stake_fraction)
    return BetDecision(price, market_odd, edge, qualifies, applied_kelly, bankroll * applied_kelly)


def settle_total_bet(side: str, line: float, total_goals: int) -> str:
    """Liquida uma aposta Over/Under, incluindo push em linhas inteiras."""
    if total_goals == line:
        return "Push"
    if side.lower() == "over":
        return "Green" if total_goals > line else "Red"
    if side.lower() == "under":
        return "Green" if total_goals < line else "Red"
    raise ValueError("side deve ser 'Over' ou 'Under'")
