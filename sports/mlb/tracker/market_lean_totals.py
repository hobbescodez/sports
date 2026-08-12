"""
Forward-only paper test of a second, distinct totals rule from
totals_threshold.py's "model total >= market + PICK_THRESHOLD" rule -
this one derives its test line purely from the market's own real-money
over/under lean (Kalshi), never from any of this project's own model
projections. Paper/analysis only, no connection to real order placement
anywhere - same guarantee as every other tracker/ module (see
log_predictions.py's own module docstring).

Rule: for every game with a determined Kalshi over/under lean,

    test_line = market_total - 2   if Kalshi leans OVER
    test_line = market_total - 4   if Kalshi leans UNDER

and the rule always bets OVER against that adjusted line, never under.
"market_total" here is the same book-line field every other totals
metric in this project already reads (BPP falling back to DRatings -
see totals_threshold.py's module docstring for that same field) -
deliberately not Kalshi's own total_line, which is a second, separate
number. The lean and the line intentionally come from two different
real inputs: Kalshi supplies the real-money direction, the book line
supplies the number being adjusted.

The -2/-4 adjustment is a fixed constant chosen by hand, not fit to any
backtest - so unlike PICK_THRESHOLD (which WAS selected from a backtest
and therefore needs forward tracking to prove it generalizes), there's
no historical-fit concern to cordon off here. It's still tracked
forward-only from MARKET_LEAN_START_DATE rather than retroactively over
old games, for the same reason every other forward metric in this
project works that way: consistency, and no temptation to eyeball
"would this have worked historically" as if that were proof of
anything.

Because this rule's line is deliberately easier to clear than the
market's real number (2-4 runs below it), market_lean_baseline() tracks
a naive "always bet the over at the market's real, unadjusted total"
comparison over the exact same qualifying games (same Kalshi-lean
requirement, so it's apples-to-apples) - the only way to tell whether
the adjustment is adding real value or just picking an easier line that
would have hit regardless.
"""

from dataclasses import dataclass

MARKET_LEAN_START_DATE = "2026-08-11"  # date this rule shipped
OVER_LEAN_ADJUSTMENT = 2.0
UNDER_LEAN_ADJUSTMENT = 4.0


def _to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _adjusted_line(row):
    """(test_line, market_total) for a qualifying row - the rule's own
    adjusted line, and the raw market_total it was adjusted from. (None,
    None) if there's no determined Kalshi lean or no market_total to
    adjust - never a guessed/forced lean."""
    lean = row.get("kalshi_total_pick")
    market_total = _to_float(row.get("market_total"))
    if lean not in ("over", "under") or market_total is None:
        return None, None
    adjustment = OVER_LEAN_ADJUSTMENT if lean == "over" else UNDER_LEAN_ADJUSTMENT
    return market_total - adjustment, market_total


@dataclass
class RuleResult:
    qualifying: int
    pushes: int
    hits: int
    hit_rate: float | None


def _score(rows, results_by_id, start_date, line_fn):
    qualifying = 0
    pushes = 0
    hits = 0
    for row in rows:
        if (row.get("date") or "") < start_date:
            continue
        line, _market_total = line_fn(row)
        if line is None:
            continue
        result = results_by_id.get(row.get("game_id"))
        if not result:
            continue
        actual_total = _to_float(result.get("total_runs"))
        if actual_total is None:
            continue
        qualifying += 1
        if actual_total == line:
            pushes += 1
        elif actual_total > line:
            hits += 1
    decided = qualifying - pushes
    hit_rate = round(100 * hits / decided, 1) if decided else None
    return RuleResult(qualifying=qualifying, pushes=pushes, hits=hits, hit_rate=hit_rate)


def market_lean_forward(predictions_rows, results_rows, start_date=MARKET_LEAN_START_DATE):
    """The adjusted-line rule itself - see module docstring."""
    results_by_id = {r["game_id"]: r for r in results_rows if r.get("game_id")}
    return _score(predictions_rows, results_by_id, start_date, _adjusted_line)


def market_lean_baseline(predictions_rows, results_rows, start_date=MARKET_LEAN_START_DATE):
    """Naive baseline over the SAME qualifying games as market_lean_forward
    (same Kalshi-lean requirement gates both) - always bets over at the
    market's real, unadjusted total instead of the adjusted line. See
    module docstring for why this exists."""
    results_by_id = {r["game_id"]: r for r in results_rows if r.get("game_id")}

    def _unadjusted_line(row):
        _test_line, market_total = _adjusted_line(row)
        if _test_line is None:
            return None, None
        return market_total, market_total

    return _score(predictions_rows, results_by_id, start_date, _unadjusted_line)
