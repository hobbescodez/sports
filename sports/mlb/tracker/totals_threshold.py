"""
Backtests / forward-tests the "over-only, model total >= market + threshold"
totals rule against this project's own logged history (predictions_log.csv
+ results_log.csv) - a paper/analysis exercise only, with no connection to
order placement anywhere (see log_predictions.py's own module docstring
for that same no-trading guarantee, which this module inherits by only
ever reading the same two CSVs - never writing/submitting anything).

"Model total" here reuses the exact definition build.py's trigger (a)
already uses for the live report's "model vs. market" flag (see
_check_total_gap): BPP's projected total (bpp_total_proj), falling back
to DRatings' projected total (dratings_total_proj) when BPP has no row
for that game - kept consistent with the live report rather than
inventing a second, competing definition of "the model total."

Three separate things live here:
  - backtest_thresholds(): fits/tests every threshold in
    BACKTEST_THRESHOLDS against the FULL logged history - a backtest,
    free to be shaped by what's already happened, so its numbers alone
    are not proof a threshold works going forward.
  - forward_hit_rates(): the ongoing, going-forward companion - scoped to
    CANDIDATE_THRESHOLDS and only counts predictions logged on/after
    FORWARD_TRACKING_START_DATE, so it is never fit to the same data used
    to pick those candidate thresholds. main.py calls this every daily
    run, so the sample keeps growing on genuinely new, out-of-sample
    games instead of re-testing history the backtest already used.
  - qualifies() / score_row(): the actual "is this game a pick" and
    "did that pick hit" primitives, now used live by track_record.py and
    main.py to decide which games count as a tracked totals pick at all
    (see PICK_THRESHOLD below) - not just for backtest reporting.

PICK_THRESHOLD is the one threshold now used to decide, day to day,
whether a game gets a totals pick logged for tracking purposes at all
(over-only: qualifies() is never true for a game where the model leans
under or is within threshold of the market). It was chosen from the
2026-08-04 re-run of backtest_thresholds() against the full logged
history at that point (203 predictions / 172 results):

    threshold | qualifying | decided | hit rate
    0.5       | 94         | 92      | 45.7%
    1.0       | 48         | 47      | 51.1%
    1.5       | 16         | 16      | 62.5%   <- picked
    2.0       | 8          | 8       | 75.0%   (too_small: <15 decided)
    2.5       | 1          | 1       | 0.0%    (too_small: <15 decided)
    3.0       | 0          | 0       | n/a     (too_small: <15 decided)

1.5 is the best hit rate among thresholds that actually clear
MIN_SAMPLE=15 decided games. 2.0's 75% is real work but built on only 8
decided games - noise dressed up as an edge, exactly the small-sample
outlier this selection deliberately excludes rather than chasing it for
the flashier number. 2.5 (one game) and 3.0 (zero games) aren't even
tested samples. That said, 16 decided games only just clears the bar -
this is "best-supported by what we have," not "proven" - which is why
forward_hit_rates() below keeps testing it (among CANDIDATE_THRESHOLDS)
on genuinely new, out-of-sample games rather than treating the backtest
as the last word.
"""

from dataclasses import dataclass

BACKTEST_THRESHOLDS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
PICK_THRESHOLD = 1.5  # see module docstring for why - re-derive by re-running backtest_thresholds()
# Candidates for ongoing forward-tracking, re-run daily against genuinely
# new out-of-sample games. Includes PICK_THRESHOLD itself (so the
# threshold actually driving live picks keeps getting checked against
# fresh data, not just the backtest that selected it) plus its two
# lower neighbors for context/comparison.
CANDIDATE_THRESHOLDS = (0.5, 1.0, 1.5)
FORWARD_TRACKING_START_DATE = "2026-07-29"  # date this feature shipped
MIN_SAMPLE = 15  # fewer qualifying decided games than this = "too small to trust"


def _to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _model_total(row):
    bpp = _to_float(row.get("bpp_total_proj"))
    return bpp if bpp is not None else _to_float(row.get("dratings_total_proj"))


def qualifies(row, threshold=PICK_THRESHOLD):
    """Does this game get a totals pick logged at all, under the
    over-only "model total >= market + threshold" rule? False whenever
    the model leans under, agrees with the market, or doesn't clear
    threshold - never forced to a pick (see module docstring)."""
    model_total = _model_total(row)
    market_total = _to_float(row.get("market_total"))
    if model_total is None or market_total is None:
        return False
    return model_total - market_total >= threshold


def score_row(row, result, threshold=PICK_THRESHOLD):
    """For one prediction row + its result row (may be None/not-final
    yet): "hit", "push", "miss", or None (doesn't qualify for a pick, or
    no final result yet to score it against). Over-only, so a qualifying
    pick is always "hope for actual > market_total"."""
    if not qualifies(row, threshold):
        return None
    if not result:
        return None
    market_total = _to_float(row.get("market_total"))
    actual_total = _to_float(result.get("total_runs"))
    if actual_total is None:
        return None
    if actual_total == market_total:
        return "push"
    return "hit" if actual_total > market_total else "miss"


@dataclass
class ThresholdResult:
    threshold: float
    qualifying: int  # games where model_total - market_total >= threshold, with a final result
    pushes: int       # of those, actual total == market line (excluded from hit rate)
    hits: int          # of the non-push qualifying games, actual total > market line
    hit_rate: float | None
    too_small: bool


def _threshold_result(rows, results_by_id, threshold, min_sample):
    qualifying = 0
    pushes = 0
    hits = 0
    for row in rows:
        outcome = score_row(row, results_by_id.get(row.get("game_id")), threshold)
        if outcome is None:
            continue
        qualifying += 1
        if outcome == "push":
            pushes += 1
        elif outcome == "hit":
            hits += 1
    decided = qualifying - pushes
    hit_rate = round(100 * hits / decided, 1) if decided else None
    return ThresholdResult(
        threshold=threshold, qualifying=qualifying, pushes=pushes, hits=hits,
        hit_rate=hit_rate, too_small=decided < min_sample,
    )


def backtest_thresholds(predictions_rows, results_rows, thresholds=BACKTEST_THRESHOLDS, min_sample=MIN_SAMPLE):
    """Fits/tests every threshold against the full logged history. Returns
    a list of ThresholdResult, one per threshold, in the order given."""
    results_by_id = {r["game_id"]: r for r in results_rows if r.get("game_id")}
    return [_threshold_result(predictions_rows, results_by_id, t, min_sample) for t in thresholds]


def forward_hit_rates(predictions_rows, results_rows, thresholds=CANDIDATE_THRESHOLDS,
                       start_date=FORWARD_TRACKING_START_DATE, min_sample=MIN_SAMPLE):
    """Same computation as backtest_thresholds, but scoped to predictions
    logged on/after start_date - the genuinely out-of-sample, "tested
    forward on new data" companion the backtest can't be (see module
    docstring)."""
    forward_rows = [r for r in predictions_rows if (r.get("date") or "") >= start_date]
    results_by_id = {r["game_id"]: r for r in results_rows if r.get("game_id")}
    return [_threshold_result(forward_rows, results_by_id, t, min_sample) for t in thresholds]
