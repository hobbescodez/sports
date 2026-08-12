"""
Trailing-N-days accuracy history - a display layer over
predictions_log.csv + results_log.csv (scoring.py's data), not new data
infrastructure.

Moneyline side: for each of the last few days, finds that day's single
highest-conviction moneyline pick - mirroring build.py's Conviction
Board vote-counting and CONVICTION_MIN_SOURCES/CONVICTION_MAX_DISSENT
thresholds exactly, just applied to logged historical picks instead of
live source objects (a past day's Matchup objects no longer exist by
the time this runs) - then checks it against what actually happened.
Scoped to one pick per day, not all of that day's games, to keep the
table readable.

Totals side: NOT a highest-conviction single pick. Every game that day
clearing totals_threshold.py's PICK_THRESHOLD rule (over-only: model
total >= market + PICK_THRESHOLD) counts as a logged pick - could be
zero games on a quiet day, could be several. This mirrors what's
actually tracked as a "totals pick" for accuracy purposes now (see
totals_threshold.py's module docstring for why that threshold and rule),
not a separate conviction-vote concept - so a day's totals row here is a
count + hit rate, not a single matchup.
"""

from dataclasses import dataclass

from core.conviction import tally_votes
from sports.mlb.analysis.build import CONVICTION_MAX_DISSENT, CONVICTION_MIN_SOURCES
from sports.mlb.teams import full_name
from sports.mlb.tracker.totals_threshold import qualifies, score_row

_MONEYLINE_FIELDS = (("dratings_pick", "DRatings"), ("bpp_pick", "BPP"), ("mymodel_pick", "My model"))


@dataclass
class DayRecord:
    date: str
    ml_matchup: str | None = None
    ml_pick: str | None = None
    ml_pick_name: str | None = None
    ml_agree: str | None = None
    ml_actual_winner: str | None = None
    ml_correct: bool | None = None
    totals_qualifying: int = 0   # games that day clearing PICK_THRESHOLD - i.e. picks actually logged
    totals_pushes: int = 0
    totals_hits: int = 0
    totals_hit_rate: float | None = None  # None if no decided (non-push) qualifying picks yet


def _moneyline_votes(row):
    """Counted votes only - DRatings, BPP, My model. Kalshi is a real-money
    market price, not an independent prediction, so it's excluded from the
    count here too (see build.py's _moneyline_votes for the live-board
    equivalent)."""
    return [(label, row[field]) for field, label in _MONEYLINE_FIELDS if row.get(field)]


def _highest_conviction(rows, votes_fn):
    """The single game (from a day's predictions_log rows) with the most
    source agreement, via core.conviction's same generic tally used by
    build.py's live Conviction Board (same CONVICTION_MIN_SOURCES/
    CONVICTION_MAX_DISSENT bar) - never a forced pick if nothing clears
    it. Ties broken the same way the live board is sorted: more agreeing
    sources first, then more total sources present."""
    best = None
    best_key = None
    for row in rows:
        votes = votes_fn(row)
        tally = tally_votes(votes, CONVICTION_MIN_SOURCES, CONVICTION_MAX_DISSENT, lambda d: d)
        if tally is None:
            continue
        key = (tally.agree_count, tally.total_count)
        if best is None or key > best_key:
            best = (row, tally.label, tally.agree_count, tally.total_count)
            best_key = key
    return best


def build_track_record(predictions_rows, results_rows, num_days=7, exclude_date=None):
    """Returns a list of DayRecord, most recent day first, for up to the
    last `num_days` distinct dates present in predictions_rows (excluding
    exclude_date - normally today, whose games likely aren't final yet)."""
    results_by_id = {r["game_id"]: r for r in results_rows if r.get("game_id")}

    by_date = {}
    for p in predictions_rows:
        d = p.get("date")
        if d:
            by_date.setdefault(d, []).append(p)

    dates = sorted((d for d in by_date if d != exclude_date), reverse=True)[:num_days]

    records = []
    for date in dates:
        rows = by_date[date]
        rec = DayRecord(date=date)

        ml_best = _highest_conviction(rows, _moneyline_votes)
        if ml_best:
            row, pick, agree_n, total_n = ml_best
            rec.ml_matchup = f"{row['away_abbrev']} @ {row['home_abbrev']}"
            rec.ml_pick = pick
            rec.ml_pick_name = full_name(pick)
            rec.ml_agree = f"{agree_n}/{total_n}"
            result = results_by_id.get(row.get("game_id"))
            if result and result.get("winner_abbrev"):
                rec.ml_actual_winner = result["winner_abbrev"]
                rec.ml_correct = pick == result["winner_abbrev"]

        totals_misses = 0
        for row in rows:
            if not qualifies(row):
                continue
            rec.totals_qualifying += 1
            outcome = score_row(row, results_by_id.get(row.get("game_id")))
            if outcome == "push":
                rec.totals_pushes += 1
            elif outcome == "hit":
                rec.totals_hits += 1
            elif outcome == "miss":
                totals_misses += 1
            # outcome is None when the game isn't final yet - still counts
            # toward totals_qualifying (a pick was logged) but not yet
            # toward hits/misses/hit_rate.
        decided = rec.totals_hits + totals_misses
        rec.totals_hit_rate = round(100 * rec.totals_hits / decided, 1) if decided else None

        records.append(rec)

    return records
