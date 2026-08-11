"""
Orchestrates the daily MLB analysis report: fetches all four sources
(isolating failures so one bad source doesn't take down the report),
builds the joined/analyzed data, renders static HTML, and writes it to
docs/ (today-dated + a rolling index.html for GitHub Pages).

Usage: python main.py
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sports.mlb.analysis.build import build_report_data
from sports.mlb.fetch import dratings, kalshi, moundedge, mymodel, reddit, schedule, sportsbettingdime
from sports.mlb.report import fonts
from sports.mlb.report.render import render_artifact_fragment, render_report
from sports.mlb.tracker.accuracy_breakdown import build_accuracy_breakdown
from sports.mlb.tracker.log_predictions import log_todays_predictions
from sports.mlb.tracker.log_results import PREDICTIONS_LOG_PATH, RESULTS_LOG_PATH, fetch_and_log_results
from sports.mlb.tracker.market_lean_totals import (
    MARKET_LEAN_START_DATE,
    market_lean_baseline,
    market_lean_forward,
)
from sports.mlb.tracker.scoring import score_predictions
from sports.mlb.tracker.totals_threshold import (
    FORWARD_TRACKING_START_DATE,
    PICK_THRESHOLD,
    backtest_thresholds,
    forward_hit_rates,
)
from sports.mlb.tracker.track_record import build_track_record

ET = ZoneInfo("America/New_York")  # MLB slates are organized by US Eastern date
PT = ZoneInfo("America/Los_Angeles")  # "generated at" is shown in Pacific time
OUTPUT_DIR = Path(__file__).resolve().parent / "docs"
FONT_CACHE_PATH = Path(__file__).resolve().parent / "sports" / "mlb" / "report" / "inline_fonts.cache.css"


def _inline_font_css():
    """Cached so the ~150KB of embedded woff2 data isn't re-fetched every
    single day - Oswald/Inter don't change. Delete the cache file to force
    a re-fetch (e.g. if the font weights used here ever change)."""
    if FONT_CACHE_PATH.exists():
        return FONT_CACHE_PATH.read_text(encoding="utf-8")
    css = fonts.build_inline_font_css()
    FONT_CACHE_PATH.write_text(css, encoding="utf-8")
    return css


def _fetch_safe(label, fn, default):
    try:
        return fn()
    except Exception as e:
        print(f"[warn] {label} fetch failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return default


def main():
    now_et = datetime.now(ET)
    today_iso = now_et.strftime("%Y-%m-%d")
    today_display = now_et.strftime("%A, %B %-d, %Y")
    generated_at = datetime.now(PT).strftime("%Y-%m-%d %H:%M %Z")

    print(f"Building MLB daily analysis for {today_display} ({today_iso})")

    dr_games = _fetch_safe("DRatings", dratings.fetch_today_games, [])
    print(f"DRatings: {len(dr_games)} games")

    me_games = _fetch_safe("MoundEdge", moundedge.fetch_today_games, [])
    print(f"MoundEdge: {len(me_games)} games")
    slate_subtitle = me_games[0].slate_subtitle if me_games else ""

    sbd_status = _fetch_safe(
        "SportsBettingDime", sportsbettingdime.check_source,
        sportsbettingdime.SBDStatus(reachable=False, note="fetch failed"),
    )
    print(f"SportsBettingDime: {sbd_status.note}")

    reddit_result = _fetch_safe(
        "Reddit", lambda: reddit.fetch_daily_sentiment(today_iso),
        reddit.RedditResult(available=False, source="unavailable", note="fetch raised an exception"),
    )
    print(f"Reddit: source={reddit_result.source} available={reddit_result.available}")

    kalshi_games = _fetch_safe("Kalshi", lambda: kalshi.fetch_today_games(today_iso), [])
    print(f"Kalshi: {len(kalshi_games)} games")

    mymodel_games = _fetch_safe("My model", lambda: mymodel.fetch_today_games(today_iso), [])
    print(f"My model: {len(mymodel_games)} games")

    # official MLB schedule (gamePk/gameNumber per real game) - the spine
    # build.py uses to keep doubleheader games separate instead of
    # silently collapsing them; degrades gracefully to the old per-pair
    # behavior if this fetch fails, same as every other source here.
    schedule_games = _fetch_safe("Schedule", lambda: schedule.fetch_today_schedule(today_iso), [])
    print(f"Schedule: {len(schedule_games)} games")

    report_data = build_report_data(
        dr_games, me_games, reddit_result, today_iso, today_display, slate_subtitle,
        kalshi_games=kalshi_games, mymodel_games=mymodel_games, schedule_games=schedule_games,
    )
    report_data["sbd_status_note"] = sbd_status.note

    # MoundEdge (and, less commonly, DRatings) is a daily-generated page that
    # doesn't always refresh for the new day by the time this runs - detect
    # that rather than silently presenting last night's slate as today's.
    source_date_stale = bool(slate_subtitle) and today_display not in slate_subtitle
    report_data["source_date_stale"] = source_date_stale
    if source_date_stale:
        print(f"[warn] MoundEdge slate subtitle doesn't mention today ({today_display}): {slate_subtitle!r}")

    # Paper-trading accuracy tracker: logs each source's pick + Kalshi's price
    # for scoring later - a historical record only, never an order/trade.
    try:
        logged_rows = log_todays_predictions(report_data["games"], today_iso)
        print(f"Prediction tracker: logged {len(logged_rows)} games for {today_iso}")
    except Exception as e:
        print(f"[warn] Prediction tracker logging failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # checks every previously-logged prediction that isn't already recorded
    # as final, and records any that have finished since the last run -
    # historical scoring only, same no-trading guarantee as the logger above.
    try:
        newly_final = fetch_and_log_results()
        print(f"Results tracker: {len(newly_final)} game(s) newly recorded as final")
    except Exception as e:
        print(f"[warn] Results tracker fetch failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # Track Record section: trailing days' highest-conviction picks vs what
    # actually happened - pure display over the two CSVs above, so a
    # failure here shouldn't take down the rest of the report either.
    import csv as _csv

    def _read_csv(path):
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as f:
            return list(_csv.DictReader(f))

    predictions_rows, results_rows = [], []
    try:
        predictions_rows = _read_csv(PREDICTIONS_LOG_PATH)
        results_rows = _read_csv(RESULTS_LOG_PATH)
        report_data["track_record"] = build_track_record(
            predictions_rows, results_rows, num_days=7, exclude_date=today_iso,
        )
        print(f"Track record: {len(report_data['track_record'])} day(s)")
    except Exception as e:
        print(f"[warn] Track record build failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        report_data["track_record"] = []

    # Overall accuracy: every logged/scored game (not just the daily top
    # pick), accumulated across all days tracked so far - shown alongside
    # Track Record above, not replacing it. Pure re-use of scoring.py,
    # which already computes exactly this per source.
    try:
        scored = score_predictions(predictions_rows, results_rows)
        report_data["overall_accuracy"] = [
            {
                "label": label,
                "ml_games": scored["moneyline"][key].games_scored,
                "ml_correct": scored["moneyline"][key].correct_picks,
                "ml_pct": scored["moneyline"][key].accuracy_pct,
                "tot_games": scored["totals"][key].games_scored,
                "tot_avg_err": scored["totals"][key].avg_abs_error_runs,
            }
            for key, label in (("dratings", "DRatings"), ("bpp", "BPP"), ("mymodel", "My model"))
        ]
        print(f"Overall accuracy: {len(report_data['overall_accuracy'])} source(s) scored")
    except Exception as e:
        print(f"[warn] Overall accuracy build failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        report_data["overall_accuracy"] = []

    # Totals pick-rule accuracy, all logged games to date: how many games
    # ever cleared PICK_THRESHOLD (i.e. would have gotten a totals pick
    # logged under the current rule) and how that subset has done, vs.
    # every game in overall_accuracy above. Paper/analysis only, same
    # no-trading guarantee as everything else in tracker/ - see
    # totals_threshold.py's module docstring for why PICK_THRESHOLD was
    # chosen. NOTE this re-uses the same full history the backtest used
    # to pick PICK_THRESHOLD in the first place, so it's expected to look
    # decent - it is not the out-of-sample check. The Totals Threshold
    # Tracking section further down (forward_hit_rates, scoped to dates
    # on/after FORWARD_TRACKING_START_DATE) is the genuine forward test.
    try:
        report_data["totals_pick_threshold"] = PICK_THRESHOLD
        report_data["totals_pick_alltime"] = backtest_thresholds(
            predictions_rows, results_rows, thresholds=(PICK_THRESHOLD,),
        )[0]
    except Exception as e:
        print(f"[warn] Totals pick-rule all-time accuracy build failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        report_data["totals_pick_threshold"] = PICK_THRESHOLD
        report_data["totals_pick_alltime"] = None

    # "What predicts accuracy" analysis: win-pick accuracy broken out by
    # confidence tier, source-agreement count, and flagged/notable status -
    # see accuracy_breakdown.py for the not-enough-data-yet threshold.
    try:
        report_data["accuracy_breakdown"] = build_accuracy_breakdown(predictions_rows, results_rows)
        print(f"Accuracy breakdown: {report_data['accuracy_breakdown']['days_logged']} day(s) with scored games")
    except Exception as e:
        print(f"[warn] Accuracy breakdown build failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        report_data["accuracy_breakdown"] = None

    # Totals threshold forward-tracking: paper/analysis only, no order
    # placement anywhere - see totals_threshold.py's module docstring.
    # Deliberately re-run every day rather than cached, so each new day's
    # logged games extend the out-of-sample count without any extra state.
    try:
        report_data["totals_threshold_forward"] = forward_hit_rates(predictions_rows, results_rows)
        report_data["totals_threshold_start_date"] = FORWARD_TRACKING_START_DATE
        n_qualifying = sum(row.qualifying for row in report_data["totals_threshold_forward"])
        print(f"Totals threshold forward-tracking: {n_qualifying} qualifying game-threshold entries since {FORWARD_TRACKING_START_DATE}")
    except Exception as e:
        print(f"[warn] Totals threshold forward-tracking failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        report_data["totals_threshold_forward"] = []
        report_data["totals_threshold_start_date"] = FORWARD_TRACKING_START_DATE

    # Market-lean adjusted totals rule: a SECOND, distinct forward-only
    # paper test, not the same rule as totals_threshold_forward above and
    # deliberately not merged with it - see market_lean_totals.py's
    # module docstring. Paper/analysis only, no order placement.
    try:
        report_data["market_lean_forward"] = market_lean_forward(predictions_rows, results_rows)
        report_data["market_lean_baseline"] = market_lean_baseline(predictions_rows, results_rows)
        report_data["market_lean_start_date"] = MARKET_LEAN_START_DATE
        print(
            f"Market-lean totals rule: {report_data['market_lean_forward'].qualifying} qualifying "
            f"game(s) since {MARKET_LEAN_START_DATE}"
        )
    except Exception as e:
        print(f"[warn] Market-lean totals rule failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        report_data["market_lean_forward"] = None
        report_data["market_lean_baseline"] = None
        report_data["market_lean_start_date"] = MARKET_LEAN_START_DATE

    inline_font_css = _fetch_safe("Google Fonts (Oswald/Inter)", _inline_font_css, "")
    if inline_font_css:
        print(f"Inline fonts: {len(inline_font_css) // 1024}KB embedded")

    html = render_report(report_data, generated_at, inline_font_css)
    fragment_html = render_artifact_fragment(report_data, generated_at, inline_font_css)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dated_path = OUTPUT_DIR / f"{today_iso}.html"
    # was docs/index.html - moved to docs/mlb.html so the site root can be
    # a real cross-sport landing page instead of MLB claiming it outright
    # (see build_landing_page.py)
    mlb_index_path = OUTPUT_DIR / "mlb.html"
    fragment_path = OUTPUT_DIR / "artifact_fragment.html"
    dated_path.write_text(html, encoding="utf-8")
    mlb_index_path.write_text(html, encoding="utf-8")
    fragment_path.write_text(fragment_html, encoding="utf-8")

    print(f"Wrote {dated_path}, {mlb_index_path}, and {fragment_path}")


if __name__ == "__main__":
    main()
