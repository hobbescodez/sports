"""
Model-improvement groundwork (analysis only - does not touch the live
model in sports/mlb/fetch/mymodel.py or main.py's daily pipeline).

Confirmed windows actually in use by the live model (see fetch/mymodel.py):
  - Team-batting xwOBA: TEAM_XWOBA_WINDOW_DAYS = 15 days.
  - Pitcher xwOBA-allowed: only xwoba_allowed_30d is read by
    _project_runs() - a 30-day window. (A 15d version is also computed
    per pitcher but never used in the live formula.)
These are NOT the same length - the two sides of the formula are on
different windows today. This script tests WINDOW_LENGTHS for both
sides, backtested against this project's own logged history.

Three things live here:

  1. TEAM-BATTING WINDOW, RAW CORRELATION (team xwOBA at each window
     length vs. that team's actual runs scored) - the original, narrower
     probe. Isolates the team-batting side only, no pitcher data needed.

  2. PITCHER VS. TEAM-BATTING REAL-WORLD SPREAD - a same-day snapshot of
     which side of the (symmetric-by-construction) formula swings
     further in practice across real teams/pitchers.

  3. FULL TWO-SIDED MODEL BACKTEST, POINT-IN-TIME, PER WINDOW LENGTH -
     the piece the original version of this script explicitly couldn't
     do, because predictions_log.csv logs the final projected-runs
     output per game, not per-game starting pitcher IDs. That gap is
     closed here via MLB Stats API's schedule endpoint, which retains
     historical probable-pitcher data for past dates (not just "today") -
     the same endpoint mymodel.py's _fetch_probable_pitchers already
     uses, generalized to any date. For every window length, this
     recomputes what My model's formula would have projected using ONLY
     data available as of the day before each game (no lookahead):
     team xwOBA, pitcher xwOBA-allowed, and league-average-runs, each
     recomputed at that window length specifically. Scored two ways:
     moneyline accuracy (predicted winner vs. actual) and average
     absolute total-runs error.

     Deliberately uses each pitcher's RAW (unshrunk) xwOBA-allowed at
     every window length, not the live model's PITCHER_STABILIZATION_PA
     shrinkage (shipped separately, after this script last ran) - window
     length and shrinkage are two different variables, and mixing them
     into one backtest would make it impossible to tell which one is
     driving any difference found. This isolates window length only.

     Network efficiency: rather than one Statcast pull per (pitcher,
     game) pair, this pulls each unique pitcher's full needed date range
     ONCE (spanning their earliest start's window-back cutoff through
     the day before their latest start), then slices per-game,
     per-window sub-windows from that single in-memory pull. Same idea
     for team-wide xwOBA: one league-wide pull spanning the whole
     backtest period, sliced per date/window client-side, instead of a
     fresh pull per date.

Same reason every other Statcast-touching script in this project is a
probe-style one-off run via GitHub Actions workflow_dispatch: this dev
sandbox cannot reach baseballsavant.mlb.com directly.
"""

import csv
import statistics
import sys
from collections import defaultdict
from datetime import date as date_cls
from datetime import timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sports.mlb.fetch.mymodel import HEADERS, _normalize_abbrev  # noqa: E402
from sports.mlb.park_factors import park_factor  # noqa: E402
from sports.mlb.teams import abbrev_from_name  # noqa: E402

RESULTS_LOG_PATH = Path(__file__).resolve().parent.parent / "sports" / "mlb" / "data" / "results_log.csv"
WINDOW_LENGTHS = (7, 15, 30, 45)
MAX_WINDOW = max(WINDOW_LENGTHS)


def hr(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _load_results():
    with RESULTS_LOG_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("status") == "Final" and r.get("away_score") and r.get("home_score")]


def _team_xwoba_for_window(as_of_date, window_days):
    """Pulls ONE league-wide Statcast slice covering [as_of_date - 30,
    as_of_date - 1] and slices it down to the requested window
    client-side, rather than a separate network pull per window length -
    same idea as fetch_team_xwoba() in mymodel.py, generalized to accept
    any window <= 30 days from a single pull."""
    import pandas as pd
    import pybaseball as pb

    start = as_of_date - timedelta(days=30)
    end = as_of_date - timedelta(days=1)
    df = pb.statcast(str(start), str(end))
    if df is None or len(df) == 0:
        return {}

    df = df[df["woba_denom"] > 0].copy()
    if len(df) == 0:
        return {}
    df["game_date"] = pd.to_datetime(df["game_date"])
    cutoff = pd.Timestamp(as_of_date - timedelta(days=window_days))
    df = df[df["game_date"] >= cutoff]
    if len(df) == 0:
        return {}

    df["batting_team"] = df.apply(lambda r: r["away_team"] if r["inning_topbot"] == "Top" else r["home_team"], axis=1)
    df["xwoba_component"] = df["estimated_woba_using_speedangle"].fillna(df["woba_value"])

    team_xwoba = {}
    for raw_team, group in df.groupby("batting_team"):
        ab = _normalize_abbrev(raw_team)
        if not ab:
            continue
        denom_sum = group["woba_denom"].sum()
        if denom_sum:
            team_xwoba[ab] = float((group["xwoba_component"] * group["woba_denom"]).sum() / denom_sum)
    return team_xwoba


def probe_window_backtest():
    hr("1. Rolling xwOBA window backtest (team-batting side only, raw correlation)")
    results = _load_results()
    dates = sorted({r["date"] for r in results})
    print(f"Testing {len(dates)} date(s) with final results: {dates}")

    pairs_by_window = {w: [] for w in WINDOW_LENGTHS}
    for date_str in dates:
        y, m, d = (int(x) for x in date_str.split("-"))
        as_of = date_cls(y, m, d)
        games_today = [r for r in results if r["date"] == date_str]
        print(f"\n-- {date_str}: {len(games_today)} final game(s) --")

        for window_days in WINDOW_LENGTHS:
            try:
                team_xwoba = _team_xwoba_for_window(as_of, window_days)
            except Exception as e:
                print(f"  [warn] {window_days}d pull failed for {date_str}: {e}")
                continue
            n_matched = 0
            for g in games_today:
                for team, runs in ((g["away_abbrev"], g["away_score"]), (g["home_abbrev"], g["home_score"])):
                    xwoba = team_xwoba.get(team)
                    if xwoba is not None:
                        pairs_by_window[window_days].append((xwoba, float(runs)))
                        n_matched += 1
            print(f"  {window_days:>2}d window: {len(team_xwoba)} teams with data, {n_matched} team-games matched")

    hr("Window backtest results (correlation between trailing team xwOBA and that team's actual runs scored)")
    print(f"{'window':>8} {'n':>5} {'pearson_r':>10}")
    for window_days in WINDOW_LENGTHS:
        pairs = pairs_by_window[window_days]
        n = len(pairs)
        if n < 3:
            print(f"{window_days:>7}d {n:>5}  not enough data")
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        try:
            r = statistics.correlation(xs, ys)
        except statistics.StatisticsError:
            r = None
        r_str = f"{r:.3f}" if r is not None else "n/a"
        print(f"{window_days:>7}d {n:>5} {r_str:>10}")
    print("\nHigher |pearson_r| = that window length's recent team xwOBA tracked actual runs scored more closely in this sample.")
    print("Small-sample caveat: this is a relative comparison across window lengths on the same handful of logged days, not an absolute accuracy claim.")


def probe_pitcher_vs_team_spread():
    hr("2. Real-world spread: team-batting xwOBA vs. pitcher xwOBA-allowed")
    import pybaseball as pb

    today = date_cls.today()
    team_xwoba = _team_xwoba_for_window(today, 15)
    print(f"Team xwOBA (15d window, {len(team_xwoba)} teams): {sorted(team_xwoba.values())}")
    if len(team_xwoba) >= 2:
        team_stdev = statistics.stdev(team_xwoba.values())
        team_mean = statistics.mean(team_xwoba.values())
        print(f"  mean={team_mean:.4f}  stdev={team_stdev:.4f}  coefficient_of_variation={team_stdev / team_mean:.4f}")

    df = pb.statcast_pitcher_expected_stats(today.year, minPA=30)
    if df is None or len(df) == 0:
        print("statcast_pitcher_expected_stats returned no rows - skipping pitcher spread")
        return
    pitcher_xwoba_allowed = [float(v) for v in df["est_woba"].dropna().tolist()]
    print(f"Pitcher xwOBA-allowed (season, {len(pitcher_xwoba_allowed)} qualified pitchers, minPA=30)")
    if len(pitcher_xwoba_allowed) >= 2:
        p_stdev = statistics.stdev(pitcher_xwoba_allowed)
        p_mean = statistics.mean(pitcher_xwoba_allowed)
        print(f"  mean={p_mean:.4f}  stdev={p_stdev:.4f}  coefficient_of_variation={p_stdev / p_mean:.4f}")

    print("\nA higher coefficient_of_variation means that side of the formula swings further (as a %) across real teams/")
    print("pitchers - i.e. carries more real-world influence on My model's projections in practice, even though the")
    print("formula itself weights both factors identically (see module docstring).")


def _historical_probable_pitchers(date_str, timeout=20):
    """MLB Stats API's schedule endpoint retains historical probable-
    pitcher data for past dates too, not just "today" - same endpoint
    mymodel.py's _fetch_probable_pitchers uses, generalized to any date.
    Returns {(away_abbrev, home_abbrev): (away_pitcher_id, home_pitcher_id)},
    keyed on the FIRST game of that matchup that day (doubleheaders
    collapse onto one lookup here - a known simplification, see the
    per-game skip-on-missing-ID handling below rather than mis-attributing
    game 2's pitcher to game 1)."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,team"
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    out = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away, home = g["teams"]["away"], g["teams"]["home"]
            away_ab = abbrev_from_name(away["team"].get("name", ""))
            home_ab = abbrev_from_name(home["team"].get("name", ""))
            if not away_ab or not home_ab:
                continue
            key = (away_ab, home_ab)
            if key in out:
                continue  # keep game 1's starters for a doubleheader pair
            out[key] = (
                away.get("probablePitcher", {}).get("id"),
                home.get("probablePitcher", {}).get("id"),
            )
    return out


def _pitcher_xwoba_allowed_window(df, as_of_date, window_days):
    """xwOBA-allowed for one pitcher's pitch-level DataFrame, sliced to
    [as_of_date - window_days, as_of_date - 1] - same PA-weighted
    computation as mymodel.py's _window_stats, just the xwoba_allowed
    piece (this script doesn't need velocity/whiff for the backtest)."""
    import pandas as pd

    cutoff_start = pd.Timestamp(as_of_date - timedelta(days=window_days))
    cutoff_end = pd.Timestamp(as_of_date - timedelta(days=1))
    sub = df[(df["game_date"] >= cutoff_start) & (df["game_date"] <= cutoff_end)]
    pa_ending = sub[sub["woba_denom"] > 0]
    if len(pa_ending) == 0:
        return None
    component = pa_ending["estimated_woba_using_speedangle"].fillna(pa_ending["woba_value"])
    denom_sum = pa_ending["woba_denom"].sum()
    if not denom_sum:
        return None
    return float((component * pa_ending["woba_denom"]).sum() / denom_sum)


def probe_full_model_backtest():
    hr("3. Full two-sided model backtest, point-in-time, per window length")
    import pandas as pd
    import pybaseball as pb

    results = _load_results()
    dates = sorted({r["date"] for r in results})
    print(f"{len(results)} final game(s) across {len(dates)} date(s): {dates}")

    # step 1: historical probable pitchers per date (cheap MLB Stats API calls)
    print("\nLooking up historical probable pitchers per date...")
    pitchers_by_date = {}
    for date_str in dates:
        try:
            pitchers_by_date[date_str] = _historical_probable_pitchers(date_str)
        except Exception as e:
            print(f"  [warn] probable-pitcher lookup failed for {date_str}: {e}")
            pitchers_by_date[date_str] = {}

    # step 2: build the game list with resolved pitcher IDs, skip games missing either
    games = []
    skipped_no_pitcher = 0
    for r in results:
        y, m, d = (int(x) for x in r["date"].split("-"))
        as_of = date_cls(y, m, d)
        key = (r["away_abbrev"], r["home_abbrev"])
        away_pid, home_pid = pitchers_by_date.get(r["date"], {}).get(key, (None, None))
        if not away_pid or not home_pid:
            skipped_no_pitcher += 1
            continue
        games.append({
            "date": r["date"], "as_of": as_of,
            "away_abbrev": r["away_abbrev"], "home_abbrev": r["home_abbrev"],
            "away_pitcher_id": away_pid, "home_pitcher_id": home_pid,
            "away_score": float(r["away_score"]), "home_score": float(r["home_score"]),
            "winner_abbrev": r.get("winner_abbrev") or (r["away_abbrev"] if float(r["away_score"]) > float(r["home_score"]) else r["home_abbrev"]),
            "actual_total": float(r["away_score"]) + float(r["home_score"]),
        })
    print(f"{len(games)} game(s) with both starting pitchers resolved ({skipped_no_pitcher} skipped - no probable pitcher on record)")
    if not games:
        print("Nothing to backtest - stopping part 3.")
        return

    # step 3: one team-wide Statcast pull spanning the whole backtest period + MAX_WINDOW lookback
    earliest_as_of = min(g["as_of"] for g in games)
    latest_as_of = max(g["as_of"] for g in games)
    team_start = earliest_as_of - timedelta(days=MAX_WINDOW)
    team_end = latest_as_of - timedelta(days=1)
    print(f"\nPulling one league-wide Statcast slice: {team_start} to {team_end} ...")
    team_df = pb.statcast(str(team_start), str(team_end))
    if team_df is None or len(team_df) == 0:
        print("[error] league-wide team pull returned nothing - stopping part 3.")
        return
    team_df = team_df[team_df["woba_denom"] > 0].copy()
    team_df["game_date"] = pd.to_datetime(team_df["game_date"])
    team_df["batting_team"] = team_df.apply(
        lambda r: r["away_team"] if r["inning_topbot"] == "Top" else r["home_team"], axis=1
    )
    team_df["xwoba_component"] = team_df["estimated_woba_using_speedangle"].fillna(team_df["woba_value"])
    print(f"  {len(team_df)} PA-ending rows pulled league-wide.")

    def _team_xwoba_and_league(as_of, window_days):
        cutoff = pd.Timestamp(as_of - timedelta(days=window_days))
        sub = team_df[(team_df["game_date"] >= cutoff) & (team_df["game_date"] < pd.Timestamp(as_of))]
        if len(sub) == 0:
            return {}, None
        league_denom = sub["woba_denom"].sum()
        league_xwoba = float((sub["xwoba_component"] * sub["woba_denom"]).sum() / league_denom) if league_denom else None
        team_xwoba = {}
        for raw_team, group in sub.groupby("batting_team"):
            ab = _normalize_abbrev(raw_team)
            if not ab:
                continue
            denom_sum = group["woba_denom"].sum()
            if denom_sum:
                team_xwoba[ab] = float((group["xwoba_component"] * group["woba_denom"]).sum() / denom_sum)
        return team_xwoba, league_xwoba

    # step 4: one Statcast pull per unique pitcher, spanning their full needed range
    pitcher_dates = defaultdict(list)  # pitcher_id -> [as_of dates they started]
    for g in games:
        pitcher_dates[g["away_pitcher_id"]].append(g["as_of"])
        pitcher_dates[g["home_pitcher_id"]].append(g["as_of"])

    print(f"\nPulling Statcast data for {len(pitcher_dates)} unique starting pitcher(s)...")
    pitcher_dfs = {}
    for i, (pid, as_of_list) in enumerate(pitcher_dates.items(), 1):
        p_start = min(as_of_list) - timedelta(days=MAX_WINDOW)
        p_end = max(as_of_list) - timedelta(days=1)
        try:
            df = pb.statcast_pitcher(str(p_start), str(p_end), pid)
        except Exception as e:
            print(f"  [warn] pitcher {pid} pull failed: {e}")
            pitcher_dfs[pid] = None
            continue
        if df is None or len(df) == 0:
            pitcher_dfs[pid] = None
            continue
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        pitcher_dfs[pid] = df
        if i % 10 == 0 or i == len(pitcher_dates):
            print(f"  ...{i}/{len(pitcher_dates)} pitchers pulled")

    # step 5: recompute My model's projection at each window length, per game - RAW
    # xwoba_allowed (no shrinkage - see module docstring for why)
    hr("Recomputing projections per window length (raw, unshrunk pitcher term)")
    stats_by_window = {w: {"n": 0, "ml_correct": 0, "total_errors": []} for w in WINDOW_LENGTHS}

    for window_days in WINDOW_LENGTHS:
        for g in games:
            team_xwoba, league_xwoba = _team_xwoba_and_league(g["as_of"], window_days)
            away_team_xwoba = team_xwoba.get(g["away_abbrev"])
            home_team_xwoba = team_xwoba.get(g["home_abbrev"])
            if away_team_xwoba is None or home_team_xwoba is None or league_xwoba is None:
                continue

            # away team bats against the home starter, and vice versa -
            # same convention as mymodel.py's fetch_today_games()
            home_starter_df = pitcher_dfs.get(g["home_pitcher_id"])
            away_starter_df = pitcher_dfs.get(g["away_pitcher_id"])
            if home_starter_df is None or away_starter_df is None:
                continue
            away_pitcher_xwoba = _pitcher_xwoba_allowed_window(home_starter_df, g["as_of"], window_days)
            home_pitcher_xwoba = _pitcher_xwoba_allowed_window(away_starter_df, g["as_of"], window_days)
            if away_pitcher_xwoba is None or home_pitcher_xwoba is None:
                continue

            league_avg_runs = 4.5  # fixed constant here - isolating window length on the xwOBA
            # terms only; league_avg_runs_per_team_game is a scalar multiplier applied equally to
            # both teams' projections, so it cancels out of the moneyline pick entirely and only
            # rescales the total (not the per-window-length comparison of total error shape) -
            # not worth another 4x round of window-dependent MLB Stats API pulls to include.
            pk_factor = park_factor(g["home_abbrev"])

            away_proj = league_avg_runs * (away_team_xwoba / league_xwoba) * (away_pitcher_xwoba / league_xwoba) * pk_factor
            home_proj = league_avg_runs * (home_team_xwoba / league_xwoba) * (home_pitcher_xwoba / league_xwoba) * pk_factor

            predicted_winner = g["away_abbrev"] if away_proj > home_proj else g["home_abbrev"]
            predicted_total = away_proj + home_proj

            s = stats_by_window[window_days]
            s["n"] += 1
            s["ml_correct"] += int(predicted_winner == g["winner_abbrev"])
            s["total_errors"].append(abs(predicted_total - g["actual_total"]))

    hr("Full model backtest results (moneyline accuracy + avg total error, per window length)")
    print(f"{'window':>8} {'n':>5} {'ml_acc':>8} {'avg_total_err':>14}")
    for window_days in WINDOW_LENGTHS:
        s = stats_by_window[window_days]
        n = s["n"]
        if n == 0:
            print(f"{window_days:>7}d {n:>5}  no games scored")
            continue
        ml_acc = 100 * s["ml_correct"] / n
        avg_err = statistics.mean(s["total_errors"])
        print(f"{window_days:>7}d {n:>5} {ml_acc:>7.1f}% {avg_err:>13.2f}")
    print(f"\n{len(games)} games had both starting pitchers resolved; actual n per window varies further based on")
    print("whether that window/date combination returned usable Statcast data - see per-window n above, not a fixed total.")
    print("Small-sample caveat: this is backtested analysis on this project's own limited logged history, not proof")
    print("any window length is 'correct' going forward - treat the spread between window lengths' n as itself a")
    print("signal of whether the accuracy difference is real or just which window happened to have more data this period.")


def main():
    try:
        probe_window_backtest()
    except Exception as e:
        print(f"[error] window backtest failed: {e}")
        raise
    try:
        probe_pitcher_vs_team_spread()
    except Exception as e:
        print(f"[error] pitcher/team spread check failed: {e}")
        raise
    try:
        probe_full_model_backtest()
    except Exception as e:
        print(f"[error] full model backtest failed: {e}")
        raise


if __name__ == "__main__":
    main()
