"""
wta_data_fetcher.py
-------------------
Fetches WTA match data from Jeff Sackmann's GitHub repository and computes
rolling player stats and recent match history for use in pre-match prediction.

Repository: https://github.com/JeffSackmann/tennis_wta

Usage:
    python wta_data_fetcher.py

    # Or import and use directly:
    from wta_data_fetcher import WTADataFetcher
    fetcher = WTADataFetcher(years=[2023, 2024, 2025])
    fetcher.load_data()
    stats = fetcher.get_player_stats("Swiatek I.")
    h2h   = fetcher.get_head_to_head("Swiatek I.", "Sabalenka A.")
"""

import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master"

# Match-result CSV columns produced by Sackmann's repo
MATCH_COLS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num",
    "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand",
    "winner_ht", "winner_ioc", "winner_age",
    "loser_id",  "loser_seed",  "loser_entry",  "loser_name",  "loser_hand",
    "loser_ht",  "loser_ioc",   "loser_age",
    "score", "best_of", "round", "minutes",
    # Serve / return stats (winner)
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    # Serve / return stats (loser)
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
    "winner_rank", "winner_rank_points",
    "loser_rank",  "loser_rank_points",
]

ROLLING_WINDOWS = [5, 10, 20]   # match windows for rolling stats


# ---------------------------------------------------------------------------
# Fetcher class
# ---------------------------------------------------------------------------

class WTADataFetcher:
    """
    Downloads and processes WTA match data from Sackmann's GitHub repo.

    Parameters
    ----------
    years : list[int]
        Calendar years to download (e.g. [2022, 2023, 2024, 2025]).
        Defaults to the last 3 complete/current years.
    include_qualifying : bool
        Whether to include qualifying-round matches. Default False.
    request_delay : float
        Seconds to wait between HTTP requests (be polite). Default 0.5.
    """

    def __init__(self, years=None, include_qualifying=False, request_delay=0.5):
        current_year = datetime.now().year
        self.years = years or list(range(current_year - 2, current_year + 1))
        self.include_qualifying = include_qualifying
        self.request_delay = request_delay
        self.matches: pd.DataFrame = pd.DataFrame()
        self.rankings: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self):
        """Download match files and (optionally) rankings, then pre-process."""
        print(f"Loading WTA match data for years: {self.years}")
        frames = []
        for year in self.years:
            df = self._fetch_matches(year)
            if df is not None:
                frames.append(df)

        if not frames:
            raise RuntimeError("No match data could be loaded. Check your internet connection.")

        self.matches = pd.concat(frames, ignore_index=True)
        self._preprocess_matches()

        self.rankings = self._fetch_latest_rankings()
        print(f"Loaded {len(self.matches):,} matches across {len(self.years)} year(s).")
        return self

    def _fetch_matches(self, year: int) -> pd.DataFrame | None:
        url = f"{BASE_URL}/wta_matches_{year}.csv"
        print(f"  Fetching {url} …")
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), low_memory=False)
            time.sleep(self.request_delay)
            return df
        except requests.HTTPError as e:
            print(f"  Warning: could not fetch {year} ({e})")
            return None
        except Exception as e:
            print(f"  Warning: unexpected error for {year} ({e})")
            return None

    def _fetch_latest_rankings(self) -> pd.DataFrame:
        """
        Downloads the most recent available weekly rankings file.
        Sackmann stores rankings as wta_rankings_DECADE.csv, e.g. wta_rankings_20s.csv
        We try to grab the current-decade file first.
        """
        current_year = datetime.now().year
        decade_str = f"{str(current_year)[:3]}s"   # e.g. "202" -> "20s"  (handles 2020s)
        # Sackmann uses "00s", "10s", "20s"
        decade_map = {2020: "20s", 2010: "10s", 2000: "00s"}
        decade_key = (current_year // 10) * 10
        suffix = decade_map.get(decade_key, "20s")

        url = f"{BASE_URL}/wta_rankings_{suffix}.csv"
        print(f"  Fetching rankings from {url} …")
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), low_memory=False)
            # Keep only the most recent week per player
            df["ranking_date"] = pd.to_datetime(df["ranking_date"], format="%Y%m%d")
            latest = df.loc[df.groupby("player")["ranking_date"].idxmax()]
            time.sleep(self.request_delay)
            print(f"  Loaded {len(latest):,} player rankings (latest snapshot).")
            return latest.reset_index(drop=True)
        except Exception as e:
            print(f"  Warning: could not load rankings ({e})")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess_matches(self):
        df = self.matches.copy()

        # Parse date
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")

        # Drop qualifying if requested
        if not self.include_qualifying:
            df = df[~df["round"].isin(["Q1", "Q2", "Q3", "Q4", "QR"])]

        # Numeric coercion for stat columns
        stat_cols = [c for c in df.columns if c.startswith(("w_", "l_", "winner_rank", "loser_rank"))]
        df[stat_cols] = df[stat_cols].apply(pd.to_numeric, errors="coerce")

        # Derived serve stats (winner)
        df["w_1stServeIn_pct"] = df["w_1stIn"]  / df["w_svpt"]
        df["w_1stServeWon_pct"] = df["w_1stWon"] / df["w_1stIn"]
        df["w_2ndServeWon_pct"] = df["w_2ndWon"] / (df["w_svpt"] - df["w_1stIn"])
        df["w_bpSaved_pct"]    = df["w_bpSaved"] / df["w_bpFaced"].replace(0, np.nan)

        # Derived serve stats (loser)
        df["l_1stServeIn_pct"] = df["l_1stIn"]  / df["l_svpt"]
        df["l_1stServeWon_pct"] = df["l_1stWon"] / df["l_1stIn"]
        df["l_2ndServeWon_pct"] = df["l_2ndWon"] / (df["l_svpt"] - df["l_1stIn"])
        df["l_bpSaved_pct"]    = df["l_bpSaved"] / df["l_bpFaced"].replace(0, np.nan)

        df.sort_values("tourney_date", inplace=True)
        self.matches = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Player-level helpers
    # ------------------------------------------------------------------

    def _get_player_matches(self, player_name: str) -> pd.DataFrame:
        """Return all matches (as winner or loser) for a player, sorted oldest→newest."""
        df = self.matches
        mask = (df["winner_name"] == player_name) | (df["loser_name"] == player_name)
        return df[mask].copy().sort_values("tourney_date").reset_index(drop=True)

    def _flatten_player_perspective(self, player_name: str) -> pd.DataFrame:
        """
        Reshape match data so each row represents one match from the
        given player's perspective (win/loss, their stats vs opponent stats).
        """
        pm = self._get_player_matches(player_name)
        rows = []
        for _, r in pm.iterrows():
            is_winner = r["winner_name"] == player_name
            p_pfx, o_pfx = ("w_", "l_") if is_winner else ("l_", "w_")
            p_name_pfx = "winner_" if is_winner else "loser_"
            o_name_pfx = "loser_"  if is_winner else "winner_"

            row = {
                "date":       r["tourney_date"],
                "tourney":    r["tourney_name"],
                "surface":    r["surface"],
                "round":      r["round"],
                "won":        int(is_winner),
                "opponent":   r[f"{o_name_pfx}name"],
                "opp_rank":   r[f"{o_name_pfx}rank"],
                "p_rank":     r[f"{p_name_pfx}rank"],
                "p_rank_pts": r[f"{p_name_pfx}rank_points"],
                "minutes":    r["minutes"],
            }
            # Copy per-match serve/return stats
            for stat in [
                "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon",
                "SvGms", "bpSaved", "bpFaced",
                "1stServeIn_pct", "1stServeWon_pct",
                "2ndServeWon_pct", "bpSaved_pct",
            ]:
                row[f"p_{stat}"] = r.get(f"{p_pfx}{stat}", np.nan)
                row[f"o_{stat}"] = r.get(f"{o_pfx}{stat}", np.nan)
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_recent_matches(self, player_name: str, n: int = 10,
                           surface: str = None) -> pd.DataFrame:
        """
        Return the n most recent matches for a player.

        Parameters
        ----------
        player_name : str  e.g. "Swiatek I."
        n           : int  number of recent matches to return
        surface     : str  optional filter: "Hard", "Clay", "Grass", "Carpet"
        """
        df = self._flatten_player_perspective(player_name)
        if surface:
            df = df[df["surface"].str.lower() == surface.lower()]
        return df.tail(n).reset_index(drop=True)

    def get_player_stats(self, player_name: str, surface: str = None,
                         days_back: int = 365) -> dict:
        """
        Compute rolling and aggregate pre-match stats for a player.

        Returns a dict containing:
          - win rates (overall, by surface, recent windows)
          - average serve stats (1st-serve %, 2nd-serve won %, aces, DFs, BP saved %)
          - current ranking (from rankings file)
          - recent form indicators
        """
        df = self._flatten_player_perspective(player_name)
        if df.empty:
            return {"error": f"No matches found for '{player_name}'"}

        # Date filter
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
        df_period = df[df["date"] >= cutoff]
        df_surface = df_period[df_period["surface"].str.lower() == surface.lower()] \
            if surface else df_period

        stats = {"player": player_name, "surface_filter": surface or "All"}

        # --- Win rates ---
        for label, subset in [("overall", df_period), ("surface", df_surface)]:
            if len(subset):
                stats[f"win_rate_{label}"] = round(subset["won"].mean(), 4)
                stats[f"matches_{label}"]  = len(subset)
            else:
                stats[f"win_rate_{label}"] = np.nan
                stats[f"matches_{label}"]  = 0

        # --- Rolling win rates (last N matches, unfiltered) ---
        for w in ROLLING_WINDOWS:
            recent = df.tail(w)
            stats[f"win_rate_L{w}"] = round(recent["won"].mean(), 4) if len(recent) else np.nan

        # --- Serve / return averages (surface-filtered period) ---
        serve_stats = [
            "p_ace", "p_df", "p_1stServeIn_pct",
            "p_1stServeWon_pct", "p_2ndServeWon_pct", "p_bpSaved_pct",
        ]
        for col in serve_stats:
            if col in df_surface.columns:
                stats[col] = round(df_surface[col].mean(skipna=True), 4)

        # --- Average match duration ---
        stats["avg_match_minutes"] = round(df_surface["minutes"].mean(skipna=True), 1)

        # --- Current ranking ---
        if not self.rankings.empty:
            r = self.rankings[self.rankings["player"] == player_name]
            if not r.empty:
                stats["current_rank"]       = int(r.iloc[0]["rank"])
                stats["current_rank_date"]  = str(r.iloc[0]["ranking_date"].date())

        return stats

    def get_head_to_head(self, player_a: str, player_b: str,
                         surface: str = None) -> dict:
        """
        Return head-to-head record between two players.

        Parameters
        ----------
        player_a, player_b : str   player name strings (Sackmann format)
        surface            : str   optional surface filter
        """
        df = self.matches
        mask = (
            ((df["winner_name"] == player_a) & (df["loser_name"] == player_b)) |
            ((df["winner_name"] == player_b) & (df["loser_name"] == player_a))
        )
        h2h = df[mask].copy()
        if surface:
            h2h = h2h[h2h["surface"].str.lower() == surface.lower()]

        h2h = h2h.sort_values("tourney_date")
        a_wins = (h2h["winner_name"] == player_a).sum()
        b_wins = (h2h["winner_name"] == player_b).sum()

        result = {
            "player_a": player_a,
            "player_b": player_b,
            "surface_filter": surface or "All",
            "total_matches": len(h2h),
            f"{player_a}_wins": int(a_wins),
            f"{player_b}_wins": int(b_wins),
            "matches": h2h[[
                "tourney_date", "tourney_name", "surface", "round",
                "winner_name", "loser_name", "score"
            ]].to_dict(orient="records"),
        }
        return result

    def get_pre_match_features(self, player_a: str, player_b: str,
                               surface: str = None) -> dict:
        """
        Build a flat feature dict ready to pass into a prediction model.
        Includes stats for both players and H2H on the given surface.

        Parameters
        ----------
        player_a : str   first player (the one you want to predict for)
        player_b : str   second player (the opponent)
        surface  : str   match surface ("Hard", "Clay", "Grass")

        Returns
        -------
        dict with prefixed keys: a_{stat}, b_{stat}, h2h_{stat}
        """
        stats_a = self.get_player_stats(player_a, surface=surface)
        stats_b = self.get_player_stats(player_b, surface=surface)
        h2h     = self.get_head_to_head(player_a, player_b, surface=surface)

        features = {}

        # Player A features
        for k, v in stats_a.items():
            if k not in ("player", "surface_filter", "current_rank_date"):
                features[f"a_{k}"] = v

        # Player B features
        for k, v in stats_b.items():
            if k not in ("player", "surface_filter", "current_rank_date"):
                features[f"b_{k}"] = v

        # H2H features
        features["h2h_total"]  = h2h["total_matches"]
        features["h2h_a_wins"] = h2h.get(f"{player_a}_wins", 0)
        features["h2h_b_wins"] = h2h.get(f"{player_b}_wins", 0)
        features["h2h_a_win_rate"] = (
            round(features["h2h_a_wins"] / features["h2h_total"], 4)
            if features["h2h_total"] > 0 else np.nan
        )

        # Differentials (useful for models that prefer relative features)
        if "a_current_rank" in features and "b_current_rank" in features:
            features["rank_diff_a_minus_b"] = features["a_current_rank"] - features["b_current_rank"]

        for w in ROLLING_WINDOWS:
            ka, kb = f"a_win_rate_L{w}", f"b_win_rate_L{w}"
            if ka in features and kb in features:
                features[f"win_rate_diff_L{w}"] = (
                    features[ka] - features[kb]
                    if not (np.isnan(features[ka]) or np.isnan(features[kb]))
                    else np.nan
                )

        return features


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

def main():
    # ---- Load last 3 years of data ----------------------------------------
    fetcher = WTADataFetcher(years=[2023, 2024, 2025])
    fetcher.load_data()

    print("\n" + "=" * 60)
    print("EXAMPLE 1 – Recent 10 matches for Iga Swiatek (Clay)")
    print("=" * 60)
    recent = fetcher.get_recent_matches("Swiatek I.", n=10, surface="Clay")
    print(recent[["date", "tourney", "opponent", "won", "p_1stServeIn_pct",
                  "p_1stServeWon_pct", "p_bpSaved_pct"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("EXAMPLE 2 – Player stats for Aryna Sabalenka (Hard)")
    print("=" * 60)
    stats = fetcher.get_player_stats("Sabalenka A.", surface="Hard")
    for k, v in stats.items():
        print(f"  {k:35s}: {v}")

    print("\n" + "=" * 60)
    print("EXAMPLE 3 – Head-to-head: Swiatek vs Sabalenka")
    print("=" * 60)
    h2h = fetcher.get_head_to_head("Swiatek I.", "Sabalenka A.")
    print(f"  Total matches : {h2h['total_matches']}")
    print(f"  Swiatek wins  : {h2h['Swiatek I._wins']}")
    print(f"  Sabalenka wins: {h2h['Sabalenka A._wins']}")
    print("  Match history:")
    for m in h2h["matches"][-5:]:   # last 5
        print(f"    {str(m['tourney_date'])[:10]}  {m['tourney_name']:30s}"
              f"  {m['surface']:5s}  {m['winner_name']} def. {m['loser_name']}  {m['score']}")

    print("\n" + "=" * 60)
    print("EXAMPLE 4 – Pre-match feature vector (Swiatek vs Sabalenka, Clay)")
    print("=" * 60)
    features = fetcher.get_pre_match_features("Swiatek I.", "Sabalenka A.", surface="Clay")
    for k, v in features.items():
        print(f"  {k:40s}: {v}")


if __name__ == "__main__":
    main()