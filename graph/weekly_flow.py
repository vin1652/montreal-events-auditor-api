# graph/weekly_flow.py
from __future__ import annotations
import os
import sys
import json
import re
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Agents
from agents.collector import collect, save_last_run
from agents.cleaner import clean
from agents.enricher_weather import enrich_weather
from agents.ranker_faiss import rank
from agents.summarizer import summarize_to_markdown, save_report


# ---------- General helpers ----------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _upcoming_window(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    """
    Keep only events whose start_datetime is within [today_local, today_local + days).
    Uses America/Toronto timezone (Montréal local time).
    Relies on cleaner.py having added 'start_datetime' alias.
    """
    if df.empty or "start_datetime" not in df.columns:
        return df

    tz = ZoneInfo("America/Toronto")
    now_local = datetime.now(tz)
    start_of_today = datetime(now_local.year, now_local.month, now_local.day, tzinfo=tz)
    window_start = start_of_today
    window_end = window_start + timedelta(days=days)

    s = pd.to_datetime(df["start_datetime"], errors="coerce")
    # localize (treat as Montréal local if naive)
    s = s.dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
    mask = (s >= window_start) & (s < window_end)
    return df.loc[mask].copy()


def _apply_hard_filters(df: pd.DataFrame, prefs_path: str = "preferences.json") -> pd.DataFrame:
    """
    Apply strict filters from preferences.json:
      - audience_allow / exclude_children
      - emplacement_exclude (e.g., 'en ligne')
      - type_evenement_allow
      - arrondissement_allow
      - pricing: free_only / max_price 
    Works on original French columns from the dataset.
    """
    if df.empty:
        return df

    try:
        prefs = json.load(open(prefs_path, "r", encoding="utf-8"))
        hf = prefs.get("hard_filters", {})
    except Exception:
        hf = {}

    # Audience (public_cible)
    allow_aud = set(a.lower() for a in hf.get("audience_allow", []))
    exclude_children = bool(hf.get("exclude_children", True))
    if "public_cible" in df.columns:
        aud = df["public_cible"].fillna("").str.lower()
        keep = (aud == "") | (aud.isin(allow_aud) if allow_aud else True)
        if exclude_children:
            keep = keep & ~aud.str.contains(r"\benfant|jeunesse|jeunes|ados\b", regex=True)
        df = df.loc[keep].copy()

    # Emplacement exclude (e.g., "en ligne")
    excl_emp = [e.lower() for e in hf.get("emplacement_exclude", [])]
    if "emplacement" in df.columns and excl_emp:
        emp = df["emplacement"].fillna("").str.lower()
        for token in excl_emp:
            df = df.loc[~emp.str.contains(re.escape(token))].copy()

    # Type_evenement allow
    allow_types = set(t.lower() for t in hf.get("type_evenement_allow", []))
    if "type_evenement" in df.columns and allow_types:
        t = df["type_evenement"].fillna("").str.lower()
        df = df.loc[t.isin(allow_types)].copy()

    # Arrondissement allow
    allow_arr = [a for a in hf.get("arrondissement_allow", [])]
    allow_arr_lc = set(a.lower() for a in allow_arr)
    if "arrondissement" in df.columns and allow_arr:
        arr = df["arrondissement"].fillna("")
        df = df.loc[arr.str.lower().isin(allow_arr_lc)].copy()

    # Price
    free_only = bool(hf.get("free_only", False))
    max_price = hf.get("max_price")
    if "cout" in df.columns:
        cost = df["cout"].astype(str).str.lower()
        if free_only:
            df = df.loc[cost.str.contains("gratuit", na=False)].copy()
        elif isinstance(max_price, (int, float)):
            price_num = cost.str.extract(r"(\d+[\.,]?\d*)", expand=False).str.replace(",", ".", regex=False)
            price_num = pd.to_numeric(price_num, errors="coerce")
            df = df.loc[cost.str.contains("gratuit", na=False) | (price_num <= float(max_price))].copy()

    return df


def _add_borough_preference_score(df: pd.DataFrame, prefs_path: str = "preferences.json") -> pd.DataFrame:
    """
    Create a 'borough_pref' numeric score (0..1) from the order of
    'arrondissement_allow' in hard_filters. First in list gets 1.0, last gets ~0.
    If an event's arrondissement isn't in the allow list, score=0.
    """
    try:
        prefs = json.load(open(prefs_path, "r", encoding="utf-8"))
        ordered = prefs.get("hard_filters", {}).get("arrondissement_allow", [])
    except Exception:
        ordered = []

    if not ordered or "arrondissement" not in df.columns:
        df["borough_pref"] = 0.0
        return df

    # Build rank weight: first = 1.0, last = small positive (or 0 if only 1)
    n = len(ordered)
    if n == 1:
        weights = {ordered[0].lower(): 1.0}
    else:
        # linear descending weights from 1.0 to 0.1
        weights = {name.lower(): 1.0 - (i / (n - 1)) * 0.9 for i, name in enumerate(ordered)}

    arr = df["arrondissement"].fillna("").str.lower()
    df["borough_pref"] = arr.map(weights).fillna(0.0)
    return df


def _combine_scores(df: pd.DataFrame, emb_col: str = "score") -> pd.DataFrame:
    """
    Combine embedding-based rank 'score' with borough preference 'borough_pref'
    using weights from env:
      EMB_WEIGHT (default 0.7)
      BOROUGH_WEIGHT (default 0.3)
    Produces 'combined_score' and sorts by it desc.
    """
    if df.empty:
        return df

    emb_w = _env_float("EMB_WEIGHT", 0.7)
    bor_w = _env_float("BOROUGH_WEIGHT", 0.3)
    total = max(1e-6, emb_w + bor_w)
    emb_w /= total
    bor_w /= total

    # Normalize embedding score to 0..1 to combine fairly
    s = df[emb_col].astype(float)
    s_min, s_max = float(s.min()), float(s.max())
    if s_max > s_min:
        s_norm = (s - s_min) / (s_max - s_min)
    else:
        s_norm = pd.Series([0.5] * len(df), index=df.index)  # degenerate case

    # borough_pref already 0..1
    b = df.get("borough_pref", pd.Series([0.0] * len(df), index=df.index)).astype(float)

    df["combined_score"] = emb_w * s_norm + bor_w * b
    return df.sort_values("combined_score", ascending=False).reset_index(drop=True)


# ---------- Main pipeline ----------

def run():
    """
    Fast order of operations:
      1) COLLECT
      2) CLEAN (adds minimal aliases: title, description, start_datetime, etc.)
      3) FILTER window (upcoming N days)
      4) HARD FILTERS (from preferences.json)
      5) RANK with likes (embeddings)
      6) BOROUGH BOOST (order from preferences)
      7) SELECT Top-N
      8) WEATHER for Top-N only
      9) SUMMARIZE & SAVE
    """
    TOP_N = _env_int("TOP_N", 10)
    WINDOW_DAYS = _env_int("WINDOW_DAYS", 7)

    print(" Step 1/10 — COLLECT (CKAN dataset)")
    df_raw, run_iso = collect()
    print(f"   - rows fetched: {len(df_raw)}")
    if df_raw.empty:
        md = "# Montréal — Aucun événement\n\n_Aucune donnée récupérée._\n"
        path = save_report(md, run_iso)
        print(f"✅  Wrote report: {path}")
        save_last_run(run_iso)
        return

    print(" Step 2/10 — CLEAN (minimal aliases, no heavy transforms)")
    df = clean(df_raw)
    print(f"   - rows after clean: {len(df)}")
    if df.empty:
        md = "# Montréal — Vide après nettoyage\n\n_Données récupérées mais aucune ligne exploitable._\n"
        path = save_report(md, run_iso)
        print(f"  Wrote report: {path}")
        save_last_run(run_iso)
        return

    print(f" Step 3/10 — FILTER (prochaines {WINDOW_DAYS} jours)")
    df_week = _upcoming_window(df, days=WINDOW_DAYS)
    print(f"   - rows in window: {len(df_week)}")
    if df_week.empty:
        md = f"# Montréal — Prochaines {WINDOW_DAYS} jours\n\n_Aucun événement à venir._\n"
        path = save_report(md, run_iso)
        print(f" Wrote report: {path}")
        save_last_run(run_iso)
        return

    print(" Step 4/10 — HARD FILTERS (preferences.json)")
    df_hard = _apply_hard_filters(df_week)
    print(f"   - rows after hard filters: {len(df_hard)}")
    if df_hard.empty:
        md = "# Montréal — Filtres appliqués\n\n_Aucun événement ne correspond à vos filtres._\n"
        path = save_report(md, run_iso)
        print(f"  Wrote report: {path}")
        save_last_run(run_iso)
        return

        # after df_hard is created (hard filters applied)

    print(" Step 5/10 — RANK (likes embeddings via Ollama)")
    df_ranked = rank(df_hard)
    print(f"   - ranked rows: {len(df_ranked)}")

    SHORTLIST_K = _env_int("SHORTLIST_K", 30)

    print(f" Step 6/10 — SHORTLIST top {SHORTLIST_K} for deeper reasoning")
    df_short = df_ranked.head(SHORTLIST_K).copy()
    print(f"   - shortlist rows: {len(df_short)}")

    print(" Step 7/10 — WEATHER (Open-Meteo) for shortlist")
    df_short = enrich_weather(df_short)

    print(" Step 8/10 — LLM SELECTION (choose final events best for you)")
    from agents.summarizer import select_events_with_llm  
    chosen = select_events_with_llm(df_short, prefs_path="preferences.json", final_n=_env_int("TOP_N", 10))
    if chosen and isinstance(chosen, list):
        df_top = df_short[df_short["url_fiche"].isin(chosen)].copy()  #url_fiche is the unique id for events that LLM chose
        # Fallback if LLM returned fewer than needed
        if len(df_top) < _env_int("TOP_N", 10):
            needed = _env_int("TOP_N", 10) - len(df_top)
            extras = df_short[~df_short["url_fiche"].isin(chosen)].head(needed)
            df_top = pd.concat([df_top, extras], ignore_index=True)
    else:
        print("   - LLM selection failed to parse; falling back to top-N of shortlist")
        df_top = df_short.head(_env_int("TOP_N", 10)).copy()

    print(" Step 9/10 — ORDER by combined score for display (optional)")
    # keep your borough boost if you want to preserve that ordering signal:
    df_top = _add_borough_preference_score(df_top)
    df_top = _combine_scores(df_top, emb_col="score")

    print(" Step 10/10 — SUMMARIZE & SAVE (English)")
    md = summarize_to_markdown(df_top, run_iso)
    path = save_report(md, run_iso)
    save_last_run(run_iso)
    print(f"  Done. Report: {path}")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(" Pipeline failed:", repr(e), file=sys.stderr)
        raise
