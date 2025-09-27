# app/pipeline.py
from __future__ import annotations

import os
import json
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any

import pandas as pd
from dotenv import load_dotenv

# Load .env so local runs can see GROQ_API_KEY etc.
load_dotenv()

# --- Import your existing building blocks, with safe fallbacks ---

# 1) collector: must return a DataFrame with French columns (titre, description, date_debut, etc.)
try:
    from agents.collector import collect 
except Exception:
    collect = None

# 2) cleaner: keep column names French, do minimal normalizations
try:
    from agents.cleaner import clean  
except Exception:
    clean = None

# 3) ranker: uses embeddings (Ollama by default)
from agents.ranker_faiss import rank

# 4) weather enricher 
try:
    from agents.enricher_weather import enrich_weather  
except Exception:
    enrich_weather = None 

# 5) summarizer: lets the LLM pick & write the newsletter in English 
from agents.summarizer import select_events_with_llm, summarize_to_markdown 



# ---------------- helpers ----------------

def _upcoming_window(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Keep only events whose start is within [now, now+window_days)."""
    if "date_debut" not in df.columns:
        return df

    now = pd.Timestamp.now(tz="America/Toronto").tz_convert("America/Toronto")
    end = now + pd.Timedelta(days=window_days)

    s = pd.to_datetime(df["date_debut"], errors="coerce")
    # Normalize to naive for robust comparisons
    s = s.dt.tz_localize(None)
    mask = (s >= now.tz_localize(None)) & (s < end.tz_localize(None))
    return df.loc[mask].copy()


def _apply_hard_filters(df: pd.DataFrame, hard: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply your strict preferences without renaming columns.
    Expects French column names (arrondissement, type_evenement, emplacement, public_cible, cout).
    Only filters that exist in the DF are applied.
    """
    out = df.copy()
    # Arrondissements
    allow_boro = hard.get("arrondissement_allow") or []
    if allow_boro and "arrondissement" in out.columns:
        out = out[out["arrondissement"].isin(allow_boro)]

    # Event types
    allow_types = hard.get("type_evenement_allow") or []
    if allow_types and "type_evenement" in out.columns:
        out = out[out["type_evenement"].isin(allow_types)]

    # Exclude emplacement values (e.g., "en ligne")
    excl_emp = hard.get("emplacement_exclude") or []
    if excl_emp and "emplacement" in out.columns:
        out = out[~out["emplacement"].isin(excl_emp)]

    # Audience allow
    allow_aud = hard.get("audience_allow") or []
    if allow_aud and "public_cible" in out.columns:
        out = out[out["public_cible"].isin(allow_aud)]

    # Max price (best-effort: treat 'gratuit' as 0)
    max_price = hard.get("max_price")
    if max_price is not None and "cout" in out.columns:
        def _price_ok(v):
            if pd.isna(v):
                return True
            txt = str(v).strip().lower()
            if "gratuit" in txt or "free" in txt:
                return True
            # try to parse a number if present
            num = None
            for tok in txt.replace("$", " ").replace(",", " ").split():
                try:
                    num = float(tok)
                    break
                except Exception:
                    continue
            return (num is None) or (num <= float(max_price))
        out = out[out["cout"].apply(_price_ok)]

    return out


def _write_temp_prefs(preferences: dict) -> str:
    """
    Write the incoming prefs to a temp file.
    """
    os.makedirs("data", exist_ok=True)
    path = "data/_prefs_api.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preferences, f, ensure_ascii=False, indent=2)
    return path


# --------------- main orchestrator ----------------

def run_pipeline(
    preferences: dict,
    window_days: int = 7,
    shortlist_k: int = 30,
    final_n: int = 7,
) -> Tuple[str, Dict[str, Any], str]:
    """
    Orchestrates:
      1) collect -> clean -> window filter
      2) hard filters (from preferences.hard_filters)
      3) rank by "likes" embeddings
      4) shortlist, weather enrich (optional)
      5) LLM select + English newsletter
    Returns: (markdown, usage_metadata, model_used)
    """
    # 1) Collect
    if collect is None:
        raise RuntimeError("agents.collector.collect not found. ")
    df_raw, run_iso = collect()

    # 2) Minimal clean (optional)
    if clean:
        df = clean(df_raw)
    else:
        df = df_raw.copy()

    # 3) Upcoming window
    df = _upcoming_window(df, window_days=window_days)

    # 4) Hard filters
    hard = (preferences or {}).get("hard_filters", {})
    df = _apply_hard_filters(df, hard)

    if df.empty:
        md = f"# Montréal Events — Week of {run_iso[:10] if run_iso else ''}\n\n_No events matched your filters this week._\n"
        return md, {"prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}, os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # 5) Rank by "likes" string using your existing ranker (Ollama or fastembed)
    df_ranked = rank(df)  # adds "score" and sorts

    # 6) Shortlist
    df_short = df_ranked.head(shortlist_k).reset_index(drop=True)

    # 7) Weather enrichment (if available)
    if enrich_weather:
        try:
            df_short = enrich_weather(df_short)
        except Exception as e:
            print("[pipeline] weather enrichment failed:", repr(e))

    # 8) LLM selection + newsletter (English sections)
    prefs_path = _write_temp_prefs(preferences)
    chosen = select_events_with_llm(df_short, prefs_path="preferences.json", final_n=final_n)
    if chosen and isinstance(chosen, list):
        df_top = df_short[df_short["url_fiche"].isin(chosen)].copy() 
    else:
        print("   - LLM selection failed to parse; falling back to top-N of shortlist")
        df_top = df_short.head(final_n).copy()
    md, usage = summarize_to_markdown(
        df_top,
       run_iso or datetime.utcnow().isoformat()
    )
    
    # Best-effort usage/model metadata (LangChain’s ChatGroq exposes response_metadata on responses you create,
    # but here we don’t have it directly. We still return model name for transparency.)
    #usage = {"prompt_tokens": None, "completion_tokens": None, "usd": None}
    model_used = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    return md, usage, model_used
