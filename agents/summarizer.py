# agents/summarizer.py
from __future__ import annotations

import os
import json
from typing import List, Optional
from datetime import datetime

import pandas as pd
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

from dotenv import load_dotenv

# ---------- Constants ----------
REPORTS_DIR = "reports"

# Load .env so local runs pick up GROQ_API_KEY (Actions uses secrets env already)
load_dotenv()

# ---------- Model helpers ----------
def _model():
    """
    Return a Groq Chat model. If GROQ_API_KEY is missing, callers should
    handle fallback behavior (we'll return None and skip LLM paths).
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    # Low-ish temperature for consistent picks + wording
    return ChatGroq(temperature=0.2, model_name=model_name, groq_api_key=api_key)


def _load_prefs_dict(prefs_path: str) -> dict:
    try:
        with open(prefs_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------- Utilities ----------
def _fmt_date(val) -> str:
    """Format a date/datetime into a short English label. No time included"""

    if pd.isna(val):
        return ""
    try:
        ts = pd.to_datetime(val)
        return ts.tz_localize("America/Toronto", nonexistent="NaT", ambiguous="NaT").strftime("%a, %b %d")
    except Exception:
        try:
            return pd.to_datetime(val).strftime("%a, %b %d")
        except Exception:
            return str(val)


def _rows_to_min_json(df: pd.DataFrame) -> List[dict]:
    """
    Convert rows to a compact JSON structure the LLM can reason over.
    Keep only useful fields; translation is handled by prompt.
    """
    out = []
    for _, r in df.iterrows():
        out.append({
            "title": ( r.get("titre") or "")[:200],
            "url": ( r.get("url_fiche") or ""),
            "borough": ( r.get("arrondissement") or ""),
            "event_type": ( r.get("type_evenement") or ""),
            "start": str( r.get("date_debut") or ""),
            "is_free": bool(r.get("is_free", False)),
            "audience": ( r.get("public_cible") or ""),
            "temp_c": None if pd.isna(r.get("temp_c")) else float(r.get("temp_c")),
            "rain_prob": None if pd.isna(r.get("rain_prob")) else float(r.get("rain_prob")),
            "desc": (r.get("description") or "")[:700],
        })
    return out


# ---------- LLM selection (decide which events to keep) ----------
def select_events_with_llm(
    df_short: pd.DataFrame,
    prefs_path: str,
    final_n: int = 5
) -> Optional[List[str]]:
    """
    Ask the LLM to pick the best `final_n` events from the shortlist.
    Returns a list of selected URLs (strings) or None on failure.
    """
    if df_short.empty:
        return []

    llm = _model()
    if llm is None:
        # No API key → caller should fallback
        return None

    prefs = _load_prefs_dict(prefs_path)
    likes = (prefs.get("likes") or "").strip()
    hard = prefs.get("hard_filters", {})
    borough_order = hard.get("arrondissement_allow", [])

    system = SystemMessage(content=(
        "You are an events concierge. Choose the best events for the user.\n"
        "Output must be JSON only (no extra prose). English only."
    ))
    human = HumanMessage(content=(
        "Task: From the shortlist, pick the best events for the user.\n"
        f"- Return ONLY JSON: {{\"selected_urls\": [\"<url1>\", \"<url2>\", ...]}}\n"
        f"- Choose exactly {final_n} items if possible; if shortlist is smaller, choose all.\n"
        "- Consider: user likes (free-text), borough preference order (earlier is better), weather (avoid heavy rain for outdoor),\n"
        "  audience, event types, price (prefer free/low-cost when appropriate), and variety across picks when possible.\n"
        "- Translate French internally if needed, but output JSON only.\n\n"
        "User likes (free text):\n"
        f"{likes}\n\n"
        "Borough preference order (earlier is better):\n"
        f"{json.dumps(borough_order, ensure_ascii=False)}\n\n"
        "Shortlist (array of events as JSON):\n"
        f"{json.dumps(_rows_to_min_json(df_short), ensure_ascii=False)}\n\n"
        "Respond with JSON ONLY like:\n"
        "{\"selected_urls\": [\"https://example.com/event1\", \"https://example.com/event2\"]}"
    ))

    try:
        resp = llm.invoke([system, human])
        text = (resp.content or "").strip()
        data = json.loads(text)
        urls = data.get("selected_urls", [])
        # Validate against shortlist
        keep_urls = set(
 (df_short["url_fiche"] if "url_fiche" in df_short.columns else pd.Series(dtype=str)).fillna("").astype(str).tolist()
        )
        urls = [u for u in urls if u in keep_urls]
        return urls
    except Exception as e:
        print("LLM selection parse failed:", repr(e))
        return None


# ---------- Newsletter generation (LLM builds sections) ----------
def _compose_newsletter_prompt(events_json: List[dict], run_iso: str) -> List:
    """
    Let the LLM create the newsletter structure (Top Picks, Free/Low-Cost, Outdoor Options),
    with flexible counts per section. English-only Markdown.
    """
    date_label = run_iso[:10] if run_iso else datetime.now().strftime("%Y-%m-%d")

    system = SystemMessage(content=(
        "You are a concise newsletter editor.\n"
        "Write in clear, accessible English only (no French or bilingual output).\n"
        "Translate any French titles/descriptions to natural English, preserving proper nouns.\n"
        "Style: scannable Markdown, short lines, neutral/helpful tone, no hype, no invented facts."
    ))

    human = HumanMessage(content=(
        f"Create a weekly Markdown newsletter for Montreal events (week of {date_label}).\n"
        "You must build the structure yourself (do not assume fixed counts):\n"
        " - Title and a 1–2 sentence intro.\n"
        " - Sections:\n"
        "   1) Top Picks\n"
        "   2) Free or Low-Cost\n"
        "   3) Outdoor Options (note temp/rain if available)\n"
        "Rules:\n"
        " - Use only the events provided below. Do not invent events.\n"
        " - Each bullet: one line with title, borough, date/time, optional price tag (free), and brief weather tag.\n"
        " - Include the event URL on the next line after each bullet.\n"
        " - Choose an appropriate number of bullets per section (typically 3–7) based on the data; avoid duplicates across sections.\n"
        " - If a section has too few qualified events, include fewer bullets rather than forcing a number.\n"
        " - English-only output. Valid Markdown. No YAML front matter.\n\n"
        "EVENTS JSON:\n"
        f"{json.dumps(events_json, ensure_ascii=False)}"
    ))
    return [system, human]


def _default_intro(run_iso: str) -> str:
    date_label = run_iso[:10] if run_iso else datetime.now().strftime("%Y-%m-%d")
    return f"# Montréal Events — Week of {date_label}\n\nHere are this week’s highlights. Weather notes are approximate.\n"


def summarize_to_markdown(df_selected: pd.DataFrame, run_iso: str) -> str:
    """
    Produce the final English Markdown TL;DR for the already-selected events in df_selected.
    Lets the LLM decide section membership and counts.
    Falls back to a simple Top Picks list if LLM is unavailable.
    """
    if df_selected is None or df_selected.empty:
        return _default_intro(run_iso) + "\n_No events matched your filters this week._\n"

    ev_json = _rows_to_min_json(df_selected)
    print("[newsletter] Rows passed to LLM:", len(df_selected))
    llm = _model()
    if llm is None:
        print("[newsletter] GROQ_API_KEY missing or not detected; using fallback.")
    if llm is None:
        # Fallback: simple list without LLM
        bullets = []
        for _, r in df_selected.iterrows():
            title = (r.get("titre") or "").strip()
            url = (r.get("url_fiche") or "").strip()
            boro = ( r.get("arrondissement") or "").strip()
            etype = (r.get("type_evenement") or "").strip()
            start = _fmt_date(r.get("start_datetime"))
            tag_free = " — free" if bool(r.get("is_free", False)) else ""
            wx = []
            if pd.notna(r.get("temp_c")):
                try: wx.append(f"{float(r['temp_c']):.1f}°C")
                except: pass
            if pd.notna(r.get("rain_prob")):
                try: wx.append(f"{int(r['rain_prob'])}% rain")
                except: pass
            wx_str = (" — " + ", ".join(wx)) if wx else ""
            bullet = f"- **{title}** ({boro}) — {start} — {etype}{tag_free}{wx_str}\n  {url}"
            bullets.append(bullet)

        intro = _default_intro(run_iso)
        return intro + "\n## Top Picks\n" + "\n".join(bullets[:10]) + "\n"

    try:
        msgs = _compose_newsletter_prompt(ev_json, run_iso)
        resp = llm.invoke(msgs)
        text = resp.content or ""
        if not text.strip():
            raise ValueError("Empty LLM response")
        return text
    except Exception as e:
        print("[NEWSLETTER]Summarization failed, using fallback:", repr(e))
        # Minimal fallback
        bullets = []
        for _, r in df_selected.iterrows():
            title = (r.get("titre") or "").strip()
            url = (r.get("url_fiche") or "").strip()
            boro = (r.get("arrondissement") or "").strip()
            start = _fmt_date(r.get("start_datetime") )
            bullets.append(f"- **{title}** ({boro}) — {start}\n  {url}")
        return _default_intro(run_iso) + "\n## Top Picks\n" + "\n".join(bullets[:10]) + "\n"





# ---------- Save report ----------
def save_report(md: str, run_iso: str) -> str:
    """
    Save the Markdown to a fixed filename so the workflow overwrites weekly.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, "weekly_tldr.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path
