# app/main.py
from __future__ import annotations

import time
from typing import Optional, Any, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.pipeline import run_pipeline

app = FastAPI(title="Montreal Events Auditor API", version="1.0.0")


# --- Pydantic request/response models ---

class PreferencesModel(BaseModel):
    hard_filters: Dict[str, Any] = {}
    likes: str = ""


class NewsletterRequest(BaseModel):
    preferences: PreferencesModel
    window_days: int = 7
    shortlist_k: int = 30
    final_n: int = 7


class NewsletterResponse(BaseModel):
    model_used: str
    latency_ms: int
    markdown: str
    usage: Dict[str, Any]  # tokens, usd (best-effort)


# --- Health check ---

@app.get("/healthz")
def healthz():
    return {"ok": True}


# --- Main endpoint: run full pipeline for given prefs ---

@app.post("/newsletter", response_model=NewsletterResponse)
def create_newsletter(req: NewsletterRequest):
    """
    Accepts user preferences + knobs, runs your full pipeline,
    and returns an English Markdown newsletter.
    """
    t0 = time.perf_counter()
    try:
        md, usage, model_used = run_pipeline(
            preferences=req.preferences.dict(),
            window_days=req.window_days,
            shortlist_k=req.shortlist_k,
            final_n=req.final_n,
        )
    except Exception as e:
        # surface a readable error to the client
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    latency_ms = int((time.perf_counter() - t0) * 1000)
    return NewsletterResponse(
        model_used=model_used,
        latency_ms=latency_ms,
        markdown=md,
        usage=usage,
    )
