# agents/ranker_faiss.py
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from typing import List

# Embeddings via local Ollama 
from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.vectorstores import FAISS



PREFS_PATH = "preferences.json"
EMBED_MODEL = "nomic-embed-text"  
INDEX_DIR = "data/faiss_index"
EMB_CACHE = "data/embeddings.npy"


def _ensure_dirs():
    os.makedirs("data", exist_ok=True)


def _load_preferences() -> dict:
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _build_pref_text(prefs: dict) -> str:
    likes = prefs.get("likes", "").strip()
    if likes:
        return likes
    # fallback to default text if likes not provided
    return "Je préfère des événements publics intéressants à Montréal."


def _prep_event_text(row: pd.Series) -> str:
    title = (row.get("title") or "").strip()
    desc = (row.get("description") or "").strip()
    snippet = (desc[:300] + ("…" if len(desc) > 300 else ""))
    return " | ".join([p for p in [title, snippet] if p])

# L2 normalize rows of a 2D numpy array
#  FAISS inner product is equivalent to cosine similarity on L2-normalized vectors
def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def _embed_ollama(texts: List[str]) -> np.ndarray:
    """
    Embeds texts using local Ollama (nomic-embed-text by default).
    Requires: ollama serve + ollama pull nomic-embed-text
    """
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    vecs = emb.embed_documents(texts)
    return np.array(vecs, dtype="float32")


def _rank_faiss(df: pd.DataFrame, texts: List[str], pref_text: str) -> pd.DataFrame:
    X = _embed_ollama(texts)
    X = _l2_normalize(X).astype("float32")

    db = FAISS.from_embeddings(list(zip(texts, X)), embedding=None)
    try:
        db.save_local(INDEX_DIR)
    except Exception:
        pass

    pvec = _embed_ollama([pref_text])
    pvec = _l2_normalize(pvec).astype("float32")

    index = db.index
    D, I = index.search(pvec, k=X.shape[0])
    scores = D[0]

    out = df.copy()
    out["score"] = scores
    out = out.sort_values("score", ascending=False).reset_index(drop=True)

    try:
        np.save(EMB_CACHE, X)
        with open("data/pref_text.txt", "w", encoding="utf-8") as f:
            f.write(pref_text + "\n")
    except Exception:
        pass

    return out


def rank(df: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
    """
    Rank events by similarity to the free-text `likes` string from preferences.json.
    Only uses title + description embeddings.
    """
    _ensure_dirs()
    if df.empty:
        df["score"] = []
        return df

    texts = df.apply(_prep_event_text, axis=1).tolist()
    prefs = _load_preferences()
    pref_text = _build_pref_text(prefs)
    out = _rank_faiss(df, texts, pref_text)


    if top_k is not None:
        out = out.head(top_k).reset_index(drop=True)

    return out
