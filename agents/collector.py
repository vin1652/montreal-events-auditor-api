import requests
import pandas as pd
import datetime as dt
import json
import os

# CKAN API endpoint for Montreal Open Data
CKAN = "https://donnees.montreal.ca/api/3/action"
DATASET_QUERY = "evenements publics"  # dataset title

def get_resource_url():
    """Find the current resource URL for the public events dataset"""
    resp = requests.get(f"{CKAN}/package_search", params={"q": DATASET_QUERY})
    resp.raise_for_status()
    pkg = resp.json()

    resources = pkg["result"]["results"][0]["resources"]
    for r in resources:
        if r["format"].lower() in ["csv", "json"]:
            return r["url"]
    raise RuntimeError("No CSV/JSON resource found in dataset")

def load_last_run():
    """Load timestamp of last successful run"""
    path = "data/last_run.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("last_run")
    return None

def save_last_run(ts):
    """Save timestamp of latest run"""
    os.makedirs("data", exist_ok=True)
    with open("data/last_run.json", "w") as f:
        json.dump({"last_run": ts}, f)

def collect():
    """Collect fresh events since last run"""
    url = get_resource_url()
    if url.endswith(".csv"):
        df = pd.read_csv(url)
    else:
        df = pd.read_json(url)

    last = load_last_run()
    if last:
        df = df[pd.to_datetime(df["date_debut"]) >= pd.to_datetime(last)]

    now = dt.datetime.utcnow().isoformat()
    return df, now