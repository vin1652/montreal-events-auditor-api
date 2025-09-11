## Montréal Events Auditor 

Automated weekly pipeline that fetches Montréal public events, applies hard filters, ranks with a free-text “likes” prompt (via local embeddings with Ollama), enriches with weather, and generates a concise English TL;DR newsletter: reports/weekly_tldr.md.

Runs on GitHub-hosted runners (no self-host needed)

Uses free tools: Ollama embeddings (nomic-embed-text), Groq (free tier) for summarization, GitHub Actions

Keeps only the latest newsletter (overwrites each week)

## What it does

Collect: downloads the Événements publics dataset from Données Montréal

Clean (minimal): adds a few alias columns (e.g., title, start_datetime)—original French fields are kept unchanged

Filter: keeps the next N days (default 7) and applies hard_filters from preferences.json

Rank: uses Ollama embeddings to rank each event’s title + description against your free-text likes prompt

Borough boost: prefers arrondissements in your specified order

Enrich: adds approximate temp & rain probability for Top-N

Summarize: LLM makes a short English Markdown TL;DR

Publish: writes reports/weekly_tldr.md and commits it to the repo

## 🗂️ Project structure
``` bash
montreal-events-auditor/
├─ agents/
│  ├─ collector.py          # fetch dataset
│  ├─ cleaner.py            # minimal aliases 
│  ├─ enricher_weather.py   # Open-Meteo approx weather for Top-N
│  ├─ ranker_faiss.py       # Ollama embeddings + (optional) FAISS
│  └─ summarizer.py         # Groq LLM → English Markdown TL;DR
├─ graph/
│  └─ weekly_flow.py        # orchestrates the full run
├─ data/                    # cache (embeddings, last_run.json)
├─ reports/                 # weekly_tldr.md (overwritten weekly)
├─ preferences.json         # your filters + likes text
├─ requirements.txt
└─ .github/workflows/weekly.yml   # scheduled workflow
```
## ⚙️ Configure preferences

preferences.json separates hard filters (strict gates) from a natural-language likes string used for ranking:
``` bash
{
  "hard_filters": {
    "audience_allow": ["Famille", "Adultes", "Pour tous"],
    "exclude_children": true,
    "emplacement_exclude": ["en ligne"],
    "type_evenement_allow": [
      "Marché", "Parcours et visite guidée", "Portes ouvertes", "Mise en forme",
      "Atelier et cours", "Fête", "Jeux", "Film", "Musique", "Humour", "Danse",
      "Cirque", "Exposition permanente", "Littérature", "Club", "Théâtre",
      "Exposition temporaire"
    ],
    "arrondissement_allow": [
      "Ville-Marie",
      "Outremont",
      "Le Plateau-Mont-Royal",
      "Côte-des-Neiges–Notre-Dame-de-Grâce",
      "Verdun",
      "Saint-Laurent",
      "Lachine",
      "Mercier–Hochelaga-Maisonneuve",
      "LaSalle"
    ],
    "max_price": 50,
    "free_only": false
  },

  "likes": "Je préfère les événements en soirée ou le week-end, surtout musique, film, expositions et marchés, dans ou près de Ville-Marie et du Plateau. Ambiance conviviale, idéalement pas trop bondée."
}
```

Hard filters are applied before ranking (French labels match the dataset).

likes is free text (French or English); the ranker embeds this to sort the remaining events.

```bash
# 1) deps
pip install -r requirements.txt

# 2) start Ollama & pull embeddings model
ollama serve &
ollama pull nomic-embed-text

# 3) run the weekly flow
python -m graph.weekly_flow

# Result: reports/weekly_tldr.md

```

The summarizer uses Groq (free tier). Set GROQ_API_KEY in your shell if you want the polished English TL;DR; otherwise a basic fallback is used.

## ☁️ Deploy on GitHub Actions

This repo includes .github/workflows/weekly.yml. It:

runs every Thursday 23:30 America/Toronto (03:30 UTC Friday)

installs Python deps

installs & starts Ollama, pulls nomic-embed-text (cached)

runs the pipeline

overwrites reports/weekly_tldr.md

commits to the target branch

uploads the newsletter as an artifact

Set secret for Groq (optional but recommended)

Repo → Settings → Secrets and variables → Actions:

GROQ_API_KEY: your Groq API key

Change target branch or schedule

Open .github/workflows/weekly.yml and edit:

env:
  BRANCH: "main"     # change to another branch if desired
...
schedule:
  - cron: "30 3 * * FRI"  # adjust UTC schedule here

## 🔧 Tunables (env vars)

Set in the workflow or locally:

TOP_N (default 5): how many events to keep/enrich/summarize

WINDOW_DAYS (default 7): lookahead window

EMB_WEIGHT / BOROUGH_WEIGHT (defaults 0.7 / 0.3): blend between semantic match and arrondissement preference order

## 🪪 Requirements

requirements.txt (pinned to avoid resolver issues on runners):

``` bash
langchain>=0.2.14
langchain-community>=0.2.14
langchain-groq>=0.1.5
requests>=2.32.0
pandas>=2.2.2
numpy<2
python-dateutil>=2.9.0.post0
jinja2>=3.1.4
python-dotenv>=1.0.1
matplotlib>=3.8.4
faiss-cpu
```
## 🧵 How ranking works (short)

Each event → text: "{title} | {first 300 chars of description}"

likes → a single free-text string from preferences.json

Both are embedded via Ollama (nomic-embed-text) → cosine similarity

Borough boost: events in earlier arrondissements from arrondissement_allow get a higher borough_pref and a weighted combined_score from which we get a shortlist of activities

These activities (TOP_K) are fed to the LLM and the LLM acts as the final decision maker and filters the TOP_N events for me to attend!

## 🧹 Repo hygiene

Only one newsletter file is kept: reports/weekly_tldr.md

Workflow deletes old files before each run

Commits back to the branch set in BRANCH (default main)

## 🆘 Troubleshooting

Git push rejected (“fetch first”)
→ Workflow does:

git fetch origin $BRANCH
git pull --rebase --autostash origin $BRANCH


before committing/pushing.

Unstaged changes / rebase failed
→ The --autostash option temporarily stashes the generated report during rebase, then reapplies it.

## Cancel/disable the schedule

Stop a running job: Actions → the run → Cancel workflow

Pause future runs: Actions → the workflow → Disable workflow

Remove entirely: delete .github/workflows/weekly.yml
