``` mermaid
flowchart LR
    A["1. Collect & Clean<br/> CKAN: Événements publics"] --> B["2. Apply Fixed Filters<br/>(dates, boroughs, audience, type)"]
    B --> C["3. Embedding Ranker<br/> Ollama nomic-embed-text cosine vs. likes"]
    C --> D["4. Enrich with Weather<br/>(Open-Meteo)"]
    D --> E["5. LLM Selection Groq: choose final N<br/>Select & Summarize in Formatted Newsletter"]

    subgraph "GitHub Actions"
      A
      B
      C
      D
      E
    end
```
