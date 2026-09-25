# ChronoRAG

Temporal-aware retrieval for investigating synthetic security logs. ChronoRAG combines semantic search with a hard time-window filter, then re-ranks events near a **confirmed** incident. A Groq-hosted chat model turns a question into a search plan and can produce an answer with log-number citations.

This is a working **prototype**, not a complete SOC platform. The current Qdrant collection contains dense vectors only; hybrid dense/sparse search, ClickHouse, Kafka/Logstash streaming, and automated event-sequence analysis are planned rather than implemented.

## Dataset and current index

The project uses **SIM-001: Living-off-the-Land Multi-User Pivot** from the [Enterprise Attack Simulator Benchmark](https://github.com/gregdiy/cyber_simulation) by `gregdiy`. It is a synthetic security-log dataset. The upstream SIM-001 scenario has 7,920,291 logs across 25 days; this prototype currently loads one local JSONL day file, `20251221.json`.

| Scope | Log count |
| --- | ---: |
| Full upstream SIM-001 scenario | 7,920,291 |
| Local `20251221.json` file | 323,182 |
| Stored in Qdrant collection `sim001_logs_bge_base_en` | **63,840** |

The Qdrant figure is an **exact point count checked on 2026-09-25 at 15:32 UTC**. It is a snapshot, not a live badge; it will change if ingestion resumes. One JSONL record becomes one Qdrant point. Only indexed records can appear in retrieval results, so a missing result is not evidence that an event is absent from the full dataset.

The dataset file, API keys, and ingestion checkpoint are **not committed** to this repository.

## Current pipeline

```text
SIM-001 JSONL → one Document per log → UTC timestamp metadata
              → BAAI/bge-base-en embedding on local CUDA GPU
              → Qdrant dense-vector collection

User question → Groq structured query plan → Qdrant datetime filter
              → semantic candidates → optional temporal re-ranking
              → Groq answer with [log #N] citations
```

`processing/data_processing.ipynb` loads the JSONL file with `JSONLoader(json_lines=True)`, normalizes timestamps to UTC, and uploads batches to Qdrant using deterministic IDs and a resume checkpoint. Its parser **assumes timezone-free source timestamps are already UTC**; verify that assumption before using a different dataset.

`processing/query_pipeline.py` validates the model's time range, builds a Qdrant filter on `metadata.timestamp`, retrieves matching logs, and applies time decay only when an incident timestamp has been confirmed. An ordinary user-specified range keeps Qdrant's similarity ranking. The answer step sends at most eight retrieved log texts (plus a confirmed incident log, if applicable) to Groq **only after an explicit `YES` prompt**. It checks that the generated answer cites only supplied log numbers.

## Run locally

1. Create a Python environment and install the notebook dependencies:

   ```powershell
   pip install jupyter python-dotenv langchain-community langchain-huggingface langchain-qdrant langchain-groq qdrant-client jq sentence-transformers torch pydantic
   ```

   The notebook currently sets `device="cuda"`; a CUDA-capable GPU and compatible PyTorch installation are needed for that setting.

2. Obtain the SIM-001 JSON data from the [upstream project](https://github.com/gregdiy/cyber_simulation#quick-start), extract the daily JSONL file, and update `file_path` in the notebook. The current notebook also contains Windows-specific absolute paths for `processing/.env`, the checkpoint, and the import directory; change those paths for your machine.

3. Create `processing/.env` (do not commit it):

   ```dotenv
   QDRANT_URL=https://your-cluster-url
   QDRANT_API_KEY=your-qdrant-key
   GROQ_API_KEY=your-groq-key
   QDRANT_COLLECTION=sim001_logs_bge_base_en
   ```

   The current code also accepts `GRPQ_API_KEY` as a compatibility fallback for an earlier variable-name typo.

4. Open `processing/data_processing.ipynb`. To ingest, run the timestamp parser, JSONL loader, and Qdrant ingestion cells in order; the ingestion cell resumes from its local checkpoint. To **query an existing collection without starting ingestion**, skip that ingestion cell and run the separate Groq/Qdrant connection cell, then the question-input cell. For a “before an incident” question, inspect the proposed incident logs and run the confirmation cell only after verifying the actual incident event.

   Example: `Show unusual network activity on 2025-12-21 between 13:00 and 14:00 UTC.` A time such as “3:00 AM” without a date does not define an unambiguous search window and is rejected.

## Limitations and safety

- Current search is **dense-only**. Exact IPs, error codes, and other identifiers may need keyword/sparse retrieval later.
- Temporal closeness is a ranking heuristic, **not proof of causation**. The cited answer should distinguish “before” from “because of.”
- Incident candidates are semantic matches, **not automatically verified incidents**. Human confirmation is required before an incident-anchored search.
- Answers cover only the Qdrant subset currently indexed. The original SIM-001 data includes additional days and logs not loaded here.
- Retrieved log text may contain IPs, hostnames, or accounts. Review the notebook's Groq-sharing prompt before typing `YES`; never commit `.env` or saved notebook outputs containing log data.

Dataset attribution: [gregdiy/cyber_simulation](https://github.com/gregdiy/cyber_simulation). This repository contains the prototype code, not a redistributed copy of SIM-001.
