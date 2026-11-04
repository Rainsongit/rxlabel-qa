# RxLabel-QA

A production-grade RAG system that answers questions about FDA drug labels, with
citations, guardrails, caching, model routing and measured latency.

Built in small versions. Every version adds one capability and reports a
before/after number.

## Architecture

```
openFDA API → ingest.py (chunk by label section) → index.py (embeddings)
                                                         ↓
question → FastAPI /ask → retrieve top-k chunks → grounded prompt → Groq LLM
                                                         ↓
                          answer + citations + refusal flag + per-stage latency
```

## How grounding works

Each retrieved chunk is tagged with an id like `[warfarin:drug_interactions:0]`.
The system prompt tells the model to answer only from those chunks, cite an id
after every claim, and reply `Not found in the label.` when the answer isn't
there. The API reports which chunks were cited and whether the model refused.

## Evaluation

`evals/run_eval.py` runs 20 questions (15 answerable, 5 not) in two modes:
baseline (plain RAG) and grounded. It reports retrieval hit rate,
hallucination rate, false refusal rate, citation rate and p50/p95 latency.

## Quickstart

```bash
cp .env.example .env          # add your Groq key
./run_all.sh                  # install, download labels, embed, run evals
source .venv/bin/activate
uvicorn app.main:app --reload # start the API
```

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What is the boxed warning for metformin?"}'
```

## Roadmap

| Version | Feature | Metric |
|---|---|---|
| v1 ✅ | Baseline RAG + FastAPI + latency timing | p50 latency |
| v1.1 ✅ | Grounded answers with citations + refusal | hallucination rate |
| v1.2 ✅ | Eval harness (baseline vs grounded) | 5 metrics |
| v1.3 | Hybrid retrieval (BM25 + embeddings) | retrieval hit rate |
| v1.4 | Guardrails for off-topic / unsafe questions | block rate |
| v1.5 | Semantic cache | cache hit rate, latency saved |
| v1.6 | Model routing (small vs large model) | cost, latency, quality |
| v1.7 | Quantized local model comparison | speed vs quality |
| v1.8 | Docker + load test | p50 / p99 under load |

## Results

| metric | baseline | grounded |
|---|---|---|
| Hallucination rate (unanswerable questions) | 100% | 0% |
| Citation rate | 0% | 100% |
| Retrieval hit rate | 100% | 100% |
| False refusal rate | 0% | 26.7% |

Model: `openai/gpt-oss-20b` on Groq, 20 eval questions (15 answerable, 5 not).
Grounding removes hallucinations but over-refuses 4 of 15 answerable questions;
reducing that is the next version.
