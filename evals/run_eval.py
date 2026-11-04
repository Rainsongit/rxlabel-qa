"""Runs every eval question twice (baseline vs grounded) and prints the numbers
that go in the README. Results are saved to evals/results.json.

Metrics
- retrieval hit rate: the right drug appears in the top-k retrieved chunks
- hallucination rate: on questions the labels can't answer, how often the model
  answers anyway instead of refusing
- false refusal rate: on answerable questions, how often it wrongly refuses
- citation rate: answered questions that cite at least one chunk id
- latency p50 / p95
"""
import json
import time
import numpy as np
from app.rag import answer

QUESTIONS = [json.loads(line) for line in open("evals/questions.jsonl")]


def pct(n, d):
    return round(100 * n / d, 1) if d else 0.0


def run(grounded: bool) -> dict:
    rows = []
    for q in QUESTIONS:
        r = answer(q["question"], grounded)
        hit = q["expected_drug"] is not None and any(
            s["id"].startswith(q["expected_drug"] + ":") for s in r["sources"])
        rows.append({**q, **r, "retrieval_hit": hit})
        time.sleep(6)  # stay under Groq free-tier rate limits

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if not r["refused"]]
    totals = [r["latency_ms"]["total"] for r in rows]

    summary = {
        "mode": "grounded" if grounded else "baseline",
        "questions": len(rows),
        "retrieval_hit_rate": pct(sum(r["retrieval_hit"] for r in ans), len(ans)),
        "hallucination_rate": pct(sum(not r["refused"] for r in unans), len(unans)),
        "false_refusal_rate": pct(sum(r["refused"] for r in ans), len(ans)),
        "citation_rate": pct(sum(bool(r["cited"]) for r in answered), len(answered)),
        "latency_p50_ms": int(np.percentile(totals, 50)),
        "latency_p95_ms": int(np.percentile(totals, 95)),
    }
    return {"summary": summary, "rows": rows}


def main():
    results = [run(grounded=False), run(grounded=True)]
    json.dump(results, open("evals/results.json", "w"), indent=2)

    keys = list(results[0]["summary"].keys())[1:]
    print(f"\n{'metric':<22}{'baseline':>12}{'grounded':>12}")
    for k in keys:
        print(f"{k:<22}{results[0]['summary'][k]:>12}{results[1]['summary'][k]:>12}")


if __name__ == "__main__":
    main()
