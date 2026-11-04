"""Step 4: retrieve context, then ask the LLM to answer from it.

grounded=True  -> v1.1 behaviour: answer only from labels, cite chunk ids, refuse otherwise
grounded=False -> v1 baseline: plain context, no rules (kept so evals can compare)
"""
import time
from groq import Groq
from app.config import GROQ_API_KEY, GROQ_MODEL
from app.retrieve import Retriever

client = Groq(api_key=GROQ_API_KEY, max_retries=5)
retriever = Retriever()

REFUSAL = "Not found in the label."

SYSTEM_PROMPT = f"""You answer questions about FDA drug labels.
Rules:
1. Use ONLY the label excerpts provided. Never use outside knowledge.
2. After every claim, cite the excerpt id in square brackets, e.g. [metformin:warnings:0].
3. If the excerpts do not contain the answer, reply with exactly: {REFUSAL}
4. Be concise. Do not give personal medical advice."""


def build_prompt(question: str, chunks: list[dict], grounded: bool = True) -> list[dict]:
    if not grounded:
        context = "\n\n".join(c["text"] for c in chunks)
        return [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]

    # Label each chunk with its id so the model can cite it.
    context = "\n\n".join(f"[{c['id']}]\n{c['text']}" for c in chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Label excerpts:\n{context}\n\nQuestion: {question}"},
    ]


def answer(question: str, grounded: bool = True) -> dict:
    start = time.perf_counter()
    chunks = retriever.search(question)
    t_retrieve = time.perf_counter()

    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=build_prompt(question, chunks, grounded),
        temperature=0, extra_body={"reasoning_effort": "low"},
    )
    t_llm = time.perf_counter()
    text = resp.choices[0].message.content.strip()

    return {
        "answer": text,
        "refused": REFUSAL.lower() in text.lower(),
        "cited": [c["id"] for c in chunks if f"[{c['id']}]" in text],
        "sources": [{"id": c["id"], "score": round(c["score"], 3)} for c in chunks],
        "latency_ms": {
            "retrieval": round((t_retrieve - start) * 1000),
            "llm": round((t_llm - t_retrieve) * 1000),
            "total": round((t_llm - start) * 1000),
        },
    }
