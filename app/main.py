from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import answer

app = FastAPI(title="RxLabel-QA")


class Ask(BaseModel):
    question: str
    grounded: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Ask):
    return answer(body.question, body.grounded)
