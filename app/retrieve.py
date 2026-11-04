"""Step 3: find the chunks most similar to a question."""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL, CHUNKS_PATH, EMBEDS_PATH, TOP_K


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.chunks = [json.loads(line) for line in open(CHUNKS_PATH)]
        self.vectors = np.load(EMBEDS_PATH)

    def search(self, question: str, k: int = TOP_K) -> list[dict]:
        q = self.model.encode([question], normalize_embeddings=True)[0]
        scores = self.vectors @ q          # cosine similarity (vectors are normalized)
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]
