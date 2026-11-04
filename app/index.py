"""Step 2: turn every chunk into an embedding vector and save them."""
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL, CHUNKS_PATH, EMBEDS_PATH


def main():
    chunks = [json.loads(line) for line in open(CHUNKS_PATH)]
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode([c["text"] for c in chunks],
                           normalize_embeddings=True, show_progress_bar=True)
    np.save(EMBEDS_PATH, vectors)
    print(f"saved {vectors.shape} embeddings to {EMBEDS_PATH}")


if __name__ == "__main__":
    main()
