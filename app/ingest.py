"""Step 1: download FDA labels from openFDA and split them into chunks."""
import json
import requests
from app.config import DRUGS, SECTIONS, CHUNKS_PATH, CHUNK_WORDS

URL = "https://api.fda.gov/drug/label.json"


def fetch_label(drug: str) -> dict | None:
    params = {"search": f'openfda.generic_name:"{drug}"', "limit": 1}
    r = requests.get(URL, params=params, timeout=30)
    if r.status_code != 200:
        print(f"  could not fetch {drug} ({r.status_code})")
        return None
    return r.json()["results"][0]


def chunk_text(text: str, size: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def main():
    chunks = []
    for drug in DRUGS:
        print(f"fetching {drug}")
        label = fetch_label(drug)
        if not label:
            continue
        for section in SECTIONS:
            for text in label.get(section, []):
                for i, piece in enumerate(chunk_text(text)):
                    chunks.append({
                        "id": f"{drug}:{section}:{i}",
                        "drug": drug,
                        "section": section,
                        "text": piece,
                    })
    with open(CHUNKS_PATH, "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"saved {len(chunks)} chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
