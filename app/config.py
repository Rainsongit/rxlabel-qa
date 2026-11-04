import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNKS_PATH = "data/chunks.jsonl"
EMBEDS_PATH = "data/embeddings.npy"

# Label sections we index. Each becomes one or more chunks.
SECTIONS = [
    "boxed_warning", "indications_and_usage", "dosage_and_administration",
    "contraindications", "warnings_and_cautions", "warnings",
    "adverse_reactions", "drug_interactions",
]

# Start small. Add more drugs in a later version.
DRUGS = ["metformin", "lisinopril", "atorvastatin", "amoxicillin", "sertraline",
         "ibuprofen", "omeprazole", "levothyroxine", "amlodipine", "warfarin"]

CHUNK_WORDS = 200
TOP_K = 4
