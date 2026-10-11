import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

def load_index():
    emb = np.load(RESULTS_DIR / "embeddings.npy")
    with open(RESULTS_DIR / "chunks.json", "r", encoding="utf-8") as f:
        texts = json.load(f)
    with open(RESULTS_DIR / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return emb, texts, meta

def semantic_search(query, model, emb, texts, meta, k=5):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = emb @ q
    order = np.argsort(scores)[::-1][:k]
    return [
        {"rank": i + 1, "score": float(scores[idx]), "text": texts[idx], "meta": meta[idx]}
        for i, idx in enumerate(order)
    ]

if __name__ == "__main__":
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    emb, texts, meta = load_index()
    query = "что делать если украли пароль"
    for r in semantic_search(query, model, emb, texts, meta, k=3):
        print(r["rank"], round(r["score"], 4), r["meta"]["doc_id"], r["text"][:80])