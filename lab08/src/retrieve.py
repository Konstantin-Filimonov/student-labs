import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

def load_index():
    emb = np.load(RESULTS_DIR / "embeddings.npy")
    texts = json.loads((RESULTS_DIR / "chunks.json").read_text(encoding="utf-8"))
    meta = json.loads((RESULTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    return emb, texts, meta

def retrieve(query, model, emb, texts, meta, k=4, user_group=None):
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = emb @ q
    order = np.argsort(scores)[::-1]
    results = []
    for idx in order:
        m = meta[idx]
        # Фильтр по правам доступа (ИБ-мера из методички, п. 6)
        if user_group is not None and m["access_group"] not in (user_group, "public"):
            continue
        results.append({
            "chunk_id": m["chunk_id"],
            "doc_id": m["doc_id"],
            "source": m["source"],
            "category": m["category"],
            "access_group": m["access_group"],
            "text": texts[idx],
            "score": float(scores[idx]),
        })
        if len(results) >= k:
            break
    return results