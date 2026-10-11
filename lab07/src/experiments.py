import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from build_index import load_config, chunk_text

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
LONG_CORPUS_PATH = BASE_DIR / "data" / "long_corpus.jsonl"
QUERIES_PATH = BASE_DIR / "data" / "queries.jsonl"

def load_long_corpus(path=LONG_CORPUS_PATH):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs

def run_chunk_experiment(config, docs, queries, sizes=[300, 800]):
    model = SentenceTransformer(config["model_name"])
    results = []
    for size in sizes:
        texts, meta = [], []
        for doc in docs:
            chunks = chunk_text(doc["text"], size, int(size * 0.1))
            for i, ch in enumerate(chunks):
                texts.append(ch)
                meta.append({"doc_id": doc["id"], "chunk_id": i})
        emb = model.encode(texts, normalize_embeddings=True)

        metrics_per_query = []
        for q in queries:
            q_vec = model.encode([q["query"]], normalize_embeddings=True)[0]
            scores = emb @ q_vec
            order = np.argsort(scores)[::-1][:5]
            top_ids = [meta[i]["doc_id"] for i in order]
            relevant = set(q["relevant_doc_ids"])
            p = len(set(top_ids) & relevant) / 5
            metrics_per_query.append(p)
        results.append({
            "chunk_size": size,
            "num_chunks": len(texts),
            "avg_precision@5": float(np.mean(metrics_per_query)),
        })
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "chunk_experiment.csv", index=False)
    print(df)
    return df

if __name__ == "__main__":
    cfg = load_config()
    docs = load_long_corpus()
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = [json.loads(l) for l in f if l.strip()]
    # Оставляем только те запросы, где эталонные документы — из long_corpus (L*)
    long_ids = {d["id"] for d in docs}
    long_queries = [q for q in queries if set(q["relevant_doc_ids"]) & long_ids]
    # Если таких нет — создаём пару запросов вручную
    if not long_queries:
        long_queries = [
            {"query": "требования к паролям и блокировке учётных записей", "relevant_doc_ids": ["L01"]},
            {"query": "как проверять и хранить резервные копии", "relevant_doc_ids": ["L02"]},
            {"query": "что делать при атаке шифровальщика", "relevant_doc_ids": ["L03"]},
            {"query": "как защититься от SQL инъекций", "relevant_doc_ids": ["L04"]},
            {"query": "признаки компрометации учётной записи", "relevant_doc_ids": ["L05"]},
        ]
    run_chunk_experiment(cfg, docs, long_queries)