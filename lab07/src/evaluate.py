import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from search import semantic_search, load_index
from baseline import build_tfidf_index, keyword_search
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
QUERIES_PATH = BASE_DIR / "data" / "queries.jsonl"

def load_queries(path=QUERIES_PATH):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries

def compute_metrics(results, relevant_ids, k):
    top_ids = [r["meta"]["doc_id"] for r in results[:k]]
    relevant_in_top = set(top_ids) & set(relevant_ids)
    precision = len(relevant_in_top) / k
    recall = len(relevant_in_top) / len(relevant_ids) if relevant_ids else 0
    first_rank = None
    for r in results:
        if r["meta"]["doc_id"] in relevant_ids:
            first_rank = r["rank"]
            break
    rr = 1.0 / first_rank if first_rank else 0.0
    return precision, recall, rr

def run_evaluation(k=5):
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    emb, texts, meta = load_index()
    queries = load_queries()

    vectorizer, tfidf_matrix = build_tfidf_index(texts)

    rows = []
    for q in queries:
        query_text = q["query"]
        relevant = q["relevant_doc_ids"]

        # semantic
        t0 = time.perf_counter()
        sem_results = semantic_search(query_text, model, emb, texts, meta, k=k)
        sem_latency = (time.perf_counter() - t0) * 1000
        p_sem, r_sem, rr_sem = compute_metrics(sem_results, relevant, k)

        # baseline
        t0 = time.perf_counter()
        base_results = keyword_search(query_text, vectorizer, tfidf_matrix, texts, meta, k=k)
        base_latency = (time.perf_counter() - t0) * 1000
        p_base, r_base, rr_base = compute_metrics(base_results, relevant, k)

        rows.append({
            "query": query_text,
            "method": "semantic",
            "precision@k": p_sem,
            "recall@k": r_sem,
            "rr": rr_sem,
            "latency_ms": sem_latency,
            "top1": sem_results[0]["meta"]["doc_id"]
        })
        rows.append({
            "query": query_text,
            "method": "tfidf",
            "precision@k": p_base,
            "recall@k": r_base,
            "rr": rr_base,
            "latency_ms": base_latency,
            "top1": base_results[0]["meta"]["doc_id"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "search_results.csv", index=False)

    agg = df.groupby("method").agg({
        "precision@k": "mean",
        "recall@k": "mean",
        "rr": "mean",
        "latency_ms": "mean"
    }).rename(columns={"rr": "mrr"})
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg.to_dict(orient="index"), f, ensure_ascii=False, indent=2)
    print(agg)
    return df, agg

if __name__ == "__main__":
    run_evaluation()