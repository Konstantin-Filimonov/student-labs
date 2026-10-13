import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

from ingest import load_config
from retrieve import load_index
from rag import rag_answer, load_questions

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

def retrieval_metrics(run, k):
    """Recall@k и RR для одного запуска."""
    ref_ids = set(run["reference_doc_ids"])
    if not ref_ids:
        return None, None
    top_ids = [r["doc_id"] for r in run["retrieved"]]
    hit = set(top_ids) & ref_ids
    recall = len(hit) / len(ref_ids)
    rr = 0.0
    for rank, doc_id in enumerate(top_ids, start=1):
        if doc_id in ref_ids:
            rr = 1.0 / rank
            break
    return recall, rr

def has_citation(answer):
    return "[S" in answer

def run_topk_experiment(k_values=(2, 4, 8)):
    cfg = load_config()
    model = SentenceTransformer(cfg["model_name"])
    emb, texts, meta = load_index()
    questions = load_questions()

    rows = []
    for k in k_values:
        recalls, rrs, gens, rets, cites, refusals, refusal_ok = [], [], [], [], 0, 0, 0
        for q in questions:
            r = rag_answer(q["question"], model, emb, texts, meta, cfg, k=k)
            r["reference_doc_ids"] = q.get("reference_doc_ids", [])
            r["has_answer_in_corpus"] = q.get("has_answer_in_corpus", True)

            rec, rr = retrieval_metrics(r, k)
            if rec is not None:
                recalls.append(rec); rrs.append(rr)

            gens.append(r["t_generate_s"])
            rets.append(r["t_retrieve_s"])
            if has_citation(r["answer"]):
                cites += 1

            # Корректный отказ
            if not q.get("has_answer_in_corpus", True):
                refusals += 1
                if "недостаточно информации" in r["answer"].lower():
                    refusal_ok += 1

        rows.append({
            "k": k,
            "recall@k": round(sum(recalls) / len(recalls), 4) if recalls else 0,
            "mrr": round(sum(rrs) / len(rrs), 4) if rrs else 0,
            "avg_retrieve_s": round(sum(rets) / len(rets), 3),
            "avg_generate_s": round(sum(gens) / len(gens), 3),
            "citation_rate": round(cites / len(questions), 3),
            "refusal_rate": round(refusal_ok / refusals, 3) if refusals else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(df)
    return df

if __name__ == "__main__":
    run_topk_experiment()