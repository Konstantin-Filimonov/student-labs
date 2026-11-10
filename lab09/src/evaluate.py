"""Эксперименты по качеству поиска: Hit@k, Recall@k, MRR на уровне doc_id, контекст, задержка.

Запуск: py src/evaluate.py [--user-group soc]
Результаты: results/summary.csv, results/search_runs.jsonl, results/run_meta.json
Вопросы без ответа в Hit/Recall/MRR не входят (их проверяет answers.py).
"""
import argparse
import json
import platform
import sys
from datetime import datetime
from importlib import metadata

import pandas as pd

from ingest import (RESULTS_DIR, build_index, corpus_fingerprint, get_encoder, load_config,
                    load_corpus, load_questions, validate_data)
from rerank import build_reranker
from retrieve import retrieve


def unique_doc_ids(results):
    """Для документных метрик остаётся первое появление каждого doc_id."""
    return list(dict.fromkeys(r["doc_id"] for r in results))


def question_metrics(results, gold_ids):
    ids = unique_doc_ids(results)
    gold = set(gold_ids)
    found = gold & set(ids)
    first = next((i for i, d in enumerate(ids, 1) if d in gold), None)
    return {"hit": int(bool(found)),
            "recall": len(found) / len(gold),
            "rr": 1.0 / first if first is not None else 0.0}


def evaluate_experiment(exp, docs, questions, model, cfg, index_cache=None, user_group=None, reranker=None):
    """Один эксперимент -> (строка сводки, подробные записи по каждому вопросу)."""
    index_cache = index_cache if index_cache is not None else {}
    key = (exp["chunk_size"], exp["overlap"])
    if key not in index_cache:
        index_cache[key] = build_index(docs, model, exp["chunk_size"], exp["overlap"], cfg.get("lexical"))
    index = index_cache[key]
    if reranker is None:
        reranker = build_reranker(exp.get("rerank", "none"), cfg)
    use_rerank = exp.get("rerank", "none") != "none"

    runs, ans_rows, top1_ans, top1_no = [], [], [], []
    for q in questions:
        r = retrieve(q["question"], model, index, exp, cfg["rrf_c"], user_group, reranker)
        res, cand = r["results"], r["candidates"]
        top1 = cand[0]["score_dense"] if cand else 0.0
        rec = {"experiment_id": exp["id"], "question_id": q["id"], "answerable": q["answerable"],
               "gold_doc_ids": q["gold_doc_ids"],
               "candidate_doc_ids": unique_doc_ids(cand) if use_rerank else None,
               "retrieved": [{"rank": x["rank"], "chunk_id": x["chunk_id"], "doc_id": x["doc_id"],
                              "score": round(x["score"], 4)} for x in res],
               "context_chars": sum(len(x["text"]) for x in res),
               "unique_docs": len(unique_doc_ids(res)),
               "top1_dense_score": round(top1, 4),
               "t_search_ms": round(r["t_search_s"] * 1000, 3),
               "t_rerank_ms": round(r["t_rerank_s"] * 1000, 3)}
        if q["answerable"]:
            m = question_metrics(res, q["gold_doc_ids"])
            rec.update(hit=m["hit"], recall=round(m["recall"], 4), rr=round(m["rr"], 4))
            if use_rerank:
                rec["hit_candidates"] = question_metrics(cand, q["gold_doc_ids"])["hit"]
            ans_rows.append(rec)
            top1_ans.append(top1)
        else:
            top1_no.append(top1)
        runs.append(rec)

    df = pd.DataFrame(ans_rows)
    row = {"experiment_id": exp["id"], "factor": exp.get("factor", ""),
           "chunk_size": exp["chunk_size"], "overlap": exp["overlap"],
           "search_mode": exp["search_mode"], "rerank": exp.get("rerank", "none"),
           "candidate_n": exp["candidate_n"] if use_rerank else "", "top_k": exp["top_k"],
           "n_chunks": len(index.chunks), "n_answerable": len(df),
           "hit_at_k": round(df.hit.mean(), 4), "recall_at_k": round(df.recall.mean(), 4),
           "mrr": round(df.rr.mean(), 4),
           "hit_at_candidates": round(df.hit_candidates.mean(), 4) if use_rerank else "",
           "avg_context_chars": round(df.context_chars.mean(), 1),
           "avg_unique_docs": round(df.unique_docs.mean(), 2),
           "avg_t_search_ms": round(sum(r["t_search_ms"] for r in runs) / len(runs), 2),
           "avg_t_rerank_ms": round(sum(r["t_rerank_ms"] for r in runs) / len(runs), 2),
           "top1_dense_answerable": round(sum(top1_ans) / len(top1_ans), 4),
           "top1_dense_unanswerable": round(sum(top1_no) / len(top1_no), 4) if top1_no else ""}
    return row, runs


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "n/a"


def run_all(user_group=None):
    cfg = load_config()
    docs, questions = load_corpus(), load_questions()
    validate_data(docs, questions)
    model = get_encoder(cfg["model_name"])
    index_cache, rows, all_runs = {}, [], []
    for exp in cfg["experiments"]:
        if exp.get("rerank") == "cross_encoder" and not cfg.get("reranker_model"):
            print(f"[пропуск] {exp['id']}: reranker_model не задан (расширенное задание)")
            continue
        row, runs = evaluate_experiment(exp, docs, questions, model, cfg, index_cache, user_group)
        rows.append(row)
        all_runs.extend(runs)
        print(f"{exp['id']:<32} Hit@k={row['hit_at_k']:.3f} MRR={row['mrr']:.3f} "
              f"ctx={row['avg_context_chars']:.0f} chunks={row['n_chunks']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    with open(RESULTS_DIR / "search_runs.jsonl", "w", encoding="utf-8") as f:
        for r in all_runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"date": datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "platform": platform.platform(), "embedding_model": cfg["model_name"],
            "reranker_model": cfg.get("reranker_model"), "corpus_version": cfg["corpus_version"],
            "data_sha256_12": corpus_fingerprint(), "n_docs": len(docs), "n_questions": len(questions),
            "user_group": user_group, "rrf_c": cfg["rrf_c"],
            "versions": {p: _version(p) for p in ("sentence-transformers", "scikit-learn",
                                                  "numpy", "pandas", "torch")}}
    (RESULTS_DIR / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(summary[["experiment_id", "n_chunks", "hit_at_k", "recall_at_k", "mrr",
                   "avg_context_chars", "avg_unique_docs"]].to_string(index=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-group", default=None, help="роль пользователя для ACL-фильтра (по умолчанию без фильтра, как в ЛР8)")
    run_all(ap.parse_args().user_group)
