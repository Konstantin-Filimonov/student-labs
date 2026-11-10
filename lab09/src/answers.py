"""Оценка генерации (шаг 8): ответы LLM для выбранных конфигураций, повторные прогоны, шаблон для экспертной разметки.

Запуск: py src/answers.py [--experiments id1 id2 id3] [--repeats 2]
Нужен запущенный LM Studio (см. .env.example).
Результаты: results/answer_runs.jsonl (подробно) и results/answer_review.csv (для ручной оценки).
"""
import argparse
import json
import re
import sys
import time

import pandas as pd

from ingest import (RESULTS_DIR, build_index, get_encoder, load_config, load_corpus, load_questions,
                    validate_data)
from llm import get_model_name, llm_chat
from prompt import SYSTEM_PROMPT, build_context, build_user_message
from rerank import build_reranker
from retrieve import retrieve

REFUSAL_PHRASE = "недостаточно информации"
REVIEW_PATH = RESULTS_DIR / "answer_review.csv"
MANUAL_COLS = ["correctness_0_2", "grounded_yes_no", "citation_ok_yes_no", "refusal_ok_yes_no", "comment"]


def auto_flags(answer, results, gold_ids):
    cited = [int(n) for n in re.findall(r"\[S(\d+)\]", answer)]
    valid = [n for n in cited if 1 <= n <= len(results)]
    cited_docs = sorted({results[n - 1]["doc_id"] for n in valid})
    return {"has_citation": bool(cited), "all_citations_exist": len(valid) == len(cited),
            "cited_doc_ids": cited_docs,
            "cited_gold": bool(set(cited_docs) & set(gold_ids)),
            "gold_in_context": bool({r["doc_id"] for r in results} & set(gold_ids)),
            "auto_refusal": REFUSAL_PHRASE in answer.lower()}


def run(exp_ids, repeats, question_ids, user_group=None, force=False):
    if REVIEW_PATH.exists() and not force:
        sys.exit(f"{REVIEW_PATH.name} уже существует (возможно, с ручной разметкой). "
                 f"Переименуйте файл или запустите с --force.")
    cfg = load_config()
    docs, all_questions = load_corpus(), load_questions()
    validate_data(docs, all_questions)
    by_id = {q["id"]: q for q in all_questions}
    questions = [by_id[i] for i in question_ids]
    experiments = {e["id"]: e for e in cfg["experiments"]}
    model = get_encoder(cfg["model_name"])
    index_cache, runs = {}, []

    for exp_id in exp_ids:
        exp = experiments[exp_id]
        key = (exp["chunk_size"], exp["overlap"])
        if key not in index_cache:
            index_cache[key] = build_index(docs, model, *key, cfg.get("lexical"))
        reranker = build_reranker(exp.get("rerank", "none"), cfg)
        for q in questions:
            r = retrieve(q["question"], model, index_cache[key], exp, cfg["rrf_c"], user_group, reranker)
            results = r["results"]
            context = build_context(results)
            for rep in range(1, repeats + 1):
                t0 = time.perf_counter()
                try:
                    answer = llm_chat(
                        [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": build_user_message(q["question"], context)}],
                        temperature=cfg["llm"]["temperature"], max_tokens=cfg["llm"]["max_tokens"])
                except Exception as exc:  # LM Studio не запущен / нет модели
                    sys.exit(f"Ошибка обращения к LLM: {exc}\nПроверьте, что LM Studio запущен и .env заполнен.")
                t_gen = time.perf_counter() - t0
                rec = {"experiment_id": exp_id, "question_id": q["id"], "repeat": rep,
                       "question": q["question"], "answerable": q["answerable"],
                       "gold_doc_ids": q["gold_doc_ids"], "expected_facts": q["expected_facts"],
                       "answer": answer, "retrieved": [{"rank": x["rank"], "chunk_id": x["chunk_id"],
                                                       "doc_id": x["doc_id"], "text": x["text"]} for x in results],
                       "context_chars": sum(len(x["text"]) for x in results),
                       "t_search_s": round(r["t_search_s"] + r["t_rerank_s"], 4),
                       "t_generate_s": round(t_gen, 3), "prompt_version": cfg["llm"]["prompt_version"],
                       "llm_model": get_model_name(), "embedding_model": cfg["model_name"]}
                rec.update(auto_flags(answer, results, q["gold_doc_ids"]))
                runs.append(rec)
                print(f"[{exp_id}] {q['id']} #{rep}: {answer[:70].replace(chr(10), ' ')}")

    with open(RESULTS_DIR / "answer_runs.jsonl", "w", encoding="utf-8") as f:
        for rec in runs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rows = []
    for rec in runs:
        row = {k: rec[k] for k in ("experiment_id", "question_id", "repeat", "question", "answerable",
                                    "expected_facts", "answer", "gold_doc_ids", "cited_doc_ids",
                                    "has_citation", "all_citations_exist", "cited_gold", "gold_in_context",
                                    "auto_refusal", "context_chars", "t_generate_s")}
        row["expected_facts"] = "; ".join(row["expected_facts"])
        row["gold_doc_ids"] = ",".join(row["gold_doc_ids"])
        row["cited_doc_ids"] = ",".join(row["cited_doc_ids"])
        row.update({c: "" for c in MANUAL_COLS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")
    print(f"\nГотово. Заполните ручные колонки в {REVIEW_PATH.name}, затем: py src/summarize_review.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    cfg0 = load_config()["answer_eval"]
    ap.add_argument("--experiments", nargs="+", default=cfg0["experiments"])
    ap.add_argument("--repeats", type=int, default=cfg0["repeats"])
    ap.add_argument("--user-group", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.experiments, a.repeats, cfg0["question_ids"], a.user_group, a.force)
