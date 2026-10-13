import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer

from ingest import load_config, load_corpus
from retrieve import load_index, retrieve
from prompt import SYSTEM_PROMPT, build_context, build_user_message
from llm import llm_chat

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
QUESTIONS_PATH = BASE_DIR / "data" / "questions.jsonl"

def rag_answer(question, model, emb, texts, meta, cfg, k=4, user_group=None):
    t0 = time.perf_counter()
    retrieved = retrieve(question, model, emb, texts, meta, k=k, user_group=user_group)
    context = build_context(retrieved)
    t_retrieve = time.perf_counter() - t0

    t0 = time.perf_counter()
    answer = llm_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, context)},
        ],
        temperature=cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
    )
    t_generate = time.perf_counter() - t0

    return {
        "question": question,
        "answer": answer,
        "retrieved": retrieved,
        "k": k,
        "t_retrieve_s": round(t_retrieve, 3),
        "t_generate_s": round(t_generate, 3),
    }

def load_questions(path=QUESTIONS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def run_all(k=4, user_group=None, out_path=None):
    cfg = load_config()
    model = SentenceTransformer(cfg["model_name"])
    emb, texts, meta = load_index()
    questions = load_questions()

    runs = []
    for q in questions:
        r = rag_answer(q["question"], model, emb, texts, meta, cfg, k=k, user_group=user_group)
        r["question_id"] = q["id"]
        r["reference_doc_ids"] = q.get("reference_doc_ids", [])
        r["has_answer_in_corpus"] = q.get("has_answer_in_corpus", True)
        runs.append(r)
        print(f"[{q['id']}] {q['question'][:60]}... → {r['answer'][:80]}")

    out = out_path or (RESULTS_DIR / f"runs_k{k}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Сохранено: {out}")
    return runs

if __name__ == "__main__":
    run_all(k=4)