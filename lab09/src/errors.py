"""Разбор ошибок поиска (раздел 6.1 методички): вопросы, где эталонный документ не найден или найден не первым.

Запуск: py src/errors.py [--experiment base_dense_600_100_k4] [--limit 2]
Использует results/search_runs.jsonl (ID фрагментов) и детерминированное разбиение корпуса,
поэтому модели не нужны. Если есть results/answer_runs.jsonl, добавляется ответ LLM.
Результат: results/error_cases.md
"""
import argparse
import json

from ingest import RESULTS_DIR, load_config, load_corpus, load_questions, make_chunks, read_jsonl


def main(experiment_id, limit):
    cfg = load_config()
    exp = next(e for e in cfg["experiments"] if e["id"] == experiment_id)
    texts = {c["chunk_id"]: c["text"] for c in make_chunks(load_corpus(), exp["chunk_size"], exp["overlap"])}
    questions = {q["id"]: q for q in load_questions()}
    runs = [r for r in read_jsonl(RESULTS_DIR / "search_runs.jsonl")
            if r["experiment_id"] == experiment_id and r["answerable"]]
    runs.sort(key=lambda r: (r["hit"], r["rr"]))
    answers = {}
    ans_path = RESULTS_DIR / "answer_runs.jsonl"
    if ans_path.exists():
        for a in read_jsonl(ans_path):
            if a["experiment_id"] == experiment_id:
                answers.setdefault(a["question_id"], a["answer"])

    lines = [f"# Разбор ошибок: {experiment_id}\n"]
    for r in [x for x in runs if x["rr"] < 1.0][:limit]:
        q = questions[r["question_id"]]
        lines += [f"## {q['id']}: {q['question']}", f"- Эталонные документы: {', '.join(q['gold_doc_ids'])}",
                  f"- Ожидаемые факты: {'; '.join(q['expected_facts'])}",
                  f"- Hit: {r['hit']}, RR: {r['rr']}", "- Найденные фрагменты:"]
        for x in r["retrieved"]:
            mark = " (эталон)" if x["doc_id"] in q["gold_doc_ids"] else ""
            lines.append(f"  - #{x['rank']} {x['chunk_id']}{mark}, score={x['score']}: {texts[x['chunk_id']][:160]}…")
        lines.append(f"- Ответ LLM: {answers.get(q['id'], '— (запустите answers.py и включите эту конфигурацию)')}")
        lines.append("- Разбор: _заполните вручную: где ошибка — поиск или генерация, помогло ли изменение параметров_\n")
    if len(lines) == 1:
        lines.append("Ошибок поиска в этой конфигурации не найдено (все RR = 1.0). Выберите другую конфигурацию.")
    (RESULTS_DIR / "error_cases.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=load_config()["baseline"])
    ap.add_argument("--limit", type=int, default=2)
    a = ap.parse_args()
    main(a.experiment, a.limit)
