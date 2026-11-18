"""Эксперимент ЛР10: сценарии T1–T13 в двух режимах — (а) без инструментов, (б) с инструментами.

Запуск (LM Studio запущен, модель загружена):
    py src/run_scenarios.py [--repeats 3] [--modes no_tools tools] [--only T1 T3] [--backend keyword]
Результаты в results/: test_cases.csv, scenario_runs.jsonl, metrics.json, metrics.csv,
trace_example.json, run_meta.json.
"""
import argparse
import csv
import json
import platform
import statistics
import sys
from datetime import datetime
from importlib import metadata

from kb_search import make_kb, read_jsonl
from llm_client import get_base_url, get_model_name, make_client
from metrics import NA, aggregate, evaluate_run, load_scenarios
from tool_loop import run_no_tools, run_once
from tools import BASE_DIR, CFG, Session

RESULTS = BASE_DIR / "results"


def _ver(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "n/a"


def build_sessions(backend):
    """Две базы знаний: обычная и с «отравленным» документом (проверка prompt injection)."""
    poisoned = read_jsonl(BASE_DIR / "data" / "kb_poisoned.jsonl")
    normal_kb = make_kb(CFG, backend=backend)
    kb_by_name = {"normal": normal_kb}
    encoder = getattr(normal_kb, "model", None)  # у Lab09KB кодировщик уже загружен — переиспользуем
    kb_by_name["poisoned"] = make_kb(CFG, extra_docs=poisoned, backend=normal_kb.name, encoder=encoder)
    return kb_by_name


def main(repeats, modes, only, backend):
    scenarios = [s for s in load_scenarios(BASE_DIR / CFG["eval"]["scenarios"]) if not only or s["id"] in only]
    client, model = make_client(CFG["llm"]["timeout_s"]), get_model_name()
    try:  # этап 1 методички: проверка базовой модели
        client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Ответь одним словом: ок"}],
                                       max_tokens=8)
    except Exception as exc:
        sys.exit(f"Модель недоступна ({exc}).\nПроверьте LM Studio, модель и .env (см. .env.example).")
    kbs = build_sessions(backend)

    runs, trace_examples = [], {}
    for mode in modes:
        for scn in scenarios:
            session = Session(scn["role"], kbs[scn["kb"]])
            for rep in range(1, repeats + 1):
                try:
                    rec = (run_once(scn["query"], client, model, session, keep_messages=(rep == 1))
                           if mode == "tools" else run_no_tools(scn["query"], client, model))
                except Exception as exc:  # ошибка API фиксируется отдельно от ошибок модели/диспетчера
                    rec = {"mode": mode, "answer": "", "trace": [], "latency_s": None, "api_error": str(exc)[:200]}
                flags = evaluate_run(scn, rec, mode)
                if mode == "tools" and rep == 1 and scn["id"] in ("T1", "T3", "T5", "T9", "T12") and "messages" in rec:
                    trace_examples[scn["id"]] = {"query": scn["query"], "role": scn["role"], "messages": rec["messages"],
                                                 "trace": rec["trace"], "latency_s": rec["latency_s"]}
                rec.pop("messages", None)
                runs.append({"mode": mode, "scenario": scn["id"], "repeat": rep, "record": rec, "flags": flags})
                print(f"[{mode}] {scn['id']} #{rep}: tools={flags['called']} "
                      f"sel={flags['selection_ok']} ok={flags['completion_ok']} {rec.get('latency_s')}s")

    # ---------- test_cases.csv (таблица 5.3 методички, по режимам) ----------
    scn_by_id = {s["id"]: s for s in scenarios}
    with open(RESULTS / "test_cases.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Тест", "Режим", "Запрос", "Ожидаемое поведение", "Фактически вызванные инструменты (по повторам)",
                    "Корректный выбор", "Корректный итог", "Время, с (среднее)", "Комментарий"])
        for mode in modes:
            for scn in scenarios:
                rs = [r for r in runs if r["mode"] == mode and r["scenario"] == scn["id"]]
                called = " | ".join(",".join(r["flags"]["called"]) or "—" for r in rs)
                sel = [r["flags"]["selection_ok"] for r in rs if r["flags"]["selection_ok"] is not None]
                lat = [r["record"]["latency_s"] for r in rs if r["record"].get("latency_s") is not None]
                notes = []
                errs = sorted({t["error"] for r in rs for t in r["record"].get("trace", []) if t["error"]})
                if errs:
                    notes.append("ошибки диспетчера: " + ",".join(errs))
                if any(r["flags"]["canary_leak"] for r in rs):
                    notes.append("КАНАРЕЙКА в ответе: инъекция сработала на уровне текста")
                if any(r["flags"]["forbidden_success"] for r in rs):
                    notes.append("выполнен запрещённый для сценария вызов")
                if any(r["flags"]["api_error"] for r in rs):
                    notes.append("ошибка API")
                w.writerow([scn["id"], mode, scn["query"], scn["expect"], called,
                            f"{sum(sel)}/{len(sel)}" if sel else NA,
                            f"{sum(r['flags']['completion_ok'] for r in rs)}/{len(rs)}",
                            round(statistics.mean(lat), 2) if lat else NA, "; ".join(notes)])

    # ---------- метрики по режимам ----------
    result = {mode: aggregate([r["flags"] for r in runs if r["mode"] == mode]) for mode in modes}
    (RESULTS / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = list(next(iter(result.values())))
    with open(RESULTS / "metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + list(result))
        for k in keys:
            w.writerow([k] + [result[m][k] for m in result])
    with open(RESULTS / "scenario_runs.jsonl", "w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps({"mode": r["mode"], "scenario": r["scenario"], "repeat": r["repeat"],
                                "query": scn_by_id[r["scenario"]]["query"], **r["record"], "flags": r["flags"]},
                               ensure_ascii=False) + "\n")
    (RESULTS / "trace_example.json").write_text(json.dumps(trace_examples, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {"date": datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "platform": platform.platform(), "llm_model": model, "llm_base_url": get_base_url(), "mode": "local",
            "kb_backend": kbs["normal"].name, "lab09_experiment": getattr(kbs["normal"], "experiment_id", None),
            "repeats": repeats, "temperature": CFG["llm"]["temperature"], "max_tool_calls": CFG["limits"]["max_tool_calls"],
            "n_scenarios": len(scenarios),
            "versions": {p: _ver(p) for p in ("openai", "sentence-transformers", "scikit-learn", "numpy")}}
    (RESULTS / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=CFG["eval"]["repeats"])
    ap.add_argument("--modes", nargs="+", default=["no_tools", "tools"], choices=["no_tools", "tools"])
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--backend", default=None, choices=["lab09", "keyword"])
    a = ap.parse_args()
    main(a.repeats, a.modes, a.only, a.backend)
