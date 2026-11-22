"""Серия экспериментов: единый набор из 10 задач (data/scenarios.jsonl) при разных ограничениях числа раундов.

  python src/run_experiments.py --mode sim                         имитатор (проверка управляющей логики, НЕ качество модели)
  python src/run_experiments.py --mode real --rounds 2 6 --repeats 3   реальная LLM (LM Studio / Ollama / облако)

Результаты СЛИВАЮТСЯ в experiments/: строки одного label (например real_r2) заменяются, остальные сохраняются.
Ответы реальной модели не правятся вручную: что получено, то и записано.
"""
import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

from agent_loop import run_agent, sanitize
from demo_client import SimulatedModel
from kb import load_poisoned_docs, make_kb
from metrics import evaluate_run, load_scenarios, safety_probe, summarize
from tools import INCIDENTS, Session

BASE = Path(__file__).resolve().parent.parent
COLUMNS = ["label", "mode", "model", "max_rounds", "max_calls", "tasks", "runs", "tsr_pct", "valid_call_rate_pct",
           "tool_calls_total", "mean_calls", "mean_steps", "latency_mean_s", "latency_median_s", "unsafe_attempts",
           "unsafe_blocked", "safety_block_rate_pct", "abnormal_pct", "status_counts"]


def merge_csv(path, rows, label):
    old = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if r["label"] != label]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(old + rows)


def merge_jsonl(path, records, label):
    old = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            old = [line for line in f if line.strip() and json.loads(line).get("label") != label]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(old)
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def merge_meta(path, label, meta):
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[label] = meta
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sim", "real"], required=True)
    p.add_argument("--rounds", type=int, nargs="+", help="значения max_rounds (по умолчанию sim: 1 2 3 4 6; real: 2 6)")
    p.add_argument("--repeats", type=int, help="повторов каждой задачи (по умолчанию sim: 1; real: 3)")
    p.add_argument("--max-calls", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=75)
    p.add_argument("--kb", default=os.getenv("KB_BACKEND", "keyword"), choices=["keyword", "lab09"])
    p.add_argument("--scenarios", default=str(BASE / "data" / "scenarios.jsonl"))
    p.add_argument("--out", default=str(BASE / "experiments"))
    args = p.parse_args()

    rounds_list = args.rounds or ([1, 2, 3, 4, 6] if args.mode == "sim" else [2, 6])
    repeats = args.repeats or (1 if args.mode == "sim" else 3)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(args.scenarios)

    if args.mode == "real":
        from llm_client import get_base_url, get_model_name, make_client
        try:
            client, model, base = make_client(20.0), get_model_name(), get_base_url()
        except ImportError:
            sys.exit("Нужен пакет openai: pip install -r requirements.txt")
        versions = {}
        try:
            import openai
            versions["openai"] = openai.__version__
        except Exception:
            pass
    else:
        client, model, base, versions = SimulatedModel(gullible=True), "SimulatedModel(gullible=True)", "-", {}

    kbs = {}

    def session_for(scn):
        key = scn["kb"]
        if key not in kbs:
            kbs[key] = make_kb(args.kb, load_poisoned_docs() if key == "poisoned" else None,
                               lab09_src=os.getenv("LAB09_SRC"))
        return Session(scn["role"], kbs[key])

    snapshot = json.dumps(INCIDENTS, sort_keys=True)
    for max_rounds in rounds_list:
        label = f"{args.mode}_r{max_rounds}"
        runs, records = [], []
        print(f"== {label}: {len(scenarios)} задач x {repeats} повторов, модель {model}")
        for scn in scenarios:
            for rep in range(1, repeats + 1):
                if args.mode == "sim":
                    client = SimulatedModel(gullible=True)
                res = run_agent(scn["goal"], client, model, max_rounds=max_rounds, max_calls=args.max_calls,
                                max_seconds=args.max_seconds, session=session_for(scn))
                ok, why = evaluate_run(scn, res)
                runs.append({"scenario": scn["id"], "success": ok, **{k: res[k] for k in
                             ("status", "rounds", "calls", "latency_s", "trace")}})
                records.append({"label": label, "mode": args.mode, "scenario": scn["id"], "repeat": rep,
                                "status": res["status"], "rounds": res["rounds"], "calls": res["calls"],
                                "latency_s": res["latency_s"], "success": ok, "fail_reasons": why,
                                "trace": res["trace"], "sources": res["sources"],
                                "detail": res.get("detail"), "answer": sanitize(res.get("answer"), 700)})
                print(f"  {scn['id']} #{rep}: {res['status']:<20} rounds={res['rounds']} calls={res['calls']} "
                      f"{'OK' if ok else 'FAIL ' + '; '.join(why)}")
        assert json.dumps(INCIDENTS, sort_keys=True) == snapshot, "данные изменены — нарушение политики только чтения"
        row = {"label": label, "mode": args.mode, "model": model, "max_rounds": max_rounds,
               "max_calls": args.max_calls, **summarize(runs)}
        merge_csv(out / "metrics.csv", [row], label)
        merge_jsonl(out / "traces_sanitized.jsonl", records, label)
        merge_meta(out / "run_meta.json", label, {
            "date": datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
            "platform": platform.platform(), "mode": args.mode, "model": model, "base_url": base,
            "kb_backend": args.kb, "max_rounds": max_rounds, "max_calls": args.max_calls,
            "max_seconds": args.max_seconds, "repeats": repeats, "temperature": 0.0, "n_scenarios": len(scenarios),
            "versions": versions})
        print(f"   TSR={row['tsr_pct']}%  steps={row['mean_steps']}  calls={row['mean_calls']}  "
              f"latency={row['latency_mean_s']}s  abnormal={row['abnormal_pct']}%  {row['status_counts']}")

    blocked, total = safety_probe()
    with open(out / "safety_probe.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["probe_attacks", "blocked", "block_rate_pct"], [total, blocked, round(100 * blocked / total, 1)]])
    print(f"Safety probe (политика без модели): заблокировано {blocked}/{total}")


if __name__ == "__main__":
    main()
