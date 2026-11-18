"""Интерактивная демонстрация: один безопасный цикл function calling.

Запуск: py src/lab10_demo.py [--question "Покажи INC-101"] [--role soc|public] [--backend keyword] [--save-trace]
"""
import argparse
import json

from kb_search import make_kb
from llm_client import get_model_name, make_client
from tool_loop import run_once
from tools import BASE_DIR, CFG, Session

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=None)
    ap.add_argument("--role", default=CFG["default_role"], choices=list(CFG["roles"]))
    ap.add_argument("--backend", default=None, choices=["lab09", "keyword"])
    ap.add_argument("--save-trace", action="store_true", help="сохранить results/trace_demo.json")
    a = ap.parse_args()
    question = a.question or input("Вопрос к учебному помощнику: ").strip()
    session = Session(a.role, make_kb(CFG, backend=a.backend))
    out = run_once(question, make_client(CFG["llm"]["timeout_s"]), get_model_name(), session, keep_messages=a.save_trace)
    messages = out.pop("messages", None)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if messages:
        (BASE_DIR / "results" / "trace_demo.json").write_text(
            json.dumps({"question": question, "role": a.role, "messages": messages, **out}, ensure_ascii=False, indent=2),
            encoding="utf-8")
