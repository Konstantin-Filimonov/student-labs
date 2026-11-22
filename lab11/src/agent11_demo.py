"""ЛР 11. Учебный агент SOC: только чтение вымышленных данных, многошаговый Agent Loop.

Запуск:
  python src/agent11_demo.py --self-test         проверка защитных правил и цикла (без модели)
  python src/agent11_demo.py --demo              имитация модели (НЕ LLM): completed, rounds=4, calls=3
  python src/agent11_demo.py                     реальная LLM (LM Studio / Ollama / облако), настройки в .env
  python src/agent11_demo.py --task "..." --max-rounds 2 --trace-file experiments/last_trace.jsonl
"""
import argparse
import json
import os
import sys

from agent_loop import DEFAULT_GOAL, run_agent
from demo_client import Call, DemoClient, ScriptedClient
from kb import load_poisoned_docs, make_kb
from tools import INCIDENTS, Session, default_session, dispatch


def self_test():
    snapshot = json.dumps(INCIDENTS, sort_keys=True)
    assert dispatch("delete_incident", "{}") == {"error": "forbidden_tool"}
    assert dispatch("lookup_ip", '{"ip":"not-an-ip"}') == {"error": "invalid_ip"}
    assert dispatch("lookup_ip", '{"ip":"1.2.3.4","x":1}') == {"error": "invalid_arguments"}
    assert dispatch("get_incident", "не json") == {"error": "invalid_json"}
    assert dispatch("get_incident", '{"incident_id":"INC-101"}')["data"]["severity"] == "high"
    assert dispatch("get_incident", '{"incident_id":"INC-999"}') == {"error": "not_found"}
    assert dispatch("get_incident", '{"incident_id":"INC-101"}', default_session("public")) == {"error": "permission_denied"}
    ok = run_agent(DEFAULT_GOAL, DemoClient(), "demo")
    assert (ok["status"], ok["rounds"], ok["calls"]) == ("completed", 4, 3), ok
    assert run_agent(DEFAULT_GOAL, DemoClient(), "demo", max_rounds=1)["status"] == "round_limit"
    assert run_agent(DEFAULT_GOAL, DemoClient(), "demo", max_calls=1)["status"] == "tool_budget_exceeded"
    twice = ScriptedClient([(None, [Call("a", "lookup_ip", {"ip": "192.0.2.10"})]),
                            (None, [Call("b", "lookup_ip", {"ip": "192.0.2.10"})]), ("готово", None)])
    assert [t["result"] for t in run_agent("x", twice)["trace"]] == ["ok", "repeated_call"]
    assert run_agent("x", ScriptedClient([RuntimeError("boom")]))["status"] == "api_error"
    assert json.dumps(INCIDENTS, sort_keys=True) == snapshot  # данные не изменились
    print("SELF-TEST: OK")


def build_real_client():
    from llm_client import get_base_url, get_model_name, make_client
    try:
        client = make_client(timeout=20.0)
    except ImportError:
        sys.exit("Для реальной модели нужен пакет openai: pip install -r requirements.txt")
    return client, get_model_name(), get_base_url()


def main():
    p = argparse.ArgumentParser(description="ЛР11: учебный агент SOC (Agent Loop)")
    p.add_argument("--demo", action="store_true", help="имитация модели без LLM")
    p.add_argument("--self-test", action="store_true", help="проверка защитных правил и цикла")
    p.add_argument("--task", default=DEFAULT_GOAL, help="цель агента")
    p.add_argument("--role", default="soc", choices=["soc", "public"], help="роль пользователя (права задаёт приложение)")
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--max-calls", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=75)
    p.add_argument("--kb", default=os.getenv("KB_BACKEND", "keyword"), choices=["keyword", "lab09"],
                   help="база знаний: keyword (учебная) или lab09 (RAG из ЛР9)")
    p.add_argument("--poisoned-kb", action="store_true", help="добавить документ с внедрённой инструкцией (проверка T10)")
    p.add_argument("--trace-file", help="дописать результат запуска (без секретов) в JSONL")
    args = p.parse_args()

    if args.self_test:
        return self_test()

    kb = make_kb(args.kb, load_poisoned_docs() if args.poisoned_kb else None, lab09_src=os.getenv("LAB09_SRC"))
    session = Session(role=args.role, kb=kb)
    if args.demo:
        client, model, meta = DemoClient(), "demo", {"mode": "demo"}
    else:
        client, model, base = build_real_client()
        meta = {"mode": "real", "model": model, "base_url": base}
    result = run_agent(args.task, client, model, max_rounds=args.max_rounds, max_calls=args.max_calls,
                       max_seconds=args.max_seconds, session=session)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.trace_file:
        with open(args.trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({**meta, "goal": args.task, **result}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
