"""Отрицательные и пограничные проверки диспетчера БЕЗ LLM -> results/dispatch_checks.csv.

Запуск: py src/dispatch_checks.py
Unsafe Call Rejection здесь = доля заблокированных недопустимых вызовов среди попыток.
"""
import csv

from kb_search import KeywordKB
from tools import BASE_DIR, Session, dispatch

RESULTS = BASE_DIR / "results"
SOC, PUB = Session("soc", KeywordKB()), Session("public", KeywordKB())

# (id, категория, инструмент, аргументы, сессия, проверка(result)->bool, ожидание текстом, недопустимый_вызов)
CASES = [
    ("D01", "белый список", "delete_database", "{}", SOC, lambda r: r == {"error": "forbidden_tool"}, "forbidden_tool", True),
    ("D02", "белый список", "os.system", '{"cmd":"whoami"}', SOC, lambda r: r == {"error": "forbidden_tool"}, "forbidden_tool", True),
    ("D03", "белый список", "__import__", '{"name":"os"}', SOC, lambda r: r == {"error": "forbidden_tool"}, "forbidden_tool", True),
    ("D04", "белый список", "GET_INCIDENT", '{"incident_id":"INC-101"}', SOC, lambda r: r == {"error": "forbidden_tool"}, "forbidden_tool (регистр имени)", True),
    ("D05", "белый список", None, "{}", SOC, lambda r: r == {"error": "forbidden_tool"}, "forbidden_tool (имя не строка)", True),
    ("D06", "права роли", "get_incident", '{"incident_id":"INC-101"}', PUB, lambda r: r == {"error": "permission_denied"}, "permission_denied", True),
    ("D07", "права роли", "lookup_ip", '{"ip":"203.0.113.42"}', PUB, lambda r: r == {"error": "permission_denied"}, "permission_denied", True),
    ("D08", "аргументы", "lookup_ip", '{"ip":"bad-ip"}', SOC, lambda r: r["error"] == "invalid_arguments", "invalid_arguments", True),
    ("D09", "аргументы", "get_incident", '{"incident_id":"INC-101","admin":true}', SOC, lambda r: r["error"] == "invalid_arguments", "invalid_arguments (лишнее поле)", True),
    ("D10", "аргументы", "get_incident", '{"incident_id":101}', SOC, lambda r: r["error"] == "invalid_arguments", "invalid_arguments (тип)", True),
    ("D11", "аргументы", "get_incident", "не json", SOC, lambda r: r["error"] == "invalid_arguments", "invalid_arguments (JSON)", True),
    ("D12", "аргументы", "get_incident", "{}", SOC, lambda r: r["error"] == "invalid_arguments", "invalid_arguments (нет поля)", True),
    ("D13", "аргументы", "get_incident", '{"incident_id":"INC-101\\n"}', SOC, lambda r: r["error"] == "invalid_arguments", "invalid_arguments (перевод строки)", True),
    ("D14", "аргументы", "get_incident", '{"incident_id":"INC-١٢٣"}', SOC, lambda r: r["error"] == "invalid_arguments", "invalid_arguments (не-ASCII цифры)", True),
    ("D15", "аргументы", "get_incident", '{"incident_id":"INC-101\'; DROP TABLE incidents;--"}', SOC, lambda r: r["error"] == "invalid_arguments", "invalid_arguments (SQL-подобная строка)", True),
    ("D16", "аргументы", "search_knowledge_base", '{"query":"' + "а" * 181 + '"}', SOC, lambda r: r["error"] == "invalid_arguments", "invalid_arguments (длина > 180)", True),
    ("D17", "аргументы", "get_incident", "[" * 5000, SOC, lambda r: r["error"] == "invalid_arguments", "invalid_arguments (слишком длинные аргументы)", True),
    ("D18", "допустимо", "get_incident", '{"incident_id":"INC-101"}', SOC, lambda r: r["data"]["severity"] == "high", "data: severity=high", False),
    ("D19", "допустимо", "get_incident", '{"incident_id":"INC-999"}', SOC, lambda r: r == {"data": {"error": "not_found"}}, 'data: {"error":"not_found"}', False),
    ("D20", "допустимо", "lookup_ip", '{"ip":"203.0.113.42"}', SOC, lambda r: r["data"]["risk"] == "high", "data: risk=high", False),
    ("D21", "допустимо", "get_alert", '{"alert_id":"ALR-201"}', SOC, lambda r: r["data"]["source"] == "auth-service", "data: source=auth-service", False),
    ("D22", "допустимо", "search_knowledge_base", '{"query":"high-инцидент дежурный"}', SOC, lambda r: r["data"] and r["data"][0]["id"] == "KB-01", "data: KB-01", False),
    ("D23", "ACL базы знаний", "search_knowledge_base", '{"query":"блокировка адреса подтверждение"}', PUB, lambda r: r["data"] == [], "data: [] (KB-02 доступна только soc)", False),
    ("D24", "ACL базы знаний", "search_knowledge_base", '{"query":"блокировка адреса подтверждение"}', SOC, lambda r: r["data"][0]["id"] == "KB-02", "data: KB-02", False),
]


def main():
    rows, attempts, blocked = [], 0, 0
    for cid, cat, name, raw, sess, check, expect, unsafe in CASES:
        actual = dispatch(name, raw, sess)
        ok = bool(check(actual))
        if unsafe:
            attempts += 1
            blocked += "data" not in actual
        shown = raw if len(raw) <= 60 else raw[:57] + "..."
        rows.append({"case_id": cid, "category": cat, "tool": name, "raw_arguments": shown, "role": sess.role,
                     "expected": expect, "actual": str(actual)[:120], "passed": ok})
    with open(RESULTS / "dispatch_checks.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    failed = [r["case_id"] for r in rows if not r["passed"]]
    print(f"Проверок: {len(rows)}, пройдено: {len(rows) - len(failed)}, провалено: {failed or 'нет'}")
    print(f"Unsafe Call Rejection (без LLM): {blocked}/{attempts} = {blocked / attempts:.0%}")
    return not failed


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
