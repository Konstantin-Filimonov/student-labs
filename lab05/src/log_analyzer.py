"""Индивидуальное задание: локальный анализатор журналов событий (прототип).

Использование:
    python src/log_analyzer.py                       # демо на data/sample_auth.txt
    python src/log_analyzer.py путь/к/журналу.txt    # свой журнал

Данные журнала (IP-адреса, имена пользователей, имена хостов) не покидают компьютер:
запрос уходит только на localhost.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from client import LMStudioClient, LMStudioHTTPError
from config import ROOT, load_config
from prompts import LOG_SCHEMA, LOG_SYSTEM_PROMPT

CHUNK_LINES = 60  # длинные журналы режем на части, чтобы не раздувать контекст
SEV_ORDER = {"low": 1, "medium": 2, "high": 3, "critical": 4}


def _chunks(lines: List[str], size: int):
    for i in range(0, len(lines), size):
        yield lines[i:i + size]


def _parse(text: str) -> Optional[Dict[str, Any]]:
    t = re.sub(r"```(?:json)?", "", text).strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        obj = json.loads(t[s:e + 1])
        return obj if isinstance(obj.get("findings"), list) else None
    except json.JSONDecodeError:
        return None


def analyze(client: LMStudioClient, log_text: str) -> Dict[str, Any]:
    lines = [l for l in log_text.splitlines() if l.strip()]
    findings: List[Dict[str, Any]] = []
    summaries, parse_errors, structured_used = [], 0, True
    total_time = 0.0
    tokens = 0
    for part in _chunks(lines, CHUNK_LINES):
        messages = [{"role": "system", "content": LOG_SYSTEM_PROMPT},
                    {"role": "user", "content": "Журнал:\n" + "\n".join(part)}]
        fmt = {"type": "json_schema",
               "json_schema": {"name": "log_findings", "strict": True, "schema": LOG_SCHEMA}}
        try:
            res = client.chat(messages, temperature=0.1, max_tokens=1500, response_format=fmt)
        except LMStudioHTTPError as exc:
            if exc.status != 400:
                raise
            structured_used = False  # среда не поддерживает json_schema - строгий разбор вручную
            res = client.chat(messages, temperature=0.1, max_tokens=1500)
        total_time += res.total_time_s
        tokens += res.completion_tokens or 0
        obj = _parse(res.text)
        if obj is None:
            parse_errors += 1
            continue
        findings += [f for f in obj["findings"] if isinstance(f, dict)]
        if obj.get("summary"):
            summaries.append(str(obj["summary"]))
    findings.sort(key=lambda f: -SEV_ORDER.get(str(f.get("severity")).lower(), 0))
    return {"lines": len(lines), "findings": findings, "summary": " ".join(summaries),
            "parse_errors": parse_errors, "structured_output": structured_used,
            "time_s": round(total_time, 2), "completion_tokens": tokens}


def evaluate(findings: List[Dict[str, Any]], expected_path: Path) -> Dict[str, Any]:
    """Эвристическая проверка по ключевым словам (итоговая оценка - вручную)."""
    exp = json.loads(expected_path.read_text(encoding="utf-8"))
    blobs = [(f, " ".join(str(f.get(k, "")) for k in ("category", "evidence", "reason")).lower())
             for f in findings]

    def match(item):
        return [f for f, b in blobs if all(k.lower() in b for k in item["keywords"])]

    found, missed = [], []
    for item in exp["must_find"]:
        m = match(item)
        (found if m else missed).append({"id": item["id"], "description": item["description"],
                                         "severity": m[0].get("severity") if m else None})
    false_pos = []
    for item in exp["must_not_flag"]:
        bad = [f for f in match(item) if SEV_ORDER.get(str(f.get("severity")).lower(), 0) >= 2]
        if bad:
            false_pos.append({"id": item["id"], "description": item["description"],
                              "severity": bad[0].get("severity")})
    n = len(exp["must_find"])
    return {"expected": n, "found": len(found), "recall": round(len(found) / n, 2),
            "found_items": found, "missed_items": missed, "false_positives": false_pos}


def to_markdown(path: Path, result: Dict[str, Any], ev: Optional[Dict[str, Any]]) -> str:
    md = [f"# Индивидуальное задание. Анализ журнала `{path.name}`\n",
          f"Строк в журнале: {result['lines']}, находок: {len(result['findings'])}, "
          f"время: {result['time_s']} с, JSON-схема сервера: "
          f"{'да' if result['structured_output'] else 'нет (строгий разбор)'}, "
          f"ошибок разбора: {result['parse_errors']}\n",
          "| Severity | Категория | Доказательство | Причина | Рекомендация |", "|---|---|---|---|---|"]
    for f in result["findings"]:
        cells = [str(f.get(k, "")).replace("|", "/").replace("\n", " ")
                 for k in ("severity", "category", "evidence", "reason", "recommended_action")]
        md.append("| " + " | ".join(cells) + " |")
    md.append(f"\n**Вывод модели:** {result['summary']}\n")
    if ev:
        md.append(f"**Проверка по эталону:** найдено {ev['found']} из {ev['expected']} "
                  f"(recall {ev['recall']}); ложных срабатываний: {len(ev['false_positives'])}")
        if ev["missed_items"]:
            md.append("\nПропущено: " + "; ".join(m["description"] for m in ev["missed_items"]))
        if ev["false_positives"]:
            md.append("\nЛожные: " + "; ".join(m["description"] for m in ev["false_positives"]))
    return "\n".join(md) + "\n"


def run_log_analysis(client: LMStudioClient, log_path: Path, out_dir: Path) -> Dict[str, Any]:
    result = analyze(client, log_path.read_text(encoding="utf-8"))
    expected = ROOT / "data" / "expected_findings.json"
    ev = evaluate(result["findings"], expected) if log_path.name == "sample_auth.txt" else None
    result["evaluation"] = ev
    (out_dir / "log_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    (out_dir / "log_analysis.md").write_text(to_markdown(log_path, result, ev), encoding="utf-8")
    return result


def main() -> None:
    cfg = load_config()
    client = LMStudioClient(cfg.base_url, cfg.model, cfg.timeout)
    client.model = client.resolve_model(cfg.model)
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "sample_auth.txt"
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    result = run_log_analysis(client, path, out)
    print((out / "log_analysis.md").read_text(encoding="utf-8"))
    print(f"Детали: {out / 'log_analysis.json'}")
    _ = result


if __name__ == "__main__":
    main()
