"""Задание 2: пять типов запросов и простые автоматические проверки ответов."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from client import LMStudioClient
from prompts import DIALOG_PROMPTS


def _extract_json(text: str):
    t = re.sub(r"```(?:json)?", "", text).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None


def auto_check(pid: int, text: str) -> Dict[str, Any]:
    """Формальные проверки (содержательную оценку даёт человек, см. отчёт)."""
    low = text.lower()
    if pid == 1:
        n = len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])
        return {"требование": "5 предложений", "найдено_предложений": n, "ok": n == 5}
    if pid == 2:
        n = len(re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+", text, flags=re.M))
        return {"требование": "3 пункта", "найдено_пунктов": n, "ok": n == 3}
    if pid == 3:
        found = [w for w in ("норма", "подозрительно", "инцидент") if w in low]
        return {"требование": "одна из категорий", "найдено": found, "ok": len(found) >= 1}
    if pid == 4:
        obj = _extract_json(text)
        need = {"risk_level", "reason", "recommended_action"}
        ok = isinstance(obj, dict) and need.issubset(obj.keys())
        return {"требование": "валидный JSON с 3 полями", "json_валиден": obj is not None, "ok": ok}
    if pid == 5:
        has_def = "def " in text
        has_re = "re." in text or "ipaddress" in text or "split" in text
        return {"требование": "функция Python", "есть_def": has_def,
                "есть_логика_проверки": has_re, "ok": has_def,
                "примечание": "код НЕ исполнялся автоматически (безопасность); проверяется вручную"}
    return {}


def run_dialog(client: LMStudioClient, temperature: float, out_dir: Path) -> List[Dict[str, Any]]:
    results = []
    md = ["# Задание 2. Пять тестовых запросов\n"]
    for item in DIALOG_PROMPTS:
        print(f"  запрос {item['id']}/5: {item['kind']}")
        res = client.chat([{"role": "user", "content": item["prompt"]}],
                          temperature=temperature, max_tokens=700)
        check = auto_check(item["id"], res.text)
        results.append({
            "id": item["id"], "kind": item["kind"], "prompt": item["prompt"],
            "answer": res.text, "time_s": round(res.total_time_s, 2),
            "completion_tokens": res.completion_tokens,
            "tokens_per_s": round(res.tokens_per_s, 2) if res.tokens_per_s else None,
            "auto_check": check,
        })
        md += [f"## Запрос {item['id']} - {item['kind']}\n",
               f"**Промпт:** {item['prompt']}\n", "**Ответ модели:**\n",
               res.text.strip() + "\n",
               f"*Время: {res.total_time_s:.2f} с, токенов: {res.completion_tokens}. "
               f"Автопроверка: {json.dumps(check, ensure_ascii=False)}*\n"]
    (out_dir / "examples.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "dialog.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return results
