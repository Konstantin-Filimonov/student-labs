"""ЛР № 13. Подсчёт показателей GEO-аудита из ручной разметки.

Запуск из корня lab13:
  python scripts/geo_audit.py data/responses.csv

Результат: results/summary.csv. Только стандартная библиотека Python.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

REQUIRED = {
    "query_id", "intent", "system", "run", "status",
    "brand_mentioned", "brand_cited", "claims_true",
    "claims_false", "claims_unknown",
}

ROOT = Path(__file__).resolve().parent.parent


def read_rows(path: Path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or REQUIRED - set(reader.fieldnames):
            missing = sorted(REQUIRED - set(reader.fieldnames or []))
            raise ValueError("Нет столбцов: " + ", ".join(missing))
        rows = list(reader)
    if not rows:
        raise ValueError("Таблица пуста")
    for idx, row in enumerate(rows, start=2):
        if row["status"] not in {"ok", "error", "unavailable"}:
            raise ValueError(f"Строка {idx}: неверный status")
        if row["status"] == "ok":
            for flag in ("brand_mentioned", "brand_cited"):
                if row[flag] not in {"0", "1"}:
                    raise ValueError(f"Строка {idx}: {flag} должен быть 0 или 1")
            for field in ("claims_true", "claims_false", "claims_unknown"):
                if not str(row[field]).isdigit():
                    raise ValueError(f"Строка {idx}: {field} не число")
    return rows


def safe_pct(numerator, denominator):
    return round(100 * numerator / denominator, 1) if denominator else "н/д"


def calculate(rows):
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["query_id"], row["system"], row["run"])
        if key in seen:
            raise ValueError(f"Повтор уникального измерения: {key}")
        seen.add(key)
        groups[(row["system"], row["intent"])].append(row)
        groups[(row["system"], "ИТОГО")].append(row)

    result = []
    for (system, intent), block in sorted(groups.items()):
        ok = [r for r in block if r["status"] == "ok"]
        counts = {
            c: sum(int(r[c]) for r in ok)
            for c in ("claims_true", "claims_false", "claims_unknown")
        }
        inspected = sum(counts.values())
        result.append({
            "system": system,
            "intent": intent,
            "attempts": len(block),
            "answered": len(ok),
            "completion_%": safe_pct(len(ok), len(block)),
            "mention_%": safe_pct(sum(int(r["brand_mentioned"]) for r in ok), len(ok)),
            "own_citation_%": safe_pct(sum(int(r["brand_cited"]) for r in ok), len(ok)),
            "verified_claim_%": safe_pct(counts["claims_true"], inspected),
            "false_claim_%": safe_pct(counts["claims_false"], inspected),
        })
    return result


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "responses.csv")
    if not src.is_absolute():
        src = (Path.cwd() / src).resolve()
    summary = calculate(read_rows(src))
    out = ROOT / "results" / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("=== ИТОГО по системам ===")
    for row in summary:
        if row["intent"] == "ИТОГО":
            print(
                f"  {row['system']}: attempts={row['attempts']} answered={row['answered']} "
                f"completion={row['completion_%']}% mention={row['mention_%']}% "
                f"own_cite={row['own_citation_%']}% verified={row['verified_claim_%']}% "
                f"false={row['false_claim_%']}%"
            )
    print(f"Файл записан: {out}")
    print("Интерпретируйте брендовые и небрендовые сегменты отдельно!")


if __name__ == "__main__":
    main()
