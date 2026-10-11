"""Шаги 5, 6 и 9: выборка для ручной оценки, шаблон таблицы и LLM-as-a-Judge."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from config import ModelCfg, Settings
from llm_client import chat, warm_up
from parsing import clean_text, parse_judge
from prompts import (
    JUDGE_SCHEMA,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    RUBRIC_CRITERIA,
)

MANUAL_COLUMNS = ["correctness", "completeness", "instruction", "relevance", "hallucination", "comment"]


def select_sample_ids(tests: pd.DataFrame, n: int, seed: int) -> list[int]:
    """Стратифицированная выборка по difficulty с фиксированным seed.
    Выбор НЕ зависит от ответов моделей (нет «вишнёвого» отбора). Одни и те же id
    оцениваются для обеих моделей, чтобы сравнение было парным."""
    groups = {d: sorted(g["id"].tolist()) for d, g in tests.groupby("difficulty")}
    order = {"easy": 0, "medium": 1, "hard": 2}
    names = sorted(groups, key=lambda d: (order.get(d, 99), d))
    base, extra = divmod(n, len(names))
    # остаток отдаём последним группам (обычно hard) — там больше всего ошибок
    quota = {name: base + (1 if i >= len(names) - extra else 0) for i, name in enumerate(names)}
    rng = random.Random(seed)
    chosen: list[int] = []
    for name in names:
        chosen += rng.sample(groups[name], min(quota[name], len(groups[name])))
    return sorted(int(i) for i in chosen)


def build_manual_template(results: dict[str, pd.DataFrame], ids: list[int], path: Path, tests: pd.DataFrame) -> bool:
    """Создаёт manual_eval.csv. Если файл уже существует — НЕ перезаписывает (защита ручной работы)."""
    if path.exists():
        print(f"[manual] {path.name} уже существует — не перезаписываю")
        return False
    inputs = tests.set_index("id")["input"].to_dict()
    rows = []
    for row_id in ids:
        for tag in sorted(results):
            r = results[tag].set_index("id").loc[row_id]
            rows.append(
                {
                    "id": row_id,
                    "model": tag.upper(),
                    "difficulty": r["difficulty"],
                    "input": inputs[row_id],
                    "expected": r["expected"],
                    "predicted": r["predicted"],
                    "confidence": r["confidence"],
                    "reason": r["reason"] if r["reason"] else str(r["raw"])[:400],
                    **{c: "" for c in MANUAL_COLUMNS},
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[manual] создан шаблон {path.name}: {len(rows)} строк. Заполните колонки "
          f"{MANUAL_COLUMNS[:5]} (0–2 и 0/1) ДО просмотра judge_scores.csv")
    return True


def judge_one(judge: ModelCfg, event: str, answer: str, settings: Settings) -> dict:
    resp = chat(
        judge,
        JUDGE_SYSTEM_PROMPT,
        JUDGE_USER_TEMPLATE.format(event=event, answer=clean_text(answer) or "(пустой ответ)"),
        settings,
        temperature=settings.judge_temperature,
        schema=JUDGE_SCHEMA if settings.structured_output else None,
        schema_name="judge",
    )
    if resp.error:
        scores, error = None, resp.error
    else:
        scores, error = parse_judge(resp.text)
    out = {
        "judge_error": error,
        "judge_latency_s": round(resp.latency_s, 4),
        "judge_raw": resp.text,
    }
    for name in RUBRIC_CRITERIA:
        out[name] = scores[name] if scores else None
    out["hallucination"] = scores["hallucination"] if scores else None
    out["basis"] = scores["basis"] if scores else ""
    out["judge_total"] = sum(scores[c] for c in RUBRIC_CRITERIA) if scores else None
    return out


def run_judge(
    judge: ModelCfg,
    settings: Settings,
    tests: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    repeats: dict[str, pd.DataFrame],
    ids: list[int],
) -> pd.DataFrame:
    """Судья оценивает (а) выборку из основного прогона и (б) повторные ответы (для устойчивости)."""
    print(f"[judge] судья = {judge.name}")
    warm_up(judge, settings)
    inputs = tests.set_index("id")["input"].to_dict()
    rows = []

    for tag in sorted(results):
        df = results[tag].set_index("id")
        for row_id in ids:
            r = df.loc[row_id]
            rows.append(
                {"kind": "sample", "model": tag.upper(), "id": row_id, "repeat": 0,
                 **judge_one(judge, inputs[row_id], r["raw"], settings)}
            )
    for tag in sorted(repeats):
        base = results[tag].set_index("id")
        for row_id in sorted(repeats[tag]["id"].unique()):  # исходный ответ = запуск №0
            rows.append(
                {"kind": "repeat", "model": tag.upper(), "id": row_id, "repeat": 0,
                 **judge_one(judge, inputs[row_id], base.loc[row_id]["raw"], settings)}
            )
        for r in repeats[tag].to_dict("records"):
            rows.append(
                {"kind": "repeat", "model": tag.upper(), "id": r["id"], "repeat": r["repeat"],
                 **judge_one(judge, inputs[r["id"]], r["raw"], settings)}
            )
    print(f"[judge] оценено ответов: {len(rows)}")
    return pd.DataFrame(rows)
