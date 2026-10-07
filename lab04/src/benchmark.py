"""Шаги 3 и 8 методички: прогон тестового набора и повторные запуски."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from config import ModelCfg, Settings
from llm_client import chat, warm_up
from parsing import PARSE_ERROR, Parsed, parse_classification
from prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, USER_TEMPLATE

REQUIRED_COLUMNS = {"id", "input", "expected_label", "difficulty"}


def load_tests(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"В {path} нет колонок: {sorted(missing)}")
    if df["id"].duplicated().any():
        raise SystemExit("В тестовом наборе есть повторяющиеся id")
    bad = set(df["expected_label"]) - {"normal", "suspicious"}
    if bad:
        raise SystemExit(f"Недопустимые эталонные метки: {bad}")
    return df


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_once(model: ModelCfg, event: str, settings: Settings) -> dict:
    resp = chat(
        model,
        SYSTEM_PROMPT,
        USER_TEMPLATE.format(event=event),
        settings,
        schema=RESPONSE_SCHEMA if settings.structured_output else None,
        schema_name="classification",
    )
    if resp.error:
        parsed = Parsed(PARSE_ERROR, None, "", False, resp.error)
    else:
        parsed = parse_classification(resp.text)
    return {
        "predicted": parsed.label,
        "confidence": parsed.confidence,
        "reason": parsed.reason,
        "format_ok": parsed.format_ok,
        "format_error": parsed.error,
        "latency_s": round(resp.latency_s, 4),
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "raw": resp.text,
    }


def run_model(model: ModelCfg, tests: pd.DataFrame, settings: Settings, tag: str) -> pd.DataFrame:
    """Основной прогон: по одному запросу на каждый пример. Все ответы сохраняются."""
    print(f"[run] модель {tag.upper()} = {model.name}: {len(tests)} примеров")
    warm_up(model, settings)
    rows = []
    for record in tests.to_dict("records"):
        result = classify_once(model, record["input"], settings)
        rows.append(
            {
                "id": record["id"],
                "difficulty": record["difficulty"],
                "expected": record["expected_label"],
                **result,
            }
        )
        mark = "OK " if result["predicted"] == record["expected_label"] else "ERR"
        print(
            f"  id={record['id']:>2} {mark} pred={result['predicted']:<11} "
            f"{result['latency_s']:.2f}s {result['format_error']}"
        )
    return pd.DataFrame(rows)


def select_hardest(results: dict[str, pd.DataFrame], n: int) -> list[int]:
    """Выбор n самых сложных примеров по объективному правилу (фиксируется до анализа):
    1) больше суммарных ошибок обеих моделей (parse_error тоже ошибка);
    2) при равенстве — ниже средняя уверенность моделей;
    3) при равенстве — меньший id."""
    frames = []
    for df in results.values():
        part = df[["id", "expected", "predicted", "confidence"]].copy()
        part["error"] = (part["expected"] != part["predicted"]).astype(int)
        frames.append(part)
    all_rows = pd.concat(frames)
    score = all_rows.groupby("id").agg(errors=("error", "sum"), mean_conf=("confidence", "mean"))
    score["mean_conf"] = score["mean_conf"].fillna(0.0)
    score = score.reset_index().sort_values(["errors", "mean_conf", "id"], ascending=[False, True, True])
    return [int(i) for i in score["id"].head(n)]


def run_repeats(
    model: ModelCfg, tests: pd.DataFrame, ids: list[int], settings: Settings, tag: str
) -> pd.DataFrame:
    """Шаг 8: n_repeats дополнительных запусков для выбранных примеров."""
    print(f"[repeats] модель {tag.upper()}: примеры {ids} × {settings.n_repeats}")
    warm_up(model, settings)
    subset = tests[tests["id"].isin(ids)]
    rows = []
    for repeat in range(1, settings.n_repeats + 1):
        for record in subset.to_dict("records"):
            result = classify_once(model, record["input"], settings)
            rows.append(
                {
                    "id": record["id"],
                    "difficulty": record["difficulty"],
                    "expected": record["expected_label"],
                    "repeat": repeat,
                    **result,
                }
            )
    return pd.DataFrame(rows)
