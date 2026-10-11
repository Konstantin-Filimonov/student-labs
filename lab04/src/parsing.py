"""Разбор и строгая валидация ответов моделей.

Политика (фиксируется в отчёте):
* блоки <think>...</think> у «рассуждающих» моделей и обрамляющие Markdown-ограждения ```json
  срезаются до разбора; всё остальное (текст вокруг JSON, неверные поля) — нарушение формата;
* любой ответ, не прошедший валидацию, получает predicted = "parse_error" и остаётся в
  выборке — исключать такие ответы нельзя (раздел 8.4 методички).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from prompts import RUBRIC_CRITERIA

LABELS = ("normal", "suspicious")
PARSE_ERROR = "parse_error"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def clean_text(raw: str) -> str:
    text = _THINK_RE.sub("", raw or "")
    if "</think>" in text.lower():  # незакрытый/обрезанный блок рассуждений
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    text = text.strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group(1).strip()
    return text


@dataclass
class Parsed:
    label: str
    confidence: float | None
    reason: str
    format_ok: bool
    error: str = ""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_classification(raw: str) -> Parsed:
    text = clean_text(raw)
    if not text:
        return Parsed(PARSE_ERROR, None, "", False, "empty_response")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return Parsed(PARSE_ERROR, None, "", False, "invalid_json")
    if not isinstance(data, dict):
        return Parsed(PARSE_ERROR, None, "", False, "json_not_object")

    label = data.get("label")
    if not isinstance(label, str) or label.strip().lower() not in LABELS:
        return Parsed(PARSE_ERROR, None, "", False, "bad_label")
    confidence = data.get("confidence")
    if not _is_number(confidence) or not 0.0 <= float(confidence) <= 1.0:
        return Parsed(PARSE_ERROR, None, "", False, "bad_confidence")
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return Parsed(PARSE_ERROR, None, "", False, "bad_reason")

    return Parsed(label.strip().lower(), float(confidence), reason.strip(), True)


def parse_judge(raw: str) -> tuple[dict | None, str]:
    """Возвращает (оценки, ошибка). Оценки: 4 критерия (int 0–2), hallucination (bool), basis (str)."""
    text = clean_text(raw)
    if not text:
        return None, "empty_response"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "json_not_object"

    scores: dict = {}
    for name in RUBRIC_CRITERIA:
        value = data.get(name)
        if not _is_number(value) or int(value) != value or not 0 <= int(value) <= 2:
            return None, f"bad_{name}"
        scores[name] = int(value)
    halluc = data.get("hallucination")
    if not isinstance(halluc, bool):
        return None, "bad_hallucination"
    scores["hallucination"] = halluc
    basis = data.get("basis")
    scores["basis"] = json.dumps(basis, ensure_ascii=False) if basis is not None else ""
    return scores, ""
