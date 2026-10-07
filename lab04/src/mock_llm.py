"""Имитатор LLM для отладки конвейера БЕЗ реальных моделей (флаг --mock).

Ответы генерируются по простым ключевым словам со случайным шумом. Это НЕ результат
эксперимента: прогон в режиме mock помечается в run_meta/report и не должен
использоваться в отчёте.
"""
from __future__ import annotations

import json
import random
import re

from config import ModelCfg
from llm_client import LLMResponse

_counter = 0

_SUSPICIOUS_HINTS = [
    "25 неудачных", "не смог удалить", "сканирование портов", "переименовано", "без заявки",
    "бразилии", "скачала", "-encodedcommand", "txt-записи", "300 писем", "отключена",
    "личное облачное", "svc-backup", "остановлена", "сброса пароля", "разных ip",
]


def _event_from_user(user: str) -> str:
    match = re.search(r"Описание события:\n(.*?)\n\nКлассифицируй", user, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"описание события\):\n(.*?)\n\nОтвет оцениваемой", user, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else user


def chat(model: ModelCfg, system: str, user: str, *, schema_name: str) -> LLMResponse:
    global _counter
    _counter += 1
    rng = random.Random(f"{model.key}|{model.name}|{user}|{_counter}")
    weak = model.key == "B"

    if schema_name == "judge":
        base = 1 if weak else 2
        scores = {
            c: max(0, min(2, base + rng.choice([-1, 0, 0, 0, 1]) if rng.random() < 0.5 else base))
            for c in ("factual_correctness", "completeness", "instruction_following", "relevance")
        }
        payload = {
            **scores,
            "hallucination": rng.random() < (0.3 if weak else 0.1),
            "basis": {k: "mock" for k in scores},
        }
        text = json.dumps(payload, ensure_ascii=False)
        return LLMResponse(text, rng.uniform(1.0, 2.5), 400, 90)

    event = _event_from_user(user).lower()
    label = "suspicious" if any(h in event for h in _SUSPICIOUS_HINTS) else "normal"
    if rng.random() < (0.28 if weak else 0.10):  # шум классификации
        label = "normal" if label == "suspicious" else "suspicious"
    confidence = round(rng.uniform(0.6, 0.99) if weak else rng.uniform(0.7, 0.95), 2)
    reason = "Имитация объяснения по фактам из описания события."
    if weak and rng.random() < 0.3:
        reason += " Также замечен IP 192.168.77.5."  # «галлюцинация»
    text = json.dumps({"label": label, "confidence": confidence, "reason": reason}, ensure_ascii=False)
    if weak and rng.random() < 0.12:
        text = "Вот результат анализа: " + text  # нарушение формата
    latency = rng.gauss(2.6, 0.6) if weak else rng.gauss(1.3, 0.3)
    return LLMResponse(text, max(0.2, latency), 260 + len(event) // 4, 70 + len(text) // 4)
