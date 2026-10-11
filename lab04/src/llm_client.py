"""Единая точка вызова модели.

Работает с любым OpenAI-совместимым API (LM Studio, Ollama, OpenAI, OpenRouter и т. д.).
Для каждого вызова возвращает текст, время ответа и usage (если сервер его отдаёт).
Ошибки API не пробрасываются наружу, а фиксируются в поле error — такие запросы
попадают в статистику как нарушение формата и не удаляются из выборки.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from config import ModelCfg, Settings


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    input_tokens: int | None
    output_tokens: int | None
    error: str = ""


_clients: dict[tuple[str, str], object] = {}


def _get_client(base_url: str, api_key: str):
    from openai import OpenAI  # импорт здесь, чтобы --mock работал без установленного SDK

    key = (base_url, api_key)
    if key not in _clients:
        _clients[key] = OpenAI(base_url=base_url, api_key=api_key)
    return _clients[key]


def chat(
    model: ModelCfg,
    system: str,
    user: str,
    settings: Settings,
    *,
    temperature: float | None = None,
    schema: dict | None = None,
    schema_name: str = "response",
) -> LLMResponse:
    if settings.mock:
        import mock_llm

        return mock_llm.chat(model, system, user, schema_name=schema_name)

    kwargs: dict = {
        "model": model.name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": settings.temperature if temperature is None else temperature,
        settings.max_tokens_param: settings.max_tokens,
        "timeout": settings.timeout,
    }
    if settings.seed is not None:
        kwargs["seed"] = settings.seed
    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }

    client = _get_client(model.base_url, model.api_key)
    started = time.perf_counter()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 — намеренно ловим всё и фиксируем
        latency = time.perf_counter() - started
        return LLMResponse("", latency, None, None, f"api_error: {type(exc).__name__}: {exc}"[:300])
    latency = time.perf_counter() - started

    text = (resp.choices[0].message.content or "") if resp.choices else ""
    usage = getattr(resp, "usage", None)
    return LLMResponse(
        text=text,
        latency_s=latency,
        input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
    )


def warm_up(model: ModelCfg, settings: Settings) -> None:
    """Прогревочный запрос (не записывается в результаты): в LM Studio первое обращение
    к модели может включать её загрузку в память и исказить latency."""
    if settings.mock:
        return
    chat(model, "Ответь одним словом.", "Привет", settings, temperature=0.0)
