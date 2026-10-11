"""Клиент локального сервера LM Studio.

Используется нативный REST API LM Studio (/api/v0/...), потому что в ответе
он возвращает статистику генерации (tokens_per_second, time_to_first_token,
generation_time). Если такого эндпоинта нет (другая версия/среда), клиент
автоматически переключается на OpenAI-совместимый /v1/chat/completions и
считает скорость по общему времени запроса (это помечается в stats_source).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


class LMStudioError(RuntimeError):
    """Ошибка работы с локальным сервером."""


class LMStudioHTTPError(LMStudioError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


@dataclass
class ChatResult:
    text: str
    total_time_s: float                 # полное время запроса (с загрузкой модели, если она была)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    ttft_s: Optional[float] = None      # время до первого токена
    gen_time_s: Optional[float] = None  # время собственно генерации
    tokens_per_s: Optional[float] = None
    stats_source: str = "wallclock"     # lmstudio | wallclock
    finish_reason: Optional[str] = None
    runtime: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)


class LMStudioClient:
    def __init__(self, base_url: str, model: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.native = True  # пробуем /api/v0, при 404 переключимся на /v1
        self.session = requests.Session()
        # Локальному серверу ключ не нужен, но OpenAI-совместимые клиенты шлют заголовок.
        self.session.headers.update({"Authorization": "Bearer lm-studio"})

    # ---------- служебные запросы ----------
    def list_models(self) -> List[Dict[str, Any]]:
        """Список моделей. Нативный API отдаёт state/quantization/arch/контекст."""
        for path in ("/api/v0/models", "/v1/models"):
            try:
                r = self.session.get(self.base_url + path, timeout=10)
            except requests.exceptions.ConnectionError as exc:
                raise LMStudioError(
                    f"Сервер LM Studio недоступен по адресу {self.base_url}. "
                    "Откройте вкладку Developer и включите Start Server."
                ) from exc
            if r.status_code == 200:
                return r.json().get("data", [])
        raise LMStudioError("Не удалось получить список моделей ни по /api/v0, ни по /v1.")

    def get_model_record(self) -> Dict[str, Any]:
        for m in self.list_models():
            if m.get("id") == self.model:
                return m
        return {}

    def resolve_model(self, preferred: str) -> str:
        """Подбирает точный идентификатор модели среди доступных на сервере."""
        models = [m for m in self.list_models() if m.get("type") != "embeddings"]
        ids = [m["id"] for m in models]
        if preferred in ids:
            return preferred
        low = preferred.lower()
        for mid in ids:
            if low in mid.lower() or mid.lower() in low:
                return mid
        for mid in ids:
            if "qwen2.5" in mid.lower() and "3b" in mid.lower():
                return mid
        raise LMStudioError(
            f"Модель '{preferred}' не найдена. Доступные идентификаторы: {ids}. "
            "Укажите точный id в .env (MODEL_NAME)."
        )

    # ---------- основной вызов ----------
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> ChatResult:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = response_format
        if seed is not None:
            payload["seed"] = seed

        for _ in range(2):
            path = "/api/v0/chat/completions" if self.native else "/v1/chat/completions"
            t0 = time.perf_counter()
            try:
                r = self.session.post(self.base_url + path, json=payload, timeout=self.timeout)
            except requests.exceptions.ConnectionError as exc:
                raise LMStudioError("Потеряно соединение с сервером LM Studio.") from exc
            except requests.exceptions.Timeout as exc:
                raise LMStudioError(
                    f"Таймаут {self.timeout} с. Возможно, модель ещё загружается - повторите запуск."
                ) from exc
            total = time.perf_counter() - t0
            if r.status_code == 404 and self.native:
                self.native = False  # нет нативного API - переходим на /v1
                continue
            if r.status_code >= 400:
                raise LMStudioHTTPError(r.status_code, r.text[:500])
            break

        data = r.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        stats = data.get("stats") or {}

        completion_tokens = usage.get("completion_tokens")
        tps = stats.get("tokens_per_second")
        gen = stats.get("generation_time")
        source = "lmstudio" if tps else "wallclock"
        if not tps and completion_tokens:
            tps = completion_tokens / total if total > 0 else None

        return ChatResult(
            text=(choice.get("message") or {}).get("content") or "",
            total_time_s=total,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=completion_tokens,
            ttft_s=stats.get("time_to_first_token"),
            gen_time_s=gen,
            tokens_per_s=tps,
            stats_source=source,
            finish_reason=choice.get("finish_reason"),
            runtime=data.get("runtime") or {},
            model_info=data.get("model_info") or {},
        )


def _demo() -> None:
    """Быстрая проверка: python src/client.py "ваш вопрос"."""
    import sys

    from config import load_config

    cfg = load_config()
    client = LMStudioClient(cfg.base_url, cfg.model, cfg.timeout)
    client.model = client.resolve_model(cfg.model)
    question = " ".join(sys.argv[1:]) or "Объясни принцип наименьших привилегий в ИБ."
    res = client.chat([{"role": "user", "content": question}], temperature=cfg.temperature)
    print(res.text)
    print(f"\nВремя: {res.total_time_s:.2f} с | токенов: {res.completion_tokens} | "
          f"скорость: {res.tokens_per_s} ток/с ({res.stats_source})")


if __name__ == "__main__":
    _demo()
