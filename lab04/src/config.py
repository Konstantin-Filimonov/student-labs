"""Конфигурация эксперимента.

Все параметры читаются из переменных окружения / файла .env (см. .env.example).
API-ключи в результаты и отчёты не попадают (маскируются).
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
TESTS_PATH = DATA_DIR / "tests.csv"

load_dotenv(ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _get_int(name: str, default: int) -> int:
    return int(_get(name, str(default)))


def _get_float(name: str, default: float) -> float:
    return float(_get(name, str(default)).replace(",", "."))


def _get_bool(name: str, default: bool) -> bool:
    return _get(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def _opt_float(name: str) -> float | None:
    value = _get(name)
    return float(value.replace(",", ".")) if value else None


@dataclass
class ModelCfg:
    """Одна модель (или конфигурация), участвующая в эксперименте."""

    key: str  # "A", "B" или "JUDGE"
    name: str  # идентификатор модели в API (в LM Studio — имя загруженной модели)
    base_url: str
    api_key: str
    price_in: float | None = None  # цена за 1 млн входных токенов
    price_out: float | None = None  # цена за 1 млн выходных токенов

    def public(self) -> dict:
        data = asdict(self)
        data["api_key"] = "***"
        return data


@dataclass
class Settings:
    model_a: ModelCfg
    model_b: ModelCfg
    judge: ModelCfg
    temperature: float
    judge_temperature: float
    max_tokens: int
    max_tokens_param: str
    seed: int | None
    timeout: float
    structured_output: bool
    n_repeats: int  # число ДОПОЛНИТЕЛЬНЫХ запусков
    n_hard: int  # сколько самых сложных примеров повторять
    manual_n: int  # сколько ответов на модель оценивать вручную / судьёй
    sample_seed: int
    bootstrap_n: int
    price_currency: str
    price_date: str
    price_source: str
    mock: bool

    @property
    def models(self) -> dict[str, ModelCfg]:
        return {"a": self.model_a, "b": self.model_b}

    def public(self) -> dict:
        data = asdict(self)
        for key in ("model_a", "model_b", "judge"):
            data[key]["api_key"] = "***"
        return data


def _model(key: str, default_name: str, base_url: str, api_key: str) -> ModelCfg:
    return ModelCfg(
        key=key,
        name=_get(f"MODEL_{key}", default_name),
        base_url=_get(f"MODEL_{key}_BASE_URL", base_url),
        api_key=_get(f"MODEL_{key}_API_KEY", api_key),
        price_in=_opt_float(f"MODEL_{key}_PRICE_IN"),
        price_out=_opt_float(f"MODEL_{key}_PRICE_OUT"),
    )


def load_settings(mock: bool | None = None) -> Settings:
    if mock is None:
        mock = _get_bool("MOCK", False)

    base_url = _get("LLM_BASE_URL", "http://localhost:1234/v1")  # LM Studio по умолчанию
    api_key = _get("LLM_API_KEY", "lm-studio")

    default_names = {"A": "", "B": "", "JUDGE": ""}
    if mock:
        default_names = {"A": "mock-model-a", "B": "mock-model-b", "JUDGE": "mock-judge"}

    model_a = _model("A", default_names["A"], base_url, api_key)
    model_b = _model("B", default_names["B"], base_url, api_key)
    judge = _model("JUDGE", default_names["JUDGE"], base_url, api_key)

    if not mock:
        missing = [m.key for m in (model_a, model_b, judge) if not m.name]
        if missing:
            names = ", ".join(f"MODEL_{k}" for k in missing)
            raise SystemExit(
                f"Не заданы имена моделей: {names}. Скопируйте .env.example в .env и заполните."
            )

    seed = _get("SEED")
    return Settings(
        model_a=model_a,
        model_b=model_b,
        judge=judge,
        temperature=_get_float("TEMPERATURE", 0.2),
        judge_temperature=_get_float("JUDGE_TEMPERATURE", 0.0),
        max_tokens=_get_int("MAX_TOKENS", 1024),
        max_tokens_param=_get("MAX_TOKENS_PARAM", "max_tokens"),
        seed=int(seed) if seed else None,
        timeout=_get_float("REQUEST_TIMEOUT", 180),
        structured_output=_get_bool("STRUCTURED_OUTPUT", False),
        n_repeats=_get_int("N_REPEATS", 3),
        n_hard=_get_int("N_HARD", 5),
        manual_n=_get_int("MANUAL_N", 10),
        sample_seed=_get_int("SAMPLE_SEED", 42),
        bootstrap_n=_get_int("BOOTSTRAP_N", 2000),
        price_currency=_get("PRICE_CURRENCY", "USD"),
        price_date=_get("PRICE_DATE", ""),
        price_source=_get("PRICE_SOURCE", ""),
        mock=mock,
    )
