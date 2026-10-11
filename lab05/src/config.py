"""Загрузка конфигурации из .env (без внешних зависимостей)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> None:
    """Минимальный разбор .env: KEY=VALUE, комментарии после ' #' отбрасываются."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #")[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


@dataclass
class Config:
    base_url: str
    model: str
    timeout: int
    bench_runs: int
    bench_max_tokens: int
    temperature: float
    lmstudio_version: str
    sample_interval: float = 0.5


def load_config() -> Config:
    _load_env(ROOT / ".env")
    g = os.environ.get
    return Config(
        base_url=g("LMSTUDIO_URL", "http://localhost:1234").rstrip("/"),
        model=g("MODEL_NAME", "qwen2.5-3b-instruct"),
        timeout=int(g("REQUEST_TIMEOUT", "300")),
        bench_runs=max(5, int(g("BENCH_RUNS", "5"))),
        bench_max_tokens=int(g("BENCH_MAX_TOKENS", "300")),
        temperature=float(g("TEMPERATURE", "0.2")),
        lmstudio_version=g("LMSTUDIO_VERSION", "").strip(),
    )
