import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI, APIError, BadRequestError

# --------------------------------------------------------------------------
# 0. Конфигурация клиента (LM Studio = OpenAI-совместимый локальный сервер)
# --------------------------------------------------------------------------
load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
MODEL = os.getenv("LLM_MODEL", "google/gemma-4-e4b")

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# ПРАВКА: результаты пишем в lab02/results/, а не рядом со скриптом в src/,
# чтобы соответствовать структуре lab02/src + lab02/results из требований
# к оформлению репозитория.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
JSONL_PATH = os.path.join(RESULTS_DIR, "results.jsonl")
CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")

CSV_FIELDS = [
    "timestamp", "model", "experiment", "case_id", "repeat",
    "temperature", "top_p", "max_tokens", "system_id",
    "latency_s", "prompt_tokens", "completion_tokens", "total_tokens",
    "answer_preview", "error",
]


# --------------------------------------------------------------------------
# 1. Обёртка вызова модели с фиксацией параметров и ошибок
# --------------------------------------------------------------------------
def ask_model(prompt: str, system: str, temperature=None, top_p=None, max_tokens=None):
    """
    Вызывает модель через LM Studio. Добавляет в запрос только те параметры,
    которые были явно переданы (не None) - чтобы не затирать значения
    по умолчанию, когда мы их сознательно не трогаем (Этап 1).

    Возвращает кортеж (answer, latency_s, usage_dict, error_str_or_None).
    Если API вернуло ошибку о неподдерживаемом параметре - не падаем,
    а возвращаем error, чтобы это можно было зафиксировать как результат
    эксперимента (см. п.4.7 и п.8.6 методички).
    """
    kwargs = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - started
        answer = response.choices[0].message.content
        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return answer, latency, usage, None
    except (BadRequestError, APIError) as exc:
        latency = time.perf_counter() - started
        return None, latency, {}, str(exc)
    except Exception as exc:  # сетевые ошибки, сервер не запущен и т.п.
        latency = time.perf_counter() - started
        return None, latency, {}, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# 2. Логирование результата одной попытки
# --------------------------------------------------------------------------
_csv_file_initialized = os.path.exists(CSV_PATH)


def log_result(experiment, case_id, repeat, temperature, top_p, max_tokens,
                system_id, latency_s, usage, answer, error):
    global _csv_file_initialized
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "experiment": experiment,
        "case_id": case_id,
        "repeat": repeat,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "system_id": system_id,
        "latency_s": round(latency_s, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "answer": answer,
        "error": error,
    }

    # полная запись, включая полный ответ - в JSONL
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # сводная таблица с усечённым ответом - в CSV, удобно открыть в Excel
    csv_record = dict(record)
    preview = (answer or "").replace("\n", " ").replace("\r", " ")
    csv_record["answer_preview"] = (preview[:150] + "…") if len(preview) > 150 else preview
    del csv_record["answer"]

    write_header = not _csv_file_initialized
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
            _csv_file_initialized = True
        writer.writerow(csv_record)

    status = "OK " if error is None else "ERR"
    print(f"[{status}] {experiment:<11} case={case_id:<10} repeat={repeat} "
          f"t={temperature} top_p={top_p} max_tok={max_tokens} "
          f"latency={latency_s:.2f}s" + (f" error={error[:120]}" if error else ""))


# --------------------------------------------------------------------------
# 3. Этап 1. Базовая конфигурация (значения по умолчанию)
# --------------------------------------------------------------------------
BASELINE_PROMPT = (
    "Событие: в 03:17 зафиксировано 25 неудачных попыток входа в учётную "
    "запись admin с одного IP, после чего произошёл успешный вход. "
    "Проанализируй ситуацию."
)
BASELINE_SYSTEM = "Отвечай на запрос пользователя."


def stage1_baseline(repeats=3):
    print("\n=== Этап 1. Базовая конфигурация (default) ===")
    for i in range(1, repeats + 1):
        answer, latency, usage, error = ask_model(
            BASELINE_PROMPT, BASELINE_SYSTEM,
            temperature=None, top_p=None, max_tokens=None,
        )
        log_result("baseline", "default", i, None, None, None,
                    "sys_00", latency, usage, answer, error)


# --------------------------------------------------------------------------
# 4. Этап 2. Влияние system prompt (3 варианта, один и тот же user prompt)
# --------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "sys_01": "Отвечай на запрос пользователя.",
    "sys_02": (
        "Ты ассистент аналитика ИБ. Отвечай не более чем в 5 пунктах. "
        "Для каждого пункта укажи риск и рекомендуемое действие."
    ),
    "sys_03": (
        "Ты ассистент аналитика ИБ. Не выдумывай недостающие данные. "
        "Явно отделяй факты от предположений. Ответ верни в формате: "
        "Факты / Гипотезы / Действия."
    ),
}
SYSTEM_PROMPT_USER_QUERY = BASELINE_PROMPT  # тот же запрос, что и в методичке


def stage2_system_prompt():
    print("\n=== Этап 2. Влияние system prompt ===")
    for sys_id, system_text in SYSTEM_PROMPTS.items():
        answer, latency, usage, error = ask_model(
            SYSTEM_PROMPT_USER_QUERY, system_text,
            temperature=None, top_p=None, max_tokens=None,
        )
        log_result("system_prompt", sys_id, 1, None, None, None,
                    sys_id, latency, usage, answer, error)


# --------------------------------------------------------------------------
# 5. Этап 3. Влияние temperature (5 повторов на каждое значение)
# --------------------------------------------------------------------------
TEMPERATURE_PROMPT = (
    "Предложи пять названий сервиса для автоматического анализа журналов "
    "событий информационной безопасности. Для каждого названия дай "
    "пояснение в одном предложении."
)
TEMPERATURE_SYSTEM = "Отвечай на запрос пользователя."
TEMPERATURE_VALUES = [0.0, 0.3, 0.7, 1.0]


def stage3_temperature(repeats=5):
    print("\n=== Этап 3. Влияние temperature ===")
    for t in TEMPERATURE_VALUES:
        for i in range(1, repeats + 1):
            answer, latency, usage, error = ask_model(
                TEMPERATURE_PROMPT, TEMPERATURE_SYSTEM,
                temperature=t, top_p=None, max_tokens=None,
            )
            log_result("temperature", f"t={t}", i, t, None, None,
                        "sys_00", latency, usage, answer, error)


# --------------------------------------------------------------------------
# 6. Этап 4. Ограничение длины ответа
# --------------------------------------------------------------------------
LENGTH_PROMPT = (
    "Объясни студенту 4 курса архитектуру RAG-системы: ingestion, "
    "chunking, embeddings, vector database, retrieval, prompt construction "
    "и generation. Для каждого этапа укажи его назначение и одну типичную "
    "ошибку."
)
LENGTH_SYSTEM = "Отвечай на запрос пользователя."
MAX_TOKENS_VALUES = [80, 300, 900]  # малый / средний / большой лимит


def stage4_length():
    print("\n=== Этап 4. Ограничение длины ответа ===")
    for mt in MAX_TOKENS_VALUES:
        answer, latency, usage, error = ask_model(
            LENGTH_PROMPT, LENGTH_SYSTEM,
            temperature=None, top_p=None, max_tokens=mt,
        )
        log_result("max_tokens", f"max={mt}", 1, None, None, mt,
                    "sys_00", latency, usage, answer, error)


# --------------------------------------------------------------------------
# 7. Этап 6. Точка входа / автоматизация всего эксперимента
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ЛР2: параметры и конфигурация LLM")
    parser.add_argument("--stage", choices=["1", "2", "3", "4", "all"], default="all",
                         help="какой этап запустить (по умолчанию - все)")
    parser.add_argument("--repeats-baseline", type=int, default=3)
    parser.add_argument("--repeats-temperature", type=int, default=5)
    args = parser.parse_args()

    print(f"Модель: {MODEL}")
    print(f"Эндпоинт: {LLM_BASE_URL}")
    print(f"Результаты: {JSONL_PATH}\n {CSV_PATH}")

    if args.stage in ("1", "all"):
        stage1_baseline(repeats=args.repeats_baseline)
    if args.stage in ("2", "all"):
        stage2_system_prompt()
    if args.stage in ("3", "all"):
        stage3_temperature(repeats=args.repeats_temperature)
    if args.stage in ("4", "all"):
        stage4_length()

    print("\nГотово. Результаты дозаписаны в results.jsonl и results.csv.")
    print("Дальше: открыть results.csv, посчитать число уникальных ответов "
          "в Этапе 3 (temperature) и заполнить таблицы отчёта (раздел 10 методички).")


if __name__ == "__main__":
    main()
