import argparse
import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# --------------------------------------------------------------------------
# 0. Конфигурация клиента (LM Studio = OpenAI-совместимый локальный сервер)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../lab03/src
LAB_DIR = BASE_DIR.parent                            # .../lab03
load_dotenv(LAB_DIR / ".env")

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY", "lm-studio"),
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
)

MODEL = os.getenv("LLM_MODEL", "google/gemma-4-e4b")
TEMPERATURE = 0.2

# Результаты кладём в lab03/results/, а не рядом со скриптом в src/
RESULTS_DIR = LAB_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def ask(system_prompt: str, user_prompt: str) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
    )
    elapsed = time.perf_counter() - started
    return response.choices[0].message.content, elapsed


def load_prompts() -> dict[str, str]:
    prompts = {}
    for path in sorted(BASE_DIR.glob("prompt_v*.txt")):
        version = path.stem
        prompts[version] = path.read_text(encoding="utf-8").strip()
    return prompts


def load_tests(name: str) -> list[dict]:
    with open(BASE_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def run_suite(prompts: dict[str, str], tests: list[dict], output_name: str) -> None:
    if not prompts:
        raise SystemExit("Не найдено ни одного файла prompt_v*.txt в src/")

    output_path = RESULTS_DIR / output_name
    fieldnames = ["test_id", "category", "prompt_version", "input", "response", "time_sec"]

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for version, system_prompt in prompts.items():
            for test in tests:
                print(f"[{version}] {test['id']} ...", end=" ", flush=True)
                try:
                    text, sec = ask(system_prompt, test["input"])
                except Exception as exc:
                    text, sec = f"ERROR: {exc}", -1
                print(f"{sec:.2f} c" if sec >= 0 else "ERROR")

                writer.writerow({
                    "test_id": test["id"],
                    "category": test["category"],
                    "prompt_version": version,
                    "input": test["input"],
                    "response": text,
                    "time_sec": sec,
                })

    print(f"\nГотово. Результаты сохранены в {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР3: сравнение промптов аналитика ИБ")
    parser.add_argument(
        "--task",
        choices=["soc", "phishing"],
        default="soc",
        help="soc — основной набор событий; phishing — индивидуальное задание",
    )
    args = parser.parse_args()

    print(f"Модель: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")

    if args.task == "soc":
        prompts = load_prompts()
        tests = load_tests("tests.json")
        run_suite(prompts, tests, "results.csv")
    else:
        prompts = {
            "prompt_phishing": (BASE_DIR / "prompt_phishing.txt")
            .read_text(encoding="utf-8")
            .strip()
        }
        tests = load_tests("tests_phishing.json")
        run_suite(prompts, tests, "results_phishing.csv")


if __name__ == "__main__":
    main()