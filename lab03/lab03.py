import csv
import glob
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY", "lm-studio"),  # LM Studio не проверяет ключ, но поле обязательно
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
)

MODEL = os.getenv("LLM_MODEL", "gemma")


def ask(system_prompt: str, user_prompt: str) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    elapsed = time.perf_counter() - started
    return response.choices[0].message.content, elapsed


    prompts = {}
    for path in sorted(glob.glob("prompts/prompt_v*.txt")):
        version = os.path.splitext(os.path.basename(path))[0]  # prompt_v1
        with open(path, "r", encoding="utf-8") as f:
            prompts[version] = f.read().strip()
    return prompts


    with open("tests.json", "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    prompts = load_prompts()
    tests = load_tests()

    if not prompts:
        raise SystemExit("Не найдено ни одного prompts/prompt_v*.txt")

    fieldnames = ["test_id", "category", "prompt_version", "input", "response", "time_sec"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for version, system_prompt in prompts.items():
            for test in tests:
                print(f"[{version}] {test['id']} ...", end=" ", flush=True)
                try:
                    text, sec = ask(system_prompt, test["input"])
                except Exception as exc:  # сеть/сервер LM Studio недоступны и т.п.
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

    print("\nГотово. Результаты сохранены в results.csv")


if __name__ == "__main__":
    main()
