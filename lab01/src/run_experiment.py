import csv
import os

from llm_client import ask_model, MODEL_NAME, MODEL_LABEL
from prompts import PROMPTS

RESULTS_PATH = os.path.join("results", "experiment_results.csv")
FIELDNAMES = [
    "model_label",
    "model_name",
    "prompt_id",
    "prompt_type",
    "latency_sec",
    "answer",
    "error",
]


def run() -> list[dict]:
    results = []

    for item in PROMPTS:
        prompt_id = item["id"]
        prompt_type = item["type"]
        prompt_text = item["prompt"]

        print(f"[{MODEL_LABEL}] Запрос {prompt_id}/{len(PROMPTS)} ({prompt_type})...", end=" ")

        try:
            answer, latency = ask_model(prompt_text)
            results.append({
                "model_label": MODEL_LABEL,
                "model_name": MODEL_NAME,
                "prompt_id": prompt_id,
                "prompt_type": prompt_type,
                "latency_sec": round(latency, 3),
                "answer": answer,
                "error": "",
            })
            print(f"OK ({latency:.2f} c)")
        except Exception as exc:
            results.append({
                "model_label": MODEL_LABEL,
                "model_name": MODEL_NAME,
                "prompt_id": prompt_id,
                "prompt_type": prompt_type,
                "latency_sec": "",
                "answer": "",
                "error": str(exc),
            })
            print(f"ОШИБКА: {exc}")

    return results


def save_results(results: list[dict]) -> None:
    os.makedirs("results", exist_ok=True)
    file_exists = os.path.isfile(RESULTS_PATH)

    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nРезультаты дописаны в {RESULTS_PATH}")


if __name__ == "__main__":
    results = run()
    save_results(results)

    errors = sum(1 for r in results if r["error"])
    print(f"Готово. Успешно: {len(results) - errors}/{len(results)}. Ошибок: {errors}.")
