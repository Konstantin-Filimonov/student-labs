import csv
import os

from llm_client import ask_model, MODEL_NAME, MODEL_LABEL
from prompts import PROMPTS

REPEATS = 5
SELECTED_IDS = [1, 5, 10]  # можно изменить на любые 3 id из prompts.py

RESULTS_PATH = os.path.join("results", "stability_results.csv")
FIELDNAMES = ["model_label", "prompt_id", "run_number", "latency_sec", "word_count", "answer", "error"]


def run() -> list[dict]:
    selected = [p for p in PROMPTS if p["id"] in SELECTED_IDS]
    results = []

    for item in selected:
        for run_number in range(1, REPEATS + 1):
            print(f"[{MODEL_LABEL}] prompt {item['id']}, попытка {run_number}/{REPEATS}...", end=" ")
            try:
                answer, latency = ask_model(item["prompt"])
                results.append({
                    "model_label": MODEL_LABEL,
                    "prompt_id": item["id"],
                    "run_number": run_number,
                    "latency_sec": round(latency, 3),
                    "word_count": len(answer.split()),
                    "answer": answer,
                    "error": "",
                })
                print("OK")
            except Exception as exc:
                results.append({
                    "model_label": MODEL_LABEL,
                    "prompt_id": item["id"],
                    "run_number": run_number,
                    "latency_sec": "",
                    "word_count": "",
                    "answer": "",
                    "error": str(exc),
                })
                print(f"ОШИБКА: {exc}")

    return results


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    results = run()

    file_exists = os.path.isfile(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nРезультаты дописаны в {RESULTS_PATH}")
