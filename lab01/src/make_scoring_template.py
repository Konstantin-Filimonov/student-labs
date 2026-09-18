import pandas as pd

INPUT_PATH = "results/experiment_results.csv"
OUTPUT_PATH = "results/scoring_template.csv"

SCORE_COLUMNS = [
    "correctness_0_2",       # корректность
    "completeness_0_2",      # полнота
    "instruction_follow_0_2",  # следование инструкции
    "usefulness_0_2",        # практическая полезность
    "score_uncertain",       # True/False — если корректность нельзя проверить
]

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)

    for col in SCORE_COLUMNS:
        df[col] = "" if col != "score_uncertain" else False

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Шаблон для оценки сохранён: {OUTPUT_PATH}")
    print("Заполните баллы вручную (0-2) по каждому критерию для каждой строки.")
