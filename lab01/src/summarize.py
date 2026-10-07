import pandas as pd

SCORED_PATH = "results/scoring_template.csv"

SCORE_COLS = [
    "correctness_0_2",
    "completeness_0_2",
    "instruction_follow_0_2",
    "usefulness_0_2",
]

if __name__ == "__main__":
    df = pd.read_csv(SCORED_PATH)

    for col in SCORE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_score_0_8"] = df[SCORE_COLS].sum(axis=1)

    # Пропускаем строки с ошибками API при усреднении латентности/баллов
    ok = df[df["error"].isna() | (df["error"] == "")]

    summary = ok.groupby("model_label").agg(
        avg_latency_sec=("latency_sec", "mean"),
        avg_total_score=("total_score_0_8", "mean"),
        n_requests=("prompt_id", "count"),
    ).round(2)

    error_counts = df.groupby("model_label")["error"].apply(
        lambda s: (s.notna() & (s != "")).sum()
    ).rename("n_errors")

    summary = summary.join(error_counts)

    summary_by_type = ok.groupby(["model_label", "prompt_type"]).agg(
        avg_latency_sec=("latency_sec", "mean"),
        avg_total_score=("total_score_0_8", "mean"),
    ).round(2)

    summary.to_csv("results/summary.csv")
    summary_by_type.to_csv("results/summary_by_type.csv")

    print("=== Сводное сравнение моделей ===")
    print(summary)
    print("\n=== Сравнение по типам задач ===")
    print(summary_by_type)
    print("\nСохранено: results/summary.csv, results/summary_by_type.csv")
