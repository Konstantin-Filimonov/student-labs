import os
import pandas as pd

results_dir = "results"
candidates = [f for f in os.listdir(results_dir) if f.endswith(".csv") and "openrouter" in f]
raw_path = os.path.join(results_dir, candidates[0]) if candidates else os.path.join(results_dir, "raw_results.csv")

print(f"Читаю {raw_path}")

raw = pd.read_csv(raw_path, encoding="utf-8-sig")
# Заголовок первой колонки повреждён при переименовании файла — восстанавливаем
raw.columns = ["test_id"] + list(raw.columns[1:])

scores = pd.read_csv("data/scores.csv", encoding="utf-8-sig", quotechar='"', skipinitialspace=True)

def clean(df):
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    for col in ("test_id", "model"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

raw = clean(raw)
scores = clean(scores)

print("Колонки raw:", raw.columns.tolist())
print("Колонки scores:", scores.columns.tolist())

df = raw.merge(scores[["test_id", "model", "score"]], on=["test_id", "model"], how="left")

print("Пропущенных score:", df["score"].isna().sum())
if df["score"].isna().any():
    print(df[df["score"].isna()][["test_id", "model"]].to_string())

perf = df.groupby("model").agg(
    mean_latency=("latency_s", "mean"),
    median_latency=("latency_s", "median"),
    mean_ttft=("ttft_s", "mean"),
    mean_tokens_per_sec=("tokens_per_sec", "mean"),
    total_output_tokens=("output_tokens", "sum"),
    n_errors=("error", lambda x: x.notna().sum())
).round(2)

quality = df.groupby("model").agg(
    mean_score=("score", "mean"),
    sum_score=("score", "sum"),
    n_tests=("test_id", "count")
)
quality["max_possible"] = quality["n_tests"] * 2
quality["percent"] = (quality["sum_score"] / quality["max_possible"] * 100).round(1)

df["type"] = df["test_id"].str[1:3].map(lambda x: {
    "01":"classification","02":"classification","03":"classification","04":"classification","05":"classification",
    "06":"extraction","07":"extraction","08":"extraction","09":"extraction","10":"extraction",
    "11":"analysis","12":"analysis","13":"analysis","14":"analysis","15":"analysis",
    "16":"code","17":"code","18":"code","19":"code","20":"code"
}.get(x, "unknown"))

by_type = df.groupby(["model", "type"])["score"].mean().unstack()

print("\n=== Performance ===")
print(perf.to_string())
print("\n=== Quality (overall) ===")
print(quality.to_string())
print("\n=== Quality by type ===")
print(by_type.round(2).to_string())

perf.to_csv("results/summary_performance.csv", encoding="utf-8-sig")
quality.to_csv("results/summary_quality.csv", encoding="utf-8-sig")
by_type.to_csv("results/summary_quality_by_type.csv", encoding="utf-8-sig")