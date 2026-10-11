import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

raw = pd.read_csv("results/raw_results_openrouter_free.csv", encoding="utf-8-sig")
raw.columns = ["test_id"] + list(raw.columns[1:])

# График 1: качество по типам заданий
types = ["classification", "extraction", "analysis", "code"]
local_scores = [0.4, 0.7, 1.0, 1.0]
cloud_scores = [0.6, 0.5, 2.0, 1.2]

x = np.arange(len(types))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width/2, local_scores, width, label="Локальная Qwen 2.5 3B", color="#4C72B0")
ax.bar(x + width/2, cloud_scores, width, label="Облачная openrouter/free", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(types)
ax.set_ylabel("Средний балл")
ax.set_title("Качество по типам заданий (0–2 балла)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/plot_quality_by_type.png", dpi=150)
plt.close()

# График 2: распределение задержек
fig, ax = plt.subplots(figsize=(9, 5))
local_lat = raw[raw["model"] == "local"]["latency_s"].dropna()
cloud_lat = raw[raw["model"] == "cloud"]["latency_s"].dropna()

bins = np.linspace(0, max(local_lat.max(), cloud_lat.max()) + 1, 20)
ax.hist(local_lat, bins=bins, alpha=0.6, label="Локальная", color="#4C72B0")
ax.hist(cloud_lat, bins=bins, alpha=0.6, label="Облачная", color="#DD8452")
ax.set_xlabel("Полная задержка, с")
ax.set_ylabel("Число запросов")
ax.set_title("Распределение полной задержки")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plot_latency_distribution.png", dpi=150)
plt.close()

# График 3: TTFT vs полная задержка (scatter)
fig, ax = plt.subplots(figsize=(8, 5))
for model, color, label in [("local", "#4C72B0", "Локальная"), ("cloud", "#DD8452", "Облачная")]:
    sub = raw[raw["model"] == model].dropna(subset=["ttft_s", "latency_s"])
    ax.scatter(sub["ttft_s"], sub["latency_s"], alpha=0.7, color=color, label=label)
ax.set_xlabel("TTFT, с")
ax.set_ylabel("Полная задержка, с")
ax.set_title("TTFT vs полная задержка")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/plot_ttft_vs_latency.png", dpi=150)
plt.close()

print("Сохранено 3 графика в results/")