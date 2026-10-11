"""Графики для README: ресурсы по времени и скорость по запускам."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

PHASE_LABELS = {"baseline": "до загрузки", "cold_generation": "1-й запуск (загрузка)",
                "idle_after_load": "после загрузки", "generation": "генерация"}


def plot_resources(samples: List[Dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    rows = [s for s in samples if s["phase"] in PHASE_LABELS]
    if not rows:
        return False
    t = [s["t_s"] for s in rows]
    panels = [("cpu_pct", "CPU, %"), ("ram_gb", "RAM, ГБ"), ("gpu_pct", "GPU, %"), ("vram_gb", "VRAM, ГБ")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 8), sharex=True)
    for ax, (key, title) in zip(axes, panels):
        vals = [s[key] for s in rows]
        if all(v is None for v in vals):
            ax.text(0.5, 0.5, "нет измерения (GPU не найден)", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            ax.plot(t, [v if v is not None else float("nan") for v in vals], linewidth=1.4)
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        start = 0
        for i in range(1, len(rows) + 1):
            if i == len(rows) or rows[i]["phase"] != rows[start]["phase"]:
                ax.axvline(t[start], color="gray", linestyle=":", linewidth=0.8)
                if key == "cpu_pct":
                    ax.text(t[start], ax.get_ylim()[1], " " + PHASE_LABELS[rows[start]["phase"]],
                            fontsize=7, va="top", rotation=90)
                start = i
    axes[-1].set_xlabel("Время, с")
    fig.suptitle("Использование ресурсов во время эксперимента")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_benchmark(rows: List[Dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    nums = [r["run"] for r in rows]
    colors = ["tab:red" if r["run"] == 1 else "tab:blue" for r in rows]
    a1.bar(nums, [r["total_time_s"] for r in rows], color=colors)
    a1.set_title("Полное время ответа, с")
    a1.set_xlabel("№ запуска (красный - первый)")
    a2.bar(nums, [r["tokens_per_s"] or 0 for r in rows], color=colors)
    a2.set_title("Скорость генерации, ток/с")
    a2.set_xlabel("№ запуска")
    for a in (a1, a2):
        a.set_xticks(nums)
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
