"""Задания 4 и 5: замер времени/скорости и наблюдение за ресурсами."""
from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from client import ChatResult, LMStudioClient
from monitor import ResourceMonitor
from prompts import BENCH_PROMPT

COLUMNS = ["run", "kind", "total_time_s", "ttft_s", "gen_time_s",
           "prompt_tokens", "completion_tokens", "tokens_per_s", "stats_source", "finish_reason"]


def _r(x, nd=2):
    return round(x, nd) if isinstance(x, (int, float)) else None


def run_benchmark(client: LMStudioClient, monitor: ResourceMonitor, runs: int,
                  max_tokens: int, temperature: float, out_dir: Path) -> Dict[str, Any]:
    messages = [{"role": "user", "content": BENCH_PROMPT}]
    rows: List[Dict[str, Any]] = []
    first: ChatResult | None = None

    for i in range(1, runs + 1):
        if i == 1:
            monitor.set_phase("cold_generation")  # загрузка модели + генерация
            print(f"  запуск 1/{runs} (первый, может включать загрузку модели)...")
        else:
            monitor.set_phase("generation")
            print(f"  запуск {i}/{runs}...")
        res = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
        if i == 1:
            first = res
            # Пауза без нагрузки: измеряем «после загрузки» (модель в памяти, простой)
            monitor.set_phase("idle_after_load")
            time.sleep(5)
        rows.append({
            "run": i,
            "kind": "первый" if i == 1 else "повторный",
            "total_time_s": _r(res.total_time_s, 3),
            "ttft_s": _r(res.ttft_s, 3),
            "gen_time_s": _r(res.gen_time_s, 3),
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "tokens_per_s": _r(res.tokens_per_s, 2),
            "stats_source": res.stats_source,
            "finish_reason": res.finish_reason,
        })
    monitor.set_phase("done")

    with open(out_dir / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    repeats = rows[1:]
    times = [r["total_time_s"] for r in repeats]
    tps = [r["tokens_per_s"] for r in repeats if r["tokens_per_s"] is not None]
    summary = {
        "prompt": BENCH_PROMPT,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "runs": runs,
        "first_run_time_s": rows[0]["total_time_s"],
        "first_run_tokens_per_s": rows[0]["tokens_per_s"],
        "repeat_mean_time_s": _r(statistics.mean(times)),
        "repeat_stdev_time_s": _r(statistics.stdev(times)) if len(times) > 1 else None,
        "repeat_mean_tokens_per_s": _r(statistics.mean(tps)) if tps else None,
        "first_vs_repeat_ratio": _r(rows[0]["total_time_s"] / statistics.mean(times)),
        "answer_first_run": first.text if first else "",
        "runtime": first.runtime if first else {},
        "model_info": first.model_info if first else {},
    }
    return {"rows": rows, "summary": summary}


def resources_table(monitor: ResourceMonitor) -> Dict[str, Any]:
    """Таблица задания 5: до загрузки / после загрузки / во время генерации."""
    return {
        "before_load": monitor.summarize("baseline"),
        "after_load": monitor.summarize("idle_after_load"),
        "during_generation": monitor.summarize("generation"),
        "first_run_cold": monitor.summarize("cold_generation"),
    }
