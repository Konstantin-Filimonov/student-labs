"""Собирает ВСЕ данные запуска в один файл report_data.md - его достаточно отправить для отчёта."""
from __future__ import annotations

import json
from typing import Any, Dict


def _fmt(d, key):
    v = d.get(key) if d else None
    return "нет" if v is None else f"{v['mean']} (макс. {v['max']})"


def build(data: Dict[str, Any]) -> str:
    s, m, b, rs = data["sysinfo"], data["model"], data["benchmark"], data["resources"]
    md = ["# ДАННЫЕ ЗАПУСКА ЛР5 (для формирования отчёта)\n",
          f"Дата и время: {data['started']}  |  Версия промптов: {data['prompt_version']}\n",
          "## Конфигурация компьютера\n", "```json", json.dumps(s, ensure_ascii=False, indent=2), "```\n",
          "## Модель и среда\n", "```json", json.dumps(m, ensure_ascii=False, indent=2), "```\n",
          f"Холодный старт подтверждён (модель была выгружена перед замером): "
          f"**{data['cold_start_verified']}**\n",
          "## Задание 4. Время и скорость\n",
          "| № | Запуск | Время, с | TTFT, с | Вых. токенов | Ток/с | Источник |", "|---|---|---|---|---|---|---|"]
    for r in b["rows"]:
        md.append(f"| {r['run']} | {r['kind']} | {r['total_time_s']} | {r['ttft_s']} | "
                  f"{r['completion_tokens']} | {r['tokens_per_s']} | {r['stats_source']} |")
    sm = b["summary"]
    md += ["", f"Среднее время повторных запусков: **{sm['repeat_mean_time_s']} с** "
               f"(σ = {sm['repeat_stdev_time_s']}), средняя скорость повторных: "
               f"**{sm['repeat_mean_tokens_per_s']} ток/с**, первый/повторный = "
               f"**{sm['first_vs_repeat_ratio']}×**\n",
           "## Задание 5. Ресурсы (среднее, в скобках максимум)\n",
           "| Показатель | До загрузки | После загрузки | Во время генерации |", "|---|---|---|---|"]
    for key, name in (("ram_gb", "RAM, ГБ"), ("cpu_pct", "CPU, %"),
                      ("vram_gb", "VRAM, ГБ"), ("gpu_pct", "GPU, %")):
        md.append(f"| {name} | {_fmt(rs['before_load'], key)} | {_fmt(rs['after_load'], key)} | "
                  f"{_fmt(rs['during_generation'], key)} |")
    md += ["", f"Во время первого (холодного) запуска: RAM {_fmt(rs['first_run_cold'], 'ram_gb')}, "
               f"VRAM {_fmt(rs['first_run_cold'], 'vram_gb')}, "
               f"GPU {_fmt(rs['first_run_cold'], 'gpu_pct')}, CPU {_fmt(rs['first_run_cold'], 'cpu_pct')}\n",
           "## Задание 2. Ответы на 5 запросов\n"]
    for d in data["dialog"]:
        md += [f"### {d['id']}. {d['kind']}", f"**Промпт:** {d['prompt']}\n", "**Ответ:**\n",
               d["answer"].strip(), "",
               f"*Время {d['time_s']} с, токенов {d['completion_tokens']}, ток/с {d['tokens_per_s']}; "
               f"автопроверка: {json.dumps(d['auto_check'], ensure_ascii=False)}*\n"]
    md.append("## Задание 6. Температура\n")
    for t in data["temperature"]:
        md.append(f"### T = {t['temperature']} ({t['label']}); сходство {t['mean_similarity']}, "
                  f"разнообразие {t['diversity']}, идентичны: {t['identical_answers']}\n")
        for i, a in enumerate(t["answers"], 1):
            md += [f"**Повтор {i}:** {a}\n"]
    lg = data["log_analysis"]
    md += ["## Индивидуальное задание: анализ журнала\n", "```json",
           json.dumps(lg, ensure_ascii=False, indent=2), "```\n"]
    return "\n".join(md)
