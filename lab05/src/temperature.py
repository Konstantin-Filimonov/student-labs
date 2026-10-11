"""Задание 6: влияние температуры на повторяемость и разнообразие."""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from client import LMStudioClient
from prompts import TEMP_PROMPT, TEMP_REPEATS, TEMPERATURES


def _words(text: str) -> set:
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 1.0


def run_temperature(client: LMStudioClient, out_dir: Path) -> List[Dict[str, Any]]:
    results = []
    md = [f"# Задание 6. Влияние температуры\n\n**Промпт:** {TEMP_PROMPT}\n"]
    for label, temp in TEMPERATURES:
        print(f"  температура {temp} ({label})")
        answers = []
        for _ in range(TEMP_REPEATS):
            res = client.chat([{"role": "user", "content": TEMP_PROMPT}],
                              temperature=temp, max_tokens=250)
            answers.append(res.text.strip())
        sims = [_jaccard(a, b) for a, b in itertools.combinations(answers, 2)]
        sim = sum(sims) / len(sims)
        results.append({
            "label": label, "temperature": temp, "answers": answers,
            "mean_similarity": round(sim, 3),        # 1.0 = ответы одинаковые
            "diversity": round(1 - sim, 3),          # 0.0 = нет разнообразия
            "identical_answers": len(set(answers)) == 1,
        })
        md.append(f"## Температура {temp} ({label})\n")
        md.append(f"Сходство ответов между повторами (Жаккар по словам): **{sim:.2f}**, "
                  f"разнообразие: **{1 - sim:.2f}**, "
                  f"все ответы идентичны: **{'да' if len(set(answers)) == 1 else 'нет'}**\n")
        for i, a in enumerate(answers, 1):
            md.append(f"**Повтор {i}:**\n\n{a}\n")
    (out_dir / "temperature.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "temperature.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    return results
