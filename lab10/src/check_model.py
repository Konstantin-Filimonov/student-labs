"""Этап 1 методички: проверка модели без инструментов и проверка поддержки tools.

Запуск: py src/check_model.py
"""
import json
import platform
import sys

from llm_client import get_base_url, get_model_name, make_client
from tools import CFG, TOOLS

if __name__ == "__main__":
    client, model = make_client(CFG["llm"]["timeout_s"]), get_model_name()
    print(f"Python {sys.version.split()[0]}, {platform.platform()}")
    print(f"Режим: локальный; API: {get_base_url()}; модель: {model}")
    r = client.chat.completions.create(model=model, max_tokens=30,
                                       messages=[{"role": "user", "content": "Ответь одним словом: ок"}])
    print("Обычный ответ:", r.choices[0].message.content)
    r = client.chat.completions.create(
        model=model, tools=TOOLS, tool_choice="auto", max_tokens=100, temperature=0,
        messages=[{"role": "user", "content": "Покажи сведения об инциденте INC-101."}])
    calls = r.choices[0].message.tool_calls or []
    if calls:
        print("Tools поддерживаются:", [(c.function.name, c.function.arguments) for c in calls])
    else:
        print("Модель не вернула tool_calls. Проверьте поддержку tools у модели/сервера "
              "(в LM Studio — модель с поддержкой tool use) или выберите другую модель.")
        print("Ответ:", json.dumps(r.choices[0].message.content, ensure_ascii=False))
