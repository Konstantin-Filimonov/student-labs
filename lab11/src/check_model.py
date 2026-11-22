"""Проверка ДО основной работы: сервер отвечает, модель вызывает инструменты (tool_calls).

Запуск: python src/check_model.py
Если tool_calls не приходят — выберите другую модель (в LM Studio: с поддержкой tool use) и обновите .env.
"""
import json
import platform
import sys

from llm_client import get_base_url, get_model_name, make_client
from tools import TOOLS

if __name__ == "__main__":
    client, model = make_client(60.0), get_model_name()
    print(f"Python {sys.version.split()[0]}, {platform.platform()}")
    print(f"API: {get_base_url()}; модель: {model}")
    try:
        r = client.chat.completions.create(model=model, max_tokens=30, temperature=0,
                                           messages=[{"role": "user", "content": "Ответь одним словом: ок"}])
        print("Обычный ответ:", r.choices[0].message.content)
        r = client.chat.completions.create(
            model=model, tools=TOOLS, tool_choice="auto", max_tokens=100, temperature=0,
            messages=[{"role": "user", "content": "Покажи сведения об инциденте INC-101."}])
    except Exception as exc:
        sys.exit(f"Ошибка обращения к серверу ({type(exc).__name__}). Запущен ли Local Server в LM Studio, "
                 "загружена ли модель, совпадает ли имя модели в .env?")
    calls = r.choices[0].message.tool_calls or []
    if calls:
        print("Tool calling поддерживается:", [(c.function.name, c.function.arguments) for c in calls])
    else:
        print("Модель НЕ вернула tool_calls: выберите другую модель.")
        print("Ответ:", json.dumps(r.choices[0].message.content, ensure_ascii=False))
