from llm_client import ask_model, MODEL_NAME, MODEL_LABEL, list_available_models

if __name__ == "__main__":
    print(f"Проверка подключения к модели: {MODEL_NAME} ({MODEL_LABEL})")

    try:
        available = list_available_models()
        print(f"Модели, доступные на сервере сейчас: {available}")
        if MODEL_NAME not in available:
            print(
                "ВНИМАНИЕ: LLM_MODEL из .env не совпадает ни с одной из "
                "моделей, отданных сервером. Проверьте имя."
            )
    except Exception as exc:
        print(f"Не удалось получить список моделей: {exc}")

    print("-" * 60)

    answer, latency = ask_model(
        "Объясни разницу между хешированием и шифрованием."
    )
    print(answer)
    print(f"\nВремя ответа: {latency:.2f} c")
