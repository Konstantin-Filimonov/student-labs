"""Клиент совместимого с OpenAI API (LM Studio по умолчанию, как в ЛР8–10). Ключи — только из окружения/.env."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv не обязателен: переменные можно задать в окружении
    pass


def get_base_url():
    return os.getenv("LLM_BASE_URL") or os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")


def get_model_name():
    return os.getenv("LLM_MODEL") or os.getenv("LM_STUDIO_MODEL", "qwen2.5-3b-instruct")


def make_client(timeout=20.0):
    from openai import OpenAI  # импорт здесь: режимы --demo/--self-test работают без пакета openai
    api_key = os.getenv("LLM_API_KEY") or os.getenv("LM_STUDIO_API_KEY", "lm-studio")
    return OpenAI(base_url=get_base_url(), api_key=api_key, timeout=timeout, max_retries=0)
