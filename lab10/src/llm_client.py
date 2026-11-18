"""Клиент совместимого с OpenAI API (LM Studio по умолчанию, как в ЛР8–9). Ключи — только из окружения/.env."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_base_url():
    return os.getenv("LLM_BASE_URL") or os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")


def get_model_name():
    return os.getenv("LLM_MODEL") or os.getenv("LM_STUDIO_MODEL", "qwen2.5-3b-instruct")


def make_client(timeout=120.0):
    api_key = os.getenv("LLM_API_KEY") or os.getenv("LM_STUDIO_API_KEY", "lm-studio")
    return OpenAI(base_url=get_base_url(), api_key=api_key, timeout=timeout)
