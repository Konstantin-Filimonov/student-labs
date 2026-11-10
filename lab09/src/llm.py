import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        )
    return _client

def get_model_name():
    return os.getenv("LM_STUDIO_MODEL", "qwen2.5-3b-instruct")

def llm_chat(messages, temperature=0.2, max_tokens=500):
    client = get_client()
    resp = client.chat.completions.create(
        model=get_model_name(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content