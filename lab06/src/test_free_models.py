import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

candidates = [
    "qwen/qwen3-4b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "microsoft/phi-4-reasoning:free",
    "deepseek/deepseek-r1:free",
    "openrouter/free",
]

for model in candidates:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Ответь одним словом: ОК"}],
            max_tokens=5,
            timeout=30
        )
        print(f"OK   {model}: {r.choices[0].message.content!r}")
    except Exception as e:
        print(f"FAIL {model}: {str(e)[:150]}")