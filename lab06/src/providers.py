import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

local_client = OpenAI(
    base_url=os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("LOCAL_API_KEY", "lm-studio")
)

cloud_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

LOCAL_MODEL = "qwen2.5-3b-instruct"
CLOUD_MODEL = "openrouter/free"