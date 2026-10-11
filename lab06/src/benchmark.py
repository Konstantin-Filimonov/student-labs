import time
import json
import pandas as pd
from providers import local_client, cloud_client, LOCAL_MODEL, CLOUD_MODEL

def call_model(client, model, prompt, max_tokens=500, temperature=0.2, is_cloud=False, max_retries=5):
    messages = [
        {"role": "system", "content": "Отвечай точно и структурированно."},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(max_retries):
        t0 = time.perf_counter()
        first_token_time = None
        full_text = ""
        token_count = 0
        try:
            kwargs = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            if is_cloud:
                kwargs["extra_body"] = {"provider": {"allow_fallbacks": True}}

            stream = client.chat.completions.create(**kwargs)
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:                     # None-safe: пропускаем null
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - t0
                    full_text += delta
                    token_count += 1

            # Если контент пустой — считаем это неудачей и пробуем снова
            if not full_text.strip():
                raise ValueError("empty response")

            total_time = time.perf_counter() - t0
            return {
                "response": full_text,
                "ttft_s": round(first_token_time, 3) if first_token_time else None,
                "total_s": round(total_time, 3),
                "output_tokens": token_count,
                "tokens_per_sec": round(token_count / total_time, 2) if total_time > 0 else 0,
                "error": None
            }
        except Exception as e:
            err = str(e)
            if ("429" in err or "empty response" in err) and attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"  retry in {wait}s ({attempt+1}/{max_retries}): {err[:80]}")
                time.sleep(wait)
                continue
            return {
                "response": "",
                "ttft_s": None,
                "total_s": round(time.perf_counter() - t0, 3),
                "output_tokens": 0,
                "tokens_per_sec": 0,
                "error": err
            }

def run_benchmark(test_cases_path="data/test_cases.json", output_path="results/raw_results.csv"):
    with open(test_cases_path, "r", encoding="utf-8") as f:
        tests = json.load(f)

    rows = []
    for test in tests:
        test_id = test["id"]
        prompt = test["prompt"]
        print(f"Running {test_id}...")

        for model_name, client, model_id in [
            ("local", local_client, LOCAL_MODEL),
            ("cloud", cloud_client, CLOUD_MODEL)
        ]:
            res = call_model(client, model_id, prompt)
            row = {
                "test_id": test_id,
                "model": model_name,
                "latency_s": res["total_s"],
                "ttft_s": res["ttft_s"],
                "output_tokens": res["output_tokens"],
                "tokens_per_sec": res["tokens_per_sec"],
                "response": res["response"][:200],  # обрезаем для CSV
                "error": res["error"]
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df

if __name__ == "__main__":
    run_benchmark()