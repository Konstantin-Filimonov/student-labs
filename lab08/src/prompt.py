SYSTEM_PROMPT = """Ты отвечаешь по базе знаний.
Используй только факты из блока CONTEXT.
После каждого существенного утверждения указывай источник вида [S1], [S2].
Если данных для уверенного ответа нет, напиши: "В предоставленных источниках недостаточно информации".
Не придумывай документы, страницы, ссылки и факты."""

def build_context(results):
    blocks = []
    for i, item in enumerate(results, start=1):
        blocks.append(
            f"[S{i}] source={item['source']} id={item['chunk_id']}\n{item['text']}"
        )
    return "\n\n".join(blocks)

def build_user_message(question, context):
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"