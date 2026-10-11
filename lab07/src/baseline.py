from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf_index(texts):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def keyword_search(query, vectorizer, matrix, texts, meta, k=5):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    order = scores.argsort()[::-1][:k]
    return [
        {"rank": i + 1, "score": float(scores[idx]), "text": texts[idx], "meta": meta[idx]}
        for i, idx in enumerate(order)
    ]