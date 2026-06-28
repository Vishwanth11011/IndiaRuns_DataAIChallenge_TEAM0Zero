from rank_bm25 import BM25Okapi
import numpy as np

# JD query for BM25 — use the most discriminating rare skill terms
BM25_QUERY_TERMS = [
    "embedding", "retrieval", "vector", "faiss", "pinecone", "qdrant", "milvus",
    "sentence-transformers", "bge", "e5", "hybrid search", "bm25", "ndcg", "mrr",
    "learning to rank", "ltr", "xgboost", "lightgbm", "rag", "lora", "qlora",
    "recommendation system", "ranking system", "search engineer", "nlp engineer",
    "machine learning engineer", "ai engineer", "data scientist",
    "production", "deployed", "shipped", "inference", "python",
]

def build_bm25_index(candidates: list[dict], features_list: list[dict]) -> BM25Okapi:
    """Build BM25 on career descriptions + skills text."""
    corpus = []
    for feat in features_list:
        text = feat.get("embedding_text", "")
        tokens = text.lower().split()
        corpus.append(tokens)
    return BM25Okapi(corpus)

def bm25_retrieve(bm25_index, top_k: int = 1000) -> list[tuple[int, float]]:
    """Return (index, score) for top_k candidates by BM25."""
    query_tokens = " ".join(BM25_QUERY_TERMS).lower().split()
    scores = bm25_index.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]
