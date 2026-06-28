from sentence_transformers import CrossEncoder
import numpy as np
import torch

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CROSS_ENCODER_JD = """
We are hiring a Senior AI/ML Engineer. Required: production experience building embedding-based
retrieval, hybrid search (BM25 + dense), learning-to-rank, evaluation frameworks for search
(NDCG, MRR, MAP). Vector databases: FAISS, Pinecone, Qdrant, Milvus, Elasticsearch.
NLP, transformers, fine-tuning LLMs. Strong Python. Shipped ranking or recommendation systems
to real users. 5-9 years experience at a product company (not pure IT services). India-based.
"""

def cross_encode(features_list: list[dict], top_n: int = 300) -> list[tuple[int, float]]:
    """
    Re-rank the filtered candidate pool using a cross-encoder.
    features_list: already-filtered candidates (750 or fewer).
    Returns (original_index, score) sorted descending.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ WARNING: CrossEncoder running on CPU! Please enable GPU in Colab.")
        model = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
    else:
        model = CrossEncoder(
            CROSS_ENCODER_MODEL, 
            device=device, 
            model_kwargs={"torch_dtype": torch.float16}
        )

    pairs = [
        (CROSS_ENCODER_JD, feat.get("embedding_text", ""))
        for feat in features_list
    ]

    print(f"Cross-encoding {len(pairs)} candidates...")
    scores = model.predict(pairs, show_progress_bar=True, batch_size=256 if device == "cuda" else 32)

    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: -x[1])
    return [(idx, float(score)) for idx, score in indexed[:top_n]]
