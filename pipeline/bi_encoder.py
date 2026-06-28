import numpy as np
import pickle, os, torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDINGS_CACHE = ".cache/candidate_embeddings.pkl"

JD_QUERY = """
Senior AI/ML Engineer with production experience in embedding-based retrieval systems,
vector databases (FAISS, Pinecone, Qdrant, Milvus, Elasticsearch), hybrid search (BM25 + dense),
semantic search, learning-to-rank (XGBoost, LightGBM), evaluation frameworks (NDCG, MRR, MAP),
NLP, transformers, fine-tuning LLMs (LoRA, QLoRA), RAG pipelines. Strong Python.
Shipped ranking, recommendation, or search systems to real users at a product company.
5-9 years experience. Based in India.
"""

class BiEncoder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("⚠️ WARNING: No GPU detected! Embedding 100K candidates on CPU will take hours. Please enable T4 GPU in Colab Runtime settings.")
            self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        else:
            print("✅ GPU detected. Loading model in FP16 (Half Precision) for maximum speed...")
            self.model = SentenceTransformer(
                MODEL_NAME, 
                device=self.device, 
                model_kwargs={"torch_dtype": torch.float16}
            )

    def embed_jd(self) -> np.ndarray:
        """Embed the JD with BGE instruction prefix for asymmetric retrieval."""
        instruction = "Represent this job description for finding matching candidates: "
        return self.model.encode(instruction + JD_QUERY, normalize_embeddings=True)

    def embed_candidates(self, features_list: list[dict]) -> np.ndarray:
        """Embed all candidates. Cache result to disk."""
        if os.path.exists(EMBEDDINGS_CACHE):
            with open(EMBEDDINGS_CACHE, "rb") as f:
                print("Loaded embeddings from cache.")
                return pickle.load(f)

        texts = [f.get("embedding_text", "") for f in features_list]
        print(f"Embedding {len(texts):,} candidates with {MODEL_NAME}...")
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=256 if self.device == "cuda" else 64,
        )
        os.makedirs(".cache", exist_ok=True)
        with open(EMBEDDINGS_CACHE, "wb") as f:
            pickle.dump(embeddings, f)
        return embeddings

    def retrieve_top_k(self, jd_emb, candidate_embs, top_k=1000) -> list[tuple[int, float]]:
        scores = cosine_similarity(jd_emb.reshape(1, -1), candidate_embs)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]
