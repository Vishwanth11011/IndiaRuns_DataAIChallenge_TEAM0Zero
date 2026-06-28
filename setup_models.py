from sentence_transformers import SentenceTransformer, CrossEncoder

print("Downloading and caching BGE large model...")
bi_encoder = SentenceTransformer("BAAI/bge-large-en-v1.5")
print("BGE model cached.")

print("Downloading and caching Cross-Encoder model...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-Encoder model cached.")

print("Models cached. You're good to go.")
