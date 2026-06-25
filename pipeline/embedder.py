"""
pipeline/embedder.py
---------------------
Bi-Encoder embedding module using sentence-transformers.

Encodes both the JD and candidate texts into dense vector representations.
Uses batching and optional on-disk caching for the full 50K candidate set.

Model: all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - Very fast on CPU (~14K sentences/sec on M2)
  - Great semantic quality for English text
  - Fits the "no network during ranking" constraint once cached
"""

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_model_cache_path(cache_dir: str, model_name: str, dataset_hash: str) -> str:
    """Return the path for a cached embedding file."""
    safe_model = model_name.replace("/", "__")
    return os.path.join(cache_dir, f"embeddings_{safe_model}_{dataset_hash}.pkl")


def _hash_texts(texts: List[str]) -> str:
    """Compute a short hash of the text list for cache invalidation."""
    combined = "||".join(texts[:100])  # Use first 100 texts as fingerprint
    return hashlib.md5(combined.encode()).hexdigest()[:12]


class BiEncoder:
    """
    Wraps sentence-transformers SentenceTransformer for batch encoding.
    Provides disk-based caching to avoid re-embedding the full candidate set.
    """

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model name (e.g. "all-MiniLM-L6-v2")
            cache_dir: Directory to store/load embedding cache. None = no cache.
        """
        logger.info(f"Loading bi-encoder model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Bi-encoder ready. Embedding dim: {self.model.get_sentence_embedding_dimension()}")

    def encode(
        self,
        texts: List[str],
        batch_size: int = 128,
        show_progress: bool = True,
        cache_key: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode a list of texts into normalized embeddings.

        Args:
            texts: List of strings to embed
            batch_size: Batch size for encoding (tune based on RAM)
            show_progress: Show tqdm progress bar
            cache_key: If provided, try to load/save from disk cache

        Returns:
            numpy array of shape (len(texts), embedding_dim), L2-normalized
        """
        # Try cache first
        if cache_key and self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_path):
                logger.info(f"Loading embeddings from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

        logger.info(f"Encoding {len(texts):,} texts (batch_size={batch_size}) ...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
            convert_to_numpy=True,
        )

        # Save to cache
        if cache_key and self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_path, "wb") as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings to cache: {cache_path}")

        logger.info(f"Done encoding. Shape: {embeddings.shape}")
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text string. Returns 1D array of shape (dim,).
        """
        embedding = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0]
