"""
pipeline/retriever.py
----------------------
Stage 1: Bi-Encoder Retrieval using cosine similarity.

Takes the JD embedding and all candidate embeddings, computes cosine similarity
(via dot product since embeddings are L2-normalized), and retrieves the top-K
candidates for Stage 2 re-ranking.

This is the "recall" stage: fast, approximate, broad.
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity_scores(
    jd_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between the JD embedding and all candidate embeddings.

    Since both are L2-normalized (by the BiEncoder), this reduces to a dot product:
        cos_sim = jd_emb @ cand_emb.T

    Args:
        jd_embedding: 1D array of shape (dim,)
        candidate_embeddings: 2D array of shape (n_candidates, dim)

    Returns:
        1D array of shape (n_candidates,) with similarity scores in [-1, 1]
    """
    # Ensure jd_embedding is 1D
    jd_emb = jd_embedding.flatten()
    # dot product = cosine similarity for normalized vectors
    scores = candidate_embeddings @ jd_emb
    return scores


def retrieve_top_k(
    jd_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_ids: List[str],
    top_k: int = 500,
) -> List[Tuple[str, float, int]]:
    """
    Retrieve the top-K candidates by cosine similarity to the JD.

    Args:
        jd_embedding: 1D embedding for the job description
        candidate_embeddings: 2D array of shape (n_candidates, dim)
        candidate_ids: List of candidate_id strings, aligned with embeddings
        top_k: Number of candidates to retrieve

    Returns:
        List of (candidate_id, cosine_score, original_index) tuples,
        sorted by score descending.
    """
    n_candidates = len(candidate_ids)
    actual_k = min(top_k, n_candidates)

    logger.info(f"Stage 1: Computing cosine similarity over {n_candidates:,} candidates ...")
    scores = cosine_similarity_scores(jd_embedding, candidate_embeddings)

    # Get top-k indices (efficient partial sort using argpartition)
    if actual_k < n_candidates:
        top_indices = np.argpartition(scores, -actual_k)[-actual_k:]
        # Sort these top-k by score descending
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    else:
        top_indices = np.argsort(scores)[::-1]

    results = [
        (candidate_ids[i], float(scores[i]), int(i))
        for i in top_indices
    ]

    logger.info(
        f"Stage 1 complete: retrieved top-{actual_k} candidates. "
        f"Score range: [{results[-1][1]:.4f}, {results[0][1]:.4f}]"
    )
    return results
