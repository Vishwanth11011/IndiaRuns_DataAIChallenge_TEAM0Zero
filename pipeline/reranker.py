"""
pipeline/reranker.py
---------------------
Stage 2: Cross-Encoder Re-ranking.

Takes the top-K candidates from Stage 1 and performs a deep, pairwise
semantic evaluation of each (JD, candidate_text) pair using a Cross-Encoder.

The Cross-Encoder jointly processes both texts together, giving much higher
quality relevance scores than the Bi-Encoder's separate encoding, at the cost
of being O(K) forward passes (not feasible for all 50K candidates).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS-MARCO passage ranking (query-document relevance)
  - Returns raw logits → normalized via sigmoid to [0, 1]
  - Runs entirely on CPU; each forward pass ~5ms
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


class CrossEncoder:
    """
    Wraps sentence-transformers CrossEncoder for re-ranking.
    """

    def __init__(self, model_name: str):
        """
        Args:
            model_name: HuggingFace cross-encoder model name
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        try:
            from sentence_transformers import CrossEncoder as STCrossEncoder
            self.model = STCrossEncoder(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            )
        self.model_name = model_name
        logger.info("Cross-encoder ready.")

    def rerank(
        self,
        jd_text: str,
        candidates: List[Tuple[str, str, float, int]],
        top_k: int = 200,
        batch_size: int = 32,
    ) -> List[Tuple[str, float, float, int]]:
        """
        Re-rank candidates using the cross-encoder.

        Args:
            jd_text: The cleaned job description text
            candidates: List of (candidate_id, candidate_text, bi_score, orig_idx)
                        from Stage 1
            top_k: Number of candidates to return after re-ranking
            batch_size: Number of pairs per cross-encoder forward pass

        Returns:
            List of (candidate_id, cross_score_normalized, bi_score, orig_idx)
            sorted by cross_score descending, truncated to top_k.
        """
        n = len(candidates)
        logger.info(f"Stage 2: Cross-encoder scoring {n} candidates ...")

        # Build (query, document) pairs
        pairs = [(jd_text, cand_text) for _, cand_text, _, _ in candidates]

        # Run cross-encoder
        raw_scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=True,
        )

        # Normalize via sigmoid to [0, 1]
        normalized_scores = np.array([sigmoid(float(s)) for s in raw_scores])

        # Zip with candidate metadata
        results = [
            (
                candidates[i][0],   # candidate_id
                float(normalized_scores[i]),  # cross_score
                candidates[i][2],   # bi_score (from Stage 1)
                candidates[i][3],   # orig_idx
            )
            for i in range(n)
        ]

        # Sort by cross_score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Keep top-K
        results = results[:top_k]

        logger.info(
            f"Stage 2 complete: top-{top_k} selected. "
            f"Cross-score range: [{results[-1][1]:.4f}, {results[0][1]:.4f}]"
        )
        return results
