"""
pipeline/scorer.py
-------------------
Stage 3: Hybrid Fusion Scoring.

Combines the Cross-Encoder semantic score (Stage 2) with the normalized
behavioral/platform features (from feature_engineer.py) into a single
final_score per candidate.

final_score = Σ (weight_i * feature_i) - consultancy_penalty (if applicable)

The weights are tunable via config.WEIGHTS. This is the primary optimization
lever for leaderboard performance.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    WEIGHTS,
    PENALTY_CONSULTANCY_ONLY,
    TITLE_BOOST_KEYWORDS,
    TITLE_PENALTY_KEYWORDS,
    TITLE_BOOST_FACTOR,
    TITLE_PENALTY_FACTOR,
    CROSS_ENCODER_FLOOR,
    CROSS_ENCODER_FLOOR_PENALTY,
)

logger = logging.getLogger(__name__)


def compute_final_score(
    cross_score: float,
    features: Dict[str, float],
) -> float:
    """
    Compute the final hybrid score for a single candidate.

    Args:
        cross_score: Normalized cross-encoder score [0, 1]
        features: Dict of behavioral features from feature_engineer.extract_features()

    Returns:
        Final score in [0, 1] (approximately)
    """
    # Weighted sum of all components
    score = (
        WEIGHTS["cross_encoder"]           * cross_score
        + WEIGHTS["skill_match_score"]     * features.get("skill_match_score", 0)
        + WEIGHTS["industry_relevance"]    * features.get("industry_relevance", 0.4)
        + WEIGHTS["education_relevance"]   * features.get("education_relevance", 0.4)
        + WEIGHTS["github_score"]          * features.get("github_score", 0)
        + WEIGHTS["recruiter_response_rate"] * features.get("recruiter_response_rate", 0)
        + WEIGHTS["interview_completion"]  * features.get("interview_completion", 0)
        + WEIGHTS["profile_completeness"]  * features.get("profile_completeness", 0)
        + WEIGHTS["offer_acceptance"]      * features.get("offer_acceptance", 0.5)
        + WEIGHTS["days_since_active_score"] * features.get("days_since_active_score", 0)
        + WEIGHTS["skill_assess_avg"]      * features.get("skill_assess_avg", 0.3)
        + WEIGHTS["open_to_work"]          * features.get("open_to_work", 0)
        + WEIGHTS["notice_score"]          * features.get("notice_score", 0.5)
    )

    # Apply penalty for consultancy-only careers (JD explicitly says this is a red flag)
    if features.get("consultancy_only_penalty", False):
        score = score * (1.0 - PENALTY_CONSULTANCY_ONLY)

    # Apply cross-encoder floor penalty: if the semantic score is too low,
    # the candidate is likely irrelevant regardless of behavioral signals
    if cross_score < CROSS_ENCODER_FLOOR:
        score = score * (1.0 - CROSS_ENCODER_FLOOR_PENALTY)

    return round(float(score), 6)


def _get_title_modifier(candidate: dict) -> float:
    """
    Return a score modifier based on the candidate's current job title.
    +TITLE_BOOST_FACTOR for ML/AI/SWE titles (good fit for this JD)
    -TITLE_PENALTY_FACTOR for clearly irrelevant titles (Civil Eng, Accountant, etc.)
    0.0 for neutral/unknown titles
    """
    if not candidate:
        return 0.0
    title = candidate.get("profile", {}).get("current_title", "").lower()
    for keyword in TITLE_BOOST_KEYWORDS:
        if keyword in title:
            return TITLE_BOOST_FACTOR
    for keyword in TITLE_PENALTY_KEYWORDS:
        if keyword in title:
            return -TITLE_PENALTY_FACTOR
    return 0.0


def _renormalize_cross_scores(
    candidates: List[Tuple[str, float, float, int]],
) -> List[Tuple[str, float, float, int]]:
    """
    Re-normalize cross-encoder scores from their raw compressed range
    (e.g., [0.004, 0.079]) to a wider [0, 1] range using min-max scaling.
    This ensures the cross-encoder's 70% weight is meaningful.
    """
    if not candidates:
        return candidates
    scores = [c[1] for c in candidates]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return candidates
    return [
        (cid, (s - min_s) / (max_s - min_s), bi, idx)
        for cid, s, bi, idx in candidates
    ]


def score_all_candidates(
    reranked_candidates: List[Tuple[str, float, float, int]],
    all_features: Dict[str, Dict[str, float]],
    candidates_by_id: Optional[Dict[str, dict]] = None,
) -> List[Tuple[str, float, float, float]]:
    """
    Compute final hybrid scores for all re-ranked candidates.

    Args:
        reranked_candidates: Output from Stage 2 cross-encoder
        all_features: Dict mapping candidate_id → feature dict
        candidates_by_id: Optional dict for title boost/penalty lookup

    Returns:
        List of (candidate_id, final_score, cross_score, bi_score),
        sorted by final_score descending.
    """
    logger.info(f"Stage 3: Computing hybrid scores for {len(reranked_candidates)} candidates ...")

    # Re-normalize cross-encoder scores to [0,1] to ensure they carry their
    # intended 70% weight (raw scores are compressed: ~0.004-0.079)
    reranked_normalized = _renormalize_cross_scores(reranked_candidates)

    results = []
    for cand_id, cross_score, bi_score, orig_idx in reranked_normalized:
        features = all_features.get(cand_id, {})
        final_score = compute_final_score(cross_score, features)

        # Apply title-based soft boost/penalty
        if candidates_by_id:
            candidate = candidates_by_id.get(cand_id, {})
            title_mod = _get_title_modifier(candidate)
            final_score = final_score * (1.0 + title_mod)

        results.append((cand_id, round(final_score, 6), cross_score, bi_score))

    # Sort by final score descending
    results.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        f"Stage 3 complete. Score range: [{results[-1][1]:.4f}, {results[0][1]:.4f}]"
    )
    return results


def normalize_scores_to_range(
    scored_candidates: List[Tuple[str, float, float, float]],
    score_min: float = 0.20,
    score_max: float = 0.99,
) -> List[Tuple[str, float, float, float]]:
    """
    Re-scale final scores to a human-readable [score_min, score_max] range
    while preserving rank order and relative spacing.

    The sample submission shows scores from 0.99 to 0.20 (decreasing).
    This normalization ensures our output matches expected format.

    Args:
        scored_candidates: Sorted list of (candidate_id, final_score, cross_score, bi_score)
        score_min: Score for last-ranked candidate (rank 100)
        score_max: Score for top-ranked candidate (rank 1)

    Returns:
        Same list with rescaled final_scores
    """
    if not scored_candidates:
        return scored_candidates

    raw_scores = [s[1] for s in scored_candidates[:100]]
    if not raw_scores:
        return scored_candidates

    raw_min = min(raw_scores)
    raw_max = max(raw_scores)

    if raw_max == raw_min:
        # All scores are equal — distribute evenly
        n = len(raw_scores)
        arange = np.arange(n)
        rescaled = (score_max - (score_max - score_min) * arange / max(n - 1, 1)).tolist()
    else:
        raw_arr = np.array(raw_scores)
        # Linear rescale
        rescaled = (score_min + (raw_arr - raw_min) / (raw_max - raw_min) * (score_max - score_min)).tolist()

    result = []
    for i, (cand_id, _, cross_score, bi_score) in enumerate(scored_candidates[:100]):
        result.append((cand_id, round(rescaled[i], 4), cross_score, bi_score))

    # Ensure strictly non-increasing (handle floating point issues)
    for i in range(1, len(result)):
        if result[i][1] > result[i - 1][1]:
            result[i] = (result[i][0], result[i - 1][1], result[i][2], result[i][3])

    return result
