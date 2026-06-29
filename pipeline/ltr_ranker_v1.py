"""
LTR with XGBoost. All 31 features fed in; the model learns non-linear interactions.
Example: high semantic score AND high recruiter_response_rate → massive boost.
Low recency ALONE doesn't disqualify, but low recency AND low response rate = very low rank.
"""
import xgboost as xgb
import numpy as np

# Feature list in exact order for the model
FEATURE_NAMES = [
    # Career & Experience
    "experience_fit",
    "product_company_fraction",
    "services_company_fraction",
    "ml_role_fraction",
    "ml_trajectory_score",
    "tenure_score",
    "strong_product_company_flag",
    "entirely_services_career",
    # Skills
    "core_skill_overlap_score",
    "core_skill_hit_count_norm",   # normalize to [0,1]
    "preferred_skill_score",
    "verified_skill_score",
    "wrong_domain_flag",
    # Education & Certs
    "education_score",
    "certification_score",
    # Redrob Behavioral
    "open_to_work",
    "recency_score",
    "responsiveness_score",
    "notice_period_score",
    "market_demand_score",
    "github_score",
    "platform_engagement_score",
    "location_score",
    # Retrieval Scores (from earlier stages)
    "bm25_score_norm",
    "biencoder_score_norm",
    "cross_encoder_score_norm",
    # Honeypot
    "honeypot_score",
    # Disqualifier
    "is_disqualified_title",
]

def build_feature_matrix(features_list: list[dict],
                         stage_scores: dict) -> np.ndarray:
    """Assemble feature matrix for LTR."""
    X = []
    for i, feat in enumerate(features_list):
        row = []
        for fname in FEATURE_NAMES:
            if fname in feat:
                row.append(float(feat[fname]))
            elif fname in stage_scores:
                row.append(float(stage_scores[fname][i]))
            else:
                row.append(0.0)
        X.append(row)
    return np.array(X)

def ltr_rank(X: np.ndarray,
             pseudo_labels: np.ndarray,
             top_n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Train XGBoost on pseudo-labels from cross-encoder, then predict final ranking.
    Pseudo-labels: cross_encoder_score (already a quality signal).
    Hard-zero any honeypot or disqualified candidates.
    """
    # Hard filters: honeypots and disqualified titles get score=0
    honeypot_mask = X[:, FEATURE_NAMES.index("honeypot_score")] > 0.5
    disqual_mask  = X[:, FEATURE_NAMES.index("is_disqualified_title")] > 0.5
    hard_reject   = honeypot_mask | disqual_mask

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
    )
    group = [len(X)]  # all candidates are one group
    model.fit(X, pseudo_labels, group=group)

    scores = model.predict(X)
    scores[hard_reject] = -999.0  # force hard rejects to the bottom

    top_indices = np.argsort(scores)[::-1][:top_n]
    return top_indices, scores[top_indices]
