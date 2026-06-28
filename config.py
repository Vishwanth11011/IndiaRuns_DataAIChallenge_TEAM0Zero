# config.py — all tunable parameters in one place

# LTR feature weights (used for pseudo-labeling + interpretability in deck)
SIGNAL_IMPORTANCE = {
    # Stage 1+2 retrieval signals
    "cross_encoder_score":        "PRIMARY — deep JD-candidate alignment",
    "biencoder_score":            "SECONDARY — broad semantic fit",
    "bm25_score":                 "SECONDARY — exact keyword hit on rare terms",

    # Career signals (highest business importance)
    "ml_role_fraction":           "HIGH — fraction of career in ML/AI roles",
    "product_company_fraction":   "HIGH — product vs services company experience",
    "ml_trajectory_score":        "HIGH — is career moving INTO ML/AI?",
    "strong_product_company_flag":"HIGH — recognizable product company (Swiggy, Razorpay etc.)",
    "entirely_services_career":   "PENALTY — entire career at TCS/Infosys etc.",

    # Skill signals
    "verified_skill_score":       "HIGH — Redrob assessment scores are ground truth",
    "core_skill_overlap_score":   "HIGH — weighted hit on JD required skills",
    "preferred_skill_score":      "MEDIUM — nice-to-have skills",
    "wrong_domain_flag":          "PENALTY — CV/Speech specialist with no NLP",

    # Redrob behavioral (availability multiplier)
    "recency_score":              "HIGH — last login; >6 months = functionally unavailable",
    "responsiveness_score":       "HIGH — recruiter_response_rate × avg_response_time",
    "notice_period_score":        "HIGH — JD explicitly wants ≤30 days",
    "github_score":               "MEDIUM — open source/production code signal",
    "market_demand_score":        "MEDIUM — saved_by_recruiters is social proof",
    "platform_engagement_score":  "MEDIUM — composite platform health signal",

    # Hard filters
    "honeypot_score":             "DISQUALIFIER — flag and remove from top 100",
    "is_disqualified_title":      "DISQUALIFIER — non-technical roles",
}
