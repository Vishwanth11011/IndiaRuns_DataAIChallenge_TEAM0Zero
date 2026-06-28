"""
For each candidate, produce a normalized feature dict covering:
  - Career depth and quality
  - Skill relevance and verified depth
  - Education quality
  - Project/work alignment to the JD
  - Redrob behavioral signals
  - Trap/honeypot indicators
"""

from datetime import date, datetime
import re

# ── JD GROUND TRUTH: what the Senior AI Engineer role actually needs ─────────
# Hard skills: production retrieval/ranking systems
JD_CORE_SKILLS = {
    # Retrieval & Vector Infrastructure
    "faiss", "pinecone", "qdrant", "milvus", "weaviate", "opensearch",
    "elasticsearch", "chroma", "pgvector", "vespa",
    # Embeddings & Models
    "sentence-transformers", "sentence transformers", "bge", "e5",
    "openai embeddings", "cohere embeddings", "bi-encoder", "cross-encoder",
    # Ranking & Search
    "bm25", "hybrid search", "reciprocal rank fusion", "rrf", "ndcg", "mrr", "map",
    "learning to rank", "ltr", "xgboost", "lightgbm",
    # LLM & Fine-tuning
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "rag",
    "retrieval augmented generation", "llm", "langchain", "llama",
    # Core ML
    "machine learning", "deep learning", "nlp", "transformers",
    "pytorch", "tensorflow", "huggingface", "mlflow", "kubeflow",
    # Python / Engineering
    "python", "fastapi", "docker", "kubernetes", "spark", "kafka",
}

JD_PREFERRED_SKILLS = {
    "recommendation system", "ranking system", "search engine",
    "a/b testing", "feature engineering", "vector database",
    "information retrieval", "semantic search", "reranking",
}

# Titles that are red flags — person does not do ML/AI work
DISQUALIFIER_TITLES = {
    "accountant", "business analyst", "marketing manager", "hr", "sales",
    "content writer", "mechanical engineer", "civil engineer", "recruiter",
    "graphic designer", "financial analyst", "operations manager",
    "customer support", "product manager",  # PM is borderline — check career history
}

# Services firms — entire career here is a negative signal
SERVICES_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "l&t infotech", "ltimindtree", "persistent", "mastech",
}

# Product companies that are strong signals
STRONG_PRODUCT_COMPANIES = {
    "swiggy", "zomato", "zepto", "blinkit", "meesho", "razorpay",
    "phonepe", "paytm", "cred", "groww", "upstox", "slice",
    "freshworks", "zoho", "browserstack", "chargebee", "hasura",
    "sarvam", "krutrim", "setu", "hyperverge", "observe.ai",
    "facilio", "darwinbox", "leadsquared", "clevertap",
    "google", "meta", "microsoft", "amazon", "apple", "netflix",
    "uber", "airbnb", "stripe", "openai", "anthropic", "cohere",
    "flipkart", "myntra", "nykaa", "ola", "dunzo",
}


def extract_features(candidate: dict) -> dict:
    """
    Extract all features from a single candidate record.
    Returns a flat dict of named, numeric features all in [0, 1] unless noted.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    certs = candidate.get("certifications", [])
    signals = candidate.get("redrob_signals", {})

    features = {}
    features["candidate_id"] = candidate.get("candidate_id")

    # ── 1. CURRENT TITLE HARD FILTER ─────────────────────────────────────────
    current_title_lower = profile.get("current_title", "").lower()
    features["is_disqualified_title"] = float(
        any(t in current_title_lower for t in DISQUALIFIER_TITLES)
    )

    # ── 2. EXPERIENCE FIT ─────────────────────────────────────────────────────
    yoe = float(profile.get("years_of_experience", 0))
    # JD says 5-9 years, sweet spot is 6-8
    if 5 <= yoe <= 9:
        features["experience_fit"] = 1.0 - abs(yoe - 7) / 4.0
    elif yoe < 5:
        features["experience_fit"] = max(0.0, yoe / 5.0 * 0.7)  # ramp up to 70% at 5 years
    else:
        features["experience_fit"] = max(0.4, 1.0 - (yoe - 9) * 0.08)  # gentle decline after 9
    features["years_of_experience_raw"] = yoe

    # ── 3. CAREER HISTORY ANALYSIS ───────────────────────────────────────────
    # 3a. Fraction of career at product companies vs services
    total_months = sum(j.get("duration_months", 0) for j in career)
    services_months = 0
    product_months = 0
    ml_role_months = 0  # months spent in ML/AI-specific roles
    career_texts = []   # all role descriptions for semantic analysis

    ml_role_keywords = {
        "machine learning", "ml engineer", "data scientist", "ai engineer",
        "nlp engineer", "search engineer", "recommendation", "ranking engineer",
        "research scientist", "applied scientist", "mlops", "ml platform",
    }

    for job in career:
        company_lower = job.get("company", "").lower()
        title_lower = job.get("title", "").lower()
        dur = job.get("duration_months", 0)
        desc = job.get("description", "")
        industry = job.get("industry", "").lower()
        career_texts.append(f"{title_lower}. {desc}")

        # Services vs product classification
        if any(s in company_lower for s in SERVICES_FIRMS):
            services_months += dur
        elif any(p in company_lower for p in STRONG_PRODUCT_COMPANIES):
            product_months += dur

        # ML/AI role detection
        if any(kw in title_lower for kw in ml_role_keywords):
            ml_role_months += dur
        # Also check description for production ML work
        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in ["production", "deployed", "shipped", "real users", "inference"]):
            if any(kw in desc_lower for kw in ["model", "embedding", "ranking", "retrieval"]):
                ml_role_months += dur // 2  # partial credit for adjacent roles

    features["product_company_fraction"] = product_months / max(total_months, 1)
    features["services_company_fraction"] = services_months / max(total_months, 1)
    features["ml_role_fraction"] = min(ml_role_months / max(total_months, 1), 1.0)
    features["career_text_blob"] = " ".join(career_texts)  # for BM25 + embedding

    # 3b. Career trajectory — are they progressing INTO ML/AI?
    # Check if their last 2 roles are more ML-focused than their first roles
    if len(career) >= 2:
        recent_roles = career[:2]   # career_history[0] = most recent
        older_roles = career[-2:]
        recent_ml = sum(1 for j in recent_roles if any(kw in j.get("title","").lower() for kw in ml_role_keywords))
        older_ml  = sum(1 for j in older_roles  if any(kw in j.get("title","").lower() for kw in ml_role_keywords))
        features["ml_trajectory_score"] = (recent_ml / 2) - (older_ml / 2) * 0.5 + 0.5
        features["ml_trajectory_score"] = max(0.0, min(1.0, features["ml_trajectory_score"]))
    else:
        features["ml_trajectory_score"] = 0.5

    # 3c. Average tenure — penalize job-hopping (<12 months average)
    if total_months > 0 and len(career) > 0:
        avg_tenure = total_months / len(career)
        if avg_tenure >= 24:
            features["tenure_score"] = 1.0
        elif avg_tenure >= 15:
            features["tenure_score"] = 0.8
        elif avg_tenure >= 12:
            features["tenure_score"] = 0.6
        else:
            features["tenure_score"] = 0.3  # job hopper
    else:
        features["tenure_score"] = 0.5

    # 3d. Has worked at a strong product company (big positive signal)
    has_strong_product_co = any(
        any(p in j.get("company", "").lower() for p in STRONG_PRODUCT_COMPANIES)
        for j in career
    )
    features["strong_product_company_flag"] = float(has_strong_product_co)

    # 3e. Entirely services career (all jobs at services firms)
    features["entirely_services_career"] = float(
        services_months >= 0.85 * total_months and total_months > 12
    )

    # ── 4. SKILLS ANALYSIS ───────────────────────────────────────────────────
    skill_map = {s["name"].lower(): s for s in skills}

    # 4a. Core JD skill overlap (weighted by proficiency and duration)
    proficiency_weight = {"beginner": 0.25, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}
    core_skill_score = 0.0
    core_hits = 0
    for jd_skill in JD_CORE_SKILLS:
        if jd_skill in skill_map:
            s = skill_map[jd_skill]
            prof = proficiency_weight.get(s.get("proficiency", "beginner"), 0.25)
            # Duration bonus: 24+ months of a skill = full credit
            dur_months = s.get("duration_months", 0)
            dur_factor = min(dur_months / 24.0, 1.0)
            # Endorsement bonus: social proof
            endorsements = s.get("endorsements", 0)
            endorse_factor = min(endorsements / 20.0, 1.0)
            core_skill_score += prof * (0.6 + 0.25 * dur_factor + 0.15 * endorse_factor)
            core_hits += 1

    features["core_skill_overlap_score"] = min(core_skill_score / max(len(JD_CORE_SKILLS) * 0.3, 1), 1.0)
    features["core_skill_hit_count"] = core_hits

    # 4b. Preferred skill overlap
    pref_hits = sum(1 for s in JD_PREFERRED_SKILLS if s in skill_map)
    features["preferred_skill_score"] = min(pref_hits / max(len(JD_PREFERRED_SKILLS), 1), 1.0)

    # 4c. Skill assessment scores — VERIFIED skills from Redrob assessments
    # These are ground truth; a high assessment score >> a self-claimed "expert"
    assessment_scores = signals.get("skill_assessment_scores", {})
    if assessment_scores:
        jd_relevant_assessments = [
            v for k, v in assessment_scores.items()
            if any(jd_sk in k.lower() for jd_sk in JD_CORE_SKILLS)
        ]
        if jd_relevant_assessments:
            features["verified_skill_score"] = sum(jd_relevant_assessments) / len(jd_relevant_assessments) / 100.0
        else:
            # Has assessments but none are JD-relevant
            features["verified_skill_score"] = 0.3
    else:
        features["verified_skill_score"] = 0.2  # no assessments at all — uncertain

    # 4d. Compute vision / speech / robotics specialist penalty (wrong domain)
    cv_speech_skills = {"computer vision", "image classification", "object detection",
                        "speech recognition", "tts", "asr", "robotics", "ros"}
    cv_speech_count = sum(1 for s in skill_map if any(cv in s for cv in cv_speech_skills))
    nlp_ir_skills = {"nlp", "information retrieval", "search", "ranking", "embeddings",
                     "bert", "transformers", "rag", "vector", "faiss", "bm25"}
    nlp_ir_count = sum(1 for s in skill_map if any(nlp in s for nlp in nlp_ir_skills))
    features["wrong_domain_flag"] = float(cv_speech_count > 3 and nlp_ir_count == 0)

    # ── 5. EDUCATION ANALYSIS ────────────────────────────────────────────────
    tier_scores = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.25, "unknown": 0.35}
    degree_scores = {
        "ph.d": 1.0, "phd": 1.0, "doctor": 1.0,
        "m.tech": 0.9, "m.e.": 0.9, "m.s.": 0.85, "msc": 0.85, "m.sc": 0.85, "mba": 0.6,
        "b.tech": 0.75, "b.e.": 0.75, "b.sc": 0.65, "b.s.": 0.65,
    }
    cs_ai_fields = {
        "computer science", "computer engineering", "artificial intelligence",
        "machine learning", "data science", "information technology",
        "electronics", "electrical", "mathematics", "statistics",
    }

    best_edu_score = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        degree = edu.get("degree", "").lower()
        field = edu.get("field_of_study", "").lower()

        tier_score = tier_scores.get(tier, 0.35)
        deg_score = max((v for k, v in degree_scores.items() if k in degree), default=0.5)
        field_score = 0.9 if any(f in field for f in cs_ai_fields) else 0.4

        # Grade bonus (if available)
        grade = edu.get("grade", "")
        grade_bonus = 0.0
        if grade:
            gpa_match = re.search(r"(\d+\.?\d*)\s*(?:CGPA|GPA)", grade, re.IGNORECASE)
            pct_match  = re.search(r"(\d+\.?\d*)\s*%", grade)
            if gpa_match:
                gpa = float(gpa_match.group(1))
                grade_bonus = min((gpa - 6.0) / 4.0, 1.0) * 0.1 if gpa >= 6 else 0.0
            elif pct_match:
                pct = float(pct_match.group(1))
                grade_bonus = min((pct - 60) / 40, 1.0) * 0.1 if pct >= 60 else 0.0

        edu_score = tier_score * 0.4 + deg_score * 0.35 + field_score * 0.25 + grade_bonus
        best_edu_score = max(best_edu_score, edu_score)

    features["education_score"] = best_edu_score if education else 0.3

    # ── 6. CERTIFICATIONS ────────────────────────────────────────────────────
    ai_ml_cert_keywords = {
        "machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
        "aws certified", "google cloud", "azure ml", "databricks", "hugging face",
        "coursera", "deeplearning.ai", "fast.ai",
    }
    cert_score = 0.0
    for cert in certs:
        cert_name_lower = cert.get("name", "").lower()
        if any(kw in cert_name_lower for kw in ai_ml_cert_keywords):
            # Recency bonus: certifications from last 3 years count more
            cert_year = cert.get("year", 2020)
            recency = 1.0 if cert_year >= 2022 else 0.7 if cert_year >= 2020 else 0.4
            cert_score += recency * 0.25  # max 0.25 per cert

    features["certification_score"] = min(cert_score, 1.0)

    # ── 7. REDROB BEHAVIORAL SIGNALS ─────────────────────────────────────────
    # These are the most predictive signals for HIRING LIKELIHOOD, not just skill fit.
    # Use them as a multiplier on the final score.

    # 7a. Availability signals
    features["open_to_work"] = float(signals.get("open_to_work_flag", False))

    last_active_str = signals.get("last_active_date", "2024-01-01")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (date.today() - last_active).days
    except:
        days_inactive = 365

    if days_inactive <= 14:
        features["recency_score"] = 1.0
    elif days_inactive <= 30:
        features["recency_score"] = 0.9
    elif days_inactive <= 60:
        features["recency_score"] = 0.7
    elif days_inactive <= 90:
        features["recency_score"] = 0.5
    elif days_inactive <= 180:
        features["recency_score"] = 0.3
    else:
        features["recency_score"] = 0.1  # effectively unavailable

    # 7b. Responsiveness — critical for hiring pipeline
    rr = float(signals.get("recruiter_response_rate", 0.0))
    avg_rt = float(signals.get("avg_response_time_hours", 999))
    if rr >= 0.8 and avg_rt <= 24:
        features["responsiveness_score"] = 1.0
    elif rr >= 0.6 and avg_rt <= 72:
        features["responsiveness_score"] = 0.8
    elif rr >= 0.4:
        features["responsiveness_score"] = 0.6
    elif rr >= 0.2:
        features["responsiveness_score"] = 0.4
    else:
        features["responsiveness_score"] = 0.1  # treat as unavailable

    # 7c. Notice period — JD says sub-30 days ideal
    notice = int(signals.get("notice_period_days", 90))
    if notice <= 0:
        features["notice_period_score"] = 1.0   # immediately available
    elif notice <= 30:
        features["notice_period_score"] = 0.95
    elif notice <= 60:
        features["notice_period_score"] = 0.75
    elif notice <= 90:
        features["notice_period_score"] = 0.5
    else:
        features["notice_period_score"] = 0.2   # >90 days is a real friction point

    # 7d. Market demand signal — recruiters voting with their saves
    saved = int(signals.get("saved_by_recruiters_30d", 0))
    search_appear = int(signals.get("search_appearance_30d", 0))
    features["market_demand_score"] = min(saved / 15.0, 1.0) * 0.6 + min(search_appear / 200.0, 1.0) * 0.4

    # 7e. GitHub activity — production code and open-source signal
    github = float(signals.get("github_activity_score", -1))
    if github < 0:
        features["github_score"] = 0.2   # no GitHub linked — missing signal, slight penalty
    elif github >= 70:
        features["github_score"] = 1.0
    elif github >= 40:
        features["github_score"] = 0.75
    elif github >= 20:
        features["github_score"] = 0.5
    else:
        features["github_score"] = 0.25

    # 7f. Platform engagement composite
    apps_30d = int(signals.get("applications_submitted_30d", 0))
    profile_views = int(signals.get("profile_views_received_30d", 0))
    completeness = float(signals.get("profile_completeness_score", 0)) / 100.0
    interview_rate = float(signals.get("interview_completion_rate", 0.5))
    offer_acc = float(signals.get("offer_acceptance_rate", -1))
    if offer_acc < 0: offer_acc = 0.5  # no history = neutral

    features["platform_engagement_score"] = (
        completeness       * 0.20 +
        interview_rate     * 0.30 +
        offer_acc          * 0.15 +
        min(apps_30d / 5, 1.0)  * 0.15 +
        min(profile_views / 30, 1.0) * 0.10 +
        float(signals.get("verified_email", False)) * 0.05 +
        float(signals.get("verified_phone", False)) * 0.05
    )

    # 7g. Location fit — JD wants India (Pune/Noida) candidates
    country = profile.get("country", "").lower()
    location = profile.get("location", "").lower()
    relocate = bool(signals.get("willing_to_relocate", False))
    preferred_locations = {"pune", "noida", "delhi", "hyderabad", "mumbai", "bengaluru", "bangalore"}
    in_preferred_city = any(city in location for city in preferred_locations)
    is_india = country in ("india", "in", "ind")

    if is_india and in_preferred_city:
        features["location_score"] = 1.0
    elif is_india:
        features["location_score"] = 0.8
    elif relocate:
        features["location_score"] = 0.6
    else:
        features["location_score"] = 0.3  # overseas and won't relocate

    # ── 8. HONEYPOT DETECTION ────────────────────────────────────────────────
    # Honeypots are ~80 synthetic "too good to be true" profiles.
    # Characteristics: all signals maxed out (100/100 assessments, perfect response rate,
    # max github, max saved_by_recruiters) but career history doesn't add up.
    assessment_vals = list(signals.get("skill_assessment_scores", {}).values())
    all_assessments_perfect = len(assessment_vals) > 0 and all(v >= 95 for v in assessment_vals)
    all_signals_maxed = (
        rr >= 0.99 and
        github >= 98 and
        saved >= 30 and
        interview_rate >= 0.99 and
        completeness >= 99
    )
    title_career_mismatch = (
        features["is_disqualified_title"] > 0 and
        (all_assessments_perfect or all_signals_maxed)
    )
    features["honeypot_score"] = float(
        (all_assessments_perfect and all_signals_maxed) or
        title_career_mismatch or
        (all_signals_maxed and features["ml_role_fraction"] < 0.1)
    )

    # ── 9. TEXT BLOB FOR EMBEDDING ───────────────────────────────────────────
    # Carefully constructed text that emphasizes the most signal-rich parts.
    # Order: current role → career descriptions → skills with depth → education → certs
    skill_text = " ".join([
        f"{s['name']} ({s['proficiency']}, {s.get('duration_months', 0)} months)"
        for s in skills
        if s.get("proficiency") in ("advanced", "expert") or s.get("duration_months", 0) >= 18
    ])
    edu_text = " ".join([
        f"{e.get('degree', '')} {e.get('field_of_study', '')} {e.get('institution','')} {e.get('tier','')}"
        for e in education
    ])
    cert_text = " ".join([f"{c.get('name', '')} {c.get('issuer', '')}" for c in certs])
    summary = profile.get("summary", "")
    headline = profile.get("headline", "")

    # Repeat career descriptions twice — they are the richest signal
    features["embedding_text"] = " ".join([
        headline,
        summary,
        features["career_text_blob"],  # career descriptions
        features["career_text_blob"],  # repeat for emphasis
        skill_text,
        edu_text,
        cert_text,
    ])

    return features
