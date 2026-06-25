"""
pipeline/feature_engineer.py
------------------------------
Extract and normalize behavioral/numerical features from redrob_signals.

These features are combined with semantic scores in Stage 3 (Hybrid Fusion).

Key principles:
  - All output features are normalized to [0.0, 1.0]
  - Sentinel values (-1) are mapped to neutral/zero
  - Penalty flags are applied for JD-specific disqualifiers (consultancy-only careers)
  - Higher always = better (so we can do weighted sum safely)
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Optional

from config import (
    CONSULTANCY_COMPANIES,
    NOTICE_PERIOD_MAX_DAYS,
    GITHUB_SCORE_NO_ACCOUNT,
    OFFER_ACCEPTANCE_NO_HISTORY,
    PENALTY_CONSULTANCY_ONLY,
    JD_REQUIRED_SKILLS,
    RELEVANT_INDUSTRIES,
    IRRELEVANT_INDUSTRIES,
    RELEVANT_EDUCATION_FIELDS,
)

logger = logging.getLogger(__name__)

# Reference date for recency calculations
REFERENCE_DATE = date.today()


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """Parse ISO date string to date object."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _clip_norm(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clip and normalize value to [0, 1]."""
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def days_since_active_score(last_active_str: Optional[str]) -> float:
    """
    Convert last_active_date to a recency score [0, 1].
    Active yesterday → 1.0
    Active 365+ days ago → 0.0
    """
    last_active = _parse_date(last_active_str)
    if last_active is None:
        return 0.0
    days_ago = (REFERENCE_DATE - last_active).days
    # Decay: 0 days = 1.0, 365 days = 0.0, exponential decay
    score = max(0.0, 1.0 - (days_ago / 365.0))
    return round(score, 4)


def github_feature(github_score: float) -> float:
    """
    Normalize github_activity_score.
    -1 (no GitHub) → 0.0
    0–100 → 0.0–1.0
    """
    if github_score == GITHUB_SCORE_NO_ACCOUNT:
        return 0.0
    return round(_clip_norm(github_score, 0, 100), 4)


def offer_acceptance_feature(offer_rate: float) -> float:
    """
    Normalize offer_acceptance_rate.
    -1 (no history) → 0.5 (neutral, not penalized)
    0.0–1.0 → direct mapping
    """
    if offer_rate == OFFER_ACCEPTANCE_NO_HISTORY:
        return 0.5  # No history: don't penalize or reward
    return round(_clip_norm(offer_rate, 0.0, 1.0), 4)


def notice_score_feature(notice_days: int) -> float:
    """
    Convert notice period to a score where shorter = better.
    0 days → 1.0, 180 days → 0.0
    The JD explicitly prefers sub-30-day notice.
    """
    if notice_days is None:
        return 0.5
    # Extra boost for ≤30 day notice (JD explicitly prefers it)
    if notice_days <= 30:
        return 1.0
    return round(max(0.0, 1.0 - (notice_days / NOTICE_PERIOD_MAX_DAYS)), 4)


def skill_assessment_avg(skill_assessment_scores: dict) -> float:
    """
    Compute the average Redrob skill assessment score, normalized to [0, 1].
    Empty dict → 0.3 (slight penalty for no assessments completed).
    """
    if not skill_assessment_scores:
        return 0.3
    scores = list(skill_assessment_scores.values())
    avg = sum(scores) / len(scores)
    return round(_clip_norm(avg, 0, 100), 4)


def _is_consultancy_only_career(career_history: List[dict]) -> bool:
    """
    Returns True if the candidate's ENTIRE career has been at large IT
    consultancy/services companies (TCS, Infosys, Wipro, etc.).
    The JD explicitly says this is a red flag.
    """
    if not career_history:
        return False
    for job in career_history:
        company = job.get("company", "").lower().strip()
        # If ANY job is NOT at a consultancy, they're not consultancy-only
        if company not in CONSULTANCY_COMPANIES:
            return False
    return True


def skill_match_score(candidate: dict) -> float:
    """
    Compute the ratio of JD-required skills that the candidate possesses.
    Returns a value in [0, 1] where 1.0 = candidate has all required skills.
    
    This is the STRONGEST relevance signal after the cross-encoder.
    A candidate with 0 matching skills is almost certainly not a fit.
    """
    skills = candidate.get("skills", [])
    if not skills or not JD_REQUIRED_SKILLS:
        return 0.0
    
    candidate_skill_names = set()
    for skill in skills:
        name = skill.get("name", "").lower().strip()
        if name:
            candidate_skill_names.add(name)
    
    # Count how many JD-required skills the candidate has
    matched = 0
    for required in JD_REQUIRED_SKILLS:
        # Check both exact match and substring match
        if required in candidate_skill_names:
            matched += 1
        else:
            # Check if any candidate skill contains the required skill as substring
            for cand_skill in candidate_skill_names:
                if required in cand_skill or cand_skill in required:
                    matched += 1
                    break
    
    # Normalize: cap at a reasonable number (having 15+ matches = perfect)
    max_expected_matches = 15
    score = min(1.0, matched / max_expected_matches)
    return round(score, 4)


def industry_relevance_score(candidate: dict) -> float:
    """
    Score how relevant the candidate's industry experience is to ML/AI.
    
    Looks at current_industry and career_history industries.
    Returns:
      1.0 = strong tech/AI industry background
      0.5 = neutral/unknown
      0.0 = clearly irrelevant industry (construction, manufacturing)
    """
    industries = set()
    
    # Current industry
    current_industry = candidate.get("profile", {}).get("current_industry", "").lower().strip()
    if current_industry:
        industries.add(current_industry)
    
    # Career history industries  
    for job in candidate.get("career_history", []):
        ind = job.get("industry", "").lower().strip()
        if ind:
            industries.add(ind)
    
    if not industries:
        return 0.5  # Unknown = neutral
    
    relevant_count = 0
    irrelevant_count = 0
    
    for ind in industries:
        for rel in RELEVANT_INDUSTRIES:
            if rel in ind or ind in rel:
                relevant_count += 1
                break
        for irr in IRRELEVANT_INDUSTRIES:
            if irr in ind or ind in irr:
                irrelevant_count += 1
                break
    
    total = len(industries)
    if relevant_count > 0 and irrelevant_count == 0:
        return round(min(1.0, 0.6 + 0.4 * (relevant_count / total)), 4)
    elif irrelevant_count > 0 and relevant_count == 0:
        return round(max(0.0, 0.3 - 0.3 * (irrelevant_count / total)), 4)
    elif relevant_count > 0 and irrelevant_count > 0:
        # Mixed — give partial credit
        return round(0.5 * (relevant_count / (relevant_count + irrelevant_count)), 4)
    else:
        return 0.4  # No match either way — slightly below neutral


def education_relevance_score(candidate: dict) -> float:
    """
    Score how relevant the candidate's education is to ML/AI/CS.
    
    Returns:
      1.0 = CS/AI/Data Science degree
      0.5 = neutral/unknown
      0.0 = clearly unrelated field
    """
    education = candidate.get("education", [])
    if not education:
        return 0.4  # No education data = slightly below neutral
    
    best_score = 0.0
    for edu in education:
        field = edu.get("field_of_study", "").lower().strip()
        degree = edu.get("degree", "").lower().strip()
        tier = edu.get("tier", "unknown").lower()
        
        # Check field relevance
        field_relevant = False
        for rel_field in RELEVANT_EDUCATION_FIELDS:
            if rel_field in field or field in rel_field:
                field_relevant = True
                break
        
        if field_relevant:
            base = 0.8
            # Tier bonus
            if tier == "tier_1":
                base = 1.0
            elif tier == "tier_2":
                base = 0.9
            # Degree level bonus
            if "master" in degree or "m.tech" in degree or "m.s" in degree or "phd" in degree:
                base = min(1.0, base + 0.1)
            best_score = max(best_score, base)
        else:
            # Non-relevant field
            best_score = max(best_score, 0.2)
    
    return round(best_score, 4)


def extract_features(candidate: dict) -> Dict[str, float]:
    """
    Extract all behavioral and numerical features for a single candidate.
    Returns a dict of feature_name → normalized float [0, 1].

    Features:
      - profile_completeness     (0–1)
      - open_to_work             (0 or 1)
      - recruiter_response_rate  (0–1)
      - github_score             (0–1, -1 sentinel → 0)
      - interview_completion     (0–1)
      - offer_acceptance         (0–1, -1 sentinel → 0.5)
      - days_since_active_score  (0–1, recent = high)
      - skill_assess_avg         (0–1, empty → 0.3)
      - notice_score             (0–1, short = high)
      - consultancy_only_penalty (bool flag)
      - has_linkedin             (0 or 1, extra credibility)
      - connection_score         (0–1, normalized connection count)
      - endorsements_score       (0–1, normalized endorsements)
    """
    signals = candidate.get("redrob_signals", {})
    career_history = candidate.get("career_history", [])

    profile_completeness = _clip_norm(
        signals.get("profile_completeness_score", 50), 0, 100
    )
    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    recruiter_response = _clip_norm(signals.get("recruiter_response_rate", 0.0), 0, 1)
    github = github_feature(signals.get("github_activity_score", GITHUB_SCORE_NO_ACCOUNT))
    interview_completion = _clip_norm(signals.get("interview_completion_rate", 0.0), 0, 1)
    offer_acceptance = offer_acceptance_feature(
        signals.get("offer_acceptance_rate", OFFER_ACCEPTANCE_NO_HISTORY)
    )
    last_active = signals.get("last_active_date")
    active_score = days_since_active_score(last_active)
    skill_assess = skill_assessment_avg(signals.get("skill_assessment_scores", {}))
    notice = notice_score_feature(signals.get("notice_period_days", 90))
    consultancy_only = _is_consultancy_only_career(career_history)
    has_linkedin = 1.0 if signals.get("linkedin_connected", False) else 0.0

    # Connection count: log-normalize (0–500+ network)
    conn = signals.get("connection_count", 0)
    connection_score = round(min(1.0, conn / 500.0), 4)

    # Endorsements: clip at 100 and normalize
    endorsements = signals.get("endorsements_received", 0)
    endorsements_score = round(min(1.0, endorsements / 100.0), 4)

    return {
        "profile_completeness": round(profile_completeness, 4),
        "open_to_work": open_to_work,
        "recruiter_response_rate": round(recruiter_response, 4),
        "github_score": github,
        "interview_completion": round(interview_completion, 4),
        "offer_acceptance": offer_acceptance,
        "days_since_active_score": active_score,
        "skill_assess_avg": skill_assess,
        "notice_score": notice,
        "consultancy_only_penalty": consultancy_only,
        "has_linkedin": has_linkedin,
        "connection_score": connection_score,
        "endorsements_score": endorsements_score,
        # New features for improved ranking
        "skill_match_score": skill_match_score(candidate),
        "industry_relevance": industry_relevance_score(candidate),
        "education_relevance": education_relevance_score(candidate),
    }


def extract_features_batch(candidates: List[dict]) -> List[Dict[str, float]]:
    """
    Extract features for a batch of candidates.
    Returns a list of feature dicts in the same order.
    """
    return [extract_features(c) for c in candidates]
