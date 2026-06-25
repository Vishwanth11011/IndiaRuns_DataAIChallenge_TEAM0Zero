"""
pipeline/preprocessor.py
-------------------------
Text cleaning and candidate profile synthesis.

Key function: synthesize_candidate_text()
  Converts a structured candidate dict into a single rich-text string
  suitable for embedding. This is the primary input to the Bi-Encoder.

Also provides: load_jd_text()
  Extracts job description text from the .docx file.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalize text:
    - Strip HTML tags
    - Collapse whitespace
    - Remove non-printable characters
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove non-printable / control characters (keep newline, tab)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]", " ", text)
    # Collapse multiple whitespace into single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def proficiency_weight(proficiency: str) -> str:
    """Map proficiency level to a descriptive qualifier for embedding."""
    mapping = {
        "expert": "expert-level",
        "advanced": "advanced",
        "intermediate": "intermediate",
        "beginner": "beginner-level",
    }
    return mapping.get(proficiency, "")


# ─── Candidate Text Synthesis ──────────────────────────────────────────────────

def synthesize_candidate_text(candidate: dict, max_chars: int = 2000) -> str:
    """
    Synthesize a rich, embedding-ready text representation of a candidate.

    Strategy:
    1. Professional headline + summary (most signal-dense)
    2. Career history descriptions (concrete evidence of skills)
    3. Skills with proficiency (technical vocabulary for matching)
    4. Certifications (credentialing signal)
    5. Education (background context)
    6. Key metadata (years_of_experience, current_title)

    The output is truncated to max_chars to avoid exceeding model token limits.
    """
    parts = []

    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    certifications = candidate.get("certifications", [])
    education = candidate.get("education", [])

    # ── 1. Profile summary (most important for semantic matching) ──
    headline = clean_text(profile.get("headline", ""))
    summary = clean_text(profile.get("summary", ""))
    current_title = clean_text(profile.get("current_title", ""))
    yoe = profile.get("years_of_experience", 0)

    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)
    if current_title:
        parts.append(f"Current role: {current_title} with {yoe:.1f} years of experience.")

    # ── 2. Career history ──
    career_parts = []
    for job in career_history:
        title = clean_text(job.get("title", ""))
        company = clean_text(job.get("company", ""))
        desc = clean_text(job.get("description", ""))
        duration = job.get("duration_months", 0)
        company_size = job.get("company_size", "")

        job_text = ""
        if title and company:
            job_text = f"{title} at {company} ({duration} months, size: {company_size})"
        if desc:
            job_text += f": {desc}"
        if job_text:
            career_parts.append(job_text)

    if career_parts:
        parts.append("Career history: " + " | ".join(career_parts))

    # ── 3. Skills ──
    if skills:
        # Prioritize advanced/expert skills and those with more endorsements
        sorted_skills = sorted(
            skills,
            key=lambda s: (
                {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(s.get("proficiency", ""), 0),
                s.get("endorsements", 0),
                s.get("duration_months", 0),
            ),
            reverse=True,
        )
        skill_strs = [
            f"{proficiency_weight(s.get('proficiency', ''))} {s.get('name', '')}".strip()
            for s in sorted_skills[:20]  # Top 20 skills
        ]
        parts.append("Skills: " + ", ".join(filter(None, skill_strs)) + ".")

    # ── 4. Certifications ──
    if certifications:
        cert_strs = [f"{c.get('name', '')} ({c.get('issuer', '')})" for c in certifications]
        parts.append("Certifications: " + ", ".join(cert_strs) + ".")

    # ── 5. Education ──
    if education:
        edu_strs = []
        for edu in education:
            degree = edu.get("degree", "")
            field = edu.get("field_of_study", "")
            institution = edu.get("institution", "")
            tier = edu.get("tier", "")
            edu_strs.append(f"{degree} in {field} from {institution} ({tier})")
        parts.append("Education: " + " | ".join(edu_strs) + ".")

    full_text = " ".join(parts)

    # Truncate to max_chars to stay within model token limits
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]

    return full_text


# ─── JD Loader ────────────────────────────────────────────────────────────────

def load_jd_text(jd_path: str) -> str:
    """
    Extract and clean the job description text from a .docx file.
    Returns a clean string suitable for embedding.
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required: pip install python-docx")

    doc = Document(jd_path)
    paragraphs = [clean_text(para.text) for para in doc.paragraphs if para.text.strip()]
    jd_text = " ".join(paragraphs)
    logger.info(f"Loaded JD text ({len(jd_text)} chars) from {jd_path}")
    return jd_text


def load_jd_text_from_string(jd_text: str) -> str:
    """
    Clean and return a JD provided as a plain string.
    """
    cleaned = clean_text(jd_text)
    logger.info(f"Using provided JD text ({len(cleaned)} chars)")
    return cleaned


# ─── Quick EDA helper ─────────────────────────────────────────────────────────

def eda_summary(candidates: list) -> dict:
    """
    Run basic EDA on a list of candidate dicts.
    Returns a summary dict with key statistics.
    """
    import statistics

    n = len(candidates)
    if n == 0:
        return {"count": 0}

    yoe_vals = [c.get("profile", {}).get("years_of_experience", 0) for c in candidates]
    skills_counts = [len(c.get("skills", [])) for c in candidates]
    completeness = [c.get("redrob_signals", {}).get("profile_completeness_score", 0) for c in candidates]
    open_to_work = sum(1 for c in candidates if c.get("redrob_signals", {}).get("open_to_work_flag", False))

    return {
        "count": n,
        "yoe_mean": round(statistics.mean(yoe_vals), 2),
        "yoe_median": round(statistics.median(yoe_vals), 2),
        "yoe_min": round(min(yoe_vals), 2),
        "yoe_max": round(max(yoe_vals), 2),
        "skills_mean": round(statistics.mean(skills_counts), 2),
        "profile_completeness_mean": round(statistics.mean(completeness), 2),
        "open_to_work_pct": round(open_to_work / n * 100, 1),
    }
