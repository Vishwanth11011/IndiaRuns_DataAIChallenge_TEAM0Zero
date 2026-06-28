def generate_reasoning(candidate: dict, features: dict, score: float) -> str:
    """
    Generate a 1-2 sentence factual reasoning string.
    References real data: company names, skill names, signal values.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)

    # Find most relevant skills
    jd_core_lower = {
        "faiss", "pinecone", "qdrant", "milvus", "bm25", "lora", "qlora",
        "sentence-transformers", "elasticsearch", "ndcg", "xgboost", "lightgbm",
        "transformers", "rag", "nlp", "machine learning",
    }
    matched_skills = [
        s["name"] for s in candidate.get("skills", [])
        if s.get("name", "").lower() in jd_core_lower and
           s.get("proficiency") in ("advanced", "expert")
    ][:4]

    # Key behavioral signals
    rr = signals.get("recruiter_response_rate", 0)
    notice = signals.get("notice_period_days", 90)
    github = signals.get("github_activity_score", -1)
    open_to_work = signals.get("open_to_work_flag", False)

    # Build part 1: fit summary
    skill_str = ", ".join(matched_skills) if matched_skills else "relevant ML/AI skills"
    
    if company:
        part1 = f"{title} with {float(yoe):.0f} years of experience at {company}, with production-level expertise in {skill_str}."
    else:
        part1 = f"{title} with {float(yoe):.0f} years of experience, with production-level expertise in {skill_str}."

    # Build part 2: availability / behavioral note
    notes = []
    if open_to_work and rr >= 0.6 and notice <= 30:
        notes.append("Actively seeking roles, highly responsive, and available within 30 days.")
    elif open_to_work and rr >= 0.4:
        notes.append(f"Open to work with a {notice}-day notice period and good recruiter responsiveness.")
    elif not open_to_work:
        notes.append("Not currently flagged as open to work — may need passive outreach.")
    if github > 50:
        notes.append(f"Strong GitHub activity (score: {github:.0f}).")

    part2 = " ".join(notes) if notes else ""
    return (part1 + " " + part2).strip()
