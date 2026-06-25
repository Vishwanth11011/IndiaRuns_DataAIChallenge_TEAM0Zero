"""
pipeline/formatter.py
----------------------
Output generation and submission formatting.

Converts the final ranked candidate list into a valid submission.csv
matching the challenge spec:
  - Exactly 100 rows (ranks 1–100)
  - Columns: candidate_id, rank, score, reasoning
  - Scores non-increasing by rank
  - Tie-break: candidate_id ascending
  - UTF-8 encoded, comma-separated

Also provides generate_reasoning() to produce human-readable reasoning strings
in the format shown in sample_submission.csv.
"""

import csv
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def count_ai_core_skills(candidate: dict) -> int:
    """
    Count the number of 'AI core skills' a candidate has.

    AI core skills are skills relevant to the JD (AI/ML Engineer role):
    embeddings, NLP, retrieval, ranking, vector DBs, LLMs, transformers, etc.

    This matches the reasoning format in sample_submission.csv:
    "ML Engineer with 6.4 yrs; 4 AI core skills; response rate 0.88."
    """
    AI_CORE_SKILLS = {
        # Core ML/AI
        "machine learning", "deep learning", "neural networks", "transformers",
        "nlp", "natural language processing", "computer vision",
        # Retrieval & ranking
        "information retrieval", "semantic search", "vector search",
        "retrieval augmented generation", "rag", "learning to rank",
        "recommendation systems", "ranking systems",
        # Embeddings & models
        "sentence transformers", "embeddings", "openai embeddings",
        "bert", "gpt", "llm", "large language models", "fine-tuning llms",
        "lora", "qlora", "peft", "instruction tuning",
        # Vector DBs & infrastructure
        "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma",
        "opensearch", "elasticsearch", "vector database",
        # Frameworks
        "pytorch", "tensorflow", "hugging face", "huggingface", "langchain",
        "scikit-learn", "sklearn", "xgboost", "lightgbm",
        # Data & evaluation
        "mlflow", "weights & biases", "wandb", "ndcg", "mrr", "a/b testing",
        "feature engineering", "data science", "statistical modeling",
        # Other AI signals from dataset
        "image classification", "object detection", "speech recognition",
        "tts", "text to speech", "gans", "generative ai",
        "model deployment", "mlops", "bentoml", "triton",
        # Python/data stack
        "python", "sql", "spark", "airflow", "dbt", "kafka",
        "apache beam", "databricks", "snowflake",
    }
    skills = candidate.get("skills", [])
    count = 0
    for skill in skills:
        skill_name = skill.get("name", "").lower().strip()
        if skill_name in AI_CORE_SKILLS:
            count += 1
    return count


def generate_reasoning(
    candidate: dict,
    rank: int,
    final_score: float,
) -> str:
    """
    Generate a concise reasoning string for the submission CSV.

    Format (matches sample_submission.csv):
    "{title} with {yoe} yrs; {n_ai_skills} AI core skills; response rate {rate:.2f}."

    Args:
        candidate: Full candidate dict
        rank: Final rank (1-based)
        final_score: The hybrid score

    Returns:
        Reasoning string, e.g.:
        "ML Engineer with 6.4 yrs; 4 AI core skills; response rate 0.88."
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Professional")
    yoe = profile.get("years_of_experience", 0)
    response_rate = signals.get("recruiter_response_rate", 0.0)
    n_ai_skills = count_ai_core_skills(candidate)

    reasoning = (
        f"{title} with {yoe:.1f} yrs; "
        f"{n_ai_skills} AI core skills; "
        f"response rate {response_rate:.2f}."
    )
    return reasoning


def format_submission(
    scored_candidates: List[Tuple[str, float, float, float]],  # (id, final_score, ce_score, bi_score)
    candidates_by_id: Dict[str, dict],
    output_path: str,
    top_n: int = 100,
) -> str:
    """
    Generate and write the final submission CSV.

    Args:
        scored_candidates: Sorted (descending) list from scorer.score_all_candidates()
        candidates_by_id: Dict mapping candidate_id → full candidate dict
        output_path: Path to write submission.csv
        top_n: Number of rows to output (must be 100)

    Returns:
        Path to the written CSV file.
    """
    if len(scored_candidates) < top_n:
        available = len(scored_candidates)
        import logging as _log
        _log.getLogger(__name__).warning(
            f"Only {available} scored candidates available; submission will have {available} rows (need {top_n} for real submission)."
        )
        top_n = available

    # Take top-N
    top_candidates = scored_candidates[:top_n]

    # Ensure scores are non-increasing (handle floating point ties)
    prev_score = None
    rows = []
    for rank, (cand_id, final_score, ce_score, bi_score) in enumerate(top_candidates, start=1):
        if prev_score is not None and final_score > prev_score:
            final_score = prev_score  # Enforce non-increasing
        prev_score = final_score

        candidate = candidates_by_id.get(cand_id, {})
        reasoning = generate_reasoning(candidate, rank, final_score)

        rows.append({
            "candidate_id": cand_id,
            "rank": rank,
            "score": f"{final_score:.4f}",
            "reasoning": reasoning,
        })

    # Handle tie-breaking: equal scores must be sorted by candidate_id ascending
    # (this is already handled by our sort in scorer, but let's be explicit)
    # Group consecutive equal scores and sort within group by candidate_id
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) - 1 and rows[j]["score"] == rows[j + 1]["score"]:
            j += 1
        if j > i:
            # Tie group from i to j — sort by candidate_id ascending
            group = rows[i:j + 1]
            group.sort(key=lambda r: r["candidate_id"])
            # Re-assign ranks to the sorted group
            for k, row in enumerate(group):
                row["rank"] = i + 1 + k
            rows[i:j + 1] = group
        i = j + 1

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Submission written to: {output_path}")
    logger.info(f"Top-5 candidates:")
    for row in rows[:5]:
        logger.info(f"  #{row['rank']} {row['candidate_id']} ({row['score']}) — {row['reasoning']}")

    return output_path
