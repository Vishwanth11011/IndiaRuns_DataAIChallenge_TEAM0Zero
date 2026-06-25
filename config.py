"""
config.py — Centralized configuration for the IndiaRuns Candidate Ranking Pipeline
Tune Stage 3 hybrid weights here to optimize rankings.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

CANDIDATES_JSONL = os.path.join(DATASET_DIR, "candidates.jsonl")
SAMPLE_CANDIDATES_JSON = os.path.join(DATASET_DIR, "sample_candidates.json")
JD_DOCX = os.path.join(DATASET_DIR, "job_description.docx")
OUTPUT_CSV = os.path.join(BASE_DIR, "submission.csv")
EMBEDDING_CACHE_DIR = os.path.join(BASE_DIR, ".cache")

# ─── Stage 1: Bi-Encoder Retrieval ───────────────────────────────────────────
BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
# Alternative: "BAAI/bge-small-en-v1.5" — same 384-dim but better retrieval quality
# BI_ENCODER_MODEL = "BAAI/bge-small-en-v1.5"
STAGE1_TOP_K = 750          # Retrieve top-K from full candidate pool (wider net)
BATCH_SIZE = 128            # Embedding batch size (tune based on RAM)
MAX_TEXT_TOKENS = 512       # Max tokens for candidate text (model limit)

# ─── Stage 2: Cross-Encoder Re-ranking ───────────────────────────────────────
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
STAGE2_TOP_K = 300          # Keep top-K from Stage 1 to re-rank (wider to capture more relevant)

# Short JD summary for cross-encoder: the full JD is 9564 chars (~2400 tokens)
# but the cross-encoder has a 512-token combined limit (query+doc).
# Trimmed to maximize candidate text budget within the 512-token window.
JD_SUMMARY_FOR_CROSS_ENCODER = """Senior ML/AI Engineer: build candidate-JD ranking, semantic retrieval, matching systems.
Required: embeddings retrieval (sentence-transformers, BGE, E5), vector databases (Pinecone, Weaviate, Qdrant, FAISS),
Python, ranking evaluation (NDCG, MRR, MAP). 5-9 yrs applied ML/AI at product companies.
Shipped ranking/search/recommendation to production. LLM fine-tuning, learning-to-rank, NLP, information retrieval."""

# ─── Stage 3: Hybrid Fusion Weights ──────────────────────────────────────────
# These must sum to 1.0
# Cross-encoder semantic score MUST dominate to prevent irrelevant candidates
# from being promoted by high behavioral scores alone.
# *** PRIMARY TUNING LEVER — adjust these for leaderboard optimization ***
WEIGHTS = {
    "cross_encoder":           0.65,   # Deep semantic score (Stage 2) — PRIMARY signal
    "skill_match_score":       0.12,   # JD hard-skill match ratio — CRITICAL new signal
    "industry_relevance":      0.05,   # Industry alignment with JD
    "education_relevance":     0.03,   # CS/IT/AI degree relevance
    "github_score":            0.03,   # GitHub contribution activity
    "recruiter_response_rate": 0.03,   # Fraction of recruiter msgs responded to
    "interview_completion":    0.02,   # Fraction of interviews attended
    "profile_completeness":    0.02,   # Profile fill score
    "offer_acceptance":        0.01,   # Historical offer acceptance (reliability)
    "days_since_active_score": 0.02,   # Recency of login (availability signal)
    "skill_assess_avg":        0.01,   # Avg Redrob skill assessment score
    "open_to_work":            0.005,  # Explicit job-seeking flag
    "notice_score":            0.005,  # Short notice = easier to hire
}

# ─── JD-Specific Penalty Weights ─────────────────────────────────────────────
# These modifiers DOWN-WEIGHT candidates the JD explicitly says are bad fits.
# The JD says: consultancy-only backgrounds, non-active candidates are traps.
PENALTY_CONSULTANCY_ONLY = 0.30       # Fraction to reduce score if entire career = big IT services
CONSULTANCY_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "l&t infotech",
    "mindtree", "patni", "igate", "mastech", "niit technologies",
}

# ─── Title-Based Soft Boost/Penalty ──────────────────────────────────────────
# The JD is for a Senior ML/AI Engineer. We boost candidates whose current title
# signals ML/AI/data/software expertise, and penalize clearly irrelevant titles.
# These are SOFT modifiers (not hard filters) — semantic score still governs.
TITLE_BOOST_KEYWORDS = {
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "research engineer", "applied scientist",
    "recommendation", "search engineer", "ranking", "retrieval",
    "software engineer", "backend engineer", "data engineer",
    "full stack", "platform engineer", "deep learning",
    "ml", "artificial intelligence", "natural language processing",
}
TITLE_PENALTY_KEYWORDS = {
    "civil engineer", "mechanical engineer", "accountant", "hr manager",
    "content writer", "graphic designer", "sales executive", "marketing manager",
    "customer support", "operations manager", "project manager",
    "electrical engineer", "chemical engineer", "architect",
    "teacher", "professor", "lawyer", "doctor", "nurse", "pharmacist",
    "business analyst",
}
TITLE_BOOST_FACTOR = 0.35      # +35% score boost for ML/AI/SWE titles
TITLE_PENALTY_FACTOR = 0.65    # -65% score penalty for clearly irrelevant titles

# ─── Final Output ─────────────────────────────────────────────────────────────
FINAL_TOP_N = 100           # Number of candidates in final submission

# ─── Cross-Encoder Score Floor ────────────────────────────────────────────────
# If a candidate's normalized cross-encoder score is below this threshold,
# they are considered semantically irrelevant and get an additional penalty.
# This prevents behavioral signals from promoting garbage candidates.
CROSS_ENCODER_FLOOR = 0.15     # Below this = clearly not a semantic match
CROSS_ENCODER_FLOOR_PENALTY = 0.55  # 55% penalty for below-floor candidates

# ─── JD Required Skills (for skill-match scoring) ────────────────────────────
# Core skills the JD explicitly requires. Used to compute skill_match_score.
JD_REQUIRED_SKILLS = {
    # Core ML/AI - must haves
    "machine learning", "deep learning", "neural networks", "transformers",
    "nlp", "natural language processing",
    # Retrieval & ranking - the actual job
    "information retrieval", "semantic search", "vector search",
    "retrieval augmented generation", "rag", "learning to rank",
    "recommendation systems", "ranking systems",
    # Embeddings & models
    "sentence transformers", "embeddings", "bert", "gpt", "llm",
    "large language models", "fine-tuning", "lora", "qlora", "peft",
    # Vector DBs & infrastructure
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma",
    "opensearch", "elasticsearch", "vector database",
    # Frameworks
    "pytorch", "tensorflow", "hugging face", "huggingface",
    "scikit-learn", "sklearn", "xgboost", "lightgbm",
    # Evaluation
    "ndcg", "mrr", "a/b testing", "mlflow",
    # Core programming
    "python",
    # Data & infra
    "sql", "spark", "airflow", "kafka", "docker", "kubernetes",
    # General AI/data
    "data science", "statistical modeling", "feature engineering",
    "mlops", "model deployment",
}

# ─── Industry Relevance ──────────────────────────────────────────────────────
# Industries that indicate good/bad fit for an ML/AI Engineer role
RELEVANT_INDUSTRIES = {
    "information technology", "software", "technology", "artificial intelligence",
    "machine learning", "data science", "internet", "fintech", "e-commerce",
    "saas", "cloud computing", "cybersecurity", "analytics", "big data",
    "computer software", "it services",
}
IRRELEVANT_INDUSTRIES = {
    "construction", "real estate", "manufacturing", "civil engineering",
    "mechanical engineering", "agriculture", "mining", "textiles",
    "food & beverage", "hospitality", "healthcare", "pharmaceutical",
}

# ─── Education Relevance ─────────────────────────────────────────────────────
RELEVANT_EDUCATION_FIELDS = {
    "computer science", "computer engineering", "information technology",
    "data science", "artificial intelligence", "machine learning",
    "software engineering", "electrical engineering", "electronics",
    "mathematics", "statistics", "applied mathematics",
    "computational", "informatics",
}

# ─── Misc ─────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
NOTICE_PERIOD_MAX_DAYS = 180    # For normalization
GITHUB_SCORE_NO_ACCOUNT = -1    # Sentinel value for no GitHub account
OFFER_ACCEPTANCE_NO_HISTORY = -1  # Sentinel for no offer history
