# 🏆 IndiaRuns Data & AI Challenge — Candidate Ranking Pipeline

## 📖 Project Overview
The objective of this project is to build an intelligent, production-grade candidate ranking pipeline. Given a massive dataset of 100,000 candidate profiles from the Redrob platform (`candidates.jsonl`) and a specific Job Description for a "Senior AI/ML Engineer", our system processes, filters, and mathematically ranks the candidates to output the definitive top 100 matches in a structured format (`submission.csv`).

To achieve this, we moved away from simple keyword matching and built a rigorous 4-stage architecture combining state-of-the-art semantic search, behavioral signal fusion, and Learning-to-Rank (LTR) algorithms.

---

## 📊 Dataset Analysis & Schema Insights
Our approach was driven by a deep analysis of the provided `candidate_schema.json` and the dataset itself. Key insights that shaped our architecture include:

1. **Complex Nested Histories:** Candidates have rich, nested temporal data (education, employment history). We realized we needed to synthesize this into a unified `embedding_text` blob for semantic search, while simultaneously extracting hard numerical features (e.g., years of experience).
2. **Behavioral vs. Semantic Signals:** A candidate might be a perfect semantic match but have a 120-day notice period or a 0% recruiter response rate. We determined that pure semantic search was insufficient; we needed a Learning-to-Rank layer to fuse behavioral platform signals with NLP scores.
3. **Honeypots & Anomalies:** During exploratory data analysis (EDA), we discovered "honeypot" profiles—candidates with 100% skill matches but non-technical current roles (e.g., "Accountant" or "HR Manager"). This necessitated strict, programmatic anomaly detection.
4. **Verified Skills:** The schema differentiates between self-reported skills and Redrob `verified_skill_score`. We heavily prioritized the latter for ground-truth accuracy.

---

## 🏗️ The 4-Stage Architecture

```
100,000 Candidates (candidates.jsonl)
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │  Stage 1: Hybrid Retrieval (100K → 750)     │
 │  • Bi-Encoder: BAAI/bge-large-en-v1.5       │
 │  • BM25 Exact Keyword Match (faiss, qlora)  │
 │  • Merged via Reciprocal Rank Fusion (RRF)  │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │  Stage 2: Cross-Encoder Reranking           │
 │  • Model: ms-marco-MiniLM-L-6-v2            │
 │  • Deep Pairwise JD ↔ Candidate alignment   │
 │  • Filters 750 → 300 candidates             │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │  Stage 3: Learning-to-Rank (XGBoost)        │
 │  • Objective: rank:ndcg                     │
 │  • Trained on exactly 31 extracted signals  │
 │  • Learns non-linear career/skill combos    │
 │  • Filters 300 → 100 final candidates       │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │  Stage 4: Factual Reason Generation         │
 │  • Deterministic string generation          │
 │  • Cites verified skills, notice period,    │
 │    and GitHub activity for the top 100.     │
 └─────────────────────────────────────────────┘
        │
        ▼
  submission.csv (100 rows)
```

---

## ✨ Key Decisions & Innovations

### 1. 31 Rigorous Signal Extractions
Instead of relying on fuzzy heuristics, our `feature_extractor.py` algorithmically engineers 31 highly discriminative signals from the dataset. 
* **Career Trajectory (`ml_trajectory_score`)**: Analyzes temporal job changes to detect if a candidate's career is actively moving *into* ML/AI, or if they are moving *out* into management.
* **Product vs. Services Distinction**: Explicitly flags candidates working at top product companies and penalizes entire-career service engineers based on our analysis of industry hiring preferences.

### 2. XGBoost LTR Engine (`rank:ndcg`)
Instead of guessing linear weights for our 31 features, we use a true Learning-to-Rank approach. We bucket the deep semantic Cross-Encoder scores into 5-tier integer relevance labels and train an XGBoost Ranker (`objective="rank:ndcg"`) on the fly. This allows the model to learn that a high semantic score combined with a 30-day notice period is exponentially better than a high semantic score with a 120-day notice period.

### 3. Honeypot Elimination
To handle the data anomalies discovered in our analysis, the pipeline implements strict anomaly detection, masking invalid profiles' final LTR scores to `-999.0` to guarantee they are mathematically eliminated from the top 100.

### 4. Trained and Executed on Google Colab (FP16 GPU)
Running `BAAI/bge-large-en-v1.5` on 100,000 massive text blobs is computationally heavy. We trained and executed this pipeline on **Google Colab using a T4 GPU**. 
By explicitly binding our PyTorch models to `torch.float16`, we successfully utilized FP16 Tensor Cores, halving VRAM usage, allowing batch sizes of 256, and slashing a 10+ hour CPU workload down to mere minutes.

---

## 🚀 Running the Pipeline

It is highly recommended to run this on **Google Colab (T4 GPU)**. 

### 1. Prepare your Environment
Install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Run the Execution Flow
Run the full pipeline on your dataset. The script automatically handles loading, feature extraction, embedding, ranking, and validation.

```bash
python3 rank.py --candidates ./Dataset/candidates.jsonl --out ./submission.csv --validate
```

> **Note on Caching:** The first run will generate a 1.024-dimensional embedding matrix for all candidates and save it to `.cache/candidate_embeddings.pkl`. Subsequent runs will load this cache automatically, dropping execution time to under 60 seconds!

---

## 📁 Project Structure

```
IndiaRuns_Datachallenge/
├── rank.py                     # Main execution and validation flow
├── config.py                   # Centralized weights and hyperparameter config
├── setup_models.py             # Script to pre-download HuggingFace models
├── requirements.txt            # All necessary PyTorch/XGBoost dependencies
├── README.md                   # You are here
└── pipeline/
    ├── loader.py               # Memory-efficient JSONL streaming
    ├── feature_extractor.py    # 31-signal extraction engine and honeypot filters
    ├── bm25_retriever.py       # Exact keyword lexical search
    ├── bi_encoder.py           # BAAI/bge-large FP16 embedding logic
    ├── cross_encoder.py        # ms-marco deep semantic reranking
    ├── ltr_ranker.py           # XGBoost rank:ndcg trainer
    ├── explainer.py            # Factual reason string generator
    └── utils.py                # Reciprocal Rank Fusion (RRF) math
```
