# 🏆 IndiaRuns Data & AI Challenge — Redrob Definitive Ranking Pipeline

A state-of-the-art candidate ranking pipeline designed to strictly enforce the **Redrob Ranking Rulebook**. This architecture abandons fuzzy, ad-hoc heuristics in favor of rigorous signal extraction, treating candidate selection as a production-grade Search, Recommendation, and Learning-to-Rank (LTR) system.

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
 │  • Deep JD ↔ Candidate alignment            │
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

## ✨ Best Features & Innovations

### 1. 31 Rigorous Signal Extractions
Our `feature_extractor.py` parses complex nested JSON structures to extract highly discriminative signals:
* **Career Trajectory (`ml_trajectory_score`)**: Analyzes temporal job changes to detect if a candidate's career is actively moving *into* ML/AI, or if they are moving *out* into management.
* **Product vs. Services Distinction**: Explicitly flags candidates working at top product companies (Swiggy, Razorpay) and penalizes entire-career service engineers, matching strict industry preferences.
* **Skill Verification**: Relies heavily on `verified_skill_score` derived from Redrob assessments rather than just self-reported keywords.

### 2. Hard Disqualifiers & Honeypot Detection
Hackathons often include impossible "honeypot" profiles (e.g. 100% skill match but current job title is "Accountant" or "HR Manager"). The pipeline implements strict anomaly detection, masking their final LTR scores to `-999.0` to guarantee they are mathematically eliminated from the top 100.

### 3. GPU-Accelerated FP16 Tensor Processing
Running `BAAI/bge-large-en-v1.5` on 100,000 massive text blobs is computationally heavy. Our pipeline automatically detects CUDA environments (like Google Colab's Free T4 GPU) and explicitly binds models to `torch.float16`. This utilizes FP16 Tensor Cores, halving VRAM usage, allowing batch sizes of 256, and slashing a 10-hour CPU workload down to mere minutes.

### 4. XGBoost LTR Engine (`rank:ndcg`)
Instead of guessing linear weights for our 31 features, we use a true Learning-to-Rank approach. We bucket the deep semantic Cross-Encoder scores into 5-tier integer relevance labels and train an XGBoost Ranker (`objective="rank:ndcg"`) on the fly. This allows the model to learn that a high semantic score combined with a 30-day notice period is exponentially better than a high semantic score with a 120-day notice period.

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
