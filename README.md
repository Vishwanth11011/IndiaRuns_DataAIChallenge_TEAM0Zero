# IndiaRuns Data & AI Challenge — Multi-Stage Hybrid Candidate Ranker

A state-of-the-art candidate ranking pipeline that ranks candidates from the Redrob platform against a given Job Description using a **three-stage hybrid scoring architecture** combining dense semantic retrieval, cross-encoder re-ranking, and behavioral signal fusion.

---

## Architecture Overview

```
candidates.jsonl (~50K profiles)
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │  Stage 1: Bi-Encoder Retrieval              │
 │  Model: all-MiniLM-L6-v2 (384-dim)         │
 │  Cosine similarity over all candidates      │
 │  → Top-500 candidates                       │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │  Stage 2: Cross-Encoder Re-ranking          │
 │  Model: ms-marco-MiniLM-L-6-v2             │
 │  Pairwise JD ↔ Candidate deep scoring      │
 │  → Top-200 candidates                       │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │  Stage 3: Hybrid Fusion                     │
 │  Cross-Encoder (55%) +                      │
 │  Behavioral Signals (45%)                   │
 │  → Top-100 ranked candidates                │
 └─────────────────────────────────────────────┘
        │
        ▼
  submission.csv (100 rows)
```

### Why This Approach?

| Method | Pros | Cons |
|---|---|---|
| Keyword matching (BM25) | Fast, interpretable | Misses semantic similarity; keyword-stuffers win |
| Bi-Encoder only | Fast semantic | Less accurate than cross-encoder |
| Cross-Encoder only | Highest accuracy | Too slow for 50K candidates |
| **Hybrid (our approach)** | Best accuracy + speed | Requires 2 models |

---

## Project Structure

```
IndiaRuns_Datachallenge/
├── rank.py                     # Main CLI entrypoint
├── config.py                   # All hyperparameters and weights
├── requirements.txt
├── README.md
├── presentation_notes.md       # Approach summary for pitch deck
├── pipeline/
│   ├── data_loader.py          # Streaming JSONL parser
│   ├── preprocessor.py         # Text synthesis & JD loading
│   ├── feature_engineer.py     # Behavioral signal extraction
│   ├── embedder.py             # Bi-Encoder (Stage 1)
│   ├── retriever.py            # Cosine similarity retrieval
│   ├── reranker.py             # Cross-Encoder (Stage 2)
│   ├── scorer.py               # Hybrid Fusion (Stage 3)
│   └── formatter.py            # Submission CSV generation
└── Dataset/
    ├── candidates.jsonl        # Full dataset (~50K, ~487 MB)
    ├── sample_candidates.json  # Sample for fast testing
    ├── job_description.docx    # Target JD
    └── validate_submission.py  # Official validator
```

---

## Setup

### Prerequisites
- Python 3.10+
- pip

### Install Dependencies

```bash
cd IndiaRuns_Datachallenge
pip install -r requirements.txt
```

> **Note:** The first run downloads the embedding models (~100 MB total) from HuggingFace. Subsequent runs use the local cache. No network is required during ranking if models are already cached.

---

## Usage

### Quick Test (Sample Candidates)

```bash
# Fast smoke test on ~30 sample candidates
python rank.py \
    --candidates ./Dataset/sample_candidates.json \
    --out ./sample_output.csv \
    --validate
```

### Full Submission Run

```bash
# Rank all ~50K candidates and produce submission.csv
python rank.py \
    --candidates ./Dataset/candidates.jsonl \
    --out ./submission.csv \
    --validate
```

### With Caching (Faster Repeated Runs)

```bash
# First run: downloads models and computes embeddings, saves to .cache/
python rank.py --candidates ./Dataset/candidates.jsonl --use-cache

# Subsequent runs: loads embeddings from .cache/, skips re-embedding
python rank.py --candidates ./Dataset/candidates.jsonl --use-cache
```

### Sample Mode (5K Candidates, Very Fast)

```bash
python rank.py \
    --candidates ./Dataset/candidates.jsonl \
    --sample \
    --out ./sample_output.csv \
    --validate
```

### All Options

```
python rank.py --help

  --candidates PATH    Path to candidates.jsonl or sample_candidates.json
  --jd PATH           Path to job_description.docx (default: Dataset/job_description.docx)
  --out PATH          Output path for submission CSV (default: ./submission.csv)
  --top-k-stage1 N    Top-K to retrieve in Stage 1 (default: 500)
  --top-k-stage2 N    Top-K to re-rank in Stage 2 (default: 200)
  --use-cache         Use/save cached embeddings
  --sample            Process only first 5K candidates (fast testing)
  --eda               Print EDA summary before running
  --validate          Run validate_submission.py after generating output
```

---

## Stage 3 Weight Tuning

The key optimization lever is the `WEIGHTS` dict in [`config.py`](./config.py):

```python
WEIGHTS = {
    "cross_encoder":           0.55,  # Deep semantic score — primary signal
    "github_score":            0.08,  # GitHub activity (production code signal)
    "recruiter_response_rate": 0.08,  # Availability/engagement signal
    "interview_completion":    0.07,  # Reliability signal
    "profile_completeness":    0.05,  # Seriousness of job search
    "offer_acceptance":        0.05,  # Reliability/fit signal
    "days_since_active_score": 0.05,  # Recency = availability
    "skill_assess_avg":        0.04,  # Platform-verified skills
    "open_to_work":            0.02,  # Explicit job-seeking flag
    "notice_score":            0.01,  # Short notice preferred (JD says ≤30 days)
}
```

Adjust and re-run to optimize ranking quality.

---

## Validate Output

```bash
python Dataset/validate_submission.py submission.csv
```

Expected output:
```
Submission is valid.
```

---

## Performance

| Stage | Time (50K candidates, M1 Mac) | Memory |
|---|---|---|
| Text synthesis | ~10s | ~1 GB |
| Bi-Encoder embedding | ~3–5 min | ~2 GB |
| Stage 1 retrieval | <1s | negligible |
| Stage 2 re-ranking (top-500) | ~30–60s | <1 GB |
| Stage 3 + formatting | <1s | negligible |
| **Total (first run)** | **~5–7 min** | **~3 GB peak** |
| **With cache** | **~1–2 min** | **~1 GB** |

---

## Environment

```yaml
python_version: "3.11"
uses_gpu_for_inference: false
has_network_during_ranking: false  # models cached locally
pre_computation_required: false    # optional cache speeds things up
```
