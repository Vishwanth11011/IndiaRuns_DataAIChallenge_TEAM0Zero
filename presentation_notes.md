# Presentation Notes — Multi-Stage Hybrid Candidate Ranker

## IndiaRuns Data & AI Challenge Submission

---

## 1. The Problem: Why Keyword Matching Fails

Traditional ATS systems rank candidates by counting keyword matches between the job description and the candidate profile. This approach is fundamentally broken for three reasons:

**Problem 1 — Keyword Stuffers Win**
A candidate who lists "Machine Learning, NLP, Vector Search, RAG, LangChain, Pinecone, FAISS" in their skills section will outrank a genuine ML engineer who describes their work in natural language ("built retrieval system using dense embeddings for product search").

**Problem 2 — Missing Signals**
Keyword matching ignores:
- How long a candidate has actually used a skill
- Whether they've shipped it to production
- Whether they're actually available to respond to a recruiter

**Problem 3 — The JD Speaks Differently Than the Profile**
The JD says "production experience with embeddings-based retrieval systems." A qualified candidate might write "built recommendation engine using dense vectors at 10M user scale" — the words don't overlap, but the meaning is identical.

---

## 2. Our Solution: The Three-Stage Hybrid Architecture

### Stage 1: Semantic Retrieval (Recall)
- **Model:** `all-MiniLM-L6-v2` (sentence-transformers)
- **What it does:** Converts the JD and all 50K candidate profiles into 384-dimensional dense vectors. Computes cosine similarity to find the top-500 semantically similar candidates.
- **Why it works:** Embeddings capture semantic meaning, not just vocabulary. "Recommendation engine with dense vectors" and "retrieval system using embeddings" map to similar vector representations.
- **Speed:** <5 minutes for 50K candidates on CPU.

### Stage 2: Deep Re-Ranking (Precision)
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **What it does:** For each of the top-500 candidates, the cross-encoder jointly processes the JD + candidate text as a single sequence. This allows attention to flow between both documents, producing much higher quality relevance scores.
- **Why it's better than Bi-Encoder alone:** The Bi-Encoder encodes JD and candidate separately (fast but loses interaction). The Cross-Encoder sees them together (slow but accurate). We use it only on top-500, making it feasible.
- **Speed:** ~60 seconds for 500 pairs on CPU.

### Stage 3: Hybrid Fusion (Signal Integration)
- **What it does:** Combines the Cross-Encoder score with 9 behavioral signals from the Redrob platform.
- **Formula:** `final_score = 0.55 × cross_encoder + 0.08 × github + 0.08 × recruiter_response + ...`
- **The insight:** A semantically perfect candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is **not actually available**. The behavioral signals encode real hiring probability, not just paper fit.

---

## 3. JD-Specific Design Decisions

This JD (Senior ML/AI Engineer for Redrob's ranking team) had several explicit traps in it:

### Trap 1: AI Keywords Don't Equal AI Expertise
The JD says: *"A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit."*

**Our response:** We synthesize candidate text from career history descriptions (rich, narrative text) not just the skills list. This means "built a recommendation engine at 10M scale" gets captured in the embedding.

### Trap 2: Consultancy-Only Backgrounds Are a Red Flag
The JD explicitly says: *"People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture...) in their entire career"* are not a fit.

**Our response:** We implemented a `consultancy_only_penalty` that applies a 25% score reduction to candidates whose entire career is at the 15 listed IT services firms. This is applied *after* semantic scoring to avoid over-indexing on a single signal.

### Trap 3: Inactive Candidates Rank Lower
The JD says: *"A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is, for hiring purposes, not actually available."*

**Our response:** `days_since_active_score` and `recruiter_response_rate` together contribute 13% of the final score. Candidates who've been inactive for 6+ months get near-zero scores on both these signals.

### Trap 4: Short Notice Period Is Explicitly Preferred
The JD says: *"We'd love sub-30-day notice. We can buy out up to 30 days."*

**Our response:** `notice_score` gives 1.0 to candidates with ≤30 day notice and decays linearly to 0 at 180 days.

---

## 4. Why This Beats the Competition

| Approach | Semantic Quality | Behavioral Signals | Trap Awareness |
|---|---|---|---|
| BM25 / Keyword Match | ❌ | ❌ | ❌ |
| TF-IDF + Skills Count | ⚠️ Weak | ❌ | ❌ |
| Bi-Encoder Only | ✅ Good | ❌ | ❌ |
| **Our Hybrid System** | **✅ Excellent** | **✅ 9 signals** | **✅ Explicit penalties** |

---

## 5. Key Takeaways for the Pitch Deck

1. **"We don't rank resumes, we rank hiring probability"** — the combination of semantic fit AND behavioral availability is unique.

2. **The cross-encoder is the secret weapon** — by using it only on top-500 (not all 50K), we get research-grade accuracy at production speed.

3. **Stage 3 weights are the competitive moat** — the 9 behavioral signal weights can be tuned with recruiter feedback data to continuously improve. This is a learning system, not a static ranker.

4. **JD-aware penalty logic** — rather than treating all candidates equally, we embed domain knowledge from the JD (consultancy red flag, notice period preference) directly into the scoring function.

5. **Production ready** — runs in <5 minutes on CPU, no GPU required, fully offline after first model download.
