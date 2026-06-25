#!/usr/bin/env python3
"""
rank.py — IndiaRuns Data & AI Challenge
========================================
Main CLI entrypoint for the Multi-Stage Hybrid Candidate Ranking Pipeline.

Usage:
    # Rank all candidates from full dataset (recommended for submission):
    python rank.py --candidates ./Dataset/candidates.jsonl --out ./submission.csv

    # Quick test on sample candidates:
    python rank.py --candidates ./Dataset/sample_candidates.json --out ./sample_output.csv

    # Use cached embeddings (skip re-embedding):
    python rank.py --candidates ./Dataset/candidates.jsonl --out ./submission.csv --use-cache

    # Custom JD path:
    python rank.py --candidates ./Dataset/candidates.jsonl --jd ./Dataset/job_description.docx

    # Sample mode (process only first 5K candidates, fast):
    python rank.py --candidates ./Dataset/candidates.jsonl --sample

Pipeline:
    Stage 1: Bi-Encoder (all-MiniLM-L6-v2) → top-500 by cosine similarity
    Stage 2: Cross-Encoder (ms-marco-MiniLM-L-6-v2) → top-200 re-ranked
    Stage 3: Hybrid Fusion → top-100 final output

Author: IndiaRuns Challenge Submission
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

import config
from pipeline.data_loader import load_candidates, stream_candidates_jsonl
from pipeline.preprocessor import synthesize_candidate_text, load_jd_text, eda_summary
from pipeline.feature_engineer import extract_features, skill_match_score
from pipeline.embedder import BiEncoder
from pipeline.retriever import retrieve_top_k
from pipeline.reranker import CrossEncoder
from pipeline.scorer import score_all_candidates, normalize_scores_to_range
from pipeline.formatter import format_submission

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IndiaRuns Multi-Stage Hybrid Candidate Ranker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=config.CANDIDATES_JSONL,
        help="Path to candidates.jsonl or sample_candidates.json",
    )
    parser.add_argument(
        "--jd",
        type=str,
        default=config.JD_DOCX,
        help="Path to job_description.docx",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=config.OUTPUT_CSV,
        help="Output path for submission.csv",
    )
    parser.add_argument(
        "--top-k-stage1",
        type=int,
        default=config.STAGE1_TOP_K,
        help=f"Top-K candidates to retrieve in Stage 1 (default: {config.STAGE1_TOP_K})",
    )
    parser.add_argument(
        "--top-k-stage2",
        type=int,
        default=config.STAGE2_TOP_K,
        help=f"Top-K candidates to re-rank in Stage 2 (default: {config.STAGE2_TOP_K})",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached embeddings if available (saves time on repeated runs)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample mode: process only first 5K candidates (for fast testing)",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Print EDA summary before running the pipeline",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_submission.py on the output after generating it",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full ranking pipeline end-to-end."""
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("IndiaRuns Multi-Stage Hybrid Ranker")
    logger.info("=" * 60)

    # ── Step 0: Validate inputs ────────────────────────────────────────────────
    if not os.path.exists(args.candidates):
        logger.error(f"Candidates file not found: {args.candidates}")
        sys.exit(1)
    if not os.path.exists(args.jd):
        logger.error(f"JD file not found: {args.jd}")
        sys.exit(1)

    # ── Step 1: Load candidates ────────────────────────────────────────────────
    logger.info(f"\n[Phase 1] Loading candidates from: {args.candidates}")
    candidates = load_candidates(args.candidates, sample_mode=args.sample)
    logger.info(f"Loaded {len(candidates):,} candidates")

    if args.eda:
        summary = eda_summary(candidates)
        logger.info(f"EDA Summary: {summary}")

    # ── Step 2: Load JD ────────────────────────────────────────────────────────
    logger.info(f"Loading JD from: {args.jd}")
    jd_text = load_jd_text(args.jd)
    logger.info(f"JD loaded ({len(jd_text)} chars)")

    # ── Step 3: Synthesize candidate texts ────────────────────────────────────
    logger.info(f"\n[Phase 1] Synthesizing candidate text representations ...")
    t0 = time.time()
    candidate_texts = [synthesize_candidate_text(c) for c in candidates]
    candidate_ids = [c["candidate_id"] for c in candidates]
    logger.info(f"Text synthesis done in {time.time() - t0:.1f}s. "
                f"Avg text length: {sum(len(t) for t in candidate_texts) // len(candidate_texts)} chars")

    # ── Step 4: Extract behavioral features ───────────────────────────────────
    logger.info(f"\n[Phase 2] Extracting behavioral features ...")
    t0 = time.time()
    all_features = {c["candidate_id"]: extract_features(c) for c in candidates}
    logger.info(f"Feature extraction done in {time.time() - t0:.1f}s")

    # ── Step 5: Bi-Encoder embeddings ──────────────────────────────────────────
    logger.info(f"\n[Phase 3 - Stage 1] Bi-Encoder embedding ...")
    cache_dir = config.EMBEDDING_CACHE_DIR if args.use_cache else None
    bi_encoder = BiEncoder(config.BI_ENCODER_MODEL, cache_dir=cache_dir)

    # Embed JD
    jd_embedding = bi_encoder.encode_single(jd_text)

    # Embed all candidates (with optional caching)
    cache_key = f"candidates_{len(candidates)}" if args.use_cache else None
    t0 = time.time()
    candidate_embeddings = bi_encoder.encode(
        candidate_texts,
        batch_size=config.BATCH_SIZE,
        show_progress=True,
        cache_key=cache_key,
    )
    logger.info(f"Embedding done in {time.time() - t0:.1f}s")

    # ── Step 6: Stage 1 — Cosine retrieval ────────────────────────────────────
    # Retrieve a wider pool first (2x), then re-rank using skill+title signals
    wide_k = args.top_k_stage1 * 2  # Retrieve 2x for pre-filtering
    logger.info(f"\n[Phase 3 - Stage 1] Cosine retrieval (top-{wide_k} for pre-filtering) ...")
    t0 = time.time()
    top_k_results = retrieve_top_k(
        jd_embedding,
        candidate_embeddings,
        candidate_ids,
        top_k=wide_k,
    )
    logger.info(f"Stage 1 retrieval done in {time.time() - t0:.1f}s")

    # ── Step 6b: Pre-filter — Boost skill-matched candidates, penalize irrelevant titles ──
    logger.info(f"[Phase 3 - Stage 1b] Pre-filtering with skill-match + title relevance ...")
    candidates_by_id_temp = {c["candidate_id"]: c for c in candidates}
    boosted_results = []
    for cand_id, bi_score, orig_idx in top_k_results:
        candidate = candidates_by_id_temp.get(cand_id, {})
        # Compute skill match boost (0-1)
        sm_score = skill_match_score(candidate)
        # Compute title relevance boost
        title = candidate.get("profile", {}).get("current_title", "").lower()
        title_boost = 0.0
        for kw in config.TITLE_BOOST_KEYWORDS:
            if kw in title:
                title_boost = 0.3
                break
        for kw in config.TITLE_PENALTY_KEYWORDS:
            if kw in title:
                title_boost = -0.4
                break
        # Industry relevance
        industry = candidate.get("profile", {}).get("current_industry", "").lower()
        industry_boost = 0.0
        for ind in config.RELEVANT_INDUSTRIES:
            if ind in industry or industry in ind:
                industry_boost = 0.15
                break
        for ind in config.IRRELEVANT_INDUSTRIES:
            if ind in industry or industry in ind:
                industry_boost = -0.2
                break
        # Hybrid pre-filter score: combine cosine with skill/title/industry
        hybrid_score = bi_score + 0.25 * sm_score + 0.15 * title_boost + 0.10 * industry_boost
        boosted_results.append((cand_id, hybrid_score, bi_score, orig_idx))

    # Re-sort by hybrid score and take top-K
    boosted_results.sort(key=lambda x: x[1], reverse=True)
    top_k_results = [
        (cand_id, bi_score, orig_idx)
        for cand_id, _, bi_score, orig_idx in boosted_results[:args.top_k_stage1]
    ]
    logger.info(f"Stage 1 pre-filter complete: {len(top_k_results)} candidates retained (from {wide_k} initial)")

    # Prepare for Stage 2: add candidate texts
    stage1_with_text = [
        (cand_id, candidate_texts[orig_idx], bi_score, orig_idx)
        for cand_id, bi_score, orig_idx in top_k_results
    ]

    # ── Step 7: Stage 2 — Cross-Encoder re-ranking ───────────────────────────────────
    logger.info(f"\n[Phase 3 - Stage 2] Cross-Encoder re-ranking (top-{args.top_k_stage2}) ...")
    cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
    t0 = time.time()
    # Use the focused JD summary instead of the full 9564-char JD.
    # The cross-encoder has a 512-token combined limit; the full JD gets heavily
    # truncated and loses its discriminative signal. The summary captures must-haves.
    jd_for_cross_encoder = getattr(config, "JD_SUMMARY_FOR_CROSS_ENCODER", jd_text)
    reranked = cross_encoder.rerank(
        jd_for_cross_encoder,
        stage1_with_text,
        top_k=args.top_k_stage2,
    )
    logger.info(f"Stage 2 done in {time.time() - t0:.1f}s")

    # ── Step 8: Stage 3 — Hybrid Fusion ────────────────────────────────────────────
    logger.info(f"\n[Phase 3 - Stage 3] Hybrid Fusion scoring ...")
    candidates_by_id = {c["candidate_id"]: c for c in candidates}
    t0 = time.time()
    final_scored = score_all_candidates(reranked, all_features, candidates_by_id)
    # Normalize scores to [0.20, 0.99] range matching sample submission format
    final_scored = normalize_scores_to_range(final_scored, score_min=0.20, score_max=0.99)
    logger.info(f"Stage 3 done in {time.time() - t0:.1f}s")

    # ── Step 9: Format output ──────────────────────────────────────────────────
    logger.info(f"\n[Phase 4] Formatting submission CSV ...")
    output_path = format_submission(
        final_scored,
        candidates_by_id,
        output_path=args.out,
        top_n=config.FINAL_TOP_N,
    )

    # ── Step 10: Validate (optional) ──────────────────────────────────────────
    if args.validate:
        logger.info(f"\n[Phase 4] Validating submission ...")
        validator_path = os.path.join(os.path.dirname(args.candidates), "validate_submission.py")
        if os.path.exists(validator_path):
            import subprocess
            result = subprocess.run(
                [sys.executable, validator_path, output_path],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info(f"✅ Validation PASSED: {result.stdout.strip()}")
            else:
                logger.error(f"❌ Validation FAILED:\n{result.stdout}\n{result.stderr}")
        else:
            logger.warning(f"validate_submission.py not found at {validator_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    logger.info(f"\n{'=' * 60}")
    logger.info(f"✅ PIPELINE COMPLETE in {total_time:.1f}s")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Total candidates processed: {len(candidates):,}")
    logger.info(f"   Stage 1 retrieved: {args.top_k_stage1}")
    logger.info(f"   Stage 2 re-ranked: {args.top_k_stage2}")
    logger.info(f"   Final submission: top-{config.FINAL_TOP_N}")

    # Print top 10
    logger.info(f"\nTop 10 ranked candidates:")
    logger.info(f"{'Rank':<6} {'Candidate ID':<15} {'Score':<8} Reasoning")
    logger.info(f"{'-' * 80}")
    import csv
    with open(output_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 10:
                break
            logger.info(f"  #{row['rank']:<5} {row['candidate_id']:<15} {row['score']:<8} {row['reasoning']}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
