"""
Usage: python rank.py --candidates ./Dataset/candidates.jsonl.gz --out ./submission.csv [--use-cache]
"""
import argparse, csv, time, os
import numpy as np
from pipeline.loader           import load_all_candidates
from pipeline.feature_extractor import extract_features
from pipeline.bm25_retriever   import build_bm25_index, bm25_retrieve
from pipeline.bi_encoder       import BiEncoder
from pipeline.cross_encoder    import cross_encode
from pipeline.ltr_ranker       import build_feature_matrix, ltr_rank, FEATURE_NAMES
from pipeline.explainer        import generate_reasoning
from pipeline.utils            import reciprocal_rank_fusion, normalize_array

def validate_submission(path: str):
    import pandas as pd
    df = pd.read_csv(path)

    assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"], "Wrong columns"
    assert len(df) == 100, f"Expected 100 rows, got {len(df)}"
    assert df["rank"].tolist() == list(range(1, 101)), "Ranks must be 1-100 sequential"
    assert df["score"].is_monotonic_decreasing, "Scores must be strictly decreasing"
    assert df["score"].between(0, 1).all(), "Scores must be in [0, 1]"
    assert df["candidate_id"].str.match(r"^CAND_\d{7}$").all(), "Invalid candidate_id format"
    assert df["candidate_id"].nunique() == 100, "Duplicate candidate_ids found"
    assert df["reasoning"].str.len().between(20, 500).all(), "Reasoning too short or too long"

    print("✅ Submission validated. All checks passed.")
    print(df[["candidate_id", "rank", "score", "reasoning"]].head(10).to_string())

def main(candidates_path: str, output_path: str, use_cache: bool, validate: bool = False):
    t0 = time.time()

    # ── Load ─────────────────────────────────────────────────────────────────
    candidates = load_all_candidates(candidates_path)

    # ── Extract features for ALL 100K ────────────────────────────────────────
    print("Extracting features...")
    all_features = [extract_features(c) for c in candidates]

    # ── Stage 1: Hybrid Retrieval (100K → 750) ───────────────────────────────
    print("Stage 1: Hybrid retrieval...")
    encoder = BiEncoder()
    jd_emb  = encoder.embed_jd()
    
    # Check if cache is strictly requested or we should delete it
    if not use_cache and os.path.exists(".cache/candidate_embeddings.pkl"):
        print("Ignoring cache, re-computing embeddings...")
        os.remove(".cache/candidate_embeddings.pkl")
        
    cand_embs = encoder.embed_candidates(all_features)   # uses cache if available

    bm25_index = build_bm25_index(candidates, all_features)
    bm25_ranked    = bm25_retrieve(bm25_index, top_k=1500)
    bienc_ranked   = encoder.retrieve_top_k(jd_emb, cand_embs, top_k=1500)
    fused_750      = reciprocal_rank_fusion([bm25_ranked, bienc_ranked], top_n=750)

    # Pull features and candidates for the 750
    fused_indices  = [idx for idx, _ in fused_750]
    fused_scores   = {idx: score for idx, score in fused_750}
    stage1_cands   = [candidates[i] for i in fused_indices]
    stage1_feats   = [all_features[i] for i in fused_indices]

    # ── Stage 2: Cross-encoder (750 → 300) ───────────────────────────────────
    print("Stage 2: Cross-encoder reranking 750 → 300...")
    ce_ranked = cross_encode(stage1_feats, top_n=300)

    ce_indices  = [stage1_feats[i]["candidate_id"] for i, _ in ce_ranked]
    stage2_cands = [stage1_cands[i] for i, _ in ce_ranked]
    stage2_feats = [stage1_feats[i] for i, _ in ce_ranked]
    ce_scores    = np.array([s for _, s in ce_ranked])

    # ── Stage 3: LTR (300 → 100) ─────────────────────────────────────────────
    print("Stage 3: LTR gradient boosting 300 → 100...")
    # Inject retrieval scores into features
    bm25_norm = normalize_array(np.array([fused_scores.get(fused_indices[i], 0) for i in range(len(stage2_feats))]))
    ce_norm   = normalize_array(ce_scores)

    stage_scores = {
        "bm25_score_norm":         bm25_norm,
        "biencoder_score_norm":    normalize_array(np.array([
            float(cand_embs[fused_indices[i]] @ jd_emb) for i in range(len(stage2_feats))
        ])),
        "cross_encoder_score_norm": ce_norm,
        "core_skill_hit_count_norm": normalize_array(np.array([
            f.get("core_skill_hit_count", 0) for f in stage2_feats
        ])),
    }

    X            = build_feature_matrix(stage2_feats, stage_scores)
    # XGBoost rank:ndcg requires integer relevance labels (e.g. 0-4)
    # Convert ce_norm (0.0 - 1.0 floats) to 0-4 integers
    pseudo_labels = (ce_norm * 4).astype(int)
    top_indices, raw_scores = ltr_rank(X, pseudo_labels, top_n=100)

    # ── Post-processing (new) ─────────────────────────────────────────────────────
    from pipeline.ranker import apply_domain_and_services_penalty, normalize_scores_power
    from pipeline.ensemble import apply_availability_multiplier, enforce_availability_in_top_n

    stage2_top_feats = [stage2_feats[i] for i in top_indices]
    stage2_top_indices = list(top_indices)

    # Step 1: graduated domain/services penalty (replaces old hard caps)
    scores = apply_domain_and_services_penalty(raw_scores, stage2_top_feats)

    # Step 2: existing availability multiplier (keep from earlier patch — still helps)
    scores = apply_availability_multiplier(scores, stage2_top_feats)

    # Step 3: re-sort after steps 1 & 2
    resort_order       = np.argsort(scores)[::-1]
    stage2_top_indices = [stage2_top_indices[i] for i in resort_order]
    scores             = scores[resort_order]
    stage2_top_feats   = [stage2_top_feats[i] for i in resort_order]

    # Step 4: NEW — hard structural cap on unavailable candidates in top 15
    stage2_top_indices, scores, stage2_top_feats = enforce_availability_in_top_n(
        stage2_top_indices, scores, stage2_top_feats,
        top_n_strict=15,
        max_unavailable_in_top_n=1,
    )

    # After structural cap re-sorts, clamp any score at position i
    # to be strictly less than the score at position i-1
    # This handles the case where demoted high-scoring candidates
    # would create an inversion after normalization
    for i in range(1, len(scores)):
        if scores[i] > scores[i-1]:
            scores[i] = scores[i-1] - 1e-6

    # Step 5: power-transform normalize
    final_scores = normalize_scores_power(scores, power=0.4)

    # ── Sanity checks before writing ──────────────────────────────────────────────
    target_ids = {
        "CAND_0054829", "CAND_0046924",  # CV engineers
        "CAND_0075138", "CAND_0022870", "CAND_0024878",  # services firms
    }
    stage2_top_candidate_ids = [stage2_cands[i].get("candidate_id", "") for i in stage2_top_indices]
    still_present_ranks = [
        (cid, rank+1) for rank, cid in enumerate(stage2_top_candidate_ids) if cid in target_ids
    ]
    print(f"Previously-leaked candidates still in top 100: {still_present_ranks}")
    print(f"  (target: empty list, or only at very low ranks 90+)")

    unavailable_in_top15 = sum(
        1 for f in stage2_top_feats[:15] if not f.get("open_to_work", True)
    )
    print(f"Not-open-to-work in top 15: {unavailable_in_top15} (target: ≤1)")

    # ── Write submission.csv ──────────────────────────────────────────────────
    print("Writing submission.csv...")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (i, score) in enumerate(zip(stage2_top_indices, final_scores), start=1):
            cand = stage2_cands[i]
            feat = stage2_feats[i]
            reasoning = generate_reasoning(cand, feat, score)
            writer.writerow([
                cand.get("candidate_id", ""),
                rank,
                f"{score:.6f}",
                reasoning,
            ])

    print(f"\n✅ Done in {time.time()-t0:.1f}s — {output_path}")

    if validate:
        print("\nValidating submission...")
        validate_submission(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="./submission.csv")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    main(args.candidates, args.out, args.use_cache, args.validate)
