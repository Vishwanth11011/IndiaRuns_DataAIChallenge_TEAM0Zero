import numpy as np
from datetime import date, datetime

def apply_availability_multiplier(
    final_scores: np.ndarray,
    features_list: list[dict],
    penalty_closed: float = 0.80,
    penalty_inactive: float = 0.75,
) -> np.ndarray:
    """
    Soft down-weight for candidates who are not actively available.
    Does NOT disqualify — just adjusts relative ranking.

    penalty_closed:   multiplier for open_to_work=False (default: 0.80 → 20% down-weight)
    penalty_inactive: multiplier for last_active > 60 days (default: 0.75 → 25% down-weight)
    Combined: a candidate who is both closed AND inactive gets 0.80 * 0.75 = 0.60x
    """
    adjusted = final_scores.copy()

    for i, feat in enumerate(features_list):
        multiplier = 1.0

        if not feat.get("open_to_work", True):
            multiplier *= penalty_closed

        last_active_str = feat.get("last_active_date", "2024-01-01")
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_inactive = (date.today() - last_active).days
        except:
            days_inactive = 999

        if days_inactive > 60:
            multiplier *= penalty_inactive

        adjusted[i] *= multiplier

    return adjusted

def enforce_availability_in_top_n(
    candidate_ids: list[str],
    scores: np.ndarray,
    features_list: list[dict],
    top_n_strict: int = 15,
    max_unavailable_in_top_n: int = 1,
) -> tuple[list[str], np.ndarray, list[dict]]:
    """
    Hard structural rule: among the top `top_n_strict` ranks, allow at most
    `max_unavailable_in_top_n` candidates who are not open to work.
    Excess unavailable candidates are demoted below rank `top_n_strict`,
    making room for the next-best available candidate to move up.

    This does NOT remove unavailable candidates from the top 100 entirely —
    a Meta/Netflix-caliber candidate still deserves a strong rank — it just
    prevents them from dominating the most visible top-15 slots ahead of
    equally-qualified candidates who are actually reachable right now.
    """
    # Sort by score descending first (assumes input is already sorted, but be safe)
    order = np.argsort(scores)[::-1]
    sorted_ids    = [candidate_ids[i] for i in order]
    sorted_scores = scores[order]
    sorted_feats  = [features_list[i] for i in order]

    available_idx   = []
    unavailable_idx = []
    for i, feat in enumerate(sorted_feats):
        if feat.get("open_to_work", False):
            available_idx.append(i)
        else:
            unavailable_idx.append(i)

    # Build final order: fill top_n_strict slots prioritizing available candidates,
    # allow only max_unavailable_in_top_n unavailable ones in that window
    final_order = []
    avail_ptr, unavail_ptr = 0, 0
    unavailable_placed_in_strict_zone = 0

    for rank_pos in range(len(sorted_ids)):
        if rank_pos < top_n_strict:
            # In the strict zone: prefer available, allow limited unavailable
            if (unavail_ptr < len(unavailable_idx) and
                unavailable_placed_in_strict_zone < max_unavailable_in_top_n and
                (avail_ptr >= len(available_idx) or
                 sorted_scores[unavailable_idx[unavail_ptr]] > sorted_scores[available_idx[avail_ptr]] + 0.15)):
                # Only let an unavailable candidate through if they're CLEARLY
                # better (0.15+ score gap) than the next available one
                final_order.append(unavailable_idx[unavail_ptr])
                unavail_ptr += 1
                unavailable_placed_in_strict_zone += 1
            elif avail_ptr < len(available_idx):
                final_order.append(available_idx[avail_ptr])
                avail_ptr += 1
            elif unavail_ptr < len(unavailable_idx):
                final_order.append(unavailable_idx[unavail_ptr])
                unavail_ptr += 1
        else:
            # Outside strict zone: normal merge by remaining score order
            break

    # Append everyone remaining (both pools), re-merged by original score order
    remaining = sorted(
        available_idx[avail_ptr:] + unavailable_idx[unavail_ptr:],
        key=lambda i: -sorted_scores[i]
    )
    final_order.extend(remaining)

    new_ids    = [sorted_ids[i] for i in final_order]
    new_scores = np.array([sorted_scores[i] for i in final_order])
    new_feats  = [sorted_feats[i] for i in final_order]

    return new_ids, new_scores, new_feats
