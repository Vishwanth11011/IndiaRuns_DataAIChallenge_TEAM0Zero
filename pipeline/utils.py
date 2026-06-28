import numpy as np

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
    top_n: int = 750,
) -> list[tuple[int, float]]:
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (idx, _) in enumerate(ranked_list):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_items[:top_n]

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize a numpy array to [0, 1]."""
    if len(arr) == 0:
        return arr
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)
