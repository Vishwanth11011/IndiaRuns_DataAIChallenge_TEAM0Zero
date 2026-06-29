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
