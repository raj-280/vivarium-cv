import numpy as np
from core.config import LEVEL_THRESHOLDS

def calc_level(mask, roi_height: int) -> tuple[float, str]:
    filled_pixels = np.sum(mask > 0)
    total_pixels  = roi_height * mask.shape[1]
    pct = round((filled_pixels / total_pixels) * 100, 1) if total_pixels > 0 else 0.0

    if pct < LEVEL_THRESHOLDS["CRITICAL"]:
        status = "CRITICAL"
    elif pct < LEVEL_THRESHOLDS["LOW"]:
        status = "LOW"
    else:
        status = "OK"

    return pct, status