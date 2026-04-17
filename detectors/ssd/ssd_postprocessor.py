# detectors/ssd/postprocessor.py
"""
Parses raw SSD MobileNet ONNX output tensors → DetectionResult.

Standard TF Object Detection API SSD export produces 4 output tensors:
    - 'detection_boxes'       shape (1, N, 4)   — [ymin, xmin, ymax, xmax] normalised 0-1
    - 'detection_scores'      shape (1, N)       — confidence per box
    - 'detection_classes'     shape (1, N)       — class index (1-based, 0 = background)
    - 'num_detections'        shape (1,)         — valid detections count

If your ONNX export uses different output names, update OUTPUT_NAMES below.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from core.config import SSD_CONF_THRESHOLD, SSD_CLASS_MAP
from core.schemas import DetectionResult, LevelReading
from core.exceptions import InferenceError

# ── Output tensor name mapping ────────────────────────────────────────────────
# Keys are what we expect; update if your ONNX export uses different names.
OUTPUT_NAMES = {
    "boxes":          "detection_boxes",
    "scores":         "detection_scores",
    "classes":        "detection_classes",
    "num_detections": "num_detections",
}

MOUSE_CLASS_ID = 1   # class 1 = mouse in TF-OD-API (0 is background)


def parse_ssd_results(
    outputs: dict[str, np.ndarray],
    cage_id: str,
    water_reading: LevelReading,
    food_reading: LevelReading,
    inference_start_ns: int,
) -> DetectionResult:
    """
    Convert raw ONNX Runtime output dict → DetectionResult.

    Args:
        outputs:            {output_name: np.ndarray} from ort_session.run()
        cage_id:            Cage identifier string.
        water_reading:      Pre-computed water LevelReading.
        food_reading:       Pre-computed food LevelReading.
        inference_start_ns: time.perf_counter_ns() captured before ort.run().

    Returns:
        Fully populated DetectionResult.
    """
    inference_ms = int((time.perf_counter_ns() - inference_start_ns) / 1_000_000)

    try:
        scores         = _get_tensor(outputs, "scores").flatten()          # (N,)
        classes        = _get_tensor(outputs, "classes").flatten()         # (N,)
        num_detections = int(_get_tensor(outputs, "num_detections")[0])    # scalar
    except KeyError as e:
        raise InferenceError(
            f"SSD output tensor not found: {e}. "
            f"Expected names: {list(OUTPUT_NAMES.values())}. "
            f"Got: {list(outputs.keys())}"
        ) from e
    except Exception as e:
        raise InferenceError(f"Failed to parse SSD outputs: {e}") from e

    # Only look at valid detections (num_detections tells us how many are real)
    scores  = scores[:num_detections]
    classes = classes[:num_detections].astype(int)

    # Apply confidence threshold and count mice
    confident_mask = scores >= SSD_CONF_THRESHOLD
    mouse_count    = int(np.sum(
        (classes == MOUSE_CLASS_ID) & confident_mask
    ))

    return DetectionResult(
        cage_id     = cage_id,
        timestamp   = datetime.now(tz=timezone.utc),
        mouse_count = mouse_count,
        water       = water_reading,
        food        = food_reading,
        inference_ms= inference_ms,
        image_path  = None,
    )


def extract_boxes(
    outputs: dict[str, np.ndarray],
    conf_threshold: float = SSD_CONF_THRESHOLD,
) -> list[dict]:
    """
    Extract all confident detections as a list of dicts.
    Boxes are in normalised [ymin, xmin, ymax, xmax] format (0-1).

    Returns list of:
        {
            "class_id": int,
            "label":    str,
            "conf":     float,
            "box_norm": (ymin, xmin, ymax, xmax),
        }
    """
    try:
        boxes          = _get_tensor(outputs, "boxes")[0]          # (N, 4)
        scores         = _get_tensor(outputs, "scores").flatten()  # (N,)
        classes        = _get_tensor(outputs, "classes").flatten().astype(int)
        num_detections = int(_get_tensor(outputs, "num_detections")[0])
    except Exception as e:
        raise InferenceError(f"Failed to extract SSD boxes: {e}") from e

    out = []
    for i in range(num_detections):
        if scores[i] < conf_threshold:
            continue
        out.append({
            "class_id": int(classes[i]),
            "label":    SSD_CLASS_MAP.get(int(classes[i]), f"class_{classes[i]}"),
            "conf":     float(round(scores[i], 3)),
            "box_norm": tuple(float(v) for v in boxes[i]),  # (ymin, xmin, ymax, xmax)
        })
    return out


# ── Private ───────────────────────────────────────────────────────────────────

def _get_tensor(outputs: dict[str, np.ndarray], key: str) -> np.ndarray:
    """Look up a tensor by its logical name using OUTPUT_NAMES mapping."""
    tensor_name = OUTPUT_NAMES[key]
    if tensor_name not in outputs:
        raise KeyError(tensor_name)
    return outputs[tensor_name]