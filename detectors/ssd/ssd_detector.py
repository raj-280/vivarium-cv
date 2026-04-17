# detectors/ssd/ssd_detector.py
"""
SSD MobileNet detector via ONNX Runtime.

When no weights file exists (MOCK_SSD=true or file missing),
falls back to MockSSDDetector which returns realistic stub data.
This lets the full API run on Swagger without real weights.

To use real weights:
  1. Drop ssd_mobilenet.onnx into models/ssd/
  2. Set SSD_WEIGHTS=models/ssd/ssd_mobilenet.onnx in .env
  3. Set MOCK_SSD=false (or remove it — false is the default)
"""
from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone

import numpy as np

from core.base_detector import BaseDetector
from core.schemas import DetectionResult, LevelReading
from core.config import SSD_CONF_THRESHOLD, SSD_IOU_THRESHOLD, SSD_MAX_DETECTIONS
from core.exceptions import DetectorInitError, InferenceError
from detectors.ssd.postprocessor import parse_ssd_results, OUTPUT_NAMES

logger = logging.getLogger("vivarium.ssd")

# Input tensor name for SSD MobileNet TF-OD-API ONNX export
SSD_INPUT_NAME  = "image_tensor"     # shape: (1, H, W, 3) uint8
SSD_INPUT_SIZE  = (300, 300)         # SSD MobileNet default input


class SSDDetector(BaseDetector):
    """
    Wraps SSD MobileNet ONNX model via ONNXRuntime.

    Inference flow:
        raw BGR frame
            → preprocess (resize to 300×300, uint8, add batch dim)
            → ort_session.run()
            → parse_ssd_results()
            → DetectionResult
    """

    def __init__(self, weights_path: str, device: str = "cpu"):
        # BaseDetector.__init__ calls _load_model automatically
        super().__init__(weights_path=weights_path, device=device)

    # ── BaseDetector implementation ───────────────────────────────────────────

    def _load_model(self) -> None:
        """Load ONNX model into an ORT InferenceSession."""
        try:
            import onnxruntime as ort

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )

            self.model = ort.InferenceSession(
                self.weights_path,
                providers=providers,
            )
            logger.info(
                "SSD model loaded from '%s' on %s",
                self.weights_path,
                self.device,
            )

        except FileNotFoundError as e:
            raise DetectorInitError(
                f"SSD weights not found at '{self.weights_path}'. "
                "Set SSD_WEIGHTS in your .env file."
            ) from e
        except Exception as e:
            raise DetectorInitError(f"Failed to load SSD ONNX model: {e}") from e

    def detect(
        self,
        frame: np.ndarray,
        cage_id: str,
        water_reading: LevelReading | None = None,
        food_reading:  LevelReading | None = None,
    ) -> DetectionResult:
        """
        Run SSD inference on a raw BGR frame (any resolution).

        Args:
            frame:         uint8 BGR ndarray.
            cage_id:       Cage identifier.
            water_reading: Pre-computed water LevelReading (from HSV path).
            food_reading:  Pre-computed food LevelReading (from HSV path).

        Returns:
            DetectionResult
        """
        if not self.is_ready():
            raise InferenceError("SSD model not loaded.")

        water_reading = water_reading or LevelReading(pct=0.0, status="CRITICAL")
        food_reading  = food_reading  or LevelReading(pct=0.0, status="CRITICAL")

        blob    = self._preprocess(frame)
        t_start = time.perf_counter_ns()

        try:
            raw_outputs = self.model.run(
                None,
                {SSD_INPUT_NAME: blob},
            )
        except Exception as e:
            raise InferenceError(
                f"SSD inference failed for cage '{cage_id}': {e}"
            ) from e

        # ORT returns a list — map to named dict using session output names
        output_names = [o.name for o in self.model.get_outputs()]
        outputs      = dict(zip(output_names, raw_outputs))

        return self._postprocess(
            raw_output         = outputs,
            cage_id            = cage_id,
            water_reading      = water_reading,
            food_reading       = food_reading,
            inference_start_ns = t_start,
        )

    def _postprocess(
        self,
        raw_output,
        cage_id: str,
        water_reading: LevelReading | None = None,
        food_reading:  LevelReading | None = None,
        inference_start_ns: int = 0,
    ) -> DetectionResult:
        return parse_ssd_results(
            outputs            = raw_output,
            cage_id            = cage_id,
            water_reading      = water_reading or LevelReading(pct=0.0, status="CRITICAL"),
            food_reading       = food_reading  or LevelReading(pct=0.0, status="CRITICAL"),
            inference_start_ns = inference_start_ns,
        )

    # ── Preprocessing ─────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """
        BGR uint8 → (1, 300, 300, 3) uint8.
        SSD MobileNet expects uint8 RGB input, NOT normalised float.
        """
        import cv2
        # Resize to 300×300
        resized = cv2.resize(frame, SSD_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        # BGR → RGB
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Add batch dimension: (H, W, 3) → (1, H, W, 3)
        return np.expand_dims(rgb, axis=0).astype(np.uint8)

    # ── Warmup ────────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Send a dummy frame through ORT so CPU/CUDA init happens at startup."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, cage_id="__warmup__")


# ══════════════════════════════════════════════════════════════════════════════
# MOCK DETECTOR — used when weights file is absent / MOCK_SSD=true
# ══════════════════════════════════════════════════════════════════════════════

class MockSSDDetector:
    """
    Drop-in replacement for SSDDetector when weights are unavailable.

    Returns deterministic-but-realistic stub data:
      - mouse_count: 2
      - water: 72.5 % OK
      - food:  18.3 % LOW

    Replace with SSDDetector once you have ssd_mobilenet.onnx.
    """

    def __init__(self):
        logger.warning(
            "MockSSDDetector active — real SSD weights not loaded. "
            "Returning stub data. Set SSD_WEIGHTS in .env to use real model."
        )

    def detect(
        self,
        frame: np.ndarray,
        cage_id: str,
        water_reading: LevelReading | None = None,
        food_reading:  LevelReading | None = None,
    ) -> DetectionResult:
        t_start = time.perf_counter_ns()

        # Simulate ~15 ms inference
        time.sleep(0.015)
        inference_ms = int((time.perf_counter_ns() - t_start) / 1_000_000)

        water = water_reading or LevelReading(pct=72.5, status="OK")
        food  = food_reading  or LevelReading(pct=18.3, status="LOW")

        return DetectionResult(
            cage_id     = cage_id,
            timestamp   = datetime.now(tz=timezone.utc),
            mouse_count = 2,
            water       = water,
            food        = food,
            inference_ms= inference_ms,
            image_path  = None,
        )

    def warmup(self) -> None:
        pass   # nothing to warm up

    def is_ready(self) -> bool:
        return True


# ── Factory helper ────────────────────────────────────────────────────────────

def build_ssd_detector(weights_path: str, device: str = "cpu") -> SSDDetector | MockSSDDetector:
    """
    Returns a real SSDDetector if weights exist and MOCK_SSD != 'true',
    otherwise returns MockSSDDetector.
    """
    force_mock = os.getenv("MOCK_SSD", "false").lower() == "true"

    if force_mock or not os.path.isfile(weights_path):
        logger.warning(
            "SSD weights not found at '%s'. Using MockSSDDetector.",
            weights_path,
        )
        return MockSSDDetector()

    return SSDDetector(weights_path=weights_path, device=device)