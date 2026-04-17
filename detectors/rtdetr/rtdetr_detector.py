# detectors/rtdetr/rtdetr_detector.py
"""
RT-DETR (Real-Time Detection Transformer) detector via Ultralytics.

RT-DETR-nano runs on CPU at ~200ms per frame — noticeably better
accuracy than YOLOv8n on small/partially-occluded objects like mice.

To use:
  1. Fine-tune: yolo train model=rtdetr-n.pt data=your_dataset.yaml epochs=50
  2. Set RTDETR_WEIGHTS=models/rtdetr/best.pt in .env
  3. Set BACKEND=rtdetr in .env
"""
from __future__ import annotations

import time
import logging

import numpy as np

from core.base_detector import BaseDetector
from core.schemas import DetectionResult, LevelReading
from core.config import RTDETR_CONF_THRESHOLD, RTDETR_IOU_THRESHOLD
from core.exceptions import DetectorInitError, InferenceError
from detectors.rtdetr.postprocessor import parse_rtdetr_results

logger = logging.getLogger("vivarium.rtdetr")

RTDETR_INPUT_SIZE = 640  # RT-DETR standard input size


class RTDETRDetector(BaseDetector):
    """
    Wraps RT-DETR-nano via Ultralytics for mouse detection.

    Inference flow:
        raw BGR frame (any resolution)
            → model.predict()           # Ultralytics handles resize internally
            → parse_rtdetr_results()    # our postprocessor
            → DetectionResult

    Level readings (water / food) are injected from the pipeline
    because they come from the separate OpenCV HSV path.
    """

    def __init__(self, weights_path: str, device: str = "cpu"):
        # BaseDetector.__init__ calls _load_model automatically
        super().__init__(weights_path=weights_path, device=device)

    # ── BaseDetector implementation ───────────────────────────────

    def _load_model(self) -> None:
        """Load RT-DETR weights via Ultralytics."""
        try:
            from ultralytics import RTDETR
            self.model = RTDETR(self.weights_path)
            self.model.to(self.device)
            logger.info(
                "RT-DETR model loaded from '%s' on %s",
                self.weights_path,
                self.device,
            )
        except FileNotFoundError as e:
            raise DetectorInitError(
                f"RT-DETR weights not found at '{self.weights_path}'. "
                "Set RTDETR_WEIGHTS in your .env file."
            ) from e
        except Exception as e:
            raise DetectorInitError(f"Failed to load RT-DETR model: {e}") from e

    def detect(
        self,
        frame: np.ndarray,
        cage_id: str,
        water_reading: LevelReading | None = None,
        food_reading:  LevelReading | None = None,
    ) -> DetectionResult:
        """
        Run RT-DETR inference on a raw BGR frame (any resolution).

        Args:
            frame:         uint8 BGR ndarray — Ultralytics resizes internally.
            cage_id:       Cage identifier for the result.
            water_reading: Pre-computed water LevelReading (from HSV path).
            food_reading:  Pre-computed food LevelReading (from HSV path).

        Returns:
            DetectionResult with mouse_count, water, food, timing.
        """
        if not self.is_ready():
            raise InferenceError("RT-DETR model is not loaded.")

        water_reading = water_reading or LevelReading(pct=0.0, status="CRITICAL")
        food_reading  = food_reading  or LevelReading(pct=0.0, status="CRITICAL")

        t_start = time.perf_counter_ns()

        try:
            results = self.model.predict(
                source=frame,
                conf=RTDETR_CONF_THRESHOLD,
                iou=RTDETR_IOU_THRESHOLD,
                imgsz=RTDETR_INPUT_SIZE,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            raise InferenceError(
                f"RT-DETR inference failed for cage '{cage_id}': {e}"
            ) from e

        return self._postprocess(
            raw_output=results,
            cage_id=cage_id,
            water_reading=water_reading,
            food_reading=food_reading,
            inference_start_ns=t_start,
        )

    def _postprocess(
        self,
        raw_output,
        cage_id: str,
        water_reading: LevelReading | None = None,
        food_reading:  LevelReading | None = None,
        inference_start_ns: int = 0,
    ) -> DetectionResult:
        """Delegates to the standalone postprocessor module."""
        return parse_rtdetr_results(
            results=raw_output,
            cage_id=cage_id,
            water_reading=water_reading or LevelReading(pct=0.0, status="CRITICAL"),
            food_reading=food_reading   or LevelReading(pct=0.0, status="CRITICAL"),
            inference_start_ns=inference_start_ns,
        )

    # ── Warmup ────────────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Send a dummy frame through RT-DETR once so the first real
        cage frame doesn't pay the model initialisation cost.
        """
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, cage_id="__warmup__")
        logger.info("RT-DETR warmup complete.")