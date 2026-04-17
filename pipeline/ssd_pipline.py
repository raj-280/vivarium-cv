# pipeline/ssd_pipeline.py
"""
SSD pipeline — mirrors YOLOPipeline interface exactly.

Dev B owns this file. Swap MockSSDDetector → SSDDetector
by dropping weights into models/ssd/ and setting SSD_WEIGHTS in .env.
"""
from __future__ import annotations

import os
import logging

import cv2
import numpy as np

from core.schemas import DetectionResult
from core.exceptions import VivariumCVError
from preprocessing.frame_preprocessor import FramePreprocessor
from preprocessing.background_subtractor import BackgroundSubtractor
from level_estimation.water_level import WaterLevelEstimator
from level_estimation.food_level import FoodLevelEstimator
from detectors.ssd.ssd_detector import build_ssd_detector

logger = logging.getLogger("vivarium.ssd_pipeline")


class SSDPipeline:
    """
    Full inference pipeline for one frame using SSD MobileNet.

    Steps:
      1. Resize raw frame to 640×640 (letterbox)
      2. Estimate water level (HSV segmentation on jug ROI)
      3. Estimate food level  (HSV segmentation on hopper ROI)
      4. Run SSD mouse detection
      5. Optionally save flagged frames to disk

    Usage:
        pipeline = SSDPipeline()
        result   = pipeline.run(frame, cage_id="cage_01")
    """

    def __init__(self, cage_type: str = "default"):
        weights = os.getenv("SSD_WEIGHTS", "models/ssd/ssd_mobilenet.onnx")
        device  = os.getenv("SSD_DEVICE",  "cpu")

        self.preprocessor  = FramePreprocessor(cage_type=cage_type)
        self.bg_subtractor = BackgroundSubtractor()
        self.water_est     = WaterLevelEstimator()
        self.food_est      = FoodLevelEstimator()

        # build_ssd_detector returns MockSSDDetector if weights are missing
        self.detector = build_ssd_detector(weights_path=weights, device=device)
        self.detector.warmup()

        logger.info(
            "SSDPipeline ready — detector: %s",
            type(self.detector).__name__,
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        frame: np.ndarray,
        cage_id: str,
        save_flagged: bool = False,
        output_dir:   str  = "flagged_frames",
    ) -> DetectionResult:
        """
        Full pipeline for one frame.

        Args:
            frame:        Raw BGR frame from camera or upload (any resolution).
            cage_id:      Cage identifier written into the DetectionResult.
            save_flagged: Save frame to disk when water or food is CRITICAL.
            output_dir:   Directory for flagged frames.

        Returns:
            DetectionResult
        """
        if frame is None or frame.size == 0:
            raise VivariumCVError(f"Empty frame passed for cage '{cage_id}'.")

        # ── Step 1: resize to 640×640 (letterbox) ────────────────────────────
        frame_640 = self.preprocessor.resize(frame)

        # ── Step 2: level estimation ──────────────────────────────────────────
        water_roi     = self.preprocessor.apply_roi(frame_640, zone="jug")
        food_roi      = self.preprocessor.apply_roi(frame_640, zone="hopper")
        water_reading = self.water_est.read(water_roi)
        food_reading  = self.food_est.read(food_roi)

        # ── Step 3: SSD mouse detection ───────────────────────────────────────
        # SSD preprocesses internally (300×300 resize) — pass the 640 frame
        result = self.detector.detect(
            frame         = frame_640,
            cage_id       = cage_id,
            water_reading = water_reading,
            food_reading  = food_reading,
        )

        # ── Step 4: optional frame saving ────────────────────────────────────
        if save_flagged and self._should_flag(result):
            image_path = self._save_frame(frame_640, cage_id, output_dir)
            result     = result.model_copy(update={"image_path": image_path})

        return result

    def set_reference_frame(self, frame: np.ndarray) -> None:
        """Store a clean background frame for motion detection."""
        frame_640 = self.preprocessor.resize(frame)
        self.bg_subtractor.set_reference(frame_640)

    def has_motion(self, frame: np.ndarray) -> bool:
        """Quick motion gate — skip inference if cage is static."""
        if not self.bg_subtractor.has_reference():
            return True
        frame_640 = self.preprocessor.resize(frame)
        return self.bg_subtractor.has_motion(frame_640)

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns 640×640 frame annotated with ROI zones +
        water/food mask overlays. Useful during camera calibration.
        """
        frame_640 = self.preprocessor.resize(frame)
        viz       = self.preprocessor.draw_debug(frame)

        water_roi = self.preprocessor.apply_roi(frame_640, zone="jug")
        food_roi  = self.preprocessor.apply_roi(frame_640, zone="hopper")

        from core.config import ROI_ZONES
        zones = ROI_ZONES["default"]
        jx, jy, jw, jh = zones["jug"]
        hx, hy, hw, hh = zones["hopper"]

        viz[jy:jy+jh, jx:jx+jw] = self.water_est.debug_frame(water_roi)
        viz[hy:hy+hh, hx:hx+hw] = self.food_est.debug_frame(food_roi)

        return viz

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _should_flag(result: DetectionResult) -> bool:
        return (
            result.water.status == "CRITICAL"
            or result.food.status == "CRITICAL"
        )

    @staticmethod
    def _save_frame(frame: np.ndarray, cage_id: str, output_dir: str) -> str:
        """Save frame as JPEG and return the file path."""
        from datetime import datetime, timezone
        import os

        os.makedirs(output_dir, exist_ok=True)
        ts   = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(output_dir, f"{cage_id}_{ts}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path