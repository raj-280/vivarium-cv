# pipeline/pipeline_factory.py
import os


def get_pipeline():
    backend = os.getenv("BACKEND", "rtdetr").lower()

    if backend == "rtdetr":
        from pipeline.rtdetr_pipeline import RTDETRPipeline
        return RTDETRPipeline()

    elif backend == "yolo":
        from pipeline.yolo_pipeline import YOLOPipeline
        return YOLOPipeline()

    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            "Set BACKEND=rtdetr or BACKEND=yolo in your .env file."
        )