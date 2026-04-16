import os

def get_pipeline():
    backend = os.getenv("BACKEND", "yolo").lower()
    if backend == "yolo":
        from pipeline.yolo_pipeline import YOLOPipeline
        return YOLOPipeline()
    elif backend == "ssd":
        from pipeline.ssd_pipeline import SSDPipeline
        return SSDPipeline()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use yolo or ssd.")