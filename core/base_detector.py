from abc import ABC, abstractmethod
from .schemas import DetectionResult

class BaseDetector(ABC):

    @abstractmethod
    def detect(self, frame) -> DetectionResult:
        pass

    @abstractmethod
    def preprocess(self, frame):
        pass

    @abstractmethod
    def draw_results(self, frame, result: DetectionResult):
        pass