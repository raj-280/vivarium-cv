from abc import ABC, abstractmethod

class BaseLevelEstimator(ABC):

    @abstractmethod
    def estimate_pct(self, mask, roi_height: int) -> float:
        pass

    @abstractmethod
    def get_status(self, pct: float) -> str:
        pass