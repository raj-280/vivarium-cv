from abc import ABC, abstractmethod

class BasePreprocessor(ABC):

    @abstractmethod
    def resize(self, frame):
        pass

    @abstractmethod
    def normalize(self, frame):
        pass

    @abstractmethod
    def apply_roi(self, frame, roi):
        pass