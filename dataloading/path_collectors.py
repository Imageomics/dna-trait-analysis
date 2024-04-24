from abc import ABC, abstractmethod

class PathCollector(ABC):
    @abstractmethod
    def get_path(self, *args, **kwargs):
        pass
