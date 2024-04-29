import os

from abc import abstractmethod, ABC


class DataPreprocessor(ABC):
    def __init__(self, output_dir) -> None:
        super().__init__()
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.process_ran = False
        
    def process(self, *args, **kwargs):
        self.process_ran = True
        self._process(*args, **kwargs)
        
    def save_result(self, output_suffix):
        assert self.process_ran, "DataPreprocess did not run process, so save_result() cannot be run."
        self._save_result(os.path.join(self.output_dir, output_suffix))
        
    @abstractmethod
    def _save_result(self, path):
        pass
    
    @abstractmethod
    def _process(self):
        pass
    
class ButterflyPatternizePreprocessor(DataPreprocessor):
    def _process(self, pca_csv_path):
        pass
    
    def _save_result(self, path) -> None:
        pass
    