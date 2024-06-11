from abc import ABC, abstractmethod
from typing import List

class Genotype:
    """This class is the abstraction of the input DNA data
    that goes into training and evaluation
    """
    def __init__(self, data, **kwargs):
        """
            data: a numpy array where each row is a genotype / SNP
            **kwargs: all information will be put into the metadata property
        """
        self.data = data
        self.metadata = kwargs

class GenotypeDataLoader(ABC):
    @abstractmethod
    def load_data(self) -> List[Genotype]:
        """
        Should load a list of Genotype object
        """
        pass        

class ButterflyJigginDataLoader(GenotypeDataLoader):
    def __init__(self, root, species, gene) -> None:
        super().__init__()
        

class DataHandler:
    @classmethod
    def t(cls, x):
        pass
    
if __name__ == "__main__":
    import numpy as np
    data = np.random.rand(5, 3)
    print(data.shape)
    x = Genotype(data, color="red", species="erato")
    print(x.metadata)