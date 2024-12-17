from typing import List
from dataclasses import dataclass

@dataclass
class GlobalButterflyMetadataConfigs():
    species: List[str]
    wings: List[str]
    phenotypes: List[str]
    
    def __post_init__(self):
        pass