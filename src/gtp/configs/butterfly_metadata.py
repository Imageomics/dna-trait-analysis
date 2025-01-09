from dataclasses import dataclass
from typing import List


@dataclass
class GlobalButterflyMetadataConfigs:
    species: List[str]
    wings: List[str]
    phenotypes: List[str]
    number_of_chromosomes: int = 21

    def __post_init__(self):
        pass
