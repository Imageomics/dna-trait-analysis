from dataclasses import dataclass


@dataclass
class ExperimentConfigs:
    genotype_scope: str  # Either 'genome', 'chromosomes', 'genes'

    def __post_init__(self):
        pass
