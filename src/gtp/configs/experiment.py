from dataclasses import dataclass


@dataclass
class ExperimentConfigs:
    genotype_scope: str  # Either 'genome', 'chromosomes', 'genes'
    do_subset: bool # Whether or not to create a subset of the original data for testing purposes.

    def __post_init__(self):
        pass
