from dataclasses import dataclass
from typing import Any, Dict

from gtp.configs.butterfly_metadata import GlobalButterflyMetadataConfigs
from gtp.configs.dev import DevConfigs
from gtp.configs.experiment import ExperimentConfigs
from gtp.configs.io import IOConfigs


@dataclass
class GenotypeToPhenotypeConfigs:
    src_yaml: Dict[Any, Any]
    io: IOConfigs
    global_butterfly_metadata: GlobalButterflyMetadataConfigs
    experiment: ExperimentConfigs
    dev: DevConfigs

    def __post_init__(self):
        self.io = IOConfigs(**self.io)
        self.global_butterfly_metadata = GlobalButterflyMetadataConfigs(
            **self.global_butterfly_metadata
        )
        self.experiment = ExperimentConfigs(**self.experiment)
        self.dev = DevConfigs(**self.dev)
