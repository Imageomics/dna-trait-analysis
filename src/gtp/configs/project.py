from typing import Dict, Any
from dataclasses import dataclass

from gtp.configs.io import IOConfigs
from gtp.configs.dev import DevConfigs

@dataclass
class GenotypeToPhenotypeConfigs():
    src_yaml: Dict[Any, Any]
    io: IOConfigs
    dev: DevConfigs
    
    def __post_init__(self):
        self.io = IOConfigs(**self.io)
        self.dev = DevConfigs(**self.dev)