from dataclasses import dataclass

from gtp.options.base import BaseOptions


@dataclass(kw_only=True)
class CalculateEpistasisOptions(BaseOptions):
    out_dims: int = 1
    out_dims_start_idx: int = 0
    species: str = "erato"
    color: str = "total"
    wing: str = "forewings"
    exp_name: str = "debug"
    verbose: bool = False
    process_all: bool = False
    filter_value: str = "attribution" # [attribution, pcc, pvalue]
    top_k: int = 50
    alpha: float = 0.05
