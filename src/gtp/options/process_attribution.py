from dataclasses import dataclass

from gtp.options.base import BaseOptions


@dataclass(kw_only=True)
class ProcessAttributionOptions(BaseOptions):
    attr_method: str = "lrp"  # See gtp.evaluation.AttributionMethod for options
    batch_size: int = 64
    num_workers: int = 8
    drop_out_prob: float = 0.75
    out_dims: int = 1
    out_dims_start_idx: int = 0
    insize: int = 3
    hidden_dim: int = 10
    species: str = "erato"
    chromosome: int = 1
    color: str = "total"
    wing: str = "forewings"
    exp_name: str = "debug"
    verbose: bool = False
    process_all: bool = False
