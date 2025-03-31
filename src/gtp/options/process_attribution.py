from dataclasses import dataclass

from gtp.options.base import BaseOptions


@dataclass(kw_only=True)
class ProcessAttributionOptions(BaseOptions):
    attr_method: str = "lrp"  # See gtp.evaluation.AttributionMethod for options
    window_size: int = 10_000
    batch_size: int = 64
    num_workers: int = 8
    drop_out_prob: float = 0.75
    out_dims: int = 1
    out_dims_attribution: int = 1  # Number of dimensions to run attribution methods on
    out_dims_start_idx: int = 0
    out_dims_start_idx_attribution: int = (
        0  # Start index of the output to run attribution methods on
    )
    insize: int = 3
    hidden_dim: int = 10
    species: str = "erato"
    chromosome: int = 1
    color: str = "total"
    wing: str = "forewings"
    exp_name: str = "debug"
    verbose: bool = False
    force_reprocess: bool = False
    process_all: bool = False
