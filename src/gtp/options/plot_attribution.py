from dataclasses import dataclass
from typing import List

from gtp.options.base import BaseOptions


@dataclass(kw_only=True)
class PlotAttributionOptions(BaseOptions):
    attr_method: str = "lrp"  # See gtp.evaluation.AttributionMethod for options
    species: str = "erato"
    chromosome: int = 1
    color: str = "total"
    wing: str = "forewings"
    exp_name: str = "debug"
    top_n: int = -1
    verbose: bool = False
    img_width: int = 800
    img_height: int = 450
    plot_one_chromosome: bool = False
    process_all: bool = False
    run_test: bool = False
    out_dims_attribution: int = (
        1  # Number of dimensions to use when plotting methods on
    )
    out_dims_start_idx_attribution: int = (
        0  # Start index of the attribution output methods on
    )
    attribution_aggregation: str = (
        "mean"  # How to aggregate the attributions if multiple dimensions are given
    )
