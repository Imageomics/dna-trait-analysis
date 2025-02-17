from dataclasses import dataclass

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
    process_all: bool = False
    run_test: bool = False
