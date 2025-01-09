from dataclasses import dataclass

from gtp.options.base import BaseOptions


@dataclass(kw_only=True)
class TrainingOptions(BaseOptions):
    epochs: int = 100
    batch_size: int = 64
    lr: float = 0.0002
    drop_out_prob: float = 0.75
    out_dims: int = 1
    out_dims_start_idx: int = 0
    insize: int = 3
    hidden_dim: int = 10
    seed: int = 2
    species: str = "erato"
    chromosome: int = 1
    top_k_chromosome_training: bool = False
    top_k_chromosome_training_path: str = False
    color: str = "total"
    wing: str = "forewings"
    exp_name: str = "debug"
    verbose: bool = False
    scheduler: str = "none"
    optimizer: str = "adam"
    save_stat: str = "pearson"
