import os
import tempfile
from pathlib import Path
from typing import Union


def set_gpu(device=0) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if isinstance(device, list):
        device = ",".join(device)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"


def set_tempdir(path) -> None:
    tempfile.tempdir = str(path)


def set_hf_home(path) -> None:
    os.environ["HF_HOME"] = str(path)


def setup_notebook(
    device: Union[int, list[int]] = 0,
    tmp_dir: Union[str, Path] = "/home/carlyn.1/tmp",
    hf_cache_dir: Union[str, Path] = "/local/scratch/carlyn.1/hf_cache",
):
    set_gpu(device)
    set_tempdir(tmp_dir)
    set_hf_home(hf_cache_dir)


def get_scratch_dir() -> Path:
    hostname = os.uname()[1]
    match hostname:
        case "cse-cnc196909s.coeit.osu.edu":
            scratch_dir = "/local/scratch"
        case "cse-cnc197066s.coeit.osu.edu":
            scratch_dir = "/local/scratch_1"
        case _:
            scratch_dir = "/local/scratch"

    return Path(scratch_dir)
