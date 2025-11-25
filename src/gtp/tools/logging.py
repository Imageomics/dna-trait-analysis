from pathlib import Path


class ExperimentLogger:
    def __init__(self, logging_dir, exp_name="debug", log_fname="out", verbose=False):
        self.outdir = Path(logging_dir, exp_name)
        self.log_fname = log_fname
        self.verbose = verbose
        self.outdir.mkdir(parents=True, exist_ok=True)

    def get_log_location(self, log_name=None):
        log_name = log_name if log_name else self.log_fname
        return self.outdir / (log_name + ".log")

    def log(self, x, log_name=None, skip_print=False):
        with open(self.get_log_location(log_name=log_name), "a") as f:
            f.write(x + "\n")
            if self.verbose and not skip_print:
                print(x)

    def create_file_path(self, file_name):
        return self.outdir / file_name
