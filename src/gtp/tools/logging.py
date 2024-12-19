import os


class ExperimentLogger:
    def __init__(self, logging_dir, exp_name="debug", log_fname="out", verbose=False):
        self.outdir = os.path.join(logging_dir, exp_name)
        self.log_fname = log_fname
        self.verbose = verbose
        os.makedirs(self.outdir, exist_ok=True)

    def log(self, x, log_name=None):
        log_name = log_name if log_name else self.log_fname
        with open(os.path.join(self.outdir, f"{log_name}.log"), "a") as f:
            f.write(x + "\n")
            if self.verbose:
                print(x)
