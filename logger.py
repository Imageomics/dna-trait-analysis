import os

class Logger:
    def __init__(self, args, exp_name="debug"):
        self.args = args
        self.outdir = os.path.join(args.output_dir, args.exp_name, exp_name)
        os.makedirs(self.outdir, exist_ok=True)

    def log(self, x):
        with open(os.path.join(self.outdir, "out.log"), 'a') as f:
            f.write(x + '\n')
            if self.args.verbose:
                print(x)
