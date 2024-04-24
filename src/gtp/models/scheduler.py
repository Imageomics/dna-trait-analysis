from torch.optim import lr_scheduler

class NoScheduler:
    def step(self, *args, **kwargs):
        return

class Scheduler:
    def __init__(self, args, optimizer):
        self.args = args
        self.optimier = optimizer
        if self.args.scheduler == "none":
            self.scheduler = NoScheduler()
        elif self.args.scheduler == "step":
            self.scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.5)
        elif self.args.scheduler == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
        elif self.args.scheduler == "cosine_restart":
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=2e-5)
        else:
            raise NotImplementedError(f"Given scheduler: {self.args.scheduler} is not implemented.")
    
    def step(self, *args, **kwargs):
        self.scheduler.step(*args, **kwargs)
        