from torch.optim import lr_scheduler


class NoScheduler:
    def step(self, *args, **kwargs):
        return


class Scheduler:
    def __init__(self, scheduler, optimizer):
        self.optimier = optimizer
        if scheduler == "none":
            self.scheduler = NoScheduler()
        elif scheduler == "step":
            self.scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        elif scheduler == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=20, eta_min=1e-5
            )
        elif scheduler == "cosine_restart":
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=2e-5
            )
        else:
            raise NotImplementedError(
                f"Given scheduler: {scheduler} is not implemented."
            )

    def step(self, *args, **kwargs):
        self.scheduler.step(*args, **kwargs)
