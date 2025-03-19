import time
from functools import wraps


def profile_exe_time(verbose=False):
    def outer(func):
        @wraps(func)
        def inner(*args, **kwargs):
            st = time.perf_counter()
            rv = func(*args, **kwargs)
            et = time.perf_counter()
            out_str = time.strftime("%H:%M:%S", time.gmtime(et - st))
            if verbose:
                print(f"{func.__name__} exe time: {out_str}")
            return rv

        return inner

    return outer


class ExecutionTimer:
    def __init__(self, name=""):
        self.name = name
        self.end_time = None
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    def restart(self):
        self.end_time = None
        self.start_time = time.perf_counter()

    def get_elapsed_time(self):
        if self.end_time is None:
            print("Timer hasn't been stopped yet. Run the stop() method first.")
        return self.end_time - self.start_time

    def print_elapsed_time(self):
        et = self.get_elapsed_time()
        print(f"{self.name}: {et:.4f}")
