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
