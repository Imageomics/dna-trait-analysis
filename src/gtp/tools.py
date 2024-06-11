import time
import functools

def dna_to_vector(x):
    print(x)
    

def profile_exe_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        st = time.perf_counter()
        rv = func(*args, **kwargs)
        et = time.perf_counter()
        out_str = time.strftime("%H:%M:%S", time.gmtime(et - st))
        print(f"{func.__name__} exe time: {out_str}")
        return rv
    return wrapper
    
