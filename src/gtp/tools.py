import functools
import time

import numpy as np
import scipy.stats as stats
from tqdm import tqdm


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


def calc_pvalue_linear(snps, y):
    # TODO: may be worth making a non-linear version of this
    pvals = []
    for snp in tqdm(range(snps.shape[1]), desc="Running Linear regression on SNPS"):
        # 1, 0, 0 = 0 | 0, 1, 0 = 1 | 0, 0, 1 = 2
        snp_categories = (snps[:, snp].astype(np.uint8) * np.array([0, 1, 2])).sum(1)
        if len(np.unique(snp_categories, return_counts=True)[0]) > 1:
            result = stats.linregress(snp_categories, y)
            pval = float(result.pvalue)
        else:
            pval = 1.0
        pvals.append(pval)
    return pvals


def filter_topk_snps(scores, k=200):
    """Returns the indices of the top k scores

    Args:
        scores (list[float]): float scores per snp
        k (int, optional): Number of top snps to return indicies for. Defaults to 200.
    """
    sorted_scores = np.argsort(scores)
    return sorted_scores[-k:]
