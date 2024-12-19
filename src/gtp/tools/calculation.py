import numpy as np
from scipy import stats
from tqdm.auto import tqdm


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
