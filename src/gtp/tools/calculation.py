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


def filter_topk_snps(scores, k=200):
    """Returns the indices of the top k scores

    Args:
        scores (list[float]): float scores per snp
        k (int, optional): Number of top snps to return indicies for. Defaults to 200.
    """
    sorted_scores = np.argsort(scores)
    return sorted_scores[-k:]


def gather_model_predictions_and_actuals(model, dataloader):
    model.eval()
    actual = []
    predicted = []
    for batch in dataloader:
        model.zero_grad()
        data, pca = batch
        out = model(data.cuda())
        actual.extend(pca[:, 0].detach().cpu().numpy().tolist())
        predicted.extend(out[:, 0].detach().cpu().numpy().tolist())
    return actual, predicted
