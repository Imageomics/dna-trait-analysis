from collections import defaultdict

import numpy as np


def align_data(dna_data, dna_camids, pheno_data):
    # Remove Extra DNA data
    idx_to_rm = []
    for i, camid in enumerate(dna_camids):
        if camid not in pheno_data.camid.values.tolist():
            idx_to_rm.append(i)
    for rmi in reversed(idx_to_rm):
        dna_data = np.delete(dna_data, rmi, axis=0)
        dna_camids = np.delete(dna_camids, rmi, axis=0)

    # Remove Extra Phenotype data
    idx_to_rm = []
    for i, camid in enumerate(pheno_data.camid):
        if camid not in dna_camids.tolist():
            idx_to_rm.append(i)
    for rmi in reversed(idx_to_rm):
        pheno_data.drop(index=rmi)

    # Sort
    pheno_data.sort_values(
        by="camid",
        key=lambda column: column.map(lambda e: dna_camids.tolist().index(e)),
        inplace=True,
    )

    assert (
        dna_camids.shape[0] == pheno_data.shape[0] == dna_data.shape[0]
    ), "Unequal X and Y in data"
    for x, y in zip(pheno_data.camid.values.tolist(), dna_camids.tolist()):
        assert x == y, f"{x} != {y}. Data not aligned"

    return dna_data, dna_camids, pheno_data


def butterfly_states_to_ml_ready(df):
    # 0:
    # 1:
    state_map = defaultdict(
        lambda: [0, 0, 0],
        {
            "0|0": [1, 0, 0],
            "1|0": [0, 1, 0],
            "0|1": [0, 1, 0],
            "1|1": [0, 0, 1],
        },
    )

    ml_ready = df.map(lambda x: state_map[x])
    ml_ready = np.array(ml_ready.values.tolist())

    return ml_ready
