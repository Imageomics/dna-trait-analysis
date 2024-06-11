from collections import defaultdict

import numpy as np

import swifter

def butterfly_states_to_ml_ready(df):
    state_map = defaultdict(lambda: [0, 0, 0], {
        "0|0" : [1, 0, 0],
        "1|0" : [0, 1, 0],
        "0|1" : [0, 1, 0],
        "1|1" : [0, 0, 1],
    })

    ml_ready = df.map(lambda x: state_map[x])
    ml_ready = np.array(ml_ready.values.tolist())
    
    return ml_ready