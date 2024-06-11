import numpy as np

def extract_dna(x):
    dna_lines = []
    with open(x) as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if line != "":
                dna_lines.append(line)
                
    return dna_lines

def compose_sequence(x):
    seq = ""
    for line in x:
        seq += line
    return seq

def sequence_to_features(x):
    feat_map = {
        'A' : [1, 0, 0, 0],
        'C' : [0, 1, 0, 0],
        'G' : [0, 0, 1, 0],
        'T' : [0, 0, 0, 1],
    }

    feats = [ feat_map[a] for a in x ]
    return np.array(feats)

if __name__ == "__main__":
    dna_path = "hmelpomene_optix_bac.fasta"
    data = extract_dna(dna_path)
    sequence = compose_sequence(data)
    #print(sequence)
    print(f"Length of seqence: {len(sequence)}")
    features = sequence_to_features(sequence)
    print(features)