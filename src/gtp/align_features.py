import os
from copy import copy
from dataclasses import dataclass

@dataclass
class AlignmentRow:
    """
    See column information at: https://lastz.github.io/lastz/#fmt_text
    """
    row_id: int
    sample_name: str
    score: str
    name1: str
    strand1: str
    size1: str
    zstart1: str
    end1: str
    name2: str
    strand2: str
    size2: str
    zstart2: str
    end2: str
    identity: str
    idPct: str
    coverage: str
    covPct: str

    def get_coverage_percentage(self):
        percent = float(self.covPct[:-1]) # remove %
        return percent / 100
    
    def get_reference_length(self):
        return int(self.end1) - int(self.zstart1)
    
    def get_sample_length(self):
        return int(self.end2) - int(self.zstart2)

@dataclass
class Alignment:
        sample_name: str
        blocks: list[AlignmentRow]

def get_alignment_rows(root_path):
    m_aligns, e_aligns = [], []
    for root, dirs, paths in os.walk(root_path):
        for p in paths:
            name, ext = os.path.splitext(p)
            if ext != '.tsv': continue
            species = root.split(os.path.sep)[-1]
            sample_name = name.split("_")[0]
            
            rows = []
            with open(os.path.join(root, p), 'r') as f:
                lines = f.readlines()
                for i in range(1, len(lines)):
                    row = AlignmentRow(i, sample_name, *lines[i].strip().split("\t"))
                    rows.append(row)
            
            if species == "melpo":
                m_aligns.append(rows)
            elif species == "erato":
                e_aligns.append(rows)
    
    return m_aligns, e_aligns

def get_align_score(blocks, score_method="score"):
    if score_method == "score":
        return sum([int(x.score) for x in blocks])
    if score_method == "coverage":
        return sum([x.get_coverage_percentage() for x in blocks])

    raise NotImplementedError(f"Score method {score_method} not implemented")

def get_alignment(rows, i=0, blocks=[], score_method="score", score_map={}):
    blocks = copy(blocks)

    align_id = "_".join(sorted(list(map(lambda x: str(x.row_id), blocks))))
    if align_id == "": align_id = "root"
    if align_id in score_map: return score_map[align_id]["blocks"]

    # End condition
    if i >= len(rows): return blocks

    row = rows[i]
    conflict = False
    # Checking for conflict
    for block in blocks:
        # Does the current sample conflict with our blocks?
        conflict = conflict or (row.zstart1 >= block.zstart1 and row.zstart1 <= block.end1)
        conflict = conflict or (row.end1 >= block.zstart1 and row.end1 <= block.end1)
        # Does the current reference conflict with our blocks?
        conflict = conflict or (row.zstart2 >= block.zstart2 and row.zstart2 <= block.end2)
        conflict = conflict or (row.end2 >= block.zstart2 and row.end2 <= block.end2)

    if conflict:
        # Don't add the block if there is a conflict
        blocks = get_alignment(rows, i+1, blocks, score_map=score_map)
        score = get_align_score(blocks, score_method=score_method)
        score_map[align_id]["score"] = score
    else:
        blocks_no_add = get_alignment(rows, i+1, blocks, score_map=score_map)
        blocks_add = get_alignment(rows, i+1, blocks + [row], score_map=score_map)

        no_add_score = get_align_score(blocks_no_add, score_method=score_method)
        add_score = get_align_score(blocks_add, score_method=score_method)
        if add_score >= no_add_score:
            blocks = blocks_add
            score = add_score
        if add_score < no_add_score:
            blocks = blocks_no_add
            score = no_add_score

    score_map[align_id] = {
        "score" : score,
        "blocks" : blocks
    }

    return blocks

if __name__ == "__main__":
    dataset_path = "/local/scratch/carlyn.1/dna/seqs/alignments"
    melpo_alignments, erato_alignments = get_alignment_rows(dataset_path)
    for align in melpo_alignments:
        rows = sorted(align, key=lambda x: x.zstart2) # Sort by starting index of sample block
        blocks = get_alignment(align, score_method="score")
        alignment = Alignment(rows[0].sample_name, blocks)
        print(f"Score for {alignment.sample_name} is: {get_align_score(score_method='score')}")
        exit()
