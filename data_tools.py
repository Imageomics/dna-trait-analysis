import numpy as np

class VCF:
    def __init__(self, specimen, chromo, pos, ref_alle, alt_alle, lft, rht):
        self.specimen = specimen
        self.chromo = chromo
        self.pos = pos
        self.ref_alle = ref_alle
        self.alt_alle = alt_alle
        self.lft = lft
        self.rht = rht

    def __str__(self):
        o_str =  f"Specimen: {self.specimen}\n"
        o_str += f"Chromosome: {self.chromo}\n"
        o_str += f"Position: {self.pos}\n"
        o_str += f"Reference Allele: {self.ref_alle}\n"
        o_str += f"Alternate Allele: {self.alt_alle}\n"
        o_str += f"Genotype: {self.lft} | {self.rht}\n"
        return o_str

    def __repr__(self):
        return self.__str__()

def get_data_matrix_from_vcfs(vcfs):
    data = []
    for vcf in vcfs:
        if vcf.lft == 0 and vcf.rht == 0:
            data.append([1, 0, 0, 0])
        elif vcf.lft == 0 and vcf.rht == 1:
            data.append([0, 1, 0, 0])
        elif vcf.lft == 1 and vcf.rht == 0:
            data.append([0, 0, 1, 0])
        elif vcf.lft == 1 and vcf.rht == 1:
            data.append([0, 0, 0, 1])
    return np.array(data)

def parse_patternize_csv(path):
    data = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            cols = line.split(",")
            name = cols[0].strip('\"')
            vals = [float(x) for x in cols[1:]]
            data[name] = np.array(vals)

    return data


def parse_vcfs(path):
    vcfs = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            cols = line.split("\t")
            chromo, pos, ref_alle, alt_alle = cols[:4]
            for x in cols[4:]:
                specimen, genotype = x.split("=")
                geno_parts = genotype.split("|")
                if len(geno_parts) != 2:
                    geno_parts = genotype.split("/")
                lft, rht = [int(y) for y in geno_parts]
                vcf = VCF(specimen, chromo, pos, ref_alle, alt_alle, lft, rht)
                vcfs.append(vcf)
    return vcfs