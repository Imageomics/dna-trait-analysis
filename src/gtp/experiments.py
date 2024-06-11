import os
from dataclasses import dataclass

@dataclass
class Experiment:
    gene: str
    gene_vcf_path: str
    gene_vcf_tsv_path: str
    metadata_path: str
    pca_type: str
    pca_loading_path: str
    wing_side: str
    species: str

    def get_experiment_name(self):
        return f"{self.species}_{self.gene}_{self.wing_side}_{self.pca_type}"

def get_all_genes():
    return {
        "erato" : ["Herato1001_wntA", "Herato1003_elf1a", 
                            "Herato1301_vvl", "Herato1505_cortex", 
                            "Herato1801_optix"],
        "melpomene" : ["Hmel210001o_wntA", "Hmel210001o_elf1a", 
                            "Hmel213001o_vvl", "Hmel215003o_cortex", 
                            "Hmel218003o_optix"]
    }

def get_all_large_genes():
    return {
        "erato" : ["chrom1801", "Herato1001", "Herato1301", "Herato1505"],
        "melpomene" : ["Hmel218003o", "Hmel210001o", "Hmel213001o", "Hmel215003o"]
    }

def get_all_colors():
    return ["color_1", "color_2", "color_3", "total"]

def get_all_wing_sides():
    return ["forewings", "hindwings"]

def get_all_species():
    return ["erato", "melpomene"]

def get_valid_gene(species, gene, is_large=False):
    if is_large:
        genes = get_all_large_genes()[species]
        valid_gene = [vg for vg in genes if gene == vg]
    else:
        genes = get_all_genes()[species]
        valid_gene = [vg for vg in genes if gene == vg.split("_")[1]]
    if len(valid_gene) == 0:
        if gene not in genes:
            raise NotImplementedError(f"{gene} not a valid gene option")
        valid_gene = [gene]
    valid_gene = valid_gene[0]
    return valid_gene
    
def validate_experiment_params(species, wing, color):
    if species not in get_all_species():
        raise NotImplementedError(f"{species} not a valid species")
    if wing not in get_all_wing_sides():
        raise NotImplementedError(f"{wing} not a valid wing side")
    if color not in get_all_colors():
        raise NotImplementedError(f"{color} not a valid color option")

def get_experiment(species, gene, wing, color, is_large=False):
    validate_experiment_params(species, wing, color)
    valid_gene = get_valid_gene(species, gene, is_large)

    gene_vcf_path, metadata_path, pca_loading_path, gene_vcf_tsv_path = get_data_paths(species, valid_gene, wing, color)
    experiment = Experiment(
        gene=gene,
        wing_side=wing,
        species=species,
        pca_type=color,
        gene_vcf_path=gene_vcf_path,
        metadata_path=metadata_path,
        pca_loading_path=pca_loading_path,
        gene_vcf_tsv_path=gene_vcf_tsv_path
    )
    
    return experiment

def get_all_gene_experiments(species, wing, color):
    validate_experiment_params(species, wing, color)
    genes = get_all_genes()[species]
    experiments = []
    for gene in genes:
        experiments.append(get_experiment(species, gene, wing, color))
        
    return experiments

def get_root_paths():
    vcf_root = "/local/scratch/carlyn.1/dna/vcfs"
    pca_root = "/local/scratch/carlyn.1/dna/colors"
    return vcf_root, pca_root

def get_data_paths(species, gene, wing, color):
    vcf_root, pca_root = get_root_paths()
    if len(gene.split("_")) == 2:
        gene_vcf_path=os.path.join(os.path.join(vcf_root, gene + "_vcfs.npz"))
        metadata_path=os.path.join(vcf_root, gene + "_names.json")
    else:
        gene_vcf_path=os.path.join(os.path.join(vcf_root, species, "chrom", gene + "_vcfs.npz"))
        metadata_path=os.path.join(vcf_root, species, "chrom", gene + "_names.json")
        
    pca_loading_path=os.path.join(pca_root, f"{species}_{wing}_PCA", f"PCA_{color}_loadings.csv")
    gene_vcf_tsv_path=os.path.join(vcf_root, species, f"{gene}.tsv") #FIXME This is broken for large / whole chromosome 
    
    return gene_vcf_path, metadata_path, pca_loading_path, gene_vcf_tsv_path
        

def get_all_experiments():
    genes = get_all_genes()
    colors = get_all_colors()
    wing_sides = get_all_wing_sides()
    all_species = get_all_species()

    experiments = []

    for species in all_species:
        for gene in genes[species]:
            for color in colors:
                for wing_side in wing_sides:
                    gene_vcf_path, metadata_path, pca_loading_path, gene_vcf_path = get_data_paths(species, gene, wing_side, color)
                    experiment = Experiment(
                        gene=gene,
                        wing_side=wing_side,
                        species=species,
                        pca_type=color,
                        gene_vcf_path=gene_vcf_path,
                        metadata_path=metadata_path,
                        pca_loading_path=pca_loading_path,
                        gene_vcf_tsv_path=gene_vcf_path
                    )
                    experiments.append(experiment)

    return experiments