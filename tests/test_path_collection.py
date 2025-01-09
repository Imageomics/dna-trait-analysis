import os

def test_raw_input_data_path_collection():
    from gtp.configs.loaders import load_configs
    from gtp.dataloading.path_collectors import get_raw_genotype_input_directory, get_raw_phenotype_input_directory

    configs = load_configs("configs/default.yaml")
    
    try:
        input_geno_dir = get_raw_genotype_input_directory(configs.io)
        input_pheno_dir = get_raw_phenotype_input_directory(configs.io)
    except:
        assert False, "Assertion Error in collecting raw input data paths"
    
    assert os.path.exists(input_geno_dir), "Input raw genotype data directory does not existing"
    assert os.path.exists(input_pheno_dir), "Input raw phenotype data directory does not existing"
    