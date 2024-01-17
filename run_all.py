import subprocess
import os
from copy import copy

python_script = "train.py"
epochs = 200
out_dims = 50
base_cmd = ["python", python_script, "--epochs", str(epochs), "--out_dims", str(out_dims)]

vcf_root = "/local/scratch/carlyn.1/dna/vcfs"
pca_root = "/local/scratch/carlyn.1/dna/colors"

genes = {
    "erato" : ["Herato1001_wntA", "Herato1003_elf1a", 
                        "Herato1301_vvl", "Herato1505_cortex", 
                        "Herato1801_optix"],
    "melpomene" : ["Hmel210001o_wntA", "Hmel210001o_elf1a", 
                        "Hmel213001o_vvl", "Hmel215003o_cortex", 
                        "Hmel218003o_optix"]
}

colors = ["color_1", "color_2", "color_3"]
wing_sides = ["forewings", "hindwings"]
all_species = ["erato", "melpomene"]

commands = []

for species in all_species:
    for gene in genes[species]:
        for color in colors:
            for wing_side in wing_sides:
                cmd = copy(base_cmd)
                cmd += ["--input_data", os.path.join(vcf_root, gene + "_vcfs.npz")]
                cmd += ["--input_metadata", os.path.join(vcf_root, gene + "_names.json")]
                loading_path = os.path.join(pca_root, f"{species}_{wing_side}_PCA", f"PCA_{color}_loadings.csv")
                cmd += ["--pca_loadings", loading_path]
                cmd += ["--exp_name", f"{gene}_{wing_side}_{color}"]
                commands.append(cmd)

print(f"Total # of commands: {len(commands)}")

avail_gpus = [0,1,2,3,4,5]
processes = {}

for cmd in commands:
    print(f"Running command:")
    print(" ".join(cmd))
    gpu = avail_gpus.pop(0)
    print(f"Using GPU: {gpu}")
    proc_call = "conda run -n dna " + " ".join(cmd)
    new_env = os.environ.copy()
    new_env['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
    process = subprocess.Popen(proc_call, 
                               shell=True, 
                               executable="/bin/bash",
                               env=new_env
                               )
    processes[gpu] = process
    while not avail_gpus:
        os.wait()
        gpu_del_keys = []
        for gpu_key in processes:
            if processes[gpu_key].poll() is None:
                gpu_del_keys.append(gpu_key)
        
        for gpu_key in gpu_del_keys:
            del processes[gpu_key]
            avail_gpus.append(gpu_key)
