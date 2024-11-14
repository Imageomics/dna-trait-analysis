export CUDA_VISIBLE_DEVICES=7

python -m gtp.train_whole_genome --top_k_chromosome_training \
    --top_k_chromosome_training_path "/home/carlyn.1/dna-trait-analysis/plot_results/pvalue_erato_forewings_color_3/top_k_snps_erato_forewings_color_3.npy" \
    --species erato \
    --color color_3 \
    --wing forewings \
    --epochs 100 \
    --batch_size 32 \
    --exp_name testing_topk \
    #--genome_folder "/local/scratch/david/geno-pheno-data/dna/processed/genome" \
    #--phenotype_folder "/local/scratch/david/geno-pheno-data/colors/processed" \