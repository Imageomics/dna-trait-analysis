export CUDA_VISIBLE_DEVICES=5

python -m gtp.train_whole_genome --chromosome 13 --species erato --color color_2 --wing forewings --epochs 20 --batch_size 32 --exp_name whole_genome_pval