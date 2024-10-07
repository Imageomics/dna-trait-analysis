export CUDA_VISIBLE_DEVICES=7

python -m gtp.train_whole_genome --chromosome 18 --species erato --color color_3 --wing forewings --epochs 20 --batch_size 32 --exp_name testing