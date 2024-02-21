CUDA_VISIBLE_DEVICES=0 nohup python train.py --epochs 300 --out_dims 1 --species erato --all_genes --color color_1 --wing hindwings --exp_name one_dim > ehc1.out &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --epochs 300 --out_dims 1 --species erato --all_genes --color color_2 --wing hindwings --exp_name one_dim > ehc2.out &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --epochs 300 --out_dims 1 --species erato --all_genes --color color_3 --wing hindwings --exp_name one_dim > ehc3.out &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --epochs 300 --out_dims 1 --species erato --all_genes --color total --wing hindwings --exp_name one_dim > eht.out &
CUDA_VISIBLE_DEVICES=4 nohup python train.py --epochs 300 --out_dims 1 --species erato --all_genes --color color_1 --wing forewings --exp_name one_dim > efc1.out &
CUDA_VISIBLE_DEVICES=5 nohup python train.py --epochs 300 --out_dims 1 --species erato --all_genes --color color_2 --wing forewings --exp_name one_dim > efc2.out &
CUDA_VISIBLE_DEVICES=6 nohup python train.py --epochs 300 --out_dims 1 --species erato --all_genes --color color_3 --wing forewings --exp_name one_dim > efc3.out &
CUDA_VISIBLE_DEVICES=7 nohup python train.py --epochs 300 --out_dims 1 --species erato --all_genes --color total --wing forewings --exp_name one_dim > eft.out &