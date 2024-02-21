CUDA_VISIBLE_DEVICES=0 nohup python train.py --epochs 300 --species melpomene --all_genes --out_dims 1 --color color_1 --wing hindwings --exp_name one_dim > mhc1.out &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --epochs 300 --species melpomene --all_genes --out_dims 1 --color color_2 --wing hindwings --exp_name one_dim > mhc2.out &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --epochs 300 --species melpomene --all_genes --out_dims 1 --color color_3 --wing hindwings --exp_name one_dim > mhc3.out &
CUDA_VISIBLE_DEVICES=3 nohup python train.py --epochs 300 --species melpomene --all_genes --out_dims 1 --color total --wing hindwings --exp_name one_dim > mht.out &
CUDA_VISIBLE_DEVICES=4 nohup python train.py --epochs 300 --species melpomene --all_genes --out_dims 1 --color color_1 --wing forewings --exp_name one_dim > mfc1.out &
CUDA_VISIBLE_DEVICES=5 nohup python train.py --epochs 300 --species melpomene --all_genes --out_dims 1 --color color_2 --wing forewings --exp_name one_dim > mfc2.out &
CUDA_VISIBLE_DEVICES=6 nohup python train.py --epochs 300 --species melpomene --all_genes --out_dims 1 --color color_3 --wing forewings --exp_name one_dim > mfc3.out &
CUDA_VISIBLE_DEVICES=7 nohup python train.py --epochs 300 --species melpomene --all_genes --out_dims 1 --color total --wing forewings --exp_name one_dim > mft.out &