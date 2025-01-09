export CUDA_VISIBLE_DEVICES=0 

python -m gtp.pipelines.train --epochs 20 --species erato --color color_1 --wing hindwings --chromosome 10 --exp_name base --batch_size 32
python -m gtp.pipelines.train --epochs 20 --species erato --color color_1 --wing hindwings --chromosome 11 --exp_name base --batch_size 32
python -m gtp.pipelines.train --epochs 20 --species erato --color color_1 --wing hindwings --chromosome 13 --exp_name base --batch_size 32
python -m gtp.pipelines.train --epochs 20 --species melpomene --color color_1 --wing hindwings --chromosome 1 --exp_name base --batch_size 32
python -m gtp.pipelines.train --epochs 20 --species melpomene --color color_1 --wing hindwings --chromosome 5 --exp_name base --batch_size 32
python -m gtp.pipelines.train --epochs 20 --species melpomene --color color_1 --wing hindwings --chromosome 15 --exp_name base --batch_size 32