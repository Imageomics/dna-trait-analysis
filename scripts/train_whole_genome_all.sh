exp_name=whole_genome_pval
species_list="erato melpomene"
color_list="color_1 color_2 color_3 total"
wing_list="forewings"
epochs=20
batch_size=32
start_chromosome=1
end_chromosome=21

export CUDA_VISIBLE_DEVICES=6

for color in $color_list
do
    for wing in $wing_list
    do
        for species in $species_list
        do
            for chromosome in $(seq $start_chromosome $end_chromosome)
            do
                cmd="python -m gtp.train_whole_genome"
                cmd="${cmd} --color ${color}"
                cmd="${cmd} --wing ${wing}"
                cmd="${cmd} --species ${species}"
                cmd="${cmd} --chromosome ${chromosome}"
                cmd="${cmd} --epochs ${epochs}"
                cmd="${cmd} --batch_size ${batch_size}"
                cmd="${cmd} --exp_name ${exp_name}"
                echo "Running command: ${cmd}"
                $cmd
            done
        done
    done
done

#python -m gtp.train_whole_genome --chromosome 1 --species erato --color total --wing forewings 
#--epochs 20 --batch_size 32 --exp_name whole_genome_pval