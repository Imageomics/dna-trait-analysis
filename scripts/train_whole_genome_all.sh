exp_name=pca_10
species_list="erato melpomene"
color_list="color_3 total color_1 color_2"
wing_list="forewings hindwings"
epochs=50
batch_size=32
start_chromosome=1
end_chromosome=21
out_dims=10
out_dims_start_idx=0
available_gpus=(1 2 3 4)

commands_to_execute=()
for wing in $wing_list
do
    for color in $color_list
    do
        for species in $species_list
        do
            for chromosome in $(seq $start_chromosome $end_chromosome)
            do
                cmd="python -m gtp.pipelines.train"
                cmd="${cmd} --color ${color}"
                cmd="${cmd} --wing ${wing}"
                cmd="${cmd} --species ${species}"
                cmd="${cmd} --chromosome ${chromosome}"
                cmd="${cmd} --epochs ${epochs}"
                cmd="${cmd} --batch_size ${batch_size}"
                cmd="${cmd} --exp_name ${exp_name}"
                cmd="${cmd} --out_dims ${out_dims}"
                cmd="${cmd} --out_dims_start_idx ${out_dims_start_idx}"
                commands_to_execute=("${commands_to_execute[@]}" "${cmd}")
            done
        done
    done
done

echo "Queuing a total of ${#commands_to_execute[@]} commands"

N=${#available_gpus[@]}
for cmd in "${commands_to_execute[@]}";
do 
   ((i=i%N)); ((i++==0)) && wait
   gpu=${available_gpus[$i-1]}
   export CUDA_VISIBLE_DEVICES=$gpu 
   echo "(GPU ${gpu}) Running command: ${cmd}"
   $cmd &
done
