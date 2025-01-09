exp_name=base
species_list="erato melpomene"
color_list="color_3 total color_1 color_2"
wing_list="forewings hindwings"
epochs=20
batch_size=32
start_chromosome=1
end_chromosome=21
available_gpus=(4 5 6)

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
