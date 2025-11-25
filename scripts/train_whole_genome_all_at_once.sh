exp_name=deepnet_8_4_all
species_list="erato melpomene"
color_list="color_3 total color_1 color_2"
wing_list="forewings hindwings"
epochs=150
model=deepnet2
batch_size=32
start_chromosome=1
end_chromosome=21
out_dims=1
configs=configs/hindwing_variant.yaml
out_dims_start_idx=0
tmpdir=/local/scratch/carlyn.1/tmp
available_gpus=(0 1 2 3)

commands_to_execute=()
for wing in $wing_list
do
    for color in $color_list
    do
        for species in $species_list
        do
            cmd="uv run python -m gtp.pipelines.train"
            cmd="${cmd} --color ${color}"
            cmd="${cmd} --wing ${wing}"
            cmd="${cmd} --species ${species}"
            cmd="${cmd} --chromosome all"
            cmd="${cmd} --epochs ${epochs}"
            cmd="${cmd} --batch_size ${batch_size}"
            cmd="${cmd} --exp_name ${exp_name}"
            cmd="${cmd} --out_dims ${out_dims}"
            cmd="${cmd} --out_dims_start_idx ${out_dims_start_idx}"
            cmd="${cmd} --model ${model}"
            cmd="${cmd} --configs ${configs}"
            commands_to_execute=("${commands_to_execute[@]}" "${cmd}")
        done
    done
done

echo "Queuing a total of ${#commands_to_execute[@]} commands"

N=${#available_gpus[@]}
for cmd in "${commands_to_execute[@]}";
do 
   ((i=i%N)); ((i++==0)) && wait
   gpu=${available_gpus[$i-1]}
   export TMPDIR=$tmpdir
   export CUDA_VISIBLE_DEVICES=$gpu 
   echo "(GPU ${gpu}) Running command: ${cmd}"
   $cmd &
done
