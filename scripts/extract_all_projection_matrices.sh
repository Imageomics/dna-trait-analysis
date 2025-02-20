root_patternize_recolorize_dir="/local/scratch/carlyn.1/dna/patternize_recolorize/"
root_post_process_dir="/local/scratch/carlyn.1/dna/projection_matrices/"
species_list="erato melpomene"
color_list="1 2 3 all"
wing_list="forewings hindwings"

mkdir $root_post_process_dir

commands_to_execute=()
for wing in $wing_list
do
    for species in $species_list
    do
        local_dir="${species}_pipeline_${wing}/tmp/"
        for color in $color_list
        do
            in_file="${root_patternize_recolorize_dir}${local_dir}h_pca_${color}.rds"
            out_color_name="color_${color}"
            if [ $color == "all" ]
            then
                out_color_name="color_total"
            fi
            out_file="${root_post_process_dir}${species}_${wing}_${out_color_name}.csv"
            cmd="Rscript scripts/r/extract_projection_matrix.R ${in_file} ${out_file}"
            echo "Running command: ${cmd}"
            $cmd
        done
    done
done
