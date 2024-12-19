from argparse import ArgumentParser


def get_training_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--drop_out_prob", type=float, default=0.75)
    parser.add_argument("--out_dims", type=int, default=1)
    parser.add_argument("--out_dims_start_idx", type=int, default=0)
    parser.add_argument("--insize", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--species", type=str, default="erato")
    parser.add_argument(
        "--genome_folder",
        type=str,
        default="/local/scratch/carlyn.1/dna/vcfs/processed/genome",
    )
    parser.add_argument(
        "--phenotype_folder",
        type=str,
        default="/local/scratch/carlyn.1/dna/colors/processed",
    )
    parser.add_argument(
        "--split_data_folder",
        type=str,
        default="/home/carlyn.1/dna-trait-analysis/data",
    )
    parser.add_argument("--chromosome", type=int, default=1)
    parser.add_argument(
        "--top_k_chromosome_training", action="store_true", default=False
    )
    parser.add_argument("--top_k_chromosome_training_path", type=str, default=False)
    parser.add_argument("--color", type=str, default="total")
    parser.add_argument("--wing", type=str, default="forewings")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument(
        "--output_dir", type=str, default="/local/scratch/carlyn.1/dna/results"
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--scheduler", type=str, default="none")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--save_stat", type=str, default="pearson")

    return parser.parse_args()
