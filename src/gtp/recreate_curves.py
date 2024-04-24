import os
from argparse import ArgumentParser
from create_curve_from_sliding_window import create_curve

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    return args

def recreate_curves(results_dir):
    for root, dirs, paths, in os.walk(results_dir):
        for p in paths:
            fname, ext = os.path.splitext(p)
            if ext != ".txt": continue
            fname_parts = fname.split("_")
            new_name = "_".join(fname_parts[:-2]) + "_curve.png"
            out_path = os.path.join(root, new_name)
            if os.path.exists(out_path):
                os.remove(out_path)
            create_curve(os.path.join(root, p), out_path, title=" ".join(fname_parts[:4]))

if __name__ == "__main__":
    args = get_args()
    recreate_curves(args.results_dir)