import numpy as np
from argparse import ArgumentParser

import matplotlib.pyplot as plt

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--logfile", type=str, default="erato_sliding_window.txt")
    args = parser.parse_args()
    return args

def create_curve(filepath, outpath="sliding_window_curve.png", title="Sliding Window Analysis"):
    with open(filepath, "r") as f:
        lines = f.readlines()

    x = []
    y1 = []
    y2 = []
    for line in lines:
        parts = line.split(" ")
        start = int(parts[1][1:-1])
        end = int(parts[2][:-1])
        val = float(parts[10][1:-1])
        test = float(parts[14][1:-2])

        x.append(f"{start}-{end}")
        y1.append(val)
        y2.append(test)

    x = np.arange(len(y1))
    plt.plot(x, y1, color='blue', label="Val loss")
    plt.plot(x, y2, color='green', label="Test loss")
    plt.xlabel('Window Position')
    plt.ylabel('Regression Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(outpath)
    plt.close()


if __name__ == "__main__":
    args = get_args()
    create_curve(args.logfile)

    