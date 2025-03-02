import numpy as np

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Sort samples by generative uncertainty score")
    parser.add_argument(
        "--path",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument("--name", type=str, required=False, default=None, help="Name of the experiment")
    parser.add_argument("--N", type=int, required=False, default=10000, help="Number of samples to sort")
    parser.add_argument("--reverse", type=str, required=False, default="false", help="Sort in reverse order")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    scores = np.load(f"{args.path}/{args.name}.npy")
    print(scores.shape)
    if args.reverse == "true":
        idx_sorted = np.argsort(scores)[::-1][: args.N]
    elif args.reverse == "false":
        idx_sorted = np.argsort(scores)[: args.N]
    else:
        raise ValueError(f"Invalid reverse value: {args.reverse}")
    np.save(f"{args.path}/idx_sorted_{args.N}_{args.name}.npy", idx_sorted)
