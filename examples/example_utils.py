import argparse


def parse_args(description: str) -> tuple[bool, int | None]:
    parser = argparse.ArgumentParser(
        description=description,
    )
    parser.add_argument(
        "-no-viz",
        action="store_true",
        default=False,
        help="Do not visualize results in the MuJoCo viewer",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=-1,
        help="Seed for random sampling. Must be >= 0. If not set, a random seed will be used",
    )
    args = parser.parse_args()
    viz = not args.no_viz
    seed = args.seed
    if seed < 0:
        seed = None
    return viz, seed
