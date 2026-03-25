#!/usr/bin/env python3
"""Create a deterministic DeepDive RL validation subset parquet.

Example:
  python examples/carr_deepsearch/scripts/prepare_rl_val_subset.py \
    --input examples/carr_deepsearch/data/rl_val.parquet \
    --subset-size 64 \
    --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic rl_val subset parquet.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("examples/carr_deepsearch/data/rl_val.parquet"),
        help="Input rl_val parquet path.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=64,
        help="Number of rows to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path. Defaults to rl_val_subset_<size>_seed<seed>.parquet next to input.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    if args.output is None:
        output_path = input_path.with_name(f"rl_val_subset_{args.subset_size}_seed{args.seed}.parquet")
    else:
        output_path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")
    if output_path.exists() and not args.force:
        print(f"Subset already exists: {output_path}")
        return

    df = pd.read_parquet(input_path)
    if args.subset_size > len(df):
        raise ValueError(f"subset_size={args.subset_size} exceeds dataset size {len(df)}")

    subset_df = df.sample(n=args.subset_size, random_state=args.seed).sort_index().reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_parquet(output_path)

    print(f"Wrote {len(subset_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
