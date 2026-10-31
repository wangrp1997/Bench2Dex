#!/usr/bin/env python3
"""Build the trajectory-level train/validation split for task 09."""

from __future__ import annotations

import argparse

from bimanual_flow import build_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train_groups", type=int, default=40)
    parser.add_argument("--val_groups", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    split = build_split(
        args.dataset_dir,
        args.output,
        train_groups=args.train_groups,
        val_groups=args.val_groups,
        seed=args.seed,
    )
    print(
        f"wrote {args.output}: {len(split['train_files'])} train files "
        f"({len(split['train_groups'])} groups), {len(split['val_files'])} val files "
        f"({len(split['val_groups'])} groups)"
    )


if __name__ == "__main__":
    main()

