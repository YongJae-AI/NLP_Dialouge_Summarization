"""
Placeholder script for two-stage (turn-level → dialogue-level) experiments.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_checkpoint", type=str, required=True)
    parser.add_argument("--stage2_checkpoint", type=str, required=True)
    args = parser.parse_args()
    print("Two-stage pipeline placeholder.")
    print(
        f"stage1_checkpoint={args.stage1_checkpoint}, "
        f"stage2_checkpoint={args.stage2_checkpoint}",
    )


if __name__ == "__main__":
    main()

