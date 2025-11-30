"""
Placeholder script for decode sweep experiments.
This reuses checkpoints produced by training scripts.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--beam_sizes", type=str, default="4,6,8")
    args = parser.parse_args()
    print("Decode sweep placeholder. Implement as needed.")
    print(f"checkpoint={args.checkpoint}, beam_sizes={args.beam_sizes}")


if __name__ == "__main__":
    main()

