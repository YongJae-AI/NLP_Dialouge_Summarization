"""
Placeholder script for style prompt ablation experiments.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_base", type=str, required=True)
    parser.add_argument("--style_prompts", type=str, required=True)
    args = parser.parse_args()
    print("Style prompt ablation placeholder.")
    print(f"config_base={args.config_base}, style_prompts={args.style_prompts}")


if __name__ == "__main__":
    main()

