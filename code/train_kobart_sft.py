import argparse

from dialogsum.train import run_sft_training
from dialogsum.utils import ModelConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gogamza/kobart-base-v1")
    parser.add_argument("--encoder_max_len", type=int, default=1024)
    parser.add_argument("--decoder_max_len", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--style_prompt", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/kobart-base-sft")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = ModelConfig(
        model_name=args.model_name,
        model_type="kobart",
        encoder_max_len=args.encoder_max_len,
        decoder_max_len=args.decoder_max_len,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        style_prompt=args.style_prompt,
        use_wandb=False,
    )
    run_sft_training(
        model_cfg=cfg,
        encoder_max_len=cfg.encoder_max_len,
        decoder_max_len=cfg.decoder_max_len,
        style_prompt=cfg.style_prompt,
        checkpoint_base_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()

