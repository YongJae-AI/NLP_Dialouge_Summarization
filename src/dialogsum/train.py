import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import get_data_collator, load_datasets
from .model_kobart import load_kobart_model_and_tokenizer
from .model_t5 import load_t5_model_and_tokenizer
from .utils import ModelConfig, get_checkpoint_dir, set_seed


class CSVLoggerCallback(TrainerCallback):
    """간단한 CSV 로그 콜백 (eval 단계에서 ROUGE를 스텝별 기록)."""

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8-sig") as f:
                f.write("step,rouge1,rouge2,rougeL\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        r1 = metrics.get("eval_rouge1")
        r2 = metrics.get("eval_rouge2")
        rL = metrics.get("eval_rougeL")
        if r1 is None or r2 is None or rL is None:
            return
        step = state.global_step
        with open(self.log_path, "a", encoding="utf-8-sig") as f:
            f.write(f"{step},{r1},{r2},{rL}\n")


class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with optional weighted sampler for train set."""

    def __init__(self, *args, train_weights=None, **kwargs):
        self.train_weights = train_weights
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self.train_weights is None:
            return super().get_train_dataloader()

        sampler = WeightedRandomSampler(
            weights=self.train_weights,
            num_samples=len(self.train_weights),
            replacement=True,
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def _prepare_model_and_tokenizer(cfg: ModelConfig):
    if cfg.model_type.lower() == "kobart":
        model, tokenizer = load_kobart_model_and_tokenizer(cfg.model_name)
    else:
        model, tokenizer = load_t5_model_and_tokenizer(cfg.model_name)
    return model, tokenizer


def build_trainer(
    model,
    tokenizer,
    datasets: Dict[str, Any],
    cfg: ModelConfig,
    output_dir: str,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    train_weights: Optional[torch.Tensor] = None,
) -> Tuple[Seq2SeqTrainer, Seq2SeqTrainingArguments]:
    logging_dir = f"{output_dir}/logs"
    report_to = ["wandb"] if use_wandb and wandb_project else []

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=cfg.fp16,
        logging_dir=logging_dir,
        logging_steps=100,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rougeL",
        run_name=wandb_run_name,
        report_to=report_to,
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
    )

    data_collator = get_data_collator(tokenizer)

    def compute_metrics(eval_pred):
        from evaluate import load

        metric = load("rouge")
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        result = {k: round(v * 100, 4) for k, v in result.items()}
        return result

    callbacks = []
    if cfg.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
            ),
        )
    csv_log_path = os.path.join(output_dir, "logs", "metrics.csv")
    callbacks.append(CSVLoggerCallback(csv_log_path))

    trainer = WeightedSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        train_weights=train_weights,
    )
    return trainer, args


def run_sft_training(
    model_cfg: ModelConfig,
    encoder_max_len: int,
    decoder_max_len: int,
    style_prompt: Optional[str],
    checkpoint_base_dir: str,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    len_weight: Optional[float] = None,
    len_top_pct: float = 0.3,
) -> Tuple[Any, Any]:
    set_seed(42)

    model, tokenizer = _prepare_model_and_tokenizer(model_cfg)

    if model_cfg.model_type.lower() == "kobart":
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
    else:
        tokenizer.bos_token = None

    datasets = load_datasets(
        tokenizer=tokenizer,
        encoder_max_len=encoder_max_len,
        decoder_max_len=decoder_max_len,
        style_prompt=style_prompt,
        model_type=model_cfg.model_type,
    )

    train_weights = None
    if len_weight is not None and len_weight > 1.0:
        lengths = datasets["train"].df["dialogue"].astype(str).str.len().to_numpy()
        threshold = np.quantile(lengths, 1 - len_top_pct)
        weights = np.ones_like(lengths, dtype=np.float32)
        weights[lengths >= threshold] = len_weight
        train_weights = torch.tensor(weights, dtype=torch.double)

    output_dir = get_checkpoint_dir(checkpoint_base_dir, wandb_run_name or "run")

    # 선택적으로 wandb 초기화 (API 키는 환경변수나 wandb.login으로 설정)
    if model_cfg.use_wandb and model_cfg.wandb_project:
        try:
            import wandb  # type: ignore

            wandb.init(
                project=model_cfg.wandb_project,
                entity=model_cfg.wandb_entity,
                name=model_cfg.wandb_run_name,
            )
        except Exception:
            pass

    trainer, _ = build_trainer(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        cfg=model_cfg,
        output_dir=output_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        train_weights=train_weights,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer
