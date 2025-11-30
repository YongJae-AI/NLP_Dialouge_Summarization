from typing import Any, Dict, Optional, Tuple

import numpy as np
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .data import get_data_collator, load_datasets
from .model_kobart import load_kobart_model_and_tokenizer
from .model_t5 import load_t5_model_and_tokenizer
from .utils import ModelConfig, get_checkpoint_dir, set_seed


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
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_rougeL",
        run_name=wandb_run_name,
        report_to=report_to,
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

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
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
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer
