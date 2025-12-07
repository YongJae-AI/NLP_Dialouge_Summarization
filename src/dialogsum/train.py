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

from .data import DataConfig, get_data_collator, load_datasets
from .model_kobart import load_kobart_model_and_tokenizer
from .model_t5 import load_t5_model_and_tokenizer
from .utils import ModelConfig, get_checkpoint_dir, set_seed


class SamplePrintCallback(TrainerCallback):
    """저장 주기마다 작은 샘플을 CLI에 출력."""

    def __init__(
        self,
        sample_dataset,
        tokenizer,
        encoder_max_len: int,
        decoder_max_len: int,
        model_type: str,
        save_steps: int,
    ):
        self.sample_dataset = sample_dataset
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.model_type = model_type
        self.save_steps = save_steps

    def on_log(self, args, state, control, **kwargs):
        if state.global_step == 0 or self.save_steps <= 0:
            return
        if state.global_step % self.save_steps != 0:
            return
        model = kwargs.get("model")
        if model is None:
            return
        model.eval()
        import torch

        # 최대 5개 샘플만 출력
        for idx in range(min(5, len(self.sample_dataset))):
            item = self.sample_dataset[idx]
            input_ids = torch.tensor([item["input_ids"]]).to(model.device)
            attention_mask = torch.tensor([item["attention_mask"]]).to(model.device)
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.decoder_max_len,
                    num_beams=4,
                )
            pred = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            fname = item.get("fname", f"sample_{idx}")
            dialogue = item.get("dialogue", "")
            print(f"[step {state.global_step}] sample {idx} ({fname})")
            print("dialogue:", str(dialogue)[:200].replace("\n", " "))
            print("pred   :", pred[:200])
        model.train()


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

    def __init__(self, *args, train_weights=None, train_log_path: str | None = None, kd_alpha: float | None = None, **kwargs):
        self.train_weights = train_weights
        self.train_log_path = train_log_path
        self.kd_alpha = kd_alpha
        if self.train_log_path:
            os.makedirs(os.path.dirname(self.train_log_path), exist_ok=True)
            if not os.path.exists(self.train_log_path):
                with open(self.train_log_path, "w", encoding="utf-8-sig") as f:
                    f.write("step,loss,grad_norm\n")
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

    def log(self, logs: Dict[str, float]) -> None:
        # grad norm 계산 및 로그에 추가
        grad_norm = None
        if "loss" in logs:
            grad_norm = compute_grad_norm(self.model)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm

        super().log(logs)

        # 별도 CSV에도 기록
        if self.train_log_path and "loss" in logs and grad_norm is not None:
            step = self.state.global_step
            with open(self.train_log_path, "a", encoding="utf-8-sig") as f:
                f.write(f"{step},{logs.get('loss', '')},{grad_norm}\n")

    def compute_loss(self, model, inputs, return_outputs=False):
        kd_alpha = self.kd_alpha
        teacher_labels = inputs.pop("teacher_labels", None)
        outputs = model(**inputs)
        if kd_alpha is None or teacher_labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        # 기본 CE
        labels = inputs.get("labels")
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_gold = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Teacher CE
        t_labels = teacher_labels.to(logits.device)
        shift_t_labels = t_labels[:, 1:].contiguous()
        loss_teacher = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_t_labels.view(-1))

        loss = (1 - kd_alpha) * loss_gold + kd_alpha * loss_teacher
        return (loss, outputs) if return_outputs else loss


def _prepare_model_and_tokenizer(cfg: ModelConfig):
    if cfg.model_type.lower() == "kobart":
        model, tokenizer = load_kobart_model_and_tokenizer(cfg.model_name)
    else:
        model, tokenizer = load_t5_model_and_tokenizer(cfg.model_name)
    return model, tokenizer


def compute_grad_norm(model, norm_type: float = 2.0) -> float | None:
    import torch

    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return None
    device = parameters[0].grad.device
    total = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total.item()


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
    extra_callbacks: Optional[list] = None,
    kd_alpha: Optional[float] = None,
) -> Tuple[Seq2SeqTrainer, Seq2SeqTrainingArguments]:
    logging_dir = f"{output_dir}/logs"
    report_to = ["wandb"] if use_wandb and wandb_project else []
    train_log_path = f"{logging_dir}/train_metrics.csv"

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_train_epochs if cfg.num_train_epochs is not None else 100,
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
        overwrite_output_dir=True,
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
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

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
        train_log_path=train_log_path,
        kd_alpha=kd_alpha,
    )
    return trainer, args


def run_sft_training(
    model_cfg: ModelConfig,
    encoder_max_len: int,
    decoder_max_len: int,
    style_prompt: Optional[str],
    checkpoint_base_dir: str,
    data_cfg: Optional["DataConfig"] = None,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    len_weight: Optional[float] = None,
    len_top_pct: float = 0.3,
    teacher_preds_path: Optional[str] = None,
    kd_alpha: Optional[float] = None,
) -> Tuple[Any, Any]:
    set_seed(42)

    model, tokenizer = _prepare_model_and_tokenizer(model_cfg)

    if model_cfg.model_type.lower() == "kobart":
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
    else:
        tokenizer.bos_token = None

    teacher_map = None
    if teacher_preds_path:
        t_df = pd.read_csv(resolve_path(teacher_preds_path))
        teacher_map = {str(r["fname"]): str(r["summary"]) for _, r in t_df.iterrows() if pd.notna(r.get("summary"))}

    datasets = load_datasets(
        tokenizer=tokenizer,
        encoder_max_len=encoder_max_len,
        decoder_max_len=decoder_max_len,
        style_prompt=style_prompt,
        encoder_template=model_cfg.encoder_template,
        decoder_prefix=model_cfg.decoder_prefix,
        model_type=model_cfg.model_type,
        data_cfg=data_cfg,
        teacher_map=teacher_map,
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
        extra_callbacks=[
            SamplePrintCallback(
                sample_dataset=datasets["dev"],
                tokenizer=tokenizer,
                encoder_max_len=encoder_max_len,
                decoder_max_len=decoder_max_len,
                model_type=model_cfg.model_type,
                save_steps=model_cfg.save_steps,
            ),
        ],
        kd_alpha=kd_alpha,
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer
