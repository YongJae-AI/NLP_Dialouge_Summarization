import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from dialogsum.utils import load_yaml_config, resolve_path


def make_prompt(dialogue: str, style_prompt: str) -> str:
    return f"{style_prompt}\n\n대화:\n{dialogue}\n\n답변:"


class DpoDataset(Dataset):
    def __init__(self, items: List[Dict], tokenizer, encoder_max_len: int, decoder_max_len: int, style_prompt: str):
        self.items = items
        self.tokenizer = tokenizer
        self.encoder_max_len = encoder_max_len
        self.decoder_max_len = decoder_max_len
        self.style_prompt = style_prompt

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        prompt = make_prompt(it["dialogue"], self.style_prompt)
        enc = self.tokenizer(
            prompt,
            max_length=self.encoder_max_len,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        with self.tokenizer.as_target_tokenizer():
            chosen = self.tokenizer(
                it["chosen"],
                max_length=self.decoder_max_len,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )["input_ids"]
            rejected = self.tokenizer(
                it["rejected"],
                max_length=self.decoder_max_len,
                padding="max_length",
                truncation=True,
                return_tensors=None,
            )["input_ids"]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "chosen_labels": chosen,
            "rejected_labels": rejected,
        }


def log_prob(labels, logits):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    logprobs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    loss = torch.nn.functional.nll_loss(
        logprobs.view(-1, logprobs.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    # loss is negative log prob; flip sign and mean over tokens that are not ignored
    mask = (shift_labels != -100).float()
    tok_cnt = mask.sum(dim=-1) + 1e-8
    neg_log_prob = loss.view(labels.size(0), -1) * mask
    sent_logprob = -(neg_log_prob.sum(dim=-1) / tok_cnt)
    return sent_logprob


class SimpleDPOTrainer(Trainer):
    def __init__(self, beta: float, *args, **kwargs):
        self.beta = beta
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        chosen_labels = inputs.pop("chosen_labels")
        rejected_labels = inputs.pop("rejected_labels")
        # forward once; reuse encoder via teacher forcing
        outputs_c = model(**inputs, labels=chosen_labels)
        outputs_r = model(**inputs, labels=rejected_labels)
        logp_c = log_prob(chosen_labels, outputs_c.logits)
        logp_r = log_prob(rejected_labels, outputs_r.logits)
        diff = logp_c - logp_r
        loss = -torch.nn.functional.logsigmoid(self.beta * diff).mean()
        if return_outputs:
            return loss, outputs_c
        return loss


def load_dpo_data(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            items.append(rec)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config (model/data/prompt).")
    ap.add_argument("--dpo_data", required=True, help="JSONL with dialogue, chosen, rejected.")
    ap.add_argument("--output_dir", default="checkpoints/dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    model_name = cfg["model"]["name"]
    encoder_max_len = cfg["data"]["encoder_max_len"]
    decoder_max_len = cfg["data"]["decoder_max_len"]
    style_prompt = cfg.get("style_prompt", "")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    data = load_dpo_data(resolve_path(args.dpo_data))
    ds = DpoDataset(data, tokenizer, encoder_max_len, decoder_max_len, style_prompt)
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        import torch

        input_ids = [b["input_ids"] for b in batch]
        attn = [b["attention_mask"] for b in batch]
        chosen = [b["chosen_labels"] for b in batch]
        rejected = [b["rejected_labels"] for b in batch]

        max_in = max(len(x) for x in input_ids)
        max_ch = max(len(x) for x in chosen)
        max_rj = max(len(x) for x in rejected)

        def pad_seq(seq_list, max_len):
            out = []
            for s in seq_list:
                s = list(s)
                s = s + [pad_id] * (max_len - len(s))
                out.append(s)
            return torch.tensor(out, dtype=torch.long)

        in_ids = pad_seq(input_ids, max_in)
        attn_ids = pad_seq(attn, max_in)
        ch_ids = pad_seq(chosen, max_ch)
        rj_ids = pad_seq(rejected, max_rj)

        ch_ids[ch_ids == pad_id] = -100
        rj_ids[rj_ids == pad_id] = -100

        return {
            "input_ids": in_ids,
            "attention_mask": attn_ids,
            "chosen_labels": ch_ids,
            "rejected_labels": rj_ids,
        }

    training_args = TrainingArguments(
        output_dir=resolve_path(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        max_steps=args.max_steps,
        fp16=False,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = SimpleDPOTrainer(
        beta=args.beta,
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(resolve_path(args.output_dir))
    tokenizer.save_pretrained(resolve_path(args.output_dir))


if __name__ == "__main__":
    main()
