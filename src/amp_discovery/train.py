"""Training entrypoint for AMP fine-tuning with LoRA."""

import yaml, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import DatasetDict
from amp_discovery.data import load_escape, LABELS

def tokenize_fn(batch, tokenizer):
    return tokenizer(batch["sequence"], truncation=True, padding="max_length", max_length=128)

def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ds = load_escape()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=len(LABELS), problem_type="multi_label_classification")
    if cfg.get("use_lora", True):
        lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=cfg.get("lora_r", 16), lora_alpha=32, lora_dropout=0.05, target_modules=["query","value"])
        model = get_peft_model(model, lora)
    args = TrainingArguments(**cfg["training_args"])
    trainer = Trainer(model=model, args=args, train_dataset=ds["fold1"], eval_dataset=ds["fold2"], tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()
    print("Training complete.")

if __name__ == "__main__":
    import sys; train(sys.argv[1])
