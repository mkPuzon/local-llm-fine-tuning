import argparse
import torch
import yaml

from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    required=True,
    help="Path to YAML configuration file",
)
args = parser.parse_args()

config = load_config(args.config)


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

experiment_cfg = config["experiment"]
model_cfg = config["model"]
lora_cfg = config["lora"]
training_cfg = config["training"]

experiment_name = experiment_cfg["name"]
max_seq_length = model_cfg["max_seq_length"]


# --------------------------------------------------
# Model
# --------------------------------------------------

dtype = getattr(torch, model_cfg["dtype"])

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_cfg["name"],
    max_seq_length=max_seq_length,
    load_in_4bit=model_cfg["load_in_4bit"],
    full_finetuning=model_cfg["full_finetuning"],
    dtype=dtype,
    device_map=model_cfg["device_map"],
)


# --------------------------------------------------
# LoRA
# --------------------------------------------------

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_cfg["r"],
    lora_alpha=lora_cfg["lora_alpha"],
    target_modules=lora_cfg["target_modules"],

    finetune_vision_layers=lora_cfg["finetune_vision_layers"],
    finetune_language_layers=lora_cfg["finetune_language_layers"],
    finetune_attention_modules=lora_cfg["finetune_attention_modules"],
    finetune_mlp_modules=lora_cfg["finetune_mlp_modules"],

    lora_dropout=lora_cfg["lora_dropout"],
    bias=lora_cfg["bias"],
    use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],

    random_state=lora_cfg["random_state"],
    max_seq_length=max_seq_length,
)


# --------------------------------------------------
# Training
# --------------------------------------------------

output_dir = f"../checkpoints/{experiment_name}"

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_train,
    eval_dataset=formatted_val,
    processing_class=tokenizer,

    args=SFTConfig(
        max_seq_length=max_seq_length,
        completion_only_loss=training_cfg["completion_only_loss"],

        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],

        eval_strategy=training_cfg["eval_strategy"],

        warmup_steps=training_cfg["warmup_steps"],
        num_train_epochs=training_cfg["num_train_epochs"],
        optim=training_cfg["optim"],

        seed=experiment_cfg["seed"],
        dataset_num_proc=training_cfg["dataset_num_proc"],

        output_dir=output_dir,
        logging_steps=training_cfg["logging_steps"],
    ),
)

trainer.train()