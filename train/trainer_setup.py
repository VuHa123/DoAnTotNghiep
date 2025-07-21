import os
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback, Trainer
from huggingface_hub import upload_folder
import wandb

def preprocess_dataset(dataset, tokenizer):
    def preprocess(batch):
        prompts = [
            f"<|system|>\nTrả lời một câu hỏi tâm lý của người dùng.\n<|user|>\n{q}\n<|assistant|>\n{a}"
            for q, a in zip(batch["question"], batch["answer"])
        ]
        tokenized = tokenizer(prompts, truncation=True, max_length=1024, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

    tokenized = dataset.map(
        preprocess,
        batched=True,
        batch_size=32,
        remove_columns=dataset.column_names,
        desc="🔠 Tokenizing"
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


class CheckpointPush(TrainerCallback):
    def __init__(self, repo_id: str, token: str, save_steps: int):
        self.repo_id = repo_id
        self.token = token
        self.save_steps = save_steps

    def on_save(self, args, state, control, **kwargs):
        if state.is_local_process_zero and state.global_step % self.save_steps == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            upload_folder(
                folder_path=ckpt_path,
                repo_id=self.repo_id,
                token=self.token,
                path_in_repo=f"checkpoint-{state.global_step}"
            )
            print(f"📤 Pushed checkpoint-{state.global_step} to Hugging Face Hub.")
        return control


def setup_trainer(model, tokenizer, train_data, eval_data, repo_id, hf_token, wandb_key=None):
    # === Setup WandB ===
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        os.environ["WANDB_MODE"] = "online"
        wandb.init(
            project="MentalGPT",
            name=f"{repo_id.split('/')[-1]}",
            settings=wandb.Settings(save_code=False, _disable_stats=True)
        )
        report_to = "wandb"
    else:
        report_to = "none"

    # === Training config ===
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 1
    batch_size = per_device_train_batch_size * gradient_accumulation_steps
    num_epochs = 2

    total_steps = int(len(train_data) * num_epochs / batch_size)
    warmup_steps = int(total_steps * 0.03)
    logging_steps = int(800 / batch_size)
    eval_steps = int(1000 / batch_size) if eval_data else None
    save_steps = 50

    args = TrainingArguments(
        output_dir="MentalGPT_SFT",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=total_steps,
        warmup_steps=warmup_steps,
        learning_rate=2e-4,
        fp16=False,
        bf16=torch.cuda.is_available(),
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_strategy="steps",
        evaluation_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to=report_to,
        save_on_each_node=False,
        dataloader_num_workers=4
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        callbacks=[CheckpointPush(repo_id, hf_token, save_steps)]
    )
    return trainer
