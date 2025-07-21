# trainer_setup.py
import os
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from transformers import Trainer
from huggingface_hub import upload_folder
import wandb


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

def preprocess_dataset(dataset, tokenizer):
    if tokenizer is None:
        return dataset
    def tokenize(sample):
        return tokenizer(sample["text"], truncation=True, padding="max_length", max_length=1024)
    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

def setup_trainer(model, tokenizer, train_data, eval_data, repo_id, hf_token, wandb_key=None):
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

    per_device_train_batch_size = 4
    gradient_accumulation_steps = 8
    batch_size = per_device_train_batch_size * gradient_accumulation_steps
    num_epochs = 2
    total_steps = int(len(train_data) * num_epochs / batch_size)

    args = TrainingArguments(
        output_dir="MentalGPT",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=int(total_steps * 0.03),
        max_steps=total_steps,
        learning_rate=2e-4,
        fp16=False,
        bf16=torch.cuda.is_available(),
        logging_steps=int(800 / batch_size),
        eval_steps=int(1000 / batch_size) if eval_data else None,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=int(1600 / batch_size),
        save_total_limit=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to=report_to,
        save_on_each_node=False,
        logging_dir=None,
        dataloader_num_workers=4
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        max_seq_length=2048,
        packing=False,
        args=args,
        dataset_num_proc=4,
        callbacks=[CheckpointPush(repo_id, hf_token, args.save_steps)]
    )
    return trainer
