# training_config.py
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

def preprocess_dataset(dataset, tokenizer):
    def preprocess(batch):
        prompts = [
            f"<|system|>\nTrả lời một câu hỏi tâm lý của người dùng.\n<|user|>\n{q}\n<|assistant|>\n{a}"
            for q, a in zip(batch["question"], batch["answer"])
        ]
        tokenized = tokenizer(prompts, truncation=True, max_length=512, padding="max_length")
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

def setup_trainer(model, tokenizer, train_data, eval_data, output_dir="./llama3_finetuned_colab"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=2,
        lr_scheduler_type='cosine',
        warmup_steps=50,
        optim='adamw_torch',
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        evaluation_strategy="epoch",
        fp16=False,
        bf16=torch.cuda.is_available(),
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )
    return trainer
