import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

def setup_trainer(model, tokenizer, train_dataset, eval_dataset=None, output_dir="./llama3_finetuned_colab"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=2,
        lr_scheduler_type='cosine',
        warmup_steps=500,
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    return trainer
