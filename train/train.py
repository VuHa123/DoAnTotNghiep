from model_loader import load_model
from dataset_loader import load_dataset, preprocess_dataset
from trainer_setup import setup_trainer

if __name__ == "__main__":
    # Load model + tokenizer
    model, tokenizer = load_model()

    # Load and preprocess dataset
    dataset = load_dataset("mental_health_eng_viet.csv")
    tokenized_dataset = preprocess_dataset(dataset, tokenizer)

    # Optional split
    split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # Setup trainer
    trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset)

    # Train
    trainer.train()
