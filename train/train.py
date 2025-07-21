# train.py
from dataset_loader import load_dataset
from model_loader import load_model
from trainer_setup import preprocess_dataset, setup_trainer

if __name__ == "__main__":
    print("🚀 Bắt đầu huấn luyện!")

    # Load dataset
    dataset = load_dataset()
    # Split tập train/test nếu muốn (ở đây train 100%)
    train_data = dataset.train_test_split(test_size=0.1)["train"]
    eval_data = dataset.train_test_split(test_size=0.1)["test"]

    # Load model và tokenizer
    model, tokenizer = load_model()

    # Tiền xử lý
    train_data = preprocess_dataset(train_data, tokenizer)
    eval_data = preprocess_dataset(eval_data, tokenizer)

    # Tạo trainer
    trainer = setup_trainer(model, tokenizer, train_data, eval_data)

    # Huấn luyện
    trainer.train()
