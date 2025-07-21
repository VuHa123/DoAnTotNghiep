# train.py
import os
import argparse
import torch
from huggingface_hub import snapshot_download
from dataset_loader import load_dataset
from model_loader import load_model
from trainer_setup import preprocess_dataset, setup_trainer

def find_checkpoint(local_dir: str):
    for root, dirs, _ in os.walk(local_dir):
        for d in dirs:
            if "checkpoint" in d.lower():
                return os.path.join(root, d)
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for DentalGPT")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace access token")
    parser.add_argument("--repo", type=str, default="NV9523/DentalGPT_SFT", help="HuggingFace repo name")
    parser.add_argument("--wandb_key", type=str, default=None, help="Weights & Biases API key (optional)")
    return parser.parse_args()

if __name__ == "__main__":
    print("Bắt đầu huấn luyện!")

    args = parse_args()

    # Tải về repo để tìm checkpoint
    local_repo = snapshot_download(repo_id=args.repo, token=args.hf_token)
    checkpoint = find_checkpoint(local_repo)

    # Load và chia dataset
    dataset = load_dataset()
    split_data = dataset.train_test_split(test_size=0.1)
    train_data_raw = split_data["train"]
    eval_data_raw = split_data["test"]

    # Load model và tokenizer
    model, tokenizer = load_model()

    # Preprocess dữ liệu
    train_data = preprocess_dataset(train_data_raw, tokenizer)
    eval_data = preprocess_dataset(eval_data_raw, tokenizer)

    # Setup trainer với checkpoint push + wandb (nếu có)
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        eval_data=eval_data,
        repo_id=args.repo,
        hf_token=args.hf_token,
        wandb_key=args.wandb_key
    )

    # Resume nếu có checkpoint
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")
        original_load_rng_state = trainer._load_rng_state

        def patched_load_rng_state(checkpoint_dir):
            rng_file = os.path.join(checkpoint_dir, "rng_state.pth")
            if os.path.isfile(rng_file):
                return torch.load(rng_file, weights_only=False)
            return None

        trainer._load_rng_state = patched_load_rng_state
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print("Huấn luyện từ đầu...")
        trainer.train()

    # Push model/tokenizer lên HuggingFace
    model.push_to_hub(args.repo, token=args.hf_token, use_temp_dir=False)
    tokenizer.push_to_hub(args.repo, token=args.hf_token, use_temp_dir=False)

    # Kết thúc W&B
    if args.wandb_key:
        import wandb
        wandb.finish()
