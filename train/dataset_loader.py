# dataset_loader.py
from huggingface_hub import hf_hub_download, list_repo_files
import pandas as pd
from datasets import Dataset

def load_dataset():
    REPO_ID = "NV9523/mental_health_dataset"
    REPO_TYPE = "dataset"

    files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    train_files = [f for f in files if f.startswith("train/") and f.endswith(".parquet")]

    dfs = []
    for file in train_files:
        print(f"Tải file: {file}")
        path = hf_hub_download(repo_id=REPO_ID, filename=file, repo_type=REPO_TYPE)
        df = pd.read_parquet(path)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    # df = df.head(10)
    print("Dataset loaded với", len(df), "dòng")
    return Dataset.from_pandas(df)
