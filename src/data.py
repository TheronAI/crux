import os
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, DistributedSampler


def get_tokenizer() -> AutoTokenizer:
    # This works, it's established, you can change it here anytime if needed
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def download_dataset(
    down_dir: Path, target_folder: str, cache_folder: str, repo_id: str
) -> Path:
    target_dir = down_dir / target_folder
    cache_dir = down_dir / cache_folder

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Checking for dataset {repo_id}...")

    # Ensure files are on disk at a distinct location
    snapshot_download(
        repo_id,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
    return target_dir


def create_datasets(
    tokenizer: AutoTokenizer,
    data_path: str,
    seq_len: int = 512,
    hf_dataset_name: str = "Marcus2112/minipile_density-proportioned_pico",
):
    down_dir = Path(data_path).parent
    target_folder = Path(data_path).name
    cache_folder = f"{target_folder}_Cache"

    dataset_dir = download_dataset(
        down_dir=down_dir,
        target_folder=target_folder,
        cache_folder=cache_folder,
        repo_id=hf_dataset_name,
    )

    # load datasets from parquet files
    dataset_format = "parquet"
    train_dataset = load_dataset(
        "parquet",
        data_files={
            "train": str(dataset_dir / "data" / f"train-*.{dataset_format}"),
        },
        cache_dir=str(dataset_dir.parent / cache_folder),
        split="train",
    )

    val_dataset = load_dataset(
        "parquet",
        data_files={
            "validation": str(dataset_dir / "data" / f"validation-*.{dataset_format}"),
        },
        cache_dir=str(dataset_dir.parent / cache_folder),
        split="validation",
    )

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=seq_len,
            return_special_tokens_mask=True,
        )

    train_dataset = train_dataset.map(
        tokenize, batched=True, remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        tokenize, batched=True, remove_columns=val_dataset.column_names
    )

    return train_dataset, val_dataset


def create_dataloader(
    dataset,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    num_replicas: int = 1,
    rank: int = 0,
) -> DataLoader:
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        drop_last=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
    )
