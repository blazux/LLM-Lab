import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import random
import os
import pickle


def load_tokenizer(tokenizer_name: str, cache_dir: str = "cache"):
    """Load or cache tokenizer"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenizer.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
            if cached.get("name") == tokenizer_name:
                return cached["tokenizer"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(cache_file, "wb") as f:
        pickle.dump({"name": tokenizer_name, "tokenizer": tokenizer}, f)

    return tokenizer


def create_token_stream(datasets_config: list, tokenizer):
    """
    Create a streaming token generator from multiple datasets

    datasets_config: list of dicts with keys:
        - name: dataset name (e.g., "HuggingFaceFW/fineweb-edu")
        - split: dataset split (default "train")
        - subset: dataset subset/config name (e.g., "fra_Latn" for fineweb-2)
        - weight: sampling weight (optional, default 1.0)

    Examples:
        [{"name": "HuggingFaceFW/fineweb-edu", "split": "train"}]
        [{"name": "HuggingFaceFW/fineweb-2", "subset": "fra_Latn", "split": "train"}]
    """
    if not datasets_config:
        raise ValueError("No datasets specified")

    # Load datasets
    loaded_datasets = []
    weights = []

    for ds_config in datasets_config:
        ds_name = ds_config["name"]
        ds_split = ds_config.get("split", "train")
        ds_subset = ds_config.get("subset", None)
        ds_weight = ds_config.get("weight", 1.0)

        try:
            if ds_subset:
                ds = load_dataset(ds_name, name=ds_subset, split=ds_split, streaming=True)
            else:
                ds = load_dataset(ds_name, split=ds_split, streaming=True)
            loaded_datasets.append(ds)
            weights.append(ds_weight)
            print(f"✓ Loaded dataset: {ds_name}" + (f" ({ds_subset})" if ds_subset else ""))
        except Exception as e:
            print(f"✗ Failed to load dataset {ds_name}" + (f" ({ds_subset})" if ds_subset else ""))
            print(f"  Error: {e}")
            # If it's a config missing error, provide helpful hint
            if "Config name is missing" in str(e):
                print(f"  Hint: This dataset requires a subset/config name (e.g., subset='fra_Latn')")
            continue

    if not loaded_datasets:
        raise ValueError("No datasets could be loaded")

    # Normalize weights
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    # Interleave if multiple datasets
    if len(loaded_datasets) > 1:
        dataset = interleave_datasets(
            loaded_datasets,
            probabilities=probabilities,
            seed=random.randint(0, 1_000_000)
        )
    else:
        dataset = loaded_datasets[0]

    def token_stream():
        """Generator that yields tokenized sequences"""
        for item in dataset:
            text = item.get("text", "")
            if not text:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if tokens:
                yield tokens

    return token_stream


class StreamingTokenDataset(IterableDataset):
    """Streaming dataset that yields fixed-length sequences from a token stream"""

    def __init__(
        self,
        token_stream_fn,
        seq_len: int,
        split: str = "train",
        split_ratio: float = 0.9,
        start_offset: int = 0
    ):
        """
        Args:
            token_stream_fn: function that yields lists of tokens
            seq_len: sequence length
            split: "train" or "val"
            split_ratio: fraction of data for training (e.g., 0.9 = 90% train, 10% val)
            start_offset: number of tokens already seen (for resuming from checkpoint)
        """
        self.token_stream_fn = token_stream_fn
        self.seq_len = seq_len
        self.split = split
        self.split_ratio = split_ratio
        self.start_offset = start_offset

    def __iter__(self):
        buffer = []
        skipped = 0

        for tokens in self.token_stream_fn():
            # Skip already-seen tokens when resuming
            if skipped < self.start_offset:
                skipped += len(tokens)
                continue

            buffer.extend(tokens)

            # Yield fixed-length sequences
            while len(buffer) >= self.seq_len:
                seq = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]

                # Split train/val randomly for each sequence
                rand_val = random.random()
                if self.split == "train" and rand_val >= self.split_ratio:
                    continue
                if self.split == "val" and rand_val < self.split_ratio:
                    continue

                yield torch.tensor(seq, dtype=torch.long)


def lm_collate_fn(batch):
    """
    Collate function for language modeling
    Converts batch of sequences [B, L] to (input, target) pairs
    where target is input shifted by 1
    """
    batch = torch.stack(batch)
    x = batch[:, :-1].contiguous()
    y = batch[:, 1:].contiguous()
    return x, y
