import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
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
        - data_files: local file(s) for a builder such as "text" or "json"
        - revision: immutable Hub commit for reproducible runs (optional)
        - text_field: text column name (default "text")
        - skip / take: document range, applied before tokenization (optional)

    Examples:
        [{"name": "HuggingFaceFW/fineweb-edu", "split": "train"}]
        [{"name": "HuggingFaceFW/fineweb-2", "subset": "fra_Latn", "split": "train"}]
    """
    if not datasets_config:
        raise ValueError("No datasets specified")

    # Fail explicitly: dropping a failed source silently changes an experiment.
    import math
    loaded_datasets = []
    weights = []
    for ds_config in datasets_config:
        ds_name = ds_config["name"]
        weight = ds_config.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{ds_name}: sampling weight must be finite and positive")
        skip = ds_config.get("skip", 0)
        take = ds_config.get("take")
        if type(skip) is not int or skip < 0:
            raise ValueError(f"{ds_name}: skip must be a non-negative integer")
        if take is not None and (type(take) is not int or take <= 0):
            raise ValueError(f"{ds_name}: take must be a positive integer")
        options = {key: ds_config[key] for key in ("data_files", "revision") if ds_config.get(key)}
        ds = load_dataset(ds_name, name=ds_config.get("subset"),
                          split=ds_config.get("split", "train"), streaming=True, **options)
        if skip:
            ds = ds.skip(skip)
        if take is not None:
            ds = ds.take(take)
        # Normalize schemas before interleaving corpora with different metadata.
        field = ds_config.get("text_field", "text")
        ds = ds.select_columns([field])
        if field != "text":
            ds = ds.rename_column(field, "text")
        loaded_datasets.append(ds)
        weights.append(weight)
        print(f"✓ Loaded dataset: {ds_name}")

    # Normalize weights
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    # Interleave if multiple datasets (fixed seed so the stream order — and the
    # train/val partition derived from it — is reproducible across runs)
    if len(loaded_datasets) > 1:
        dataset = interleave_datasets(
            loaded_datasets,
            probabilities=probabilities,
            seed=42
        )
    else:
        dataset = loaded_datasets[0]

    eos_id = tokenizer.eos_token_id

    def token_stream():
        """Generator that yields tokenized documents (EOS-terminated)"""
        for item in dataset:
            text = item.get("text", "")
            if not isinstance(text, str):
                raise ValueError("The selected text column must contain strings")
            if not text.strip():
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if tokens:
                # Terminate each document with EOS so packed sequences have
                # explicit boundaries (also used to build document masks)
                if eos_id is not None:
                    tokens = tokens + [eos_id]
                yield tokens

    return token_stream


def validate_dataset_separation(training, validation):
    """Reject overlapping row ranges of the same explicit source.

    This catches configuration mistakes, not duplicated content across corpora.
    Revisions are deliberately ignored: two versions may contain the same rows.
    """
    import json
    def source(entry):
        return (entry["name"], entry.get("subset") or "default", entry.get("split", "train"),
                json.dumps(entry.get("data_files"), sort_keys=True))
    for train in training:
        for val in validation or []:
            if source(train) != source(val):
                continue
            a, b = train.get("skip", 0), val.get("skip", 0)
            end_a = a + train["take"] if train.get("take") is not None else float("inf")
            end_b = b + val["take"] if val.get("take") is not None else float("inf")
            if max(a, b) < min(end_a, end_b):
                raise ValueError("Training and validation document ranges overlap; use different splits or disjoint skip/take ranges.")


class StreamingTokenDataset(IterableDataset):
    """Streaming dataset that yields fixed-length sequences from a token stream"""

    def __init__(
        self,
        token_stream_fn,
        seq_len: int,
        split: str = "train",
        split_ratio: float = 0.9,
        start_offset: int = 0,
        with_doc_ids: bool = False
    ):
        """
        Args:
            token_stream_fn: function that yields lists of tokens
            seq_len: sequence length
            split: "train", "val", or "all" (an already separated source)
            split_ratio: fraction of data for training (e.g., 0.9 = 90% train, 10% val)
            start_offset: number of tokens already seen (for resuming from checkpoint)
        """
        if split not in {'train', 'val', 'all'}:
            raise ValueError(f"Unknown split: {split}")
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2 for next-token prediction")
        self.token_stream_fn = token_stream_fn
        self.seq_len = seq_len
        self.split = split
        self.split_ratio = split_ratio
        # start_offset = TRAIN tokens already consumed (x.numel() sum). Resume
        # skips whole train sequences while walking the stream from the start,
        # so the deterministic train/val partition stays identical — skipping
        # raw tokens would re-phase seq_idx and leak val data into training.
        self.start_offset = start_offset
        self.with_doc_ids = with_doc_ids

    def __iter__(self):
        buffer = []
        doc_buffer = []
        doc_counter = 0
        skip_train_seqs = self.start_offset // max(1, self.seq_len - 1)
        skipped_train = 0
        seq_idx = 0

        # Deterministic train/val partition: sequence i goes to val when
        # (i % 100) falls in the top (1 - split_ratio) fraction. This keeps the
        # val set identical across runs and evals (unlike a random split, which
        # would compare val losses on different data every time).
        val_threshold = int(self.split_ratio * 100)

        for tokens in self.token_stream_fn():
            buffer.extend(tokens)
            if self.with_doc_ids:
                doc_buffer.extend([doc_counter] * len(tokens))
            doc_counter += 1

            # Yield fixed-length sequences
            while len(buffer) >= self.seq_len:
                seq = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                if self.with_doc_ids:
                    docs = doc_buffer[:self.seq_len]
                    doc_buffer = doc_buffer[self.seq_len:]

                is_val = (seq_idx % 100) >= val_threshold
                seq_idx += 1
                if self.split == "train" and is_val:
                    continue
                if self.split == "val" and not is_val:
                    continue

                # Resume: skip train sequences already consumed
                if self.split in {"train", "all"} and skipped_train < skip_train_seqs:
                    skipped_train += 1
                    continue

                if self.with_doc_ids:
                    yield torch.tensor(seq, dtype=torch.long), torch.tensor(docs, dtype=torch.long)
                else:
                    yield torch.tensor(seq, dtype=torch.long)


def lm_collate_fn_docs(batch):
    """Collate for packed sequences with document ids.

    Returns (input, target, doc_ids) — doc_ids aligned with the input tokens,
    used to build intra-document attention masks.
    """
    seqs = torch.stack([b[0] for b in batch])
    docs = torch.stack([b[1] for b in batch])
    x = seqs[:, :-1].contiguous()
    y = seqs[:, 1:].contiguous()
    return x, y, docs[:, :-1].contiguous()


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
