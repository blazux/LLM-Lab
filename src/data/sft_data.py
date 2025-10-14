import torch
from torch.utils.data import IterableDataset, Dataset
from datasets import load_dataset, interleave_datasets, get_dataset_config_info
import random


def list_dataset_splits(dataset_name: str, subset: str = None):
    """
    List available splits for a dataset

    Args:
        dataset_name: HuggingFace dataset name
        subset: Optional subset/config name

    Returns:
        List of available split names
    """
    try:
        if subset:
            info = get_dataset_config_info(dataset_name, config_name=subset)
        else:
            info = get_dataset_config_info(dataset_name)
        return list(info.splits.keys())
    except Exception as e:
        print(f"⚠️  Could not retrieve splits info: {e}")
        return []


def create_sft_dataset(datasets_config: list, tokenizer, max_seq_len: int, split: str = "train"):
    """
    Create a dataset for supervised fine-tuning from conversational/instruction data

    datasets_config: list of dicts with keys:
        - name: dataset name (e.g., "HuggingFaceTB/smoltalk2")
        - split: dataset split (default "train")
        - subset: dataset subset/config name (e.g., "SFT")
        - weight: sampling weight (optional, default 1.0)

    Returns:
        SFTDataset instance
    """
    if not datasets_config:
        raise ValueError("No datasets specified")

    # Load datasets
    loaded_datasets = []
    weights = []

    for ds_config in datasets_config:
        ds_name = ds_config["name"]
        ds_split = ds_config.get("split", split)
        ds_subset = ds_config.get("subset", None)
        ds_weight = ds_config.get("weight", 1.0)

        try:
            if ds_subset:
                ds = load_dataset(ds_name, name=ds_subset, split=ds_split, streaming=True)
            else:
                ds = load_dataset(ds_name, split=ds_split, streaming=True)
            loaded_datasets.append(ds)
            weights.append(ds_weight)
            weight_str = f" (weight: {ds_weight})" if ds_weight != 1.0 else ""
            print(f"✓ Loaded SFT dataset: {ds_name}" + (f" ({ds_subset})" if ds_subset else "") + f" [split: {ds_split}]" + weight_str)
        except Exception as e:
            print(f"✗ Failed to load dataset {ds_name}" + (f" ({ds_subset})" if ds_subset else ""))
            print(f"  Error: {e}")

            # Provide helpful hints based on error type
            if "config name is missing" in str(e).lower():
                print(f"  Hint: This dataset requires a config/subset name.")
                print(f"  Check the error message above for available configs.")
            elif "split" in str(e).lower() or "bad split" in str(e).lower():
                # Try to list available splits to help the user
                available_splits = list_dataset_splits(ds_name, ds_subset)
                if available_splits:
                    print(f"  Available splits: {', '.join(available_splits[:10])}")
                    if len(available_splits) > 10:
                        print(f"  ... and {len(available_splits) - 10} more")
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

    return SFTDataset(dataset, tokenizer, max_seq_len)


class SFTDataset(IterableDataset):
    """Dataset for supervised fine-tuning on conversational/instruction data"""

    def __init__(self, hf_dataset, tokenizer, max_seq_len: int):
        """
        Args:
            hf_dataset: HuggingFace dataset (streaming or not)
            tokenizer: tokenizer instance
            max_seq_len: maximum sequence length
        """
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self):
        for item in self.dataset:
            try:
                # Try to extract conversational data
                # SmolTalk2 and similar datasets typically have a 'messages' field
                if 'messages' in item:
                    messages = item['messages']
                elif 'conversation' in item:
                    messages = item['conversation']
                elif 'text' in item:
                    # Fallback to plain text
                    text = item['text']
                    tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_seq_len)
                    if len(tokens) > 1:
                        yield torch.tensor(tokens, dtype=torch.long)
                    continue
                else:
                    continue

                # Apply chat template if tokenizer supports it
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    tokens = self.tokenizer.apply_chat_template(
                        messages,
                        truncation=True,
                        max_length=self.max_seq_len,
                        add_generation_prompt=False
                    )
                else:
                    # Fallback: concatenate messages with special tokens
                    text = ""
                    for msg in messages:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        text += f"<|{role}|>{content}<|end|>"
                    tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_seq_len)

                if len(tokens) > 1:
                    yield torch.tensor(tokens, dtype=torch.long)

            except Exception as e:
                # Skip malformed examples
                continue


def sft_collate_fn(batch):
    """
    Collate function for SFT
    Pads sequences to the same length and creates (input, target) pairs
    """
    # Find max length in batch
    max_len = max(len(seq) for seq in batch)

    # Pad sequences
    padded_batch = []
    for seq in batch:
        if len(seq) < max_len:
            # Pad with -100 (will be used for target masking)
            padding = torch.full((max_len - len(seq),), -100, dtype=torch.long)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq
        padded_batch.append(padded_seq)

    batch_tensor = torch.stack(padded_batch)

    # Create input and target
    # Input: all tokens except the last
    # Target: all tokens except the first
    x = batch_tensor[:, :-1].contiguous()
    y = batch_tensor[:, 1:].contiguous()

    # Replace -100 in input with 0 (a valid token ID for embedding layer)
    # The model won't use these positions anyway since they're padding
    x = x.masked_fill(x == -100, 0)

    # Keep -100 in targets for loss masking (ignored by cross_entropy)
    # No change needed for y since -100 is the correct value for ignoring in loss

    return x, y
