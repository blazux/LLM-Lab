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
            print(f"✓ Loaded SFT dataset: {ds_name}" + (f" ({ds_subset})" if ds_subset else "") + f" [split: {ds_split}]" + weight_str, flush=True)
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
                messages = None
                text = None

                # Try to extract conversational data in various formats
                if 'messages' in item:
                    # Chat format (e.g., ultrachat, smoltalk)
                    messages = item['messages']
                elif 'conversation' in item:
                    # Alternative chat format
                    messages = item['conversation']
                elif 'instruction' in item:
                    # Alpaca/instruction format - convert to messages
                    instruction = item.get('instruction', '')
                    input_text = item.get('input', '')
                    output = item.get('output', '')

                    # Build user message
                    if input_text:
                        user_content = f"{instruction}\n\n{input_text}"
                    else:
                        user_content = instruction

                    messages = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": output}
                    ]
                elif 'prompt' in item and 'completion' in item:
                    # Prompt/completion format
                    messages = [
                        {"role": "user", "content": item['prompt']},
                        {"role": "assistant", "content": item['completion']}
                    ]
                elif 'question' in item and 'answer' in item:
                    # Q&A format
                    messages = [
                        {"role": "user", "content": item['question']},
                        {"role": "assistant", "content": item['answer']}
                    ]
                elif 'question' in item and 'response' in item:
                    # OpenOrca format
                    system_prompt = item.get('system_prompt', '')
                    question = item['question']
                    response = item['response']

                    if system_prompt:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": response}
                        ]
                    else:
                        messages = [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": response}
                        ]
                elif 'conversations' in item:
                    # ShareGPT format - conversations with 'from' and 'value'
                    convos = item['conversations']
                    messages = []
                    for turn in convos:
                        role_map = {'human': 'user', 'gpt': 'assistant', 'system': 'system'}
                        role = role_map.get(turn.get('from', 'human'), 'user')
                        content = turn.get('value', '')
                        messages.append({"role": role, "content": content})
                elif 'text' in item:
                    # Plain text fallback
                    text = item['text']
                else:
                    continue

                # Process plain text
                if text is not None:
                    tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_seq_len)
                    if len(tokens) > 1:
                        yield torch.tensor(tokens, dtype=torch.long)
                    continue

                # Process messages format
                if messages:
                    # Always use our custom chat template format with special tokens
                    # This ensures consistency with loss masking (which looks for <|assistant|>, etc.)
                    # and inference (which applies the same format)
                    text = ""
                    for msg in messages:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        if role == 'system':
                            text += f"<|system|>{content}<|end|>\n"
                        elif role == 'user':
                            text += f"<|user|>{content}<|end|>\n"
                        elif role == 'assistant':
                            text += f"<|assistant|>{content}<|end|>\n"
                        else:
                            # Map other roles (human->user, gpt->assistant)
                            if role in ('human',):
                                text += f"<|user|>{content}<|end|>\n"
                            elif role in ('gpt', 'bot'):
                                text += f"<|assistant|>{content}<|end|>\n"
                            else:
                                text += f"<|user|>{content}<|end|>\n"
                    tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_seq_len)

                    if len(tokens) > 1:
                        yield torch.tensor(tokens, dtype=torch.long)

            except Exception:
                # Skip items that fail to process
                continue


def sft_collate_fn(batch, assistant_token_id=None, user_token_id=None, system_token_id=None):
    """
    Collate function for SFT
    Pads sequences to the same length and creates (input, target) pairs
    Only computes loss on assistant responses, not user prompts

    For multi-turn conversations, masks ALL user/system turns, not just the first one.
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
    # IMPORTANT: Use .clone() to avoid memory aliasing - otherwise modifying y also corrupts x
    x = batch_tensor[:, :-1].clone()
    y = batch_tensor[:, 1:].clone()

    # Mask loss for non-assistant tokens (only train on assistant responses)
    # For multi-turn conversations, we need to mask ALL user/system turns
    if assistant_token_id is not None:
        for i in range(y.size(0)):
            seq_len = x.size(1)
            # Start with all positions masked
            mask = torch.ones(seq_len, dtype=torch.bool)

            # Find all <|assistant|> token positions in the input sequence
            assistant_positions = (x[i] == assistant_token_id).nonzero(as_tuple=True)[0].tolist()

            # Find positions where we should stop computing loss (user/system tokens)
            stop_positions = []
            if user_token_id is not None:
                user_positions = (x[i] == user_token_id).nonzero(as_tuple=True)[0].tolist()
                stop_positions.extend(user_positions)
            if system_token_id is not None:
                system_positions = (x[i] == system_token_id).nonzero(as_tuple=True)[0].tolist()
                stop_positions.extend(system_positions)
            stop_positions = sorted(stop_positions)

            # For each <|assistant|> token, unmask until the next user/system token
            for asst_pos in assistant_positions:
                # Find the next stop position after this assistant token
                next_stop = seq_len  # Default to end of sequence
                for stop_pos in stop_positions:
                    if stop_pos > asst_pos:
                        next_stop = stop_pos
                        break
                # Unmask from assistant position to next stop (compute loss on assistant response)
                mask[asst_pos:next_stop] = False

            # Apply mask: set masked positions to -100 in targets
            y[i, mask] = -100

    # Replace -100 in input with 0 (a valid token ID for embedding layer)
    # The model won't use these positions anyway since they're padding
    x = x.masked_fill(x == -100, 0)

    # Keep -100 in targets for loss masking (ignored by cross_entropy)
    # No change needed for y since -100 is the correct value for ignoring in loss

    return x, y
