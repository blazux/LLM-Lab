import torch
from torch.utils.data import IterableDataset, Dataset
from datasets import load_dataset, interleave_datasets, get_dataset_config_info
import random


# Default chat template (standard ChatML) used when the tokenizer has none.
# The {% generation %} tags mark assistant content so that
# apply_chat_template(..., return_assistant_tokens_mask=True) can return a
# per-token assistant mask for loss masking.
# This same template is applied at export time (src/export/hub.py) so the
# tokenizer object is the single source of truth for the chat format.
DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% generation %}{{ message['content'] }}{{ '<|im_end|>' }}{% endgeneration %}"
    "{{ '\n' }}"
    "{% else %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


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


def normalize_role(role: str) -> str:
    """
    Normalize chat role names to standard 'user', 'assistant', or 'system'.
    Handles various naming conventions from different datasets.
    """
    if role is None:
        return 'user'

    role_lower = role.lower().strip()

    # System roles
    if role_lower in ('system', 'sys', 'instruction', 'context'):
        return 'system'

    # Assistant/model roles
    if role_lower in ('assistant', 'gpt', 'bot', 'chatbot', 'ai', 'model',
                       'claude', 'llama', 'vicuna', 'alpaca', 'response',
                       'answer', 'output', 'completion', 'agent'):
        return 'assistant'

    # User/human roles
    if role_lower in ('user', 'human', 'person', 'customer', 'input',
                       'question', 'query', 'prompt', 'student'):
        return 'user'

    # Default to user for unknown roles (safer than assuming assistant)
    return 'user'


def format_message(role: str, content: str) -> str:
    """Format a single message with the appropriate marker."""
    normalized = normalize_role(role)
    if normalized == 'system':
        return f"System: {content}\n\n"
    elif normalized == 'assistant':
        return f"Assistant: {content}\n\n"
    else:
        return f"User: {content}\n\n"


def format_prompt_for_generation(prompt: str, system_prompt: str = None) -> str:
    """
    Format a prompt for model generation in RLHF training.
    Returns text ending with "Assistant:" so the model continues from there.

    Args:
        prompt: The user's prompt/question
        system_prompt: Optional system prompt to prepend

    Returns:
        Formatted string like "System: ...\n\nUser: ...\n\nAssistant:"
    """
    text = ""
    if system_prompt:
        text += f"System: {system_prompt}\n\n"
    text += f"User: {prompt}\n\nAssistant:"
    return text


def format_preference_pair(prompt: str, response: str, system_prompt: str = None) -> str:
    """
    Format a prompt + response for computing log probabilities in DPO.

    Args:
        prompt: The user's prompt/question
        response: The model's response
        system_prompt: Optional system prompt to prepend

    Returns:
        Formatted string like "User: ...\n\nAssistant: response"
    """
    text = ""
    if system_prompt:
        text += f"System: {system_prompt}\n\n"
    text += f"User: {prompt}\n\nAssistant: {response}"
    return text


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

        # Prefer the tokenizer's Jinja chat template for formatting + assistant
        # loss masking. If the tokenizer has no template, install the standard
        # ChatML template as the default (single source of truth: the tokenizer
        # object, which is also what gets saved at export time).
        self.use_chat_template = False
        if hasattr(tokenizer, 'apply_chat_template'):
            if getattr(tokenizer, 'chat_template', None) is None:
                try:
                    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
                    print("✓ Tokenizer had no chat template — installed default ChatML template")
                except Exception:
                    pass
            if getattr(tokenizer, 'chat_template', None) is not None:
                self.use_chat_template = True

    @staticmethod
    def _normalize_messages(messages):
        """Normalize roles/content to the standard chat-template format."""
        norm = []
        for msg in messages:
            role = normalize_role(msg.get('role', 'user'))
            content = msg.get('content', '')
            if content is None:
                content = ''
            norm.append({"role": role, "content": str(content)})
        return norm

    def _chat_template_tokenize(self, messages):
        """
        Tokenize a conversation with the tokenizer's chat template and build a
        per-token assistant mask (1 = assistant token, train on it).

        Returns (input_ids, assistant_mask) as plain lists, or None if the
        chat-template paths fail (caller then falls back to legacy markers).
        """
        tok = self.tokenizer
        norm = self._normalize_messages(messages)
        if not norm:
            return None

        def _flatten(values):
            # Flatten a possibly nested [[...]] list (batch of one conversation)
            if hasattr(values, 'tolist'):
                values = values.tolist()
            values = list(values)
            if values and isinstance(values[0], (list, tuple)):
                return list(values[0])
            return values

        def _to_id_list(res):
            # apply_chat_template may return (depending on transformers version):
            # a flat list of ids, a nested [[ids]], a BatchEncoding/dict with
            # "input_ids", or tokenizers.Encoding objects.
            if hasattr(res, 'keys') and 'input_ids' in res:
                res = res['input_ids']
            if hasattr(res, 'ids'):  # tokenizers.Encoding
                return list(res.ids)
            if hasattr(res, 'tolist'):
                res = res.tolist()
            res = list(res)
            if res and hasattr(res[0], 'ids'):  # [Encoding]
                return list(res[0].ids)
            if res and isinstance(res[0], (list, tuple)):
                return list(res[0])
            return res

        # Preferred path: native assistant-token mask via {% generation %} tags
        try:
            out = tok.apply_chat_template(
                norm,
                tokenize=True,
                return_assistant_tokens_mask=True,
                return_dict=True,
            )
            ids = _to_id_list(out["input_ids"])
            mask = out.get("assistant_masks", None)
            if mask is not None:
                mask = _flatten(mask)
            # An all-zeros mask means the template has no {% generation %} tags
            if mask is not None and len(mask) == len(ids) and any(mask):
                return ids, mask
        except Exception:
            pass

        # Fallback path: per-message prefix tokenization; assistant spans are
        # found by diffing tokenized prefix lengths.
        try:
            full_ids = _to_id_list(tok.apply_chat_template(norm, tokenize=True))
            if not full_ids:
                return None
            mask = [0] * len(full_ids)
            prev_len = 0
            for i, msg in enumerate(norm):
                prefix_ids = _to_id_list(tok.apply_chat_template(norm[:i + 1], tokenize=True))
                cur_len = min(len(prefix_ids), len(full_ids))
                if cur_len < prev_len:
                    cur_len = prev_len
                if msg["role"] == 'assistant':
                    for j in range(prev_len, cur_len):
                        mask[j] = 1
                prev_len = cur_len
            if any(mask):
                return full_ids, mask
        except Exception:
            pass

        # Last resort: signal the caller to use the legacy marker path
        return None

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
                        role = normalize_role(turn.get('from', 'human'))
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
                    # Preferred: tokenizer chat template with assistant-token mask.
                    # Yields a dict so the collate fn can use the precomputed
                    # labels (-100 on non-assistant tokens) directly.
                    if self.use_chat_template:
                        result = self._chat_template_tokenize(messages)
                        if result is not None:
                            ids, mask = result
                            ids = ids[:self.max_seq_len]
                            mask = mask[:self.max_seq_len]
                            if len(ids) > 1:
                                input_ids = torch.tensor(ids, dtype=torch.long)
                                labels = torch.tensor(
                                    [t if m else -100 for t, m in zip(ids, mask)],
                                    dtype=torch.long
                                )
                                yield {"input_ids": input_ids, "labels": labels}
                            continue
                        # Chat-template path failed for this item — fall through
                        # to the legacy text-marker formatting below.

                    # Legacy fallback: simple text markers that the model already knows
                    # Format: "System: ...\n\nUser: ...\n\nAssistant: ...\n\n"
                    # This avoids needing special tokens and their associated training challenges
                    text = ""
                    for msg in messages:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        text += format_message(role, content)
                    tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_seq_len)

                    if len(tokens) > 1:
                        yield torch.tensor(tokens, dtype=torch.long)

            except Exception:
                # Skip items that fail to process
                continue


def find_subsequence_positions(sequence, subsequence):
    """
    Find all positions where subsequence starts in sequence.
    Returns list of starting indices.
    """
    positions = []
    seq_len = len(sequence)
    sub_len = len(subsequence)

    if sub_len == 0 or seq_len < sub_len:
        return positions

    for i in range(seq_len - sub_len + 1):
        if sequence[i:i + sub_len].tolist() == subsequence:
            positions.append(i)

    return positions


def find_all_marker_positions(sequence, marker_tokens_variants):
    """
    Find all positions of marker tokens, handling multiple tokenization variants.

    Args:
        sequence: tensor of token IDs
        marker_tokens_variants: either a list of token IDs, or a tuple of lists
                               (for markers that tokenize differently based on context)

    Returns:
        List of (start_position, marker_length) tuples
    """
    positions = []

    # Handle both single marker and tuple of variants
    if isinstance(marker_tokens_variants, tuple):
        variants = marker_tokens_variants
    else:
        variants = (marker_tokens_variants,)

    for marker_tokens in variants:
        if marker_tokens is None:
            continue
        for pos in find_subsequence_positions(sequence, marker_tokens):
            positions.append((pos, len(marker_tokens)))

    return positions


def sft_collate_fn(batch, tokenizer=None, assistant_marker_tokens=None, user_marker_tokens=None, system_marker_tokens=None):
    """
    Collate function for SFT
    Pads sequences to the same length and creates (input, target) pairs
    Only computes loss on assistant responses, not user prompts

    For multi-turn conversations, masks ALL user/system turns, not just the first one.

    Uses text markers ("Assistant:", "User:", "System:") instead of special tokens.
    The marker tokens are the tokenized versions of these strings.

    Note: marker_tokens can be either a list (single tokenization) or a tuple of lists
    (multiple tokenization variants for different contexts like start-of-text vs after-newlines).

    Batch items may be either:
    - plain 1D token tensors (legacy path — masking via text markers), or
    - dicts {"input_ids": tensor, "labels": tensor} produced by the chat-template
      path, where labels already contain -100 on non-assistant tokens.
    """
    # Normalize items: extract token sequences and optional precomputed labels
    seqs = []
    labels_list = []
    for item in batch:
        if isinstance(item, dict):
            seqs.append(item["input_ids"])
            labels_list.append(item.get("labels", None))
        else:
            seqs.append(item)
            labels_list.append(None)
    batch = seqs

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

    # Apply precomputed labels (chat-template path): targets come from the
    # shifted labels, so non-assistant tokens are already -100.
    for i, labels in enumerate(labels_list):
        if labels is None:
            continue
        shifted = labels[1:]
        row = torch.full((max_len - 1,), -100, dtype=torch.long)
        n = min(len(shifted), max_len - 1)
        if n > 0:
            row[:n] = shifted[:n]
        y[i] = row

    # Mask loss for non-assistant tokens (only train on assistant responses)
    # For multi-turn conversations, we need to mask ALL user/system turns
    # (legacy marker path — skipped for items with precomputed labels)
    if assistant_marker_tokens is not None:
        for i in range(y.size(0)):
            if labels_list[i] is not None:
                continue
            seq_len = x.size(1)
            # Start with all positions masked
            mask = torch.ones(seq_len, dtype=torch.bool)

            # Find all "Assistant:" marker positions in the input sequence
            # Handle multiple tokenization variants
            assistant_matches = find_all_marker_positions(x[i], assistant_marker_tokens)
            # Adjust to point to where the response target starts in y
            # Since y is shifted by 1 (y[i] = batch_tensor[i+1]), the first response
            # token's target is at y[pos + length - 1], not y[pos + length]
            assistant_positions = [pos + length - 1 for pos, length in assistant_matches]

            # Find positions where we should stop computing loss (User:/System: markers)
            stop_positions = []
            if user_marker_tokens is not None:
                user_matches = find_all_marker_positions(x[i], user_marker_tokens)
                # Adjust to y-space: x[pos] appears at y[pos-1], so stop before y[pos-1]
                stop_positions.extend([pos - 1 for pos, _ in user_matches])
            if system_marker_tokens is not None:
                system_matches = find_all_marker_positions(x[i], system_marker_tokens)
                stop_positions.extend([pos - 1 for pos, _ in system_matches])
            stop_positions = sorted(stop_positions)

            # For each "Assistant:" marker, unmask until the next User:/System: marker
            for asst_pos in assistant_positions:
                # Find the next stop position after this assistant marker
                next_stop = seq_len  # Default to end of sequence
                for stop_pos in stop_positions:
                    if stop_pos > asst_pos:
                        next_stop = stop_pos
                        break
                # Unmask from after assistant marker to next stop (compute loss on assistant response)
                mask[asst_pos:next_stop] = False

            # Apply mask: set masked positions to -100 in targets
            y[i, mask] = -100

    # Replace -100 in input with 0 (a valid token ID for embedding layer)
    # The model won't use these positions anyway since they're padding
    x = x.masked_fill(x == -100, 0)

    # Keep -100 in targets for loss masking (ignored by cross_entropy)
    # No change needed for y since -100 is the correct value for ignoring in loss

    return x, y
