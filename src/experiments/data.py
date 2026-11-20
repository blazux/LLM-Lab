"""Snapshot local documents and prepare tokenizers without validation leakage."""
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def documents(path, format):
    if format == 'parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=['text']):
            yield from batch.column('text').to_pylist()
    else:
        with Path(path).open(encoding='utf-8') as handle:
            for number, line in enumerate(handle, 1):
                if format == 'jsonl':
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)['text']
                    except (ValueError, KeyError, TypeError) as error:
                        raise ValueError(f'{path}:{number}: expected a JSON object with text') from error
                else:
                    yield line


def prepare_data(config, root):
    root = Path(root)
    root.mkdir()
    seen = set()
    stats = {}
    for split in ['train', 'validation']:
        output = root / f'{split}.jsonl'
        count = duplicates = 0
        limit = config.get(f'{split}_documents')
        with output.open('w', encoding='utf-8') as handle:
            for text in documents(config[split], config['format']):
                if not isinstance(text, str):
                    raise ValueError(f'{split}: document text must be a string')
                text = text.strip()
                if not text:
                    continue
                key = hashlib.sha256(' '.join(text.split()).encode()).digest()
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                handle.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                count += 1
                if limit and count >= limit:
                    break
        if not count or (limit and count < limit):
            raise ValueError(f'{split}: only {count} distinct documents available (requested {limit})')
        stats[split] = {'documents': count, 'duplicates_skipped': duplicates,
                        'path': str(output), 'sha256': digest(output),
                        'source_sha256': digest(config[split])}
    return stats


def prepare_tokenizer(config, train_path, output):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    if config['type'] == 'existing':
        tokenizer = AutoTokenizer.from_pretrained(config['path'], local_files_only=True)
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer requires an EOS token')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        from tokenizers import Tokenizer, models, decoders, pre_tokenizers, trainers
        backend = Tokenizer(models.BPE(unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=config['vocab_size'], min_frequency=config['min_frequency'],
                                      initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
                                      special_tokens=['[UNK]', '[EOS]', '[PAD]'])
        backend.train_from_iterator(documents(train_path, 'jsonl'), trainer=trainer)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]',
                                            pad_token='[PAD]', model_input_names=['input_ids', 'attention_mask'])
    tokenizer.save_pretrained(output)
    return tokenizer
