"""Train a small dense LM on a pinned, locally downloaded TinyStories subset.

See docs/reference-pretraining.md for download and execution commands.
No Hub upload is performed. Requires CUDA; all generated artifacts stay in output-dir.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))
REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
TRAIN_FILE = 'train-00000-of-00004-2d5a1467fff1081b.parquet'
VALID_FILE = 'validation-00000-of-00001-869c898b519ad725.parquet'
PROMPTS = ['Once upon a time, there was a little girl named Lily.',
           'Tom found a small dog in the garden.',
           'One day, a little rabbit wanted to find a friend.']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare_corpus(source, root, train_count, validation_count):
    """Select whole documents; remove normalized duplicates across both splits."""
    import pyarrow.parquet as pq
    seen = set()
    result = {}
    for split, filename, count in [('train', TRAIN_FILE, train_count),
                                    ('validation', VALID_FILE, validation_count)]:
        input_path = source / 'data' / filename
        output_path = root / f'{split}.jsonl'
        kept = 0
        duplicates = 0
        with output_path.open('w') as output:
            for batch in pq.ParquetFile(input_path).iter_batches(batch_size=1024, columns=['text']):
                for text in batch.column('text').to_pylist():
                    text = (text or '').strip()
                    if not text:
                        continue
                    key = hashlib.sha256(' '.join(text.split()).encode()).digest()
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                    output.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    kept += 1
                    if kept == count:
                        break
                if kept == count:
                    break
        if kept != count:
            raise ValueError(f'Only {kept} unique {split} documents, requested {count}')
        result[split] = {'documents': kept, 'duplicates_skipped': duplicates,
                         'path': str(output_path), 'sha256': digest(output_path),
                         'source_file': filename, 'source_sha256': digest(input_path)}
    return result


def train_tokenizer(train_path, output_path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.BPE(unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=8192, min_frequency=2,
                                  initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
                                  special_tokens=['[UNK]', '[EOS]', '[PAD]'])
    def texts():
        with train_path.open() as source:
            for line in source:
                yield json.loads(line)['text']
    backend.train_from_iterator(texts(), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                        eos_token='[EOS]', pad_token='[PAD]',
                                        model_input_names=['input_ids', 'attention_mask'])
    tokenizer.save_pretrained(output_path)
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--train-documents', type=int, default=40000)
    parser.add_argument('--validation-documents', type=int, default=512)
    parser.add_argument('--eval-batches', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.train_documents, args.validation_documents, args.eval_batches) < 1:
        parser.error('Counts must be positive')
    source, root = args.source_dir.resolve(), args.output_dir.resolve()
    os.environ.setdefault('HF_HOME', str(root / 'hf-cache'))
    os.environ.setdefault('HF_DATASETS_CACHE', str(root / 'dataset-cache'))
    os.environ.setdefault('MPLCONFIGDIR', str(root / 'matplotlib-cache'))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    import torch
    from torch.utils.data import DataLoader
    from config import ModelConfig, TrainingConfig
    from data import create_token_stream, StreamingTokenDataset, lm_collate_fn
    from model.factory import build_model
    from training.train import train_model, evaluate_model, precision_context
    from inference.inference import generate_text
    from export import export_to_local

    if not torch.cuda.is_available():
        raise RuntimeError('This reference run requires a CUDA GPU; refusing CPU fallback')
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(root)  # Isolate the legacy tokenizer pickle cache as well.
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f'GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; CUDA={torch.version.cuda}', flush=True)
    data = prepare_corpus(source, root, args.train_documents, args.validation_documents)
    tokenizer = train_tokenizer(Path(data['train']['path']), root / 'tokenizer')
    model_config = ModelConfig(tokenizer_name=str(root / 'tokenizer'), vocab_size=len(tokenizer),
                                d_model=384, n_layers=6, n_heads=6, n_kv_heads=2,
                                d_ff=1024, max_seq_len=256, dropout=0)
    config = TrainingConfig(max_steps=args.steps, batch_size=8, gradient_accumulation_steps=2,
                             optimizer='adamw', lr=1e-3, adamw_beta2=0.95,
                             scheduler='cosine', warmup_steps=min(50, args.steps // 10),
                             eval_every=min(100, args.steps), eval_steps=args.eval_batches,
                             gradient_checkpointing=False, save_best_only=False,
                             datasets=[{'name': 'json', 'data_files': data['train']['path']}],
                             validation_datasets=[{'name': 'json', 'data_files': data['validation']['path']}])
    manifest = {'dataset': 'roneneldan/TinyStories', 'requested_revision': REVISION,
                'dataset_license': 'cdla-sharing-1.0', 'data': data, 'seed': args.seed,
                'tokenizer_sha256': digest(root / 'tokenizer/tokenizer.json'),
                'python': platform.python_version(), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                'precision': 'float32 master weights and AdamW states; bfloat16 autocast',
                'command': sys.argv, 'source_git_head': subprocess.check_output(
                    ['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip()}
    # Capture the working-tree patch because this project has uncommitted changes.
    patch_text = subprocess.check_output(['git', '-C', str(REPO_ROOT), 'diff', 'HEAD', '--',
                                         'src', 'examples', 'requirements.txt'], text=True)
    (root / 'source.patch').write_text(patch_text)
    import shutil
    shutil.copy2(__file__, root / 'reference_pretraining.py')
    manifest['source_patch_sha256'] = digest(root / 'source.patch')
    model = build_model(model_config).to('cuda').eval()
    manifest['parameters'] = model.count_parameters()
    write_json(root / 'manifest.json', manifest)
    write_json(root / 'environment.json', dict(sorted(
        (dist.metadata['Name'], dist.version) for dist in importlib.metadata.distributions()
        if dist.metadata['Name'])))
    model_config.save(str(root / 'model_config.json'))
    config.save(str(root / 'training_config.json'))
    print(f'Parameters: {manifest["parameters"]:,}; budget: {args.steps * 8 * 2 * 255:,} tokens', flush=True)

    val_stream = create_token_stream(config.validation_datasets, tokenizer)
    val_loader = DataLoader(StreamingTokenDataset(val_stream, 256, split='all'), batch_size=8,
                             collate_fn=lm_collate_fn, generator=torch.Generator().manual_seed(0))
    initial_metrics = evaluate_model(model, val_loader, config, len(tokenizer))
    def samples(m):
        output = []
        for i, prompt in enumerate(PROMPTS):
            torch.manual_seed(args.seed + i)
            continuation = generate_text(m, tokenizer, torch.device('cuda'), prompt, max_tokens=80,
                                          strategy='top_p', top_p=0.9, temperature=0.8)
            output.append({'prompt': prompt, 'continuation': continuation})
        return output
    before = samples(model)
    write_json(root / 'before.json', {'metrics': initial_metrics, 'samples': before})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    class Monitor:
        def on_step(self, step, loss, lr, ppl):
            if not math.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            self.record({'step': step, 'train_loss': loss, 'lr': lr, 'perplexity': ppl})
        def on_eval(self, step, loss, ppl):
            self.record({'step': step, 'val_loss': loss, 'val_perplexity': ppl})
            print(f'VALIDATION step={step} loss={loss:.4f} ppl={ppl:.2f}', flush=True)
        def record(self, row):
            row['elapsed_seconds'] = time.perf_counter() - start
            with (root / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')

    torch.manual_seed(args.seed)  # Match the untrained baseline initialization.
    start = time.perf_counter()
    train_model(model_config, config, output_dir=str(root / 'run'), callback=Monitor())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    checkpoint = torch.load(root / 'run/final_model.pt', map_location='cpu', weights_only=False)
    model = build_model(checkpoint['model_config']).to('cuda').eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    after = samples(model)
    export_dir = root / 'export'
    export_to_local(str(root / 'run/final_model.pt'), str(export_dir),
                    model_name='LLM-Lab TinyStories reference',
                    model_description='Small from-scratch reference experiment; not an instruction model.',
                    license_type='cdla-sharing-1.0', datasets_used=['roneneldan/TinyStories'])
    # Fresh interpreter with no repository imports; compare mixed-precision logits
    # and deterministic generation to verify the self-contained exported model.
    x = tokenizer(PROMPTS[0], return_tensors='pt')['input_ids'].to('cuda')
    with torch.no_grad(), precision_context(torch.device('cuda')):
        logits = model(x)[0].cpu()
    torch.save({'input_ids': x.cpu(), 'logits': logits}, root / 'export-reference.pt')
    from inference.inference import generate_text as generate
    greedy_source = generate(model, tokenizer, torch.device('cuda'), PROMPTS[0], max_tokens=40, strategy='greedy')
    code = '''
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
root = sys.argv[1]
model = AutoModelForCausalLM.from_pretrained(root + '/export', trust_remote_code=True, local_files_only=True).to('cuda').eval()
tokenizer = AutoTokenizer.from_pretrained(root + '/export', local_files_only=True)
ref = torch.load(root + '/export-reference.pt', weights_only=True)
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    actual = model(ref['input_ids'].cuda()).logits.cpu()
    delta = (actual.float() - ref['logits'].float()).abs().max().item()
    torch.testing.assert_close(actual, ref['logits'], rtol=0, atol=0)
    generated = model.generate(ref['input_ids'].cuda(), max_new_tokens=40, do_sample=False,
                               eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
continuation = tokenizer.decode(generated[0, ref['input_ids'].shape[1]:], skip_special_tokens=True)
assert continuation == sys.argv[2], (continuation, sys.argv[2])
print(json.dumps({'max_logit_difference': delta, 'greedy_generation_matches': True}))
'''
    result = subprocess.run([sys.executable, '-c', code, str(root), greedy_source], cwd=root,
                             env={**os.environ, 'PYTHONPATH': '', 'HF_HUB_OFFLINE': '1'},
                             capture_output=True, text=True, timeout=120)
    (root / 'export-verification.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Export verification failed; see export-verification.log')
    export_check = json.loads(result.stdout.strip().splitlines()[-1])
    final = checkpoint['final_metrics']
    summary = {'parameters': manifest['parameters'], 'steps': checkpoint['step'],
                'tokens_seen': checkpoint['total_tokens_seen'], 'initial_validation': initial_metrics,
                'final_validation': final, 'loss_reduction': initial_metrics['val_loss'] - final['val_loss'],
                'elapsed_training_seconds': elapsed, 'training_tokens_per_second': checkpoint['total_tokens_seen'] / elapsed,
                'peak_allocated_vram_gib': peak_gb, 'export_verification': export_check,
                'before': before, 'after': after}
    write_json(root / 'results.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not final['val_loss'] < initial_metrics['val_loss']:
        raise RuntimeError('Validation loss did not improve; inspect results before extending the run')


if __name__ == '__main__':
    main()
