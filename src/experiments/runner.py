"""Persistent run lifecycle; reusable from a future GUI without shelling out."""
from contextlib import contextmanager
from dataclasses import asdict
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

from .data import digest, prepare_data, prepare_tokenizer
from .recipe import load_recipe


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def initial_weights_fingerprint(model):
    """Hash learned parameter identities and bytes, excluding topology-only nodes."""
    import torch
    result = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        tensor = parameter.detach().cpu().contiguous()
        result.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def tokenizer_fingerprint(root):
    """Include saved vocabulary, special tokens and configuration for fast or slow tokenizers."""
    files = {}
    for path in sorted(Path(root).iterdir()):
        if not path.is_file():
            continue
        if path.suffix == '.json':
            value = read_json(path)
            if isinstance(value, dict):
                value = {key: item for key, item in value.items() if key not in {'name_or_path', '_name_or_path'}}
            files[path.name] = fingerprint_json(value)
        else:
            files[path.name] = digest(path)
    if not files:
        raise ValueError('No saved tokenizer files are available for comparison')
    return fingerprint_json(files)


def protocol_fingerprints(root, recipe, initial_weights_sha256):
    manifest = read_json(root / 'manifest.json')
    data = {split+'_sha256': digest(root / 'data' / (split+'.jsonl')) for split in ['train', 'validation']}
    data.update({split+'_documents': manifest['data'][split]['documents'] for split in ['train', 'validation']})
    return dict(data=data, tokenizer_sha256=tokenizer_fingerprint(root/'tokenizer'),
                evaluation_sha256=fingerprint_json(recipe['evaluation']),
                initial_weights_sha256=initial_weights_sha256)


@contextmanager
def lock(root):
    path = root / '.run.lock'
    if path.exists():
        pid = int(path.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink()
        else:
            raise ValueError(f'Run already locked by process {pid}')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def working_directory(root):
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def check_device(recipe):
    import torch
    if recipe['experiment']['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Recipe requires CUDA; no GPU is accessible (CPU fallback is disabled)')


def assert_snapshot(root):
    manifest = read_json(root / 'manifest.json')
    for relative, expected in manifest['file_hashes'].items():
        path = root / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Run snapshot changed or missing: {relative}; refusing incompatible resume')


def samples(model, tokenizer, recipe):
    import torch
    from inference.inference import generate_text
    result = []
    for index, prompt in enumerate(recipe['evaluation']['prompts']):
        length = len(tokenizer.encode(prompt))
        count = min(recipe['evaluation']['max_new_tokens'], model.config.max_seq_len - length - 1)
        if length < 1 or count < 1:
            raise ValueError('Evaluation prompt must leave space in the model context')
        torch.manual_seed(recipe['experiment']['seed'] + index)
        result.append({'prompt': prompt, 'continuation': generate_text(model, tokenizer,
            next(model.parameters()).device, prompt, max_tokens=count, strategy='top_p',
            temperature=recipe['evaluation']['temperature'], top_p=recipe['evaluation']['top_p'])})
    return result


def baseline(root, recipe, model_config, training, tokenizer):
    import torch
    from torch.utils.data import DataLoader
    from data import create_token_stream, StreamingTokenDataset, lm_collate_fn
    from model.factory import build_model
    from training.train import evaluate_model
    torch.manual_seed(recipe['experiment']['seed'])
    model = build_model(model_config).to(recipe['experiment']['device']).eval()
    initial_weights_sha256 = initial_weights_fingerprint(model)
    loader = DataLoader(StreamingTokenDataset(create_token_stream(training.validation_datasets, tokenizer),
        model_config.max_seq_len, split='all'), batch_size=training.batch_size, collate_fn=lm_collate_fn,
        generator=torch.Generator().manual_seed(0))
    value = {'metrics': evaluate_model(model, loader, training, model_config.vocab_size),
             'samples': samples(model, tokenizer, recipe), 'parameters': model.count_parameters(),
             'initial_weights_sha256': initial_weights_sha256,
             'fingerprints': protocol_fingerprints(root, recipe, initial_weights_sha256)}
    write_json(root / 'before.json', value)
    return value


def run(recipe_path, output, stop_after=None):
    from config import ModelConfig, TrainingConfig
    recipe_path = Path(recipe_path).resolve()
    recipe = load_recipe(recipe_path)
    check_device(recipe)
    root = Path(output).resolve()
    os.environ.setdefault('HF_HOME', str(root / 'hf-cache'))
    os.environ.setdefault('HF_DATASETS_CACHE', str(root / 'dataset-cache'))
    os.environ.setdefault('MPLCONFIGDIR', str(root / 'matplotlib-cache'))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    root.mkdir(parents=True, exist_ok=False)
    with lock(root), working_directory(root):
        write_json(root / 'state.json', {'phase': 'preparing', 'name': recipe['experiment']['name'], 'step': 0})
        try:
            write_json(root / 'recipe.resolved.json', recipe)
            (root / 'recipe.toml').write_bytes(Path(recipe_path).resolve().read_bytes())
            data = prepare_data(recipe['data'], root / 'data')
            tokenizer = prepare_tokenizer(recipe['tokenizer'], data['train']['path'], root / 'tokenizer')
            model = ModelConfig(**recipe['model'], vocab_size=len(tokenizer), tokenizer_name=str(root / 'tokenizer'))
            training = TrainingConfig(**recipe['training'],
                datasets=[{'name': 'json', 'data_files': data['train']['path']}],
                validation_datasets=[{'name': 'json', 'data_files': data['validation']['path']}])
            model.save(str(root / 'model_config.json'))
            training.save(str(root / 'training_config.json'))
            manifest = {'version': 1, 'data': data, 'file_hashes': {},
                        'environment': {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}}
            for path in [root / 'recipe.resolved.json', root / 'model_config.json', root / 'training_config.json',
                         *sorted((root / 'data').glob('*')), *sorted((root / 'tokenizer').glob('*'))]:
                if path.is_file():
                    manifest['file_hashes'][str(path.relative_to(root))] = digest(path)
            write_json(root / 'manifest.json', manifest)
            import torch
            torch.set_num_threads(recipe['experiment']['threads'])
            random.seed(recipe['experiment']['seed'])
            initial = baseline(root, recipe, model, training, tokenizer)
            state = read_json(root / 'state.json')
            state.update(parameters=initial['parameters'], initial_validation=initial['metrics'])
            write_json(root / 'state.json', state)
            execute(root, recipe, model, training, stop_after=stop_after)
        except BaseException as error:
            failed(root, error)
            raise
    return read_json(root / 'state.json')


def failed(root, error):
    state = read_json(root / 'state.json')
    state.update(phase='failed', error=f'{type(error).__name__}: {error}')
    write_json(root / 'state.json', state)


def choose_checkpoint(root):
    candidates = [p for p in (root / 'checkpoints').glob('*.pt') if p.name in {'final_model.pt', 'latest_checkpoint.pt'}]
    if not candidates:
        raise ValueError('No resumable checkpoint exists in this run')
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def resume(output, additional_steps=0, stop_after=None):
    from config import ModelConfig, TrainingConfig
    root = Path(output).resolve()
    with lock(root), working_directory(root):
        assert_snapshot(root)
        recipe = read_json(root / 'recipe.resolved.json')
        check_device(recipe)
        checkpoint = choose_checkpoint(root)
        training = TrainingConfig.load(str(root / 'checkpoints/training_config.json'))
        model = ModelConfig.load(str(root / 'checkpoints/model_config.json'))
        state = read_json(root / 'state.json')
        if state['phase'] == 'completed' and not additional_steps:
            raise ValueError('Run is completed; use --additional-steps to extend it')
        if stop_after is not None:
            import torch
            current_step = torch.load(checkpoint, map_location='cpu', weights_only=False)['step']
            if stop_after <= current_step:
                raise ValueError(f'--stop-after must exceed the current update ({current_step})')
        try:
            execute(root, recipe, model, training, checkpoint, additional_steps, stop_after)
        except BaseException as error:
            failed(root, error)
            raise
    return read_json(root / 'state.json')


def execute(root, recipe, model, training, checkpoint=None, additional_steps=0, stop_after=None):
    import torch
    import math
    from transformers import AutoTokenizer
    from model.factory import build_model
    from training.train import train_model, request_stop
    from .verification import export_and_verify
    device = recipe['experiment']['device']
    torch.set_num_threads(recipe['experiment']['threads'])
    if checkpoint is None:
        torch.manual_seed(recipe['experiment']['seed'])
    state = read_json(root / 'state.json')
    resume_step = torch.load(checkpoint, map_location='cpu', weights_only=False)['step'] if checkpoint else 0
    state.update(phase='training', error=None, device=device, step=resume_step,
                 target_steps=resume_step + additional_steps if additional_steps else training.max_steps)
    write_json(root / 'state.json', state)
    start = time.perf_counter()
    previous_handlers = {}
    def stop(signum, frame):
        request_stop()
    for name in [signal.SIGINT, signal.SIGTERM]:
        previous_handlers[name] = signal.signal(name, stop)

    class Monitor:
        def record(self, row):
            row['elapsed_seconds'] = time.perf_counter() - start
            with (root / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            state.update(row)
            write_json(root / 'state.json', state)
        def on_update(self, step):
            state['step'] = step
            write_json(root / 'state.json', state)
            if stop_after is not None and step >= stop_after:
                request_stop()
        def on_metrics(self, metrics):
            if not math.isfinite(metrics['language_loss']):
                raise FloatingPointError(f'Non-finite language loss at step {metrics["step"]}')
            self.record({**metrics, 'train_loss': metrics['loss'], 'learning_rate': metrics['lr']})
        def on_evaluation_metrics(self, metrics):
            if not math.isfinite(metrics['eval_language_loss']):
                raise FloatingPointError(f'Non-finite validation loss at step {metrics["step"]}')
            self.record({**metrics, 'val_loss': metrics['eval_loss'], 'val_perplexity': metrics['eval_perplexity']})
        def on_step(self, step, loss, lr, ppl):
            if not math.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            self.record({'step': step, 'train_loss': loss, 'learning_rate': lr})
        def on_eval(self, step, loss, ppl):
            if not math.isfinite(loss):
                raise FloatingPointError(f'Non-finite validation loss at step {step}')
            self.record({'step': step, 'val_loss': loss, 'val_perplexity': ppl})
    try:
        train_model(model, training, checkpoint_path=str(checkpoint) if checkpoint else None,
                    output_dir=str(root / 'checkpoints'), additional_steps=additional_steps,
                    callback=Monitor(), device=device)
    finally:
        for name, handler in previous_handlers.items():
            signal.signal(name, handler)
    assert_snapshot(root)
    final_path = root / 'checkpoints/final_model.pt'
    saved = torch.load(final_path, map_location='cpu', weights_only=False)
    complete = saved['step'] >= saved['train_config'].max_steps
    state.update(phase='completed' if complete else 'stopped', step=saved['step'],
                  target_steps=saved['train_config'].max_steps, tokens_seen=saved['total_tokens_seen'],
                  final_metrics=saved['final_metrics'], checkpoint=str(final_path))
    write_json(root / 'state.json', state)
    trained = build_model(saved['model_config']).to(device).eval()
    trained.load_state_dict(saved['model_state_dict'])
    tokenizer = AutoTokenizer.from_pretrained(saved['model_config'].tokenizer_name, local_files_only=True)
    result = {'before': read_json(root / 'before.json'), 'after': {
        'metrics': saved['final_metrics'], 'samples': samples(trained, tokenizer, recipe)},
        'step': saved['step'], 'tokens_seen': saved['total_tokens_seen'],
        'fingerprints_after': protocol_fingerprints(root, recipe, read_json(root/'before.json').get('initial_weights_sha256'))}
    write_json(root / 'results.json', result)
    del trained, saved
    if complete and recipe['export']['enabled']:
        # Preserve earlier verified exports when extending a completed experiment.
        destination = root / 'export'
        if destination.exists():
            destination = root / f'export-step-{state["step"]}'
        state['export'] = export_and_verify(root, final_path, recipe, destination)
        write_json(root / 'state.json', state)
