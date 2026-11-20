"""Guided, bounded experiments using the existing recipe runner and run registry."""
import atexit
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
from typing import Literal
from copy import deepcopy

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from run_store import get_run_store, LANGUAGE_METRICS, clean
from . import training

router = APIRouter()
ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = 'small-model-learning'
RECIPE = ROOT / 'recipes/small-model-learning.toml'
RECIPES = {EXPERIMENT_ID: RECIPE, 'residual-connections': ROOT / 'recipes/residual-connections.toml'}
RESIDUAL_VARIANTS = [
    dict(id='with-residuals', title='With residual connections', description='Add each sublayer output to the stream that bypassed it.'),
    dict(id='without-residuals', title='Without residual connections', description='Pass through the same learned layers without the bypass additions.')]
_process = None
_active_run = None
_stop_requested = False


def hardware():
    import torch
    available = torch.cuda.is_available()
    return dict(cpu_name=platform.processor() or platform.machine(), cuda_available=available,
                cuda_device_name=torch.cuda.get_device_name(0) if available else None,
                cuda_version=torch.version.cuda)


def require_experiment(experiment_id):
    if experiment_id not in RECIPES:
        raise HTTPException(404, 'Learning experiment not found')
    return experiment_id


def recipe_for(experiment_id, variant=None):
    from experiments.recipe import load_recipe
    require_experiment(experiment_id)
    recipe = load_recipe(RECIPES[experiment_id])
    if experiment_id == 'residual-connections':
        variant = variant or 'with-residuals'
        if variant not in {row['id'] for row in RESIDUAL_VARIANTS}:
            raise HTTPException(422, 'Choose with-residuals or without-residuals.')
        if variant == 'without-residuals':
            body = recipe['model']['assembly']['nodes'][2]['body']
            additions = {node['id'] for node in body['nodes'] if node['kind'] == 'add'}
            sublayers = {edge['target']: edge['source'] for edge in body['edges']
                         if edge['target'] in additions and edge.get('targetHandle') == 'a'}
            body['nodes'] = [node for node in body['nodes'] if node['id'] not in additions]
            body['edges'] = [{**edge, 'source': sublayers.get(edge['source'], edge['source'])}
                             for edge in body['edges'] if edge['target'] not in additions]
    elif variant is not None:
        raise HTTPException(422, 'This experiment has no architecture variants.')
    return recipe, variant


def descriptor(experiment_id=EXPERIMENT_ID):
    recipe, variant = recipe_for(experiment_id)
    machine = hardware()
    preferred = 'cuda' if machine['cuda_available'] else 'cpu'
    result = dict(id=experiment_id, title='What does a small model learn?',
        description='Train a small model from random weights, then compare predictions on held-out synthetic sentences.',
        default_steps=120, min_steps=20, max_steps=300, seed=42, device=preferred,
        default_device=preferred, available_devices=['cpu', 'cuda'] if machine['cuda_available'] else ['cpu'],
        hardware=machine, model={**recipe['model'], 'vocab_size': 512},
        dataset=dict(name='Small-model sentence patterns', synthetic=True, license='MIT',
            train_documents=320, validation_documents=64,
            description='Synthetic sentences with shared vocabulary and different held-out combinations. This is not a general language benchmark.'),
        prompts=recipe['evaluation']['prompts'],
        concepts=[dict(id=id, title=title, url='https://www.thelastserver.com/understanding-ai/#tech-'+id)
                  for id, title in [('transformer', 'Transformer'), ('bpe', 'Tokenization'), ('gpt1', 'Learning from text'), ('adam', 'Optimizer updates')]],
        measurement_notes=[
            'Validation cross-entropy is averaged over predicted tokens from held-out documents; lower is better.',
            'Perplexity is exp(language cross-entropy); compare only the same tokenizer and validation data.',
            'Before and after use the same prompts, tokenizer, sampling settings and seeds.',
            'Throughput is the mean training-loop rate, including validation and checkpoint overhead; it is not a kernel benchmark.'])
    if experiment_id == 'residual-connections':
        result.update(title='What do residual connections change?',
            description='Compare the same learned layers with and without residual bypass additions.',
            variants=deepcopy(RESIDUAL_VARIANTS), default_variant=variant,
            concepts=[dict(id=id, title=title, url='https://www.thelastserver.com/understanding-ai/#tech-'+id)
                      for id, title in [('transformer', 'Transformer residual connections'), ('pre-layernorm', 'Pre-norm blocks')]])
        result['measurement_notes'].extend([
            'The two variants keep the same learned parameter names, shapes and initial values; only graph connections change.',
            'Gradient norm is the global L2 norm returned before gradient clipping. A short run may favor either variant.'])
    return result


class StartRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    steps: int = Field(default=120, ge=20, le=300, strict=True)
    seed: int = Field(default=42, ge=0, le=2147483647, strict=True)
    device: Literal['cpu', 'cuda'] | None = None
    variant: Literal['with-residuals', 'without-residuals'] | None = None


def require_run(run_id):
    if not re.fullmatch(r'[0-9a-f]{32}', run_id):
        raise HTTPException(404, 'Learning run not found')
    try:
        record = get_run_store().get(run_id)
    except KeyError:
        raise HTTPException(404, 'Learning run not found')
    if record.get('experiment_id') not in RECIPES:
        raise HTTPException(404, 'Learning run not found')
    return record


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def metric_rows(path):
    rows = []
    try:
        for line in Path(path).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # The worker may be appending the current row.
    except FileNotFoundError:
        pass
    return rows


def snapshot(record):
    root = Path(record['output_dir']) / 'experiment'
    before = read_json(root / 'before.json')
    results = read_json(root / 'results.json')
    state = read_json(root / 'state.json', {})
    rows = metric_rows(root / 'metrics.jsonl')
    metrics = []
    for row in rows:
        metrics.append({**row, 'loss': row.get('loss', row.get('train_loss')),
            'eval_loss': row.get('eval_loss', row.get('val_loss')),
            'lr': row.get('lr', row.get('learning_rate')),
            'eval_perplexity': row.get('eval_perplexity', row.get('val_perplexity'))})
    logs = []
    log_path = Path(record['output_dir']) / 'process.log'
    if log_path.exists():
        logs = [dict(message=line[-2000:], level='info', timestamp=None)
                for line in log_path.read_text(errors='replace').replace('\r', '\n').splitlines()[-80:] if line.strip()]
    phase = (record['status'] if record['status'] in {'interrupted', 'error', 'failed', 'stopped', 'completed'}
             else record.get('phase', state.get('phase', record['status'])))
    result = {**record, 'phase': phase,
        'current_step': max(record.get('current_step', 0), state.get('step', 0)),
        'metrics': metrics, 'before': before, 'after': results.get('after') if results else None,
        'results': results, 'logs': logs, 'tokens_seen': state.get('tokens_seen'),
        'fingerprints': before.get('fingerprints') if before else None,
        'tokens_measured_step': next((row['step'] for row in reversed(rows) if 'tokens_seen' in row), None),
        'elapsed_seconds': record.get('elapsed_seconds', 0)}
    saved_model = read_json(root / 'model_config.json')
    if saved_model:
        result['configuration']['model'] = saved_model
    if before:
        result['configuration']['actual_parameters'] = before['parameters']
    return clean(result)


def mirror(run_id, started, seen):
    record = get_run_store().get(run_id)
    root = Path(record['output_dir']) / 'experiment'
    state = read_json(root / 'state.json', {})
    phase = state.get('phase', 'preparing')
    # The recipe writes 'completed' before generating its after samples.
    phase = 'comparing' if phase == 'completed' else phase
    changes = dict(phase=phase, current_step=state.get('step', 0), elapsed_seconds=time.monotonic()-started)
    get_run_store().update(run_id, changes)
    if training.training_state.get('run_id') == run_id:
        training.training_state.update(changes)
    rows = metric_rows(root / 'metrics.jsonl')
    for row in rows[seen:]:
        get_run_store().append(run_id, dict(type='metrics', **row))
    return len(rows)


def run_worker(run_id, recipe_path):
    global _process, _active_run
    record = get_run_store().get(run_id)
    started = time.monotonic()
    seen = 0
    status, error = 'error', None
    try:
        if _stop_requested:
            status = 'stopped'
            return
        output = Path(record['output_dir'])
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}
        with (output / 'process.log').open('w') as log:
            _process = subprocess.Popen([sys.executable, str(ROOT/'src/cli.py'), 'run', str(recipe_path),
                '--output-dir', str(output/'experiment')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if _stop_requested:
                _process.terminate()
            while _process.poll() is None:
                seen = mirror(run_id, started, seen)
                time.sleep(.2)
            mirror(run_id, started, seen)
            saved = read_json(output/'experiment/state.json', {})
            if _stop_requested:
                status = 'stopped'
            elif _process.returncode == 0 and saved.get('phase') in {'completed', 'stopped'}:
                status = saved['phase']
            else:
                error = saved.get('error') or f'Experiment process exited with status {_process.returncode}; open the saved logs.'
    except Exception as exception:
        error = f'{type(exception).__name__}: {exception}'
    finally:
        if _process is not None and _process.poll() is None:
            _process.terminate()
            try:
                _process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _process.kill()
                _process.wait(timeout=5)
        changes = dict(is_training=False, status=status, phase=status, error=error,
                       elapsed_seconds=time.monotonic()-started)
        get_run_store().update(run_id, changes)
        get_run_store().append(run_id, dict(type='status', status=status, error=error))
        _process = None
        _active_run = None
        # Release the shared training slot only after this worker has finished
        # touching process globals, so a new run cannot be cleared by this one.
        if training.training_state.get('run_id') == run_id:
            training.training_state.update(changes)


@router.get('/experiments')
async def list_experiments():
    return {'experiments': [descriptor(id) for id in RECIPES]}


@router.get('/experiments/{experiment_id}')
async def get_experiment(experiment_id: str):
    return descriptor(experiment_id)


@router.post('/experiments/{experiment_id}/runs')
async def start_run(experiment_id: str, request: StartRequest):
    global _active_run, _stop_requested
    from experiments.recipe import dumps_recipe
    import torch
    recipe, variant = recipe_for(experiment_id, request.variant)
    if training.training_state.get('is_training'):
        raise HTTPException(409, 'Another training run is active. Stop it or wait for it to finish.')
    device = request.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        raise HTTPException(422, 'CUDA is unavailable in this Python runtime. Select CPU explicitly or use a compatible CUDA-enabled installation.')
    machine = hardware()
    configuration = dict(steps=request.steps, seed=request.seed, device=device, variant=variant,
        model={**recipe['model'], 'vocab_size': recipe['tokenizer']['vocab_size']},
        training={**recipe['training'], 'max_steps': request.steps}, dataset=descriptor(experiment_id)['dataset'],
        evaluation=recipe['evaluation'], hardware=dict(device=device,
            device_name=machine['cuda_device_name'] if device == 'cuda' else machine['cpu_name'],
            dtype='float32 weights / bfloat16 autocast' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'float32',
            torch_version=torch.__version__, cuda_version=torch.version.cuda if device == 'cuda' else None,
            threads=recipe['experiment']['threads']))
    record = get_run_store().create({**training.idle_state(), 'experiment_id': experiment_id,
        'training_type': 'learning', 'is_training': True, 'status': 'starting', 'phase': 'preparing',
        'max_steps': request.steps, 'metric_definition': LANGUAGE_METRICS, 'configuration': configuration,
        'model_config': configuration['model'], 'training_config': configuration['training']}, 'learning:'+experiment_id)
    run_id = record['run_id']
    # Save a resolved recipe outside the runner's new-directory destination.
    recipe['experiment'].update(seed=request.seed, device=device)
    recipe['training']['max_steps'] = request.steps
    text = dumps_recipe(recipe)
    recipe_path = Path(record['output_dir'])/'launch.toml'
    try:
        recipe_path.write_text(text)
    except OSError as error:
        get_run_store().update(run_id, dict(status='error', phase='error', is_training=False,
                                           error=f'Could not save the launch recipe: {error}'))
        raise HTTPException(500, 'Could not save the experiment recipe. Check the data directory permissions.')
    dict.clear(training.training_state)
    dict.update(training.training_state, record)
    _active_run, _stop_requested = run_id, False
    training.training_thread = threading.Thread(target=run_worker, args=(run_id, recipe_path), daemon=True)
    try:
        training.training_thread.start()
    except RuntimeError as error:
        _active_run = None
        training.training_state.update(status='error', phase='error', is_training=False,
                                       error=f'Could not start the experiment worker: {error}')
        raise HTTPException(500, 'Could not start the experiment worker.')
    return {'run_id': run_id}


@router.get('/runs')
async def list_runs(experiment_id: str = EXPERIMENT_ID):
    require_experiment(experiment_id)
    return {'runs': [snapshot(row) for row in get_run_store().list('learning:'+experiment_id)]}


@router.get('/runs/{run_id}')
async def get_run(run_id: str):
    return snapshot(require_run(run_id))


@router.post('/runs/{run_id}/stop')
async def stop_run(run_id: str):
    global _stop_requested
    require_run(run_id)
    if _active_run != run_id or not training.training_state.get('is_training'):
        raise HTTPException(409, 'This learning run is not active.')
    _stop_requested = True
    training.training_state.update(status='stopping')
    if _process is not None and _process.poll() is None:
        _process.terminate()
    return {'run_id': run_id, 'status': 'stopping'}


def shutdown():
    global _stop_requested
    _stop_requested = True
    process = _process
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def normalized_configuration(configuration):
    from experiments.runner import fingerprint_json
    from model.assembly_runtime import canonical_assembly
    value = deepcopy(configuration)
    model = value.get('model', {})
    model.pop('tokenizer_name', None)  # The run-local path is not a model difference.
    if model.get('assembly'):
        model['assembly_sha256'] = fingerprint_json(canonical_assembly(model.pop('assembly')))
    return value


def configuration_differences(a, b, prefix=''):
    rows = []
    for key in sorted(set(a) | set(b)):
        field = prefix+'.'+key if prefix else key
        av, bv = a.get(key), b.get(key)
        if isinstance(av, dict) and isinstance(bv, dict):
            rows.extend(configuration_differences(av, bv, field))
        elif av != bv:
            rows.append(dict(field=field, a=av, b=bv))
    return rows


@router.get('/comparisons')
async def compare_runs(run_a: str, run_b: str):
    if run_a == run_b:
        raise HTTPException(422, 'Choose two different learning runs.')
    a, b = snapshot(require_run(run_a)), snapshot(require_run(run_b))
    ac, bc = a.get('configuration', {}), b.get('configuration', {})
    af, bf = a.get('fingerprints') or {}, b.get('fingerprints') or {}
    checks = []
    def check(id, label, av, bv):
        match = None if av is None or bv is None else av == bv
        checks.append(dict(id=id, label=label, match=match,
            detail='Not recorded in one or both runs.' if match is None else 'Matches.' if match else 'Differs.'))
    check('data', 'Training and validation documents', af.get('data'), bf.get('data'))
    check('tokenizer', 'Tokenizer', af.get('tokenizer_sha256'), bf.get('tokenizer_sha256'))
    check('evaluation', 'Evaluation prompts and generation settings', af.get('evaluation_sha256'), bf.get('evaluation_sha256'))
    check('seed', 'Random seed', ac.get('seed'), bc.get('seed'))
    check('budget', 'Optimizer-update budget', ac.get('steps'), bc.get('steps'))
    check('training_settings', 'Optimizer and training settings', ac.get('training'), bc.get('training'))
    check('hardware', 'Device, precision and runtime', ac.get('hardware'), bc.get('hardware'))
    check('initial_weights', 'Initial learned parameter values', af.get('initial_weights_sha256'), bf.get('initial_weights_sha256'))
    check('metric_definition', 'Language metric definition', a.get('metric_definition'), b.get('metric_definition'))
    check('completed', 'Both runs completed', a.get('status') == 'completed', b.get('status') == 'completed')
    # Equality of two incomplete states must not certify a finished comparison.
    if a.get('status') != 'completed' or b.get('status') != 'completed':
        checks[-1].update(match=False, detail='Wait for both runs to complete; stopped or failed runs may have partial results.')
    for label, run, before in [('A', a, af), ('B', b, bf)]:
        after = (run.get('results') or {}).get('fingerprints_after')
        check('integrity_'+label.lower(), 'Run '+label+' snapshots unchanged during training', before or None, after)
    def value(run, field):
        return ((run.get('after') or {}).get('metrics') or {}).get(field)
    metrics = {}
    for id, field in [('validation_cross_entropy', 'val_language_loss'), ('validation_perplexity', 'val_perplexity')]:
        av, bv = value(a, field), value(b, field)
        metrics[id] = dict(a=av, b=bv, delta_b_minus_a=bv-av if av is not None and bv is not None else None)
    warnings = [row['label']+': '+row['detail'] for row in checks if row['match'] is not True]
    return clean(dict(run_a=a, run_b=b, configuration_differences=configuration_differences(
        normalized_configuration(ac), normalized_configuration(bc)),
        comparability=dict(comparable=not warnings, checks=checks, warnings=warnings), metrics=metrics))


atexit.register(shutdown)
