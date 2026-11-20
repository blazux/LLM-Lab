"""Strict, offline recipe validation. Paths are relative to the recipe file."""
from dataclasses import fields
import json
import math
from pathlib import Path
import tomllib

from config import ModelConfig, TrainingConfig


def dumps_recipe(recipe):
    """Serialize the recipe's explicit tables, including executable assembly DAGs."""
    def value(item):
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=False)
        if type(item) is bool:
            return str(item).lower()
        if type(item) in {int, float}:
            if not math.isfinite(item):
                raise ValueError('Recipe numbers must be finite')
            return repr(item)
        if isinstance(item, list):
            return '[' + ', '.join(value(x) for x in item) + ']'
        if isinstance(item, dict):
            return '{ ' + ', '.join(json.dumps(k) + ' = ' + value(v) for k, v in item.items()) + ' }'
        raise ValueError(f'Unsupported recipe value: {type(item).__name__}')
    lines = []
    for key, item in recipe.items():
        if not isinstance(item, dict):
            lines.append(json.dumps(key) + ' = ' + value(item))
    for key, table in recipe.items():
        if isinstance(table, dict):
            lines.extend(['', '[' + json.dumps(key) + ']'])
            lines.extend(json.dumps(k) + ' = ' + value(v) for k, v in table.items())
    return '\n'.join(lines) + '\n'


def unknown(table, allowed, label):
    if not isinstance(table, dict):
        raise ValueError(f'{label} must be a table')
    extra = set(table) - set(allowed)
    if extra:
        raise ValueError(f'{label}: unknown fields: {", ".join(sorted(extra))}')


def check_types(table, cls, label):
    for field in fields(cls):
        if field.name not in table:
            continue
        value = table[field.name]
        default = field.default
        if type(default) is bool and type(value) is not bool:
            raise ValueError(f'{label}.{field.name} must be a boolean')
        if type(default) is int and type(value) is not int:
            raise ValueError(f'{label}.{field.name} must be an integer')
        if type(default) is float and (type(value) not in {int, float} or not math.isfinite(value)):
            raise ValueError(f'{label}.{field.name} must be a finite number')


def load_recipe(path):
    path = Path(path).resolve()
    with path.open('rb') as handle:
        recipe = tomllib.load(handle)
    unknown(recipe, {'version', 'experiment', 'data', 'tokenizer', 'model', 'training', 'evaluation', 'export'}, 'recipe')
    if type(recipe.get('version')) is not int or recipe['version'] != 1:
        raise ValueError('recipe.version must be 1')
    for name in ['experiment', 'data', 'tokenizer', 'model', 'training']:
        if not isinstance(recipe.get(name), dict):
            raise ValueError(f'Missing [{name}] table')
    exp = recipe['experiment']
    unknown(exp, {'name', 'seed', 'device', 'threads'}, 'experiment')
    exp.setdefault('seed', 42)
    exp.setdefault('device', 'cuda')
    exp.setdefault('threads', 4)
    if not isinstance(exp.get('name'), str) or not exp['name'].strip():
        raise ValueError('experiment.name must be nonempty')
    if exp['device'] not in {'cpu', 'cuda'}:
        raise ValueError('experiment.device must be cpu or cuda')
    for key in ['seed', 'threads']:
        if type(exp[key]) is not int or exp[key] < (1 if key == 'threads' else 0):
            raise ValueError(f'experiment.{key} must be a valid integer')
    data = recipe['data']
    unknown(data, {'format', 'train', 'validation', 'train_documents', 'validation_documents'}, 'data')
    if data.get('format') not in {'text', 'jsonl', 'parquet'}:
        raise ValueError('data.format must be text, jsonl or parquet')
    for split in ['train', 'validation']:
        source = data.get(split)
        if not isinstance(source, str):
            raise ValueError(f'data.{split} must be a local file path')
        resolved = (path.parent / source).resolve()
        if not resolved.is_file():
            raise ValueError(f'data.{split}: file not found: {resolved}')
        data[split] = str(resolved)
        limit = data.get(f'{split}_documents')
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError(f'data.{split}_documents must be positive')
    if data['train'] == data['validation']:
        raise ValueError('Training and validation must be separate files')
    tokenizer = recipe['tokenizer']
    unknown(tokenizer, {'type', 'path', 'vocab_size', 'min_frequency'}, 'tokenizer')
    if tokenizer.get('type') == 'byte_bpe':
        unknown(tokenizer, {'type', 'vocab_size', 'min_frequency'}, 'tokenizer.byte_bpe')
        tokenizer.setdefault('vocab_size', 8192)
        tokenizer.setdefault('min_frequency', 2)
        for key, minimum in [('vocab_size', 259), ('min_frequency', 1)]:
            if type(tokenizer[key]) is not int or tokenizer[key] < minimum:
                raise ValueError(f'tokenizer.{key} must be >= {minimum}')
    elif tokenizer.get('type') == 'existing':
        unknown(tokenizer, {'type', 'path'}, 'tokenizer.existing')
        if not isinstance(tokenizer.get('path'), str):
            raise ValueError('tokenizer.path is required')
        tokenizer['path'] = str((path.parent / tokenizer['path']).resolve())
        if not Path(tokenizer['path']).is_dir():
            raise ValueError('tokenizer.path must be a local tokenizer directory')
    else:
        raise ValueError('tokenizer.type must be byte_bpe or existing')
    model = recipe['model']
    unknown(model, {f.name for f in fields(ModelConfig)} - {'tokenizer_name', 'vocab_size'}, 'model')
    train = recipe['training']
    unknown(train, {f.name for f in fields(TrainingConfig)} - {'datasets', 'validation_datasets', 'model_config_path'}, 'training')
    check_types(model, ModelConfig, 'model')
    check_types(train, TrainingConfig, 'training')
    for name, choices in {
        'attention_type': {'mha', 'mqa', 'gqa', 'mla'},
        'positional_encoding': {'rope', 'yarn', 'alibi', 'learned', 'sinusoidal'},
        'norm_type': {'rmsnorm', 'layernorm'},
        'activation': {'swiglu', 'geglu', 'reglu', 'gelu', 'relu', 'relu2', 'silu', 'leaky_relu'},
    }.items():
        if name in model and model[name] not in choices:
            raise ValueError(f'Unknown model.{name}: {model[name]}')
    try:
        cfg = ModelConfig(**model, vocab_size=tokenizer.get('vocab_size', 8192))
        training = TrainingConfig(**train)
    except (AssertionError, TypeError, ZeroDivisionError) as error:
        raise ValueError(f'Invalid configuration: {error}') from error
    # Validate common values explicitly; dataclass annotations alone do not enforce types.
    for name in ['d_model', 'n_layers', 'n_heads', 'd_ff', 'max_seq_len']:
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) < (2 if name == 'max_seq_len' else 1):
            raise ValueError(f'model.{name} must be a positive integer')
    for name in ['max_steps', 'batch_size', 'gradient_accumulation_steps', 'eval_every', 'eval_steps', 'warmup_steps']:
        if type(getattr(training, name)) is not int:
            raise ValueError(f'training.{name} must be an integer')
    if training.optimizer not in {'adamw', 'muon', 'lion', 'sophia'}:
        raise ValueError(f'Unknown optimizer: {training.optimizer}')
    if training.scheduler not in {'constant', 'cosine', 'linear', 'polynomial', 'wsd', 'adaptive'}:
        raise ValueError(f'Unknown scheduler: {training.scheduler}')
    if training.lr <= 0 or training.loss_fn not in {'cross_entropy', 'maxis'}:
        raise ValueError('Invalid learning rate or loss_fn')
    if not 0 <= cfg.dropout < 1 or training.grad_clip <= 0 or training.weight_decay < 0:
        raise ValueError('Invalid dropout, gradient clipping or weight decay')
    if training.use_8bit_optimizer and (exp['device'] != 'cuda' or training.optimizer not in {'adamw', 'muon'}):
        raise ValueError('8-bit optimizer is supported only for AdamW/Muon on CUDA')
    evaluation = recipe.setdefault('evaluation', {})
    unknown(evaluation, {'prompts', 'max_new_tokens', 'temperature', 'top_p'}, 'evaluation')
    evaluation.setdefault('prompts', ['Once upon a time'])
    evaluation.setdefault('max_new_tokens', 80)
    evaluation.setdefault('temperature', 0.8)
    evaluation.setdefault('top_p', 0.9)
    if not isinstance(evaluation['prompts'], list) or not evaluation['prompts'] or any(
            not isinstance(p, str) or not p.strip() for p in evaluation['prompts']):
        raise ValueError('evaluation.prompts must contain nonempty strings')
    if type(evaluation['max_new_tokens']) is not int or evaluation['max_new_tokens'] < 1:
        raise ValueError('evaluation.max_new_tokens must be positive')
    if evaluation['temperature'] <= 0 or not 0 < evaluation['top_p'] <= 1:
        raise ValueError('Invalid sampling temperature/top_p')
    export = recipe.setdefault('export', {})
    unknown(export, {'enabled', 'model_name', 'description', 'license'}, 'export')
    export.setdefault('enabled', True)
    if type(export['enabled']) is not bool:
        raise ValueError('export.enabled must be a boolean')
    export.setdefault('model_name', exp['name'])
    export.setdefault('description', 'From-scratch LLM-Lab pretraining experiment.')
    export.setdefault('license', 'other')
    for name in ['model_name', 'description', 'license']:
        if not isinstance(export[name], str) or not export[name].strip():
            raise ValueError(f'export.{name} must be a nonempty string')
    if export['enabled'] and cfg.positional_encoding != 'rope':
        raise ValueError('This recipe runner currently supports verified exports with RoPE; set export.enabled=false for other position encodings')
    return recipe


def describe(recipe):
    cfg = ModelConfig(**recipe['model'], vocab_size=recipe['tokenizer'].get('vocab_size', 8192))
    training = TrainingConfig(**recipe['training'])
    return {'name': recipe['experiment']['name'], 'device': recipe['experiment']['device'],
            'estimated_parameters': cfg.count_params(), 'steps': training.max_steps,
            'target_tokens': training.max_steps * training.batch_size * training.gradient_accumulation_steps * (cfg.max_seq_len - 1),
            'separate_validation': True, 'export': recipe['export']['enabled']}
