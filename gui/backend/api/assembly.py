"""The workshop validates and probes the same graph used for training/export."""
from dataclasses import fields
import threading
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
import torch
from config import ModelConfig
from model.factory import build_model
from model.assembly_runtime import AssemblyError, validate_assembly, assembly_signature
from model.assembly_templates import transformer_assembly, template_catalog, family_template, resize_assembly

router = APIRouter()

@router.get('/bricks')
def brick_catalog():
    from model.assembly_schema import MODERN_BRICKS
    return MODERN_BRICKS

probe_lock = threading.Lock()


class AssemblyRequest(BaseModel):
    model_cfg: dict[str, Any]


class TokenizerRequest(BaseModel):
    name: str


@router.post('/tokenizer')
def read_tokenizer(request: TokenizerRequest):
    from transformers import AutoTokenizer
    if not request.name.strip():
        raise HTTPException(422, 'Enter a local directory or a Hugging Face ID.')
    try:
        tokenizer = AutoTokenizer.from_pretrained(request.name)
        return {'name': request.name, 'vocab_size': len(tokenizer), 'eos_token_id': tokenizer.eos_token_id}
    except Exception as error:
        raise HTTPException(422, f'Could not read this tokenizer: {error}')


def config_from_request(data):
    data = dict(data)
    if 'architecture' in data:
        data['model_architecture'] = data.pop('architecture')
    extra = set(data) - {f.name for f in fields(ModelConfig)}
    if extra:
        raise ValueError(f'Unknown model settings: {sorted(extra)}')
    return ModelConfig(**data)


def inspection(config):
    if config.assembly is None:
        raise ValueError('Open or create a graph to use the workshop.')
    plan = validate_assembly(config.assembly, config.d_model, config.vocab_size)
    with torch.device('meta'):
        model = build_model(config)
    views = {}
    def describe(graph, path, body=False, width=None):
        width = config.d_model if width is None else width
        compiled = validate_assembly(graph, width, config.vocab_size, body=body)
        views['/'.join(path)] = {'shapes': {id: list(shape) for id, shape in compiled['shapes'].items()},
                                 'execution_order': compiled['order']}
        for node in graph['nodes']:
            if node['kind'] == 'repeat':
                describe(node['body'], path + [node['id']], True, node.get('params', {}).get('width', width))
    describe(config.assembly, [])
    return {'valid': True, 'parameters': model.count_parameters(),
            'fingerprint': assembly_signature(config)[:12],
            'views': views,
            'execution_order': plan['order'], 'expanded_bricks': plan['expanded'],
            'shapes': {id: list(shape) for id, shape in plan['shapes'].items()},
            'notes': ['B = batch size; T = number of tokens.',
                      'Connections define the computation. Moving a layer on the canvas does not change the model.',
                      'Each repeated block has independent weights.']}


@router.get('/template')
def template():
    return {'assembly': transformer_assembly(), 'd_model': 64, 'n_heads': 4, 'n_kv_heads': 2,
            'd_ff': 128, 'n_layers': 2, 'max_seq_len': 256, 'dropout': 0.0}


class FamilyTemplateRequest(BaseModel):
    family: str
    size: str = 'nano'


class ResizeRequest(AssemblyRequest):
    width: int
    layers: int
    context: int


@router.get('/templates')
def templates():
    return {'templates': template_catalog()}


@router.post('/templates/build')
def build_template(request: FamilyTemplateRequest):
    try:
        values = family_template(request.family, request.size)
        inspection(config_from_request(values))
        return values
    except (ValueError, TypeError, ImportError) as error:
        raise HTTPException(422, str(error))


@router.post('/resize')
def resize(request: ResizeRequest):
    try:
        config = resize_assembly(config_from_request(request.model_cfg), request.width, request.layers, request.context)
        from dataclasses import asdict
        return {'model_cfg': asdict(config), 'inspection': inspection(config)}
    except (ValueError, TypeError, ImportError) as error:
        raise HTTPException(422, str(error))


@router.post('/inspect')
def inspect(request: AssemblyRequest):
    try:
        return inspection(config_from_request(request.model_cfg))
    except (ValueError, TypeError, AssertionError, KeyError, ImportError) as error:
        return {'valid': False, 'errors': [{'message': str(error), 'node_id': getattr(error, 'node_id', None),
                                          'path': getattr(error, 'path', []), 'hint': getattr(error, 'hint', 'Check dimensions and connections.')}]}


@router.post('/probe')
def probe(request: AssemblyRequest):
    from .training import training_state
    if training_state['is_training'] or not probe_lock.acquire(blocking=False):
        raise HTTPException(409, 'A computation is already running. Try again when it finishes.')
    try:
        config = config_from_request(request.model_cfg)
        report = inspection(config)
        if report['parameters'] > 25_000_000:
            raise ValueError('This quick test is limited to 25 million parameters. Reduce dimensions or repetitions.')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == 'cuda' else []):
            torch.manual_seed(42)
            model = build_model(config).to(device).train()
            tokens = torch.randint(config.vocab_size, (1, min(9, config.max_seq_len)), device=device)
            logits, auxiliary = model(tokens[:, :-1])
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, config.vocab_size), tokens[:, 1:].reshape(-1))
            if auxiliary is not None: loss = loss + auxiliary
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('The computation produced non-finite values. Check the graph.')
            report.update(tested_forward=True, tested_backward=True, device=device, loss=loss.item(),
                          message='Forward and backward passes succeeded. This does not measure the quality of a trained model.')
        return report
    except (ValueError, TypeError, AssertionError, KeyError, ImportError) as error:
        raise HTTPException(422, detail={'message': str(error), 'node_id': getattr(error, 'node_id', None), 'hint': getattr(error, 'hint', '')})
    except torch.cuda.OutOfMemoryError:
        raise HTTPException(422, 'Not enough GPU memory for this test. Reduce dimensions or repetitions.')
    finally:
        probe_lock.release()
