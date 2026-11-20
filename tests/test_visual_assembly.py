"""The visual graph must change the actual model, survive training and export."""
import contextlib
from copy import deepcopy
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'gui/backend')]
import torch
from config import ModelConfig, TrainingConfig
from model.factory import build_model
from model.assembly_templates import transformer_assembly
from model.assembly_runtime import AssemblyError, validate_assembly
from optimizers.optimizers import param_groups_weight_decay


class VisualAssemblyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.device = os.environ.get('LLMLAB_TEST_DEVICE', 'cpu')
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        self.graph = transformer_assembly(32, 4, 2, 64, 2)
        self.cfg = ModelConfig(assembly=self.graph, d_model=32, vocab_size=16, max_seq_len=64)

    def model(self, graph=None):
        torch.manual_seed(42)
        cfg = ModelConfig(assembly=graph or self.graph, d_model=32, vocab_size=16, max_seq_len=64)
        return build_model(cfg).to(self.device).eval()

    def test_reference_has_real_independent_layers_and_expected_parameter_count(self):
        model = self.model()
        legacy = build_model(ModelConfig(d_model=32, vocab_size=16, max_seq_len=64, n_layers=2,
                                          n_heads=4, n_kv_heads=2, d_ff=64))
        self.assertEqual(model.count_parameters(), legacy.count_parameters())
        self.assertEqual(model.count_parameters(), self.cfg.count_params())
        first, second = model.graph.bricks['block']
        self.assertNotEqual(first.bricks['attn'].q.weight.data_ptr(), second.bricks['attn'].q.weight.data_ptr())
        changed = deepcopy(self.graph)
        changed['nodes'][2]['params']['count'] = 3
        self.assertGreater(self.model(changed).count_parameters(), model.count_parameters())

    def test_rewiring_normalization_changes_computation_without_changing_weights(self):
        graph = deepcopy(self.graph)
        body = graph['nodes'][2]['body']
        for edge in body['edges']:
            if edge['target'] == 'norm1': edge['source'] = 'attn'
            elif edge['target'] == 'attn': edge['source'] = 'entry'
            elif edge['target'] == 'add1' and edge['targetHandle'] == 'a': edge['source'] = 'norm1'
        a, b = self.model(), self.model(graph)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
        x = torch.randint(16, (1, 12), device=self.device)
        with torch.no_grad():
            self.assertGreater((a(x)[0] - b(x)[0]).abs().max().item(), 1e-4)

    def test_removing_residual_addition_changes_the_model(self):
        graph = deepcopy(self.graph)
        body = graph['nodes'][2]['body']
        body['nodes'] = [n for n in body['nodes'] if n['id'] != 'add1']
        body['edges'] = [e for e in body['edges'] if e['target'] != 'add1']
        for edge in body['edges']:
            if edge['source'] == 'add1': edge['source'] = 'attn'
        x = torch.randint(16, (1, 12), device=self.device)
        with torch.no_grad():
            self.assertGreater((self.model()(x)[0] - self.model(graph)(x)[0]).abs().max().item(), 1e-4)

    def test_positions_do_not_change_execution_and_checkpoint_cache_matches_full_forward(self):
        moved = deepcopy(self.graph)
        moved['nodes'][0]['position'] = {'x': 900, 'y': -200}
        model, other = self.model(), self.model(moved)
        x = torch.randint(16, (1, 12), device=self.device)
        with torch.no_grad():
            full = model(x)[0]
            torch.testing.assert_close(full, other(x)[0], rtol=0, atol=0)
            _, _, cache = model(x[:, :8], use_cache=True)
            tail, _, _ = model(x[:, 8:], use_cache=True, past_key_values=cache)
            torch.testing.assert_close(tail, full[:, 8:], rtol=0, atol=0)

    def test_invalid_connections_are_explained_at_the_affected_brick(self):
        cases = []
        graph = deepcopy(self.graph); graph['nodes'][2]['body']['nodes'][2]['params']['width'] = 64
        cases.append(graph)
        graph = deepcopy(self.graph); graph['nodes'].append({'id': 'unused', 'kind': 'rmsnorm'})
        cases.append(graph)
        graph = deepcopy(self.graph); graph['edges'].append({'id': 'duplicate', 'source': 'tokens', 'target': 'head', 'targetHandle': 'in'})
        cases.append(graph)
        graph = deepcopy(self.graph); graph['nodes'][2]['params']['ignored_magic'] = 4
        cases.append(graph)
        for graph in cases:
            with self.subTest(graph=graph['nodes'][-1]['id']), self.assertRaises(AssemblyError):
                validate_assembly(graph, 32, 16)
        with self.assertRaises(AssemblyError) as error:
            validate_assembly(cases[0], 32, 16)
        self.assertEqual(error.exception.node_id, 'attn')
        self.assertEqual(error.exception.path, ['block'])
        self.assertTrue(error.exception.hint)

    def test_embedding_decay_does_not_depend_on_the_visual_node_id(self):
        graph = deepcopy(self.graph)
        graph['nodes'][1]['id'] = 'arbitrary_id'
        for edge in graph['edges']:
            if edge['source'] == 'embedding': edge['source'] = 'arbitrary_id'
            if edge['target'] == 'embedding': edge['target'] = 'arbitrary_id'
        model = self.model(graph)
        decay, no_decay = param_groups_weight_decay(model, 0.1)
        embedding = model.graph.bricks['arbitrary_id'].weight
        self.assertTrue(any(p is embedding for p in no_decay['params']))
        self.assertFalse(any(p is embedding for p in decay['params']))
        from optimizers.optimizers import setup_optimizer
        renamed = deepcopy(self.graph)
        def rename(g):
            for n in g['nodes']:
                n['id'] = 'norm_' + n['id']
                if 'body' in n: rename(n['body'])
            for e in g['edges']:
                e['source'] = 'norm_' + e['source']; e['target'] = 'norm_' + e['target']
        rename(renamed)
        optimizers = setup_optimizer(self.model(renamed), TrainingConfig(optimizer='muon'))
        self.assertGreater(len(optimizers[0].param_groups[0]['params']), 0)

    def test_api_inspects_same_graph_and_rejects_invalid_training_before_starting(self):
        from api.assembly import inspect, AssemblyRequest
        from api.training import start_training, TrainingRequest, training_state
        from fastapi import HTTPException
        import asyncio
        payload = {'assembly': self.graph, 'd_model': 32, 'vocab_size': 16, 'max_seq_len': 64}
        report = inspect(AssemblyRequest(model_cfg=payload))
        self.assertTrue(report['valid'])
        self.assertEqual(report['parameters'], self.model().count_parameters())
        moved = deepcopy(payload); moved['assembly']['nodes'][0]['position']['x'] = 1234
        self.assertEqual(inspect(AssemblyRequest(model_cfg=moved))['fingerprint'], report['fingerprint'])
        wider = deepcopy(payload); wider['d_model'] = 64
        self.assertNotEqual(inspect(AssemblyRequest(model_cfg=wider))['fingerprint'], report['fingerprint'])
        self.assertEqual(report['views']['block']['shapes']['attn'], ['hidden', 32])
        bad = deepcopy(payload); bad['assembly']['edges'] = []
        self.assertFalse(inspect(AssemblyRequest(model_cfg=bad))['valid'])
        with self.assertRaises(HTTPException) as error:
            asyncio.run(start_training(TrainingRequest(model_cfg=bad, training_cfg={})))
        self.assertEqual(error.exception.status_code, 422)
        self.assertFalse(training_state['is_training'])

    def test_ffn_variants_match_their_definitions_and_parameter_counts(self):
        from model.assembly_runtime import AssemblyFFN
        import torch.nn.functional as F
        functions = {'swiglu': F.silu, 'geglu': F.gelu, 'reglu': F.relu, 'gelu': F.gelu,
                     'relu': F.relu, 'relu2': lambda x: F.relu(x).square(), 'silu': F.silu, 'leaky_relu': F.leaky_relu}
        for activation, fn in functions.items():
            for bias in [False, True]:
                with self.subTest(activation=activation, bias=bias):
                    m = AssemblyFFN(8, 12, activation, bias).to(self.device)
                    x = torch.randn(2, 5, 8, device=self.device, requires_grad=True)
                    gated = activation in {'swiglu', 'geglu', 'reglu'}
                    expected = m.down(fn(m.gate(x)) * m.up(x) if gated else fn(m.up(x)))
                    torch.testing.assert_close(m(x), expected)
                    m(x).square().mean().backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
                    count = (3 if gated else 2) * 8 * 12 + ((2 if gated else 1) * 12 + 8 if bias else 0)
                    self.assertEqual(sum(p.numel() for p in m.parameters()), count)

    def test_attention_variants_are_causal_document_isolated_and_locally_masked(self):
        from model.assembly_runtime import AssemblyAttention
        for mechanism in ['mha', 'mqa', 'gqa', 'mla']:
            for position in ['rope', 'alibi', 'none']:
                for window in [0, 3]:
                    with self.subTest(mechanism=mechanism, position=position, window=window):
                        kv = 1 if mechanism == 'mqa' else 2 if mechanism == 'gqa' else 4
                        m = AssemblyAttention(32, 4, kv, 32, 0.5,
                            {'mechanism': mechanism, 'position': position, 'window': window, 'gate': True}).to(self.device)
                        x = torch.randn(1, 8, 32, device=self.device)
                        output = m(x)
                        changed = x.clone(); changed[:, 5:] += 4
                        torch.testing.assert_close(output[:, :5], m(changed)[:, :5], rtol=0, atol=0)
                        doc = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], device=self.device)
                        changed = x.clone(); changed[:, :4] += 4
                        torch.testing.assert_close(m(x, doc)[:, 4:], m(changed, doc)[:, 4:], rtol=0, atol=0)
                        if window:
                            torch.testing.assert_close(output[:, -1], m(changed)[:, -1], rtol=0, atol=0)
                        output.square().mean().backward()
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_attention_position_selection_changes_the_computation(self):
        from model.assembly_runtime import AssemblyAttention
        torch.manual_seed(18)
        x = torch.randn(1, 12, 32, device=self.device)
        models = [AssemblyAttention(32, 4, 2, 32, 0.5, {'position': method}).to(self.device)
                  for method in ['none', 'rope', 'alibi']]
        for m in models[1:]: m.load_state_dict(models[0].state_dict())
        outputs = [m(x) for m in models]
        for i in range(3):
            for j in range(i): self.assertFalse(torch.allclose(outputs[i], outputs[j]))

    def test_absolute_positions_are_real_operations(self):
        from model.assembly_runtime import AssemblyPosition
        x = torch.zeros(1, 5, 7, device=self.device)
        sinusoidal = AssemblyPosition(7, 16, 'sinusoidal').to(self.device)
        expected = torch.tensor([0., 1., 0., 1., 0., 1., 0.], device=self.device)
        torch.testing.assert_close(sinusoidal(x)[0, 0], expected)
        self.assertFalse(torch.equal(sinusoidal(x)[:, 0], sinusoidal(x)[:, 1]))
        learned = AssemblyPosition(7, 16, 'learned').to(self.device)
        torch.testing.assert_close(learned(x)[0], learned.positions.weight[:5])
        learned(x).square().sum().backward()
        self.assertTrue(torch.isfinite(learned.positions.weight.grad).all())

    def test_incompatible_variant_options_are_rejected_instead_of_ignored(self):
        for params in [{'mechanism': 'mha', 'kv_heads': 2}, {'position': 'none', 'rope_fraction': 0.5},
                       {'mechanism': 'gqa', 'latent': 8}, {'position': 'learned'}, {'window': -1}, {'gate': 1}]:
            graph = deepcopy(self.graph)
            graph['nodes'][2]['body']['nodes'][2]['params'] = params
            with self.subTest(params=params), self.assertRaises(AssemblyError):
                validate_assembly(graph, 32, 16)

    def test_train_resume_and_export_standalone_assembly(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        from export import export_to_local
        trainer = importlib.import_module('training.train')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            words = ['[UNK]', '[EOS]', '[PAD]', 'a', 'little', 'rabbit', 'found', 'a', 'friend']
            words = list(dict.fromkeys(words))
            backend = Tokenizer(WordLevel({w:i for i,w in enumerate(words)}, unk_token='[UNK]'))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]', pad_token='[PAD]', model_input_names=['input_ids','attention_mask'])
            tokenizer.save_pretrained(root / 'tokenizer')
            from api.assembly import read_tokenizer, TokenizerRequest
            self.assertEqual(read_tokenizer(TokenizerRequest(name=str(root/'tokenizer')))['vocab_size'], len(tokenizer))
            for name in ['train', 'val']:
                (root / f'{name}.txt').write_text('a little rabbit found a friend\n' * 100)
            variant = deepcopy(self.graph)
            body = variant['nodes'][2]['body']
            body['nodes'][2]['params'] = {'mechanism': 'mla', 'heads': 4, 'latent': 8, 'position': 'alibi', 'window': 8, 'gate': True}
            body['nodes'][5]['params'] = {'hidden': 64, 'activation': 'gelu', 'bias': True}
            body['nodes'][1]['kind'] = 'layernorm'
            variant['nodes'].append({'id': 'absolute', 'kind': 'position', 'params': {'encoding': 'learned'}})
            variant['edges'][1]['source'] = 'absolute'
            variant['edges'].append({'id': 'positions', 'source': 'embedding', 'target': 'absolute', 'targetHandle': 'in'})
            cfg = ModelConfig(assembly=variant, d_model=32, vocab_size=len(tokenizer), max_seq_len=32, tokenizer_name=str(root / 'tokenizer'))
            training = TrainingConfig(max_steps=4, warmup_steps=0, batch_size=1, gradient_accumulation_steps=2, eval_every=2, eval_steps=1,
                datasets=[{'name':'text','data_files':str(root/'train.txt')}], validation_datasets=[{'name':'text','data_files':str(root/'val.txt')}])
            class Stop:
                def on_update(self, step):
                    if step == 2: trainer.request_stop()
            def run(name, callback=None, checkpoint=None):
                torch.manual_seed(42)
                with patch.object(trainer, 'load_tokenizer', return_value=tokenizer), patch.object(trainer.TrainingReport,'generate_pdf'), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    trainer.train_model(cfg, training, output_dir=str(root/name), checkpoint_path=checkpoint, callback=callback, device=self.device)
                return torch.load(root/name/'final_model.pt', map_location='cpu', weights_only=False)
            reference = run('full')
            run('stop', Stop())
            resumed = run('resumed', checkpoint=str(root/'stop/final_model.pt'))
            self.assertEqual(resumed['model_config'].assembly, variant)
            changed = deepcopy(cfg)
            changed.assembly['nodes'][2]['params']['count'] = 3
            with self.assertRaisesRegex(ValueError, 'assembly'):
                trainer.train_model(changed, training, output_dir=str(root/'changed'),
                    checkpoint_path=str(root/'stop/final_model.pt'), device=self.device)
            for key, value in reference['model_state_dict'].items():
                torch.testing.assert_close(value, resumed['model_state_dict'][key], rtol=0, atol=0)
            with contextlib.redirect_stdout(io.StringIO()):
                export_to_local(str(root/'resumed/final_model.pt'),str(root/'export'))
            model = build_model(resumed['model_config']).to(self.device).eval()
            model.load_state_dict(resumed['model_state_dict'])
            x = tokenizer('a little rabbit', return_tensors='pt')['input_ids'].to(self.device)
            with torch.no_grad(), trainer.precision_context(torch.device(self.device)):
                logits = model(x)[0].cpu()
            torch.save({'x': x.cpu(), 'logits': logits}, root/'reference.pt')
            code = '''
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
root, device = sys.argv[1:]
m=AutoModelForCausalLM.from_pretrained(root+'/export',trust_remote_code=True,local_files_only=True).to(device).eval()
t=AutoTokenizer.from_pretrained(root+'/export',local_files_only=True)
r=torch.load(root+'/reference.pt',weights_only=True)
with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device=='cuda'):
    torch.testing.assert_close(m(r['x'].to(device)).logits.cpu(),r['logits'],rtol=0,atol=0)
    result=m.generate(r['x'].to(device), max_new_tokens=4, do_sample=False, eos_token_id=t.eos_token_id,pad_token_id=t.pad_token_id)
assert result.shape[1] > r['x'].shape[1]
'''
            result = subprocess.run([sys.executable,'-I','-c',code,str(root),self.device],cwd=root,
                env={**os.environ,'HF_HOME':str(root/'hf-cache'),'HF_HUB_OFFLINE':'1'},capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == '__main__': unittest.main()
