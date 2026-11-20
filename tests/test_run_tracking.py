"""Durable run identity, non-consuming replay, restart recovery and metric semantics."""
import asyncio
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'gui/backend')]
from run_store import RunStore, LANGUAGE_METRICS, LEGACY_METRICS


class RunStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = RunStore(self.tmp.name)

    def create(self):
        return self.store.create(dict(output_dir=None, is_training=True, status='training',
            current_step=5, model_config={'d_model': 16}, training_config={'max_steps': 10}), 'project-a')

    def test_events_survive_restart_and_independent_consumers_do_not_steal_them(self):
        run = self.create()
        for step in [1, 2]:
            self.store.append(run['run_id'], {'type': 'metrics', 'step': step, 'loss': 2.0})
        first = self.store.events(run['run_id'])
        self.assertEqual(first, self.store.events(run['run_id']))
        reopened = RunStore(self.tmp.name)
        self.assertEqual(reopened.events(run['run_id'], first[0]['event_id']), first[1:])
        reopened.recover()
        state = reopened.get(run['run_id'])
        self.assertEqual(state['status'], 'interrupted')
        self.assertFalse(state['is_training'])
        self.assertEqual(state['current_step'], 5)
        self.assertEqual(state['model_config'], {'d_model': 16})

    def test_checkpoint_discovery_uses_registered_custom_folders(self):
        run = self.create()
        checkpoint = Path(run['output_dir']) / 'checkpoints' / 'final_model.pt'
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b'fixture')
        self.assertIn(checkpoint, self.store.checkpoint_files())
        self.assertEqual(self.store.list('project-b'), [])
        self.assertEqual(len(self.store.list('project-a')), 1)

    def test_events_are_scoped_and_nonfinite_values_are_not_serialized_as_json_numbers(self):
        a, b = self.create(), self.create()
        self.store.append(a['run_id'], {'loss': float('nan')})
        event = self.store.append(b['run_id'], {'loss': 3})
        self.assertIsNone(self.store.events(a['run_id'])[0]['loss'])
        self.assertEqual(self.store.events(b['run_id'])[0]['event_id'], event['event_id'])
        self.store.update(b['run_id'], {'is_training': False, 'status': 'completed'})
        self.store.recover()
        self.assertEqual(self.store.get(b['run_id'])['status'], 'completed')


class LanguageMetricsTests(unittest.TestCase):
    def test_perplexity_uses_token_weighted_cross_entropy_without_auxiliary_or_z_loss(self):
        import torch
        from config import TrainingConfig
        from training.train import evaluate_model, language_perplexity
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.))
            def forward(self, x, **kwargs):
                logits = torch.stack([x.float() * 2 + self.weight, torch.zeros_like(x).float()], -1)
                return logits, self.weight * 0 + 7
        model = Model().train()
        batches = [(torch.tensor([[0, 1]]), torch.tensor([[0, 0]])),
                   (torch.tensor([[0]]), torch.tensor([[0]]))]
        expected = (2 * math.log(2) + math.log1p(math.exp(-2))) / 3
        result = evaluate_model(model, batches, TrainingConfig(eval_steps=2, z_loss_weight=.2), 2)
        self.assertAlmostEqual(result['val_language_loss'], expected, places=6)
        self.assertAlmostEqual(result['val_perplexity'], math.exp(expected), places=6)
        self.assertGreater(result['val_auxiliary_loss'], 7)
        self.assertAlmostEqual(result['val_objective_loss'], expected + result['val_auxiliary_loss'])
        self.assertTrue(model.training)
        self.assertEqual(result['metric_definition'], LANGUAGE_METRICS)
        self.assertGreater(language_perplexity(21), math.exp(20))
        self.assertIsNone(language_perplexity(1000))
        self.assertIsNone(language_perplexity(float('nan')))


class RunApiFixture(unittest.TestCase):
    def setUp(self):
        import torch
        from fastapi.testclient import TestClient
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        import api.training as api
        import api.inference as inference
        import api.export as export
        from main import app
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = RunStore(self.root / 'registry')
        self.api = api
        for module in [api, inference, export]:
            self.enterContext(patch.object(module, 'get_run_store', lambda: self.store))
        self.enterContext(patch.object(api, 'training_state', api.PersistentState(api.idle_state())))
        self.enterContext(patch('training.report.TrainingReport.generate_pdf', return_value='omitted-in-test'))
        torch.set_num_threads(1)
        words = ['[UNK]', '[EOS]', '[PAD]', 'a', 'little', 'model', 'learns', 'text']
        backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token='[UNK]'))
        backend.pre_tokenizer = Whitespace()
        self.tokenizer = self.root / 'tokenizer'
        PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='[EOS]', pad_token='[PAD]', unk_token='[UNK]').save_pretrained(self.tokenizer)
        for name in ['train', 'validation']:
            (self.root / f'{name}.txt').write_text('a little model learns text\n' * 100)
        self.client = self.enterContext(TestClient(app))
        self.payload = dict(project_id='project-a', output_dir='/app/data', model_cfg=dict(
            tokenizer_name=str(self.tokenizer), d_model=16, n_layers=1, n_heads=2, n_kv_heads=1,
            d_ff=32, max_seq_len=16, vocab_size=8), training_cfg=dict(max_steps=4,
            batch_size=1, gradient_accumulation_steps=2, warmup_steps=0, eval_every=2, eval_steps=1,
            gradient_checkpointing=False, z_loss_weight=.1, save_best_only=False,
            datasets=[dict(name='text', data_files=str(self.root / 'train.txt'))],
            validation_datasets=[dict(name='text', data_files=str(self.root / 'validation.txt'))]))

    def wait_for_run(self):
        deadline = time.monotonic() + 30
        while self.api.training_thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertFalse(self.api.training_thread.is_alive())
        self.assertNotEqual(self.api.training_state['status'], 'error', self.api.training_state.get('error'))


class RunApiTests(RunApiFixture):
    def test_real_run_resume_restore_and_tools_use_saved_identity_configuration_and_events(self):
        failed = self.store.create({**self.api.idle_state(), 'status': 'error'})
        blocked = self.client.post('/api/training/start', json={**self.payload, 'output_dir': failed['output_dir']})
        self.assertEqual(blocked.status_code, 409)
        response = self.client.post('/api/training/start', json=self.payload)
        self.assertEqual(response.status_code, 200, response.text)
        run_id = response.json()['run_id']
        self.wait_for_run()
        state = self.client.get(f'/api/training/runs/{run_id}').json()
        self.assertEqual(state['current_step'], 4)
        self.assertEqual(state['project_id'], 'project-a')
        self.assertEqual(state['metric_definition'], LANGUAGE_METRICS)
        self.assertTrue(Path(state['output_dir'], 'final_model.pt').is_file())
        events = self.client.get(f'/api/training/runs/{run_id}/events').json()['events']
        metrics = next(event for event in events if event['type'] == 'metrics')
        self.assertGreater(metrics['loss'], metrics['language_loss'])
        self.assertAlmostEqual(metrics['perplexity'], math.exp(metrics['language_loss']))
        self.assertEqual(events, self.client.get(f'/api/training/runs/{run_id}/events').json()['events'])
        for endpoint in ['/api/inference/checkpoints', '/api/export/checkpoints']:
            paths = [row['path'] for row in self.client.get(endpoint).json()['checkpoints']]
            self.assertIn(str(Path(state['output_dir'], 'final_model.pt')), paths)
        response = self.client.post(f'/api/training/runs/{run_id}/resume', json={'additional_steps': 2})
        self.assertEqual(response.status_code, 200, response.text)
        resumed_id = response.json()['run_id']
        self.assertNotEqual(resumed_id, run_id)
        self.wait_for_run()
        resumed = self.client.get(f'/api/training/runs/{resumed_id}').json()
        self.assertEqual(resumed['current_step'], 6)
        self.assertEqual(resumed['max_steps'], 6)
        self.assertEqual(resumed['training_config']['max_steps'], 6)
        for endpoint in ['/api/inference/checkpoints', '/api/export/checkpoints']:
            scoped = self.client.get(endpoint, params={'run_id': resumed_id})
            self.assertEqual(scoped.status_code, 200, scoped.text)
            paths = [row['path'] for row in scoped.json()['checkpoints']]
            self.assertIn(str(Path(resumed['output_dir'], 'final_model.pt')), paths)
            self.assertNotIn(str(Path(state['output_dir'], 'final_model.pt')), paths)
            self.assertEqual(self.client.get(endpoint, params={'run_id': 'missing'}).status_code, 404)
        self.assertNotEqual(resumed['output_dir'], state['output_dir'])
        self.api.training_state.clear()
        dict.update(self.api.training_state, self.api.idle_state())
        self.store = RunStore(self.root / 'registry')
        restored = self.client.get('/api/training/status').json()
        self.assertEqual(restored['run_id'], resumed_id)
        self.assertEqual(restored['current_step'], 6)
        old = self.client.get(f'/api/training/runs/{run_id}/events').json()['events']
        self.assertEqual(events, old)
        self.assertEqual(self.client.get('/api/training/runs/unknown').status_code, 404)
        self.assertEqual(self.client.post('/api/training/stop?run_id=unknown').status_code, 409)

    def test_sse_replays_independently_and_resumes_after_last_event_id(self):
        run = self.store.create({**self.api.idle_state(), 'status': 'completed'})
        run_id = run['run_id']
        first = self.store.append(run_id, {'type': 'metrics', 'step': 1, 'loss': 2})
        second = self.store.append(run_id, {'type': 'metrics', 'step': 2, 'loss': 1})
        async def read(cursor, count):
            response = await self.api.stream_metrics(run_id=run_id, after=0, last_event_id=cursor)
            stream = response.body_iterator
            events = []
            try:
                async for event in stream:
                    if event['event'] == 'message':
                        events.append(json.loads(event['data']))
                        if len(events) == count:
                            break
            finally:
                await stream.aclose()
            return events
        all_events = asyncio.run(read(None, 2))
        self.assertEqual(all_events, [first, second])
        self.assertEqual(asyncio.run(read(None, 2)), all_events)
        self.assertEqual(asyncio.run(read(str(first['event_id']), 1)), [second])

    def test_legacy_import_keeps_old_metric_definition_and_does_not_invent_history(self):
        import torch
        response = self.client.post('/api/training/start', json=self.payload)
        self.wait_for_run()
        run = self.store.get(response.json()['run_id'])
        saved = torch.load(Path(run['output_dir']) / 'final_model.pt', weights_only=False)
        saved.pop('metric_definition')
        saved['best_val_loss'] = -100  # Old objective best must not compete with new language CE.
        legacy = self.root / 'legacy-output'
        legacy.mkdir()
        torch.save(saved, legacy / 'final_model.pt')
        imported = self.client.post('/api/training/runs/import', json={'output_dir': str(legacy), 'project_id': 'project-a'})
        self.assertEqual(imported.status_code, 200, imported.text)
        run_id = imported.json()['run_id']
        self.assertEqual(self.store.get(run_id)['metric_definition'], LEGACY_METRICS)
        events = self.store.events(run_id)
        self.assertEqual(sum(row['type'] == 'eval_metrics' for row in events), 1)
        self.assertFalse(any(row['type'] == 'metrics' for row in events))
        resumed = self.client.post(f'/api/training/runs/{run_id}/resume', json={'additional_steps': 2})
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.wait_for_run()
        new = self.store.get(resumed.json()['run_id'])
        self.assertEqual(new['metric_definition'], LANGUAGE_METRICS)
        checkpoint = torch.load(Path(new['output_dir']) / 'final_model.pt', weights_only=False)
        self.assertGreater(checkpoint['best_val_loss'], 0)
        self.assertTrue(Path(new['output_dir'], 'best_model.pt').is_file())
        self.assertEqual(self.store.get(run_id)['metric_definition'], LEGACY_METRICS)

    def test_stop_finishes_an_update_and_keeps_a_registered_resumable_checkpoint(self):
        original = self.api.MetricsCallback.on_update
        def stop_after_two(callback, step):
            original(callback, step)
            if step == 2:
                asyncio.run(self.api.stop_training(self.api.training_state['run_id']))
        with patch.object(self.api.MetricsCallback, 'on_update', stop_after_two):
            response = self.client.post('/api/training/start', json=self.payload)
            self.assertEqual(response.status_code, 200, response.text)
            self.wait_for_run()
        state = self.store.get(response.json()['run_id'])
        self.assertEqual(state['status'], 'stopped')
        self.assertFalse(state['is_training'])
        self.assertEqual(state['current_step'], 2)
        self.assertTrue(Path(state['output_dir'], 'final_model.pt').is_file())


if __name__ == '__main__':
    unittest.main()
