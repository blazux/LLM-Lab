"""Guided experiments use real offline data, measurements and durable run identity."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'gui/backend')]


class LearningExperimentTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from main import app
        from api import learning, training
        from run_store import RunStore
        self.learning, self.training = learning, training
        self.temp = tempfile.TemporaryDirectory(prefix='llmlab-learning-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = RunStore(self.root/'registry')
        for module in [learning, training]:
            self.enterContext(patch.object(module, 'get_run_store', lambda: self.store))
        self.enterContext(patch.object(training, 'training_state', training.PersistentState(training.idle_state())))
        self.enterContext(patch.object(learning, '_active_run', None))
        self.enterContext(patch.object(learning, '_process', None))
        self.enterContext(patch.object(learning, '_stop_requested', False))
        self.client = self.enterContext(TestClient(app))

    def test_descriptor_offline_corpus_and_safe_launch_validation(self):
        from experiments.recipe import load_recipe
        recipe = load_recipe(ROOT/'recipes/small-model-learning.toml')
        train = Path(recipe['data']['train']).read_text().splitlines()
        val = Path(recipe['data']['validation']).read_text().splitlines()
        self.assertEqual((len(train), len(val)), (320, 64))
        self.assertFalse(set(train) & set(val))
        descriptor = self.client.get('/api/learning/experiments/small-model-learning').json()
        self.assertEqual(descriptor['default_steps'], 120)
        self.assertEqual(descriptor['available_devices'][0], 'cpu')
        for payload in [{'steps': 19}, {'steps': 301}, {'seed': -1}, {'seed': True}, {'output_dir': '/tmp/unsafe'}]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post('/api/learning/experiments/small-model-learning/runs', json=payload).status_code, 422)
        self.assertEqual(self.client.get('/api/learning/runs/missing').status_code, 404)
        self.training.training_state['is_training'] = True
        self.assertEqual(self.client.post('/api/learning/experiments/small-model-learning/runs', json={}).status_code, 409)
        self.training.training_state['is_training'] = False
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(self.client.post('/api/learning/experiments/small-model-learning/runs', json={'device': 'cuda'}).status_code, 422)

    def test_real_recipe_cpu_measurements_survive_registry_reload(self):
        # Four updates exercise the complete backend while keeping tests bounded.
        source = (ROOT/'recipes/small-model-learning.toml').read_text()
        source = source.replace('max_steps = 120', 'max_steps = 4').replace('warmup_steps = 5', 'warmup_steps = 0')
        for split in ['train', 'validation']:
            source = source.replace(f'"learning-data/{split}.txt"', json.dumps(str(ROOT/f'recipes/learning-data/{split}.txt')))
        recipe = self.root/'short.toml'
        recipe.write_text(source)
        out = self.root/'experiment'
        result = subprocess.run([sys.executable, str(ROOT/'src/cli.py'), 'run', str(recipe), '--output-dir', str(out)],
            cwd=ROOT, env={**os.environ, 'HF_HUB_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1'},
            text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout[-4000:]+result.stderr[-4000:])
        measured = json.loads((out/'results.json').read_text())
        self.assertEqual(measured['step'], 4)
        self.assertGreater(measured['tokens_seen'], 0)
        before, after = measured['before'], measured['after']
        self.assertEqual([x['prompt'] for x in before['samples']], [x['prompt'] for x in after['samples']])
        for values in [before['metrics'], after['metrics']]:
            self.assertTrue(math.isfinite(values['val_language_loss']))
            self.assertAlmostEqual(math.exp(values['val_language_loss']), values['val_perplexity'])
        metrics = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
        self.assertTrue(any('language_loss' in row and 'tokens_per_second' in row for row in metrics))
        self.assertTrue(any('eval_language_loss' in row for row in metrics))
        record = self.store.create({**self.training.idle_state(), 'experiment_id': self.learning.EXPERIMENT_ID,
            'training_type':'learning', 'status':'completed', 'configuration':{'device':'cpu'}, 'max_steps':4}, 'learning:'+self.learning.EXPERIMENT_ID)
        destination = Path(record['output_dir'])/'experiment'
        out.rename(destination)
        snapshot = self.client.get('/api/learning/runs/'+record['run_id']).json()
        self.assertEqual(snapshot['results'], measured)
        self.assertEqual(snapshot['current_step'], 4)
        self.assertEqual(snapshot, self.client.get('/api/learning/runs/'+record['run_id']).json())
        self.assertEqual(len(self.client.get('/api/learning/runs').json()['runs']), 1)
        checkpoint = (destination/'checkpoints/final_model.pt').resolve()
        self.assertIn(checkpoint, self.store.checkpoint_files(record['run_id']))
        other = self.store.create({**self.training.idle_state(), 'status':'completed'})
        self.assertNotIn(checkpoint, self.store.checkpoint_files(other['run_id']))

    def test_failure_and_early_stop_keep_saved_status(self):
        def wait():
            self.training.training_thread.join(timeout=15)
            self.assertFalse(self.training.training_thread.is_alive())
        with patch.object(self.learning.subprocess, 'Popen', side_effect=OSError('fixture process failure')):
            response = self.client.post('/api/learning/experiments/small-model-learning/runs', json={'steps':20,'device':'cpu'})
            self.assertEqual(response.status_code, 200, response.text)
            wait()
        run_id = response.json()['run_id']
        saved = self.client.get('/api/learning/runs/'+run_id).json()
        self.assertEqual(saved['status'], 'error')
        self.assertIn('fixture process failure', saved['error'])
        self.assertFalse(saved['is_training'])
        self.assertEqual(self.client.post('/api/learning/runs/'+run_id+'/stop').status_code, 409)

    def test_setup_failures_do_not_reserve_training_or_leave_active_records(self):
        with patch.object(Path, 'write_text', side_effect=PermissionError('fixture readonly directory')):
            response = self.client.post('/api/learning/experiments/small-model-learning/runs', json={'device':'cpu'})
        self.assertEqual(response.status_code, 500)
        self.assertFalse(self.training.training_state.get('is_training'))
        self.assertFalse(self.store.list()[0]['is_training'])
        self.assertEqual(self.store.list()[0]['status'], 'error')
        with patch.object(self.learning.threading.Thread, 'start', side_effect=RuntimeError('fixture thread failure')):
            response = self.client.post('/api/learning/experiments/small-model-learning/runs', json={'device':'cpu'})
        self.assertEqual(response.status_code, 500)
        self.assertFalse(self.training.training_state.get('is_training'))
        self.assertEqual(self.store.list()[0]['status'], 'error')

    def test_recovered_run_uses_terminal_status_instead_of_stale_recipe_phase(self):
        record = self.store.create({**self.training.idle_state(), 'experiment_id':self.learning.EXPERIMENT_ID,
            'is_training':True, 'status':'training', 'phase':'training', 'configuration':{'device':'cpu'}},
            'learning:'+self.learning.EXPERIMENT_ID)
        root = Path(record['output_dir'])/'experiment'
        root.mkdir()
        (root/'state.json').write_text(json.dumps({'phase':'training','step':3}))
        self.store.recover()
        response = self.client.get('/api/learning/runs/'+record['run_id'])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['status'], 'interrupted')
        self.assertEqual(response.json()['phase'], 'interrupted')
        self.assertFalse(response.json()['is_training'])
        for status in ['error','failed','stopped','completed']:
            self.store.update(record['run_id'], {'status':status})
            with self.subTest(status=status):
                self.assertEqual(self.client.get('/api/learning/runs/'+record['run_id']).json()['phase'], status)

    def test_residual_variants_change_computation_not_learned_initial_values(self):
        import torch
        from config import ModelConfig
        from model.factory import build_model
        from experiments.runner import initial_weights_fingerprint
        models = []
        for variant in ['with-residuals', 'without-residuals']:
            recipe, _ = self.learning.recipe_for('residual-connections', variant)
            torch.manual_seed(42)
            models.append(build_model(ModelConfig(**recipe['model'], vocab_size=512)).eval())
        a, b = models
        self.assertEqual(initial_weights_fingerprint(a), initial_weights_fingerprint(b))
        self.assertEqual(a.count_parameters(), b.count_parameters())
        self.assertEqual(set(dict(a.named_parameters())), set(dict(b.named_parameters())))
        for name, parameter in a.named_parameters():
            torch.testing.assert_close(parameter, dict(b.named_parameters())[name], rtol=0, atol=0)
        ids = torch.tensor([[4, 7, 3, 9]])
        with torch.no_grad():
            self.assertFalse(torch.allclose(a(ids)[0], b(ids)[0]))

    def test_paired_real_runs_comparison_integrity_filters_and_mismatches(self):
        from experiments.recipe import dumps_recipe, load_recipe
        records = []
        for variant in ['with-residuals', 'without-residuals']:
            recipe, _ = self.learning.recipe_for('residual-connections', variant)
            recipe['training'].update(max_steps=4, warmup_steps=0)
            recipe_path = self.root/(variant+'.toml')
            recipe_path.write_text(dumps_recipe(recipe))
            self.assertEqual(load_recipe(recipe_path)['model']['assembly'], recipe['model']['assembly'])
            record = self.store.create({**self.training.idle_state(), 'experiment_id':'residual-connections',
                'training_type':'learning', 'status':'completed', 'metric_definition':self.learning.LANGUAGE_METRICS,
                'configuration':dict(variant=variant, steps=4, seed=42, device='cpu', model=recipe['model'],
                    training=recipe['training'], hardware={'device':'cpu','dtype':'float32'}), 'max_steps':4},
                'learning:residual-connections')
            out = Path(record['output_dir'])/'experiment'
            result = subprocess.run([sys.executable, str(ROOT/'src/cli.py'), 'run', str(recipe_path), '--output-dir', str(out)],
                cwd=ROOT, env={**os.environ, 'HF_HUB_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1'},
                text=True, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout[-4000:]+result.stderr[-4000:])
            metrics = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
            gradients = [row['gradient_norm'] for row in metrics if 'gradient_norm' in row]
            self.assertTrue(gradients)
            self.assertTrue(all(math.isfinite(value) and value >= 0 for value in gradients))
            records.append(record)
        a, b = records
        compare = lambda: self.client.get('/api/learning/comparisons', params={'run_a':a['run_id'],'run_b':b['run_id']})
        response = compare()
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result['comparability']['comparable'], result['comparability'])
        checks = {row['id']:row['match'] for row in result['comparability']['checks']}
        self.assertTrue(checks['initial_weights'])
        self.assertTrue(checks['integrity_a'])
        self.assertEqual(result['run_a']['before']['initial_weights_sha256'], result['run_b']['before']['initial_weights_sha256'])
        self.assertEqual({row['field'] for row in result['configuration_differences']}, {'variant','model.assembly_sha256'})
        values = result['metrics']['validation_cross_entropy']
        self.assertAlmostEqual(values['delta_b_minus_a'], values['b']-values['a'])
        self.assertEqual(len(self.client.get('/api/learning/runs', params={'experiment_id':'residual-connections'}).json()['runs']),2)
        self.assertEqual(self.client.get('/api/learning/runs').json()['runs'],[])
        changed = {**b['configuration'], 'seed':43}
        self.store.update(b['run_id'], {'configuration':changed})
        self.assertFalse(compare().json()['comparability']['comparable'])
        self.store.update(b['run_id'], {'configuration':b['configuration']})
        before_path = Path(b['output_dir'])/'experiment/before.json'
        before = json.loads(before_path.read_text())
        before['fingerprints']['data']['validation_sha256'] = 'different-data'
        before_path.write_text(json.dumps(before))
        failed = compare().json()['comparability']
        self.assertFalse(failed['comparable'])
        self.assertTrue(any(row['id']=='data' and row['match'] is False for row in failed['checks']))

    def test_parameterized_experiment_endpoints_reject_unknown_variants_and_foreign_runs(self):
        catalog = self.client.get('/api/learning/experiments').json()['experiments']
        self.assertEqual({row['id'] for row in catalog}, {'small-model-learning','residual-connections'})
        descriptor = self.client.get('/api/learning/experiments/residual-connections').json()
        self.assertEqual(descriptor['default_variant'], 'with-residuals')
        self.assertEqual(self.client.get('/api/learning/experiments/unknown').status_code,404)
        self.assertEqual(self.client.get('/api/learning/runs',params={'experiment_id':'unknown'}).status_code,404)
        for id,variant in [('residual-connections','unknown'),('small-model-learning','with-residuals')]:
            self.assertEqual(self.client.post(f'/api/learning/experiments/{id}/runs',json={'variant':variant}).status_code,422)
        foreign = self.store.create({**self.training.idle_state(), 'status':'completed'})
        params = {'run_a':foreign['run_id'],'run_b':'0'*32}
        self.assertEqual(self.client.get('/api/learning/comparisons',params=params).status_code,404)
        self.assertEqual(self.client.get('/api/learning/comparisons',params={'run_a':foreign['run_id'],'run_b':foreign['run_id']}).status_code,422)

    def test_tokenizer_fingerprint_covers_special_tokens_and_slow_tokenizer_artifacts(self):
        from experiments.runner import tokenizer_fingerprint
        root = self.root/'slow-tokenizer'
        root.mkdir()
        (root/'tokenizer.model').write_bytes(b'local-sentencepiece-fixture')
        config = root/'tokenizer_config.json'
        config.write_text(json.dumps({'eos_token':'[EOS]','name_or_path':'/first/run'}))
        first = tokenizer_fingerprint(root)
        config.write_text(json.dumps({'name_or_path':'/second/run','eos_token':'[EOS]'}))
        self.assertEqual(first,tokenizer_fingerprint(root))
        config.write_text(json.dumps({'eos_token':'[STOP]'}))
        self.assertNotEqual(first,tokenizer_fingerprint(root))


if __name__ == '__main__':
    unittest.main()
