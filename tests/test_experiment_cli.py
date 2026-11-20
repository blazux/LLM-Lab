"""CLI acceptance tests: real subprocesses, persisted state and standalone export."""
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from experiments.recipe import load_recipe, describe


class ExperimentCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.train = self.root / 'train.jsonl'
        self.val = self.root / 'validation.jsonl'
        for path, prefix, count in [(self.train, 'training', 32), (self.val, 'heldout', 12)]:
            path.write_text(''.join(json.dumps({'text': f'{prefix} story {i}: a little rabbit found a friend and they played together.'}) + '\n' for i in range(count)))
        self.recipe = self.root / 'recipe.toml'
        device = os.environ.get('LLMLAB_TEST_DEVICE', 'cpu')
        self.recipe.write_text(f'''version = 1
[experiment]
name = "CLI acceptance"
seed = 42
device = "{device}"
threads = 1
[data]
format = "jsonl"
train = "train.jsonl"
validation = "validation.jsonl"
[tokenizer]
type = "byte_bpe"
vocab_size = 320
[model]
d_model = 32
n_layers = 1
n_heads = 4
n_kv_heads = 2
d_ff = 64
max_seq_len = 32
dropout = 0.1
[training]
max_steps = 4
warmup_steps = 0
batch_size = 2
gradient_accumulation_steps = 2
eval_every = 2
eval_steps = 2
save_best_only = false
gradient_checkpointing = false
[evaluation]
prompts = ["story"]
max_new_tokens = 8
[export]
enabled = true
''')
        self.env = {**os.environ, 'HF_HOME': str(self.root / 'hf-cache'), 'HF_HUB_OFFLINE': '1',
                    'HF_DATASETS_CACHE': str(self.root / 'datasets-cache'),
                    'MPLCONFIGDIR': str(self.root / 'matplotlib-cache'), 'TOKENIZERS_PARALLELISM': 'false'}

    def command(self, *args, success=True):
        result = subprocess.run([sys.executable, str(ROOT / 'src/cli.py'), *map(str, args)],
                                cwd=self.root, env=self.env, capture_output=True, text=True, timeout=120)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_recipe_validation_rejects_typos_and_resolves_relative_files(self):
        recipe = load_recipe(self.recipe)
        self.assertEqual(recipe['data']['train'], str(self.train))
        self.assertEqual(describe(recipe)['target_tokens'], 4 * 2 * 2 * 31)
        result = self.command('validate', self.recipe, '--check-data')
        self.assertEqual(json.loads(result.stdout)['data']['train']['documents'], 32)
        self.recipe.write_text(self.recipe.read_text().replace('max_steps = 4', 'max_stpes = 4'))
        result = self.command('validate', self.recipe, success=False)
        self.assertIn('max_stpes', result.stderr)

    def test_run_stop_resume_export_generate_and_extend(self):
        import torch
        uninterrupted = self.root / 'continuous'
        stopped = self.root / 'stopped'
        run_result = self.command('run', self.recipe, '--output-dir', uninterrupted)
        self.assertIn(f'Device: {os.environ.get("LLMLAB_TEST_DEVICE", "cpu")}', run_result.stdout)
        self.command('run', self.recipe, '--output-dir', stopped, '--stop-after', '2')
        state = json.loads(self.command('status', stopped).stdout)
        self.assertEqual(state['phase'], 'stopped')
        self.assertEqual(state['step'], 2)
        self.assertFalse((stopped / 'export').exists())
        self.command('resume', stopped, '--stop-after', '2', success=False)
        self.assertEqual(json.loads(self.command('status', stopped).stdout)['phase'], 'stopped')
        self.train.unlink()  # Resume uses the saved dataset, not the original recipe sources.
        self.command('resume', stopped)
        state = json.loads(self.command('status', stopped).stdout)
        self.assertEqual(state['phase'], 'completed')
        self.assertTrue(state['export']['verified'])
        self.assertEqual(state['tokens_seen'], 4 * 2 * 2 * 31)
        a = torch.load(uninterrupted / 'checkpoints/final_model.pt', map_location='cpu', weights_only=False)
        b = torch.load(stopped / 'checkpoints/final_model.pt', map_location='cpu', weights_only=False)
        self.assertEqual(a['scheduler_states'], b['scheduler_states'])
        for name, value in a['model_state_dict'].items():
            torch.testing.assert_close(value, b['model_state_dict'][name], rtol=0, atol=0)
        self.command('generate', stopped, '--prompt', 'story', '--max-tokens', '4')
        self.command('export', stopped, '--output-dir', self.root / 'manual-export')
        self.assertTrue(json.loads((self.root / 'manual-export/verification.json').read_text())['verified'])
        self.command('resume', stopped, success=False)
        self.command('resume', stopped, '--additional-steps', '2')
        state = json.loads(self.command('status', stopped).stdout)
        self.assertEqual(state['step'], 6)
        self.assertTrue((stopped / 'export/verification.json').is_file())
        self.assertTrue((stopped / 'export-step-6/verification.json').is_file())
        self.command('run', self.recipe, '--output-dir', stopped, success=False)

    def test_changed_saved_data_blocks_resume(self):
        stopped = self.root / 'changed'
        self.command('run', self.recipe, '--output-dir', stopped, '--stop-after', '1')
        (stopped / 'data/train.jsonl').write_text('{}\n')
        result = self.command('resume', stopped, success=False)
        self.assertIn('snapshot changed', result.stderr.lower())
        self.assertEqual(json.loads(self.command('status', stopped).stdout)['phase'], 'stopped')

    def test_unsupported_version_and_same_validation_file_are_rejected(self):
        initial = self.recipe.read_text()
        self.recipe.write_text(self.recipe.read_text().replace('version = 1', 'version = 99'))
        self.command('validate', self.recipe, success=False)
        self.recipe.write_text(initial.replace('validation = "validation.jsonl"', 'validation = "train.jsonl"'))
        self.command('validate', self.recipe, success=False)

    def test_sigterm_finishes_an_update_and_saves_a_resumable_run(self):
        self.recipe.write_text(self.recipe.read_text().replace('max_steps = 4', 'max_steps = 5000'))
        output = self.root / 'signalled'
        with (self.root / 'signal.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(ROOT / 'src/cli.py'), 'run',
                str(self.recipe), '--output-dir', str(output)], cwd=self.root, env=self.env,
                stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline and process.poll() is None:
                    state_path = output / 'state.json'
                    if state_path.exists():
                        state = json.loads(state_path.read_text())
                        if state['phase'] == 'training' and state.get('step', 0) >= 1:
                            process.send_signal(signal.SIGTERM)
                            break
                    time.sleep(0.05)
                self.assertEqual(process.wait(timeout=30), 0, (self.root / 'signal.log').read_text())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        state = json.loads(self.command('status', output).stdout)
        self.assertEqual(state['phase'], 'stopped')
        self.assertGreaterEqual(state['step'], 1)
        self.assertLess(state['step'], 5000)
        self.assertTrue((output / 'checkpoints/final_model.pt').is_file())

    def test_wrong_numeric_types_and_unsupported_exports_fail_validation(self):
        initial = self.recipe.read_text()
        for old, new in [('max_steps = 4', 'max_steps = 4.5'), ('dropout = 0.1', 'dropout = inf'),
                         ('enabled = true', 'enabled = "true"'), ('version = 1', 'version = true')]:
            self.recipe.write_text(initial.replace(old, new))
            self.command('validate', self.recipe, success=False)
        self.recipe.write_text(initial.replace('[model]', '[model]\npositional_encoding = "learned"'))
        self.command('validate', self.recipe, success=False)


if __name__ == '__main__':
    unittest.main()
