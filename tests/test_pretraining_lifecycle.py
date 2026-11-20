"""Offline regression tests for the minimal pretraining lifecycle.

Run: python -m unittest discover -s tests -p 'test_pretraining_lifecycle.py' -v
Set LLMLAB_TEST_DEVICE=cuda to require a real CUDA GPU.
"""
import contextlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from config import ModelConfig, TrainingConfig
from export import export_to_local
from inference.inference import load_model_for_inference, generate_text
from model.factory import build_model

trainer = importlib.import_module('training.train')


class PretrainingLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('LLMLAB_TEST_DEVICE', 'cpu')
        if self.device not in {'cpu', 'cuda'}:
            raise ValueError('LLMLAB_TEST_DEVICE must be cpu or cuda')
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA tests requested but CUDA is unavailable')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        torch.set_num_threads(1)
        vocabulary = ['[UNK]', '[EOS]', '[PAD]', 'bonjour', 'le', 'petit', 'modele', 'apprend']
        backend = Tokenizer(WordLevel(dict(zip(vocabulary, range(len(vocabulary)))), unk_token='[UNK]'))
        backend.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                                 eos_token='[EOS]', pad_token='[PAD]',
                                                 model_input_names=['input_ids', 'attention_mask'])
        self.tokenizer.add_tokens(['nouveau'])  # len(tokenizer) != tokenizer.vocab_size
        self.tokenizer_path = self.root / 'tokenizer'
        self.tokenizer.save_pretrained(self.tokenizer_path)
        self.corpus = self.root / 'corpus.txt'
        self.corpus.write_text(('bonjour le petit modele apprend nouveau\n') * 400)
        self.model_config = ModelConfig(tokenizer_name=str(self.tokenizer_path), d_model=16,
                                        n_layers=1, n_heads=2, n_kv_heads=1, d_ff=32,
                                        vocab_size=8, max_seq_len=16, dropout=0.1)
        self.config = TrainingConfig(max_steps=4, batch_size=1, gradient_accumulation_steps=2,
                                     scheduler='cosine', warmup_steps=0, eval_every=2, eval_steps=1,
                                     save_best_only=False,
                                     datasets=[{'name': 'text', 'data_files': str(self.corpus)}])

    def device_context(self):
        return patch('torch.cuda.is_available', return_value=False) if self.device == 'cpu' else contextlib.nullcontext()

    def run_training(self, name, checkpoint=None, callback=None, **kwargs):
        output = self.root / name
        # Real tokenizer, dataset, model, optimizer, scheduler and checkpoints.
        # Only isolate the legacy shared pickle cache and the optional PDF report.
        with patch.object(trainer, 'load_tokenizer', return_value=self.tokenizer), \
             patch.object(trainer.TrainingReport, 'generate_pdf'), \
             self.device_context(), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trainer.train_model(self.model_config, self.config, checkpoint_path=checkpoint,
                                output_dir=str(output), callback=callback, **kwargs)
        return torch.load(output / 'final_model.pt', map_location='cpu', weights_only=False)

    def test_stop_resume_matches_uninterrupted_training_and_exports(self):
        torch.manual_seed(42)
        continuous = self.run_training('continuous')
        self.assertTrue(all(t.dtype == torch.float32 for t in continuous['model_state_dict'].values()
                            if t.is_floating_point()))
        for state in continuous['optimizer_states'][0]['state'].values():
            self.assertEqual(state['exp_avg'].dtype, torch.float32)
            self.assertEqual(state['exp_avg_sq'].dtype, torch.float32)

        class StopAfterEvaluation:
            def on_eval(self, step, *_):
                if step == 2:
                    trainer.request_stop()

        torch.manual_seed(42)
        stopped = self.run_training('stopped', callback=StopAfterEvaluation())
        self.assertEqual(stopped['step'], 2)
        intermediate = torch.load(self.root / 'stopped/latest_checkpoint.pt', weights_only=False)
        self.assertEqual(intermediate['step'], 2)
        self.assertEqual(intermediate['total_tokens_seen'], 2 * 2 * 15)
        resumed = self.run_training('resumed', str(self.root / 'stopped/latest_checkpoint.pt'))
        self.assertEqual(resumed['step'], 4)
        self.assertEqual(resumed['total_tokens_seen'], 4 * 2 * 15)
        self.assertEqual(resumed['scheduler_states'], continuous['scheduler_states'])
        self.assertEqual(resumed['best_val_loss'], continuous['best_val_loss'])
        for key, tensor in continuous['model_state_dict'].items():
            torch.testing.assert_close(resumed['model_state_dict'][key], tensor, rtol=0, atol=0)
        self.assertEqual(resumed['optimizer_states'][0]['state'][0]['step'].item(), 4)
        saved = json.loads((self.root / 'resumed/model_config.json').read_text())
        self.assertEqual(saved['vocab_size'], len(self.tokenizer))

        # Verify inference through the application's actual loader/generator.
        with patch('inference.inference.load_tokenizer', return_value=self.tokenizer), \
             self.device_context(), contextlib.redirect_stdout(io.StringIO()):
            model, tokenizer, device = load_model_for_inference(str(self.root / 'resumed/final_model.pt'))
            text = generate_text(model, tokenizer, device, 'bonjour', max_tokens=3, strategy='greedy')
        self.assertIsInstance(text, str)
        self.assertEqual(next(model.parameters()).dtype, torch.float32)
        self.assertEqual(next(model.parameters()).device.type, self.device)

        export_dir = self.root / 'export'
        with contextlib.redirect_stdout(io.StringIO()):
            export_to_local(str(self.root / 'resumed/final_model.pt'), str(export_dir))
        self.assertTrue((export_dir / 'tokenizer.json').is_file())
        self.assertTrue((export_dir / 'README.md').is_file())
        x = tokenizer('bonjour le petit', return_tensors='pt')['input_ids']
        with torch.no_grad():
            reference = model(x.to(device))[0].cpu()
        torch.save({'input_ids': x, 'logits': reference}, self.root / 'reference.pt')
        # Separate process, no src on PYTHONPATH: the exported package must stand alone.
        code = '''
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True).eval()
tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
device = sys.argv[3]
model = model.to(device=device, dtype=torch.float32)
reference = torch.load(sys.argv[2], weights_only=True)
with torch.no_grad():
    result = model(reference['input_ids'].to(device)).logits.cpu()
torch.testing.assert_close(result, reference['logits'], rtol=1e-5, atol=1e-5)
result = model.generate(**tokenizer('bonjour', return_tensors='pt').to(device), max_new_tokens=3, do_sample=False)
assert result.shape[1] > 1
'''
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONPATH': '',
               'HF_HOME': str(self.root / 'hf-cache')}
        result = subprocess.run([sys.executable, '-c', code, str(export_dir), str(self.root / 'reference.pt'), self.device],
                                cwd=self.root, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_accumulated_batches_equal_a_larger_batch(self):
        self.model_config.dropout = 0
        self.config.scheduler = 'constant'
        torch.manual_seed(7)
        # float32 isolates accumulation correctness from bf16 reduction rounding.
        with patch('torch.cuda.is_bf16_supported', return_value=False):
            accumulated = self.run_training('accumulated')
        self.config.batch_size = 2
        self.config.gradient_accumulation_steps = 1
        torch.manual_seed(7)
        with patch('torch.cuda.is_bf16_supported', return_value=False):
            batched = self.run_training('batched')
        self.assertEqual(accumulated['total_tokens_seen'], batched['total_tokens_seen'])
        for key, tensor in accumulated['model_state_dict'].items():
            torch.testing.assert_close(tensor, batched['model_state_dict'][key], rtol=1e-4, atol=1e-6)

    def test_export_preserves_realistic_rotary_heads_under_autocast(self):
        # Tiny head dimensions missed CPU-vs-CUDA RoPE rounding differences.
        from transformers import AutoModelForCausalLM
        self.model_config.d_model = 128
        self.model_config.n_heads = 2
        self.model_config.n_kv_heads = 1
        self.model_config.n_layers = 2
        self.model_config.d_ff = 256
        self.model_config.max_seq_len = 64
        self.model_config.__post_init__()
        self.corpus.write_text('bonjour le petit modele apprend nouveau\n' * 1600)
        checkpoint = self.run_training('rotary')
        model = build_model(checkpoint['model_config']).to(self.device).eval()
        model.load_state_dict(checkpoint['model_state_dict'])
        exported = self.root / 'rotary-export'
        with contextlib.redirect_stdout(io.StringIO()):
            export_to_local(str(self.root / 'rotary/final_model.pt'), str(exported))
        restored = AutoModelForCausalLM.from_pretrained(
            exported, trust_remote_code=True, local_files_only=True).to(self.device).eval()
        x = torch.randint(0, len(self.tokenizer), (1, 48), device=self.device)
        with torch.no_grad(), trainer.precision_context(torch.device(self.device)):
            expected = model(x)[0]
            actual = restored(x).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_extension_uses_new_target_for_scheduler(self):
        original = self.run_training('first')
        extended = self.run_training('extended', str(self.root / 'first/final_model.pt'), additional_steps=3)
        self.assertEqual(extended['step'], 7)
        self.assertEqual(extended['schedule_end_step'], 7)
        self.assertEqual(extended['schedule_start_step'], 4)
        self.assertEqual(extended['total_tokens_seen'] - original['total_tokens_seen'], 3 * 2 * 15)
        self.assertEqual(extended['optimizer_states'][0]['param_groups'][0]['lr'], 0)

    def test_extension_can_expand_one_source_without_resetting_optimizer_or_cursor(self):
        self.config.datasets[0]['take'] = 300
        original = self.run_training('capped-first')
        self.config.datasets[0]['take'] = 400
        checkpoint = str(self.root / 'capped-first/final_model.pt')
        with self.assertRaisesRegex(ValueError, 'changed datasets'):
            self.run_training('not-an-extension', checkpoint)
        extended = self.run_training('larger-corpus', checkpoint, additional_steps=3)
        self.assertEqual(extended['step'], 7)
        self.assertEqual(extended['total_tokens_seen'], original['total_tokens_seen'] + 90)
        self.assertEqual(extended['data_tokens_seen'], original['data_tokens_seen'] + 90)
        self.assertEqual(extended['optimizer_states'][0]['state'][0]['step'].item(), 7)
        self.assertEqual(extended['train_config'].datasets[0]['take'], 400)
        self.assertEqual(extended['schedule_start_step'], 4)

    def test_extension_rejects_changed_source_order_or_smaller_corpus(self):
        self.config.datasets[0]['take'] = 300
        self.run_training('source-first')
        checkpoint = str(self.root / 'source-first/final_model.pt')
        base = dict(self.config.datasets[0])
        for changes in [dict(take=100), dict(take=400, skip=1), dict(take=400, revision='different')]:
            self.config.datasets = [{**base, **changes}]
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'changed datasets'):
                self.run_training('rejected-source', checkpoint, additional_steps=3)
        self.assertFalse(trainer.extends_single_source([base, base], [{**base, 'take': 400}, base]))

    def test_resume_at_end_of_finite_stream_restarts_data(self):
        # 7*400 tokens => 175 sequences, of which 165 are training sequences.
        self.config.max_steps = 84  # 168 batches: crosses the finite stream boundary.
        self.config.eval_every = 83
        self.model_config.dropout = 0.1
        class Stop:
            def on_eval(self, *_):
                trainer.request_stop()
        torch.manual_seed(9)
        reference = self.run_training('finite-reference')
        torch.manual_seed(9)
        self.run_training('finite-stop', callback=Stop())
        resumed = self.run_training('finite-resume', str(self.root / 'finite-stop/final_model.pt'))
        for key, tensor in reference['model_state_dict'].items():
            torch.testing.assert_close(resumed['model_state_dict'][key], tensor, rtol=0, atol=0)

    def test_empty_training_stream_fails_instead_of_looping(self):
        self.corpus.write_text('bonjour\n')
        with self.assertRaisesRegex(ValueError, 'Training stream is empty'):
            self.run_training('empty')

    def test_pretraining_with_packed_document_masks(self):
        self.config.use_doc_masks = True
        result = self.run_training('document-masks')
        self.assertEqual(result['step'], 4)
        self.assertTrue(math.isfinite(result['final_metrics']['val_loss']))

    def test_separate_validation_source_and_resume(self):
        # Fewer than 91 sequences: the legacy sequence holdout would be empty.
        validation = self.root / 'validation.txt'
        validation.write_text('petit modele apprend\n' * 16)
        self.config.validation_datasets = [{'name': 'text', 'data_files': str(validation)}]
        class Stop:
            def on_eval(self, *_):
                trainer.request_stop()
        torch.manual_seed(23)
        reference = self.run_training('separate-reference')
        torch.manual_seed(23)
        self.run_training('separate-stop', callback=Stop())
        resumed = self.run_training('separate-resume', str(self.root / 'separate-stop/final_model.pt'))
        for key, tensor in reference['model_state_dict'].items():
            torch.testing.assert_close(resumed['model_state_dict'][key], tensor, rtol=0, atol=0)

    def test_empty_validation_is_explained(self):
        model = build_model(self.model_config)
        with self.assertRaisesRegex(ValueError, 'Validation stream is empty'):
            trainer.evaluate_model(model, [], self.config, self.model_config.vocab_size)
        self.assertTrue(model.training)

    def test_cpu_uses_adamw_even_if_bitsandbytes_is_installed(self):
        optimizers = importlib.import_module('optimizers.optimizers')
        with patch.object(optimizers, 'HAS_BNB', True):
            optimizer, = optimizers.setup_optimizer(build_model(self.model_config), self.config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.config.use_8bit_optimizer = True
        with self.assertRaisesRegex(ValueError, 'bitsandbytes and CUDA'):
            optimizers.setup_optimizer(build_model(self.model_config), self.config)

    def test_changed_recipe_cannot_silently_resume(self):
        self.run_training('before-change')
        self.config.adamw_beta1 = 0.8
        with self.assertRaisesRegex(ValueError, 'changed adamw_beta1'):
            self.run_training('changed', str(self.root / 'before-change/final_model.pt'))

    def test_adaptive_scheduler_state_survives_resume(self):
        self.config.scheduler = 'adaptive'
        self.config.eval_every = 1
        self.config.max_steps = 6
        class Stop:
            def on_eval(self, step, *_):
                if step == 4:
                    trainer.request_stop()
        torch.manual_seed(19)
        reference = self.run_training('adaptive-reference')
        torch.manual_seed(19)
        self.run_training('adaptive-stop', callback=Stop())
        resumed = self.run_training('adaptive-resume', str(self.root / 'adaptive-stop/final_model.pt'))
        self.assertEqual(resumed['scheduler_states'], reference['scheduler_states'])
        for key, tensor in reference['model_state_dict'].items():
            torch.testing.assert_close(resumed['model_state_dict'][key], tensor, rtol=0, atol=0)

    def test_interrupted_extension_can_resume_its_new_schedule(self):
        self.run_training('extension-base')
        checkpoint = str(self.root / 'extension-base/final_model.pt')
        class Stop:
            def on_eval(self, *_):
                trainer.request_stop()
        reference = self.run_training('extension-reference', checkpoint, additional_steps=3)
        stopped = self.run_training('extension-stop', checkpoint, callback=Stop(), additional_steps=3)
        self.assertEqual(stopped['step'], 6)
        self.assertEqual(stopped['train_config'].max_steps, 7)
        self.config.max_steps = 7
        resumed = self.run_training('extension-resume', str(self.root / 'extension-stop/final_model.pt'))
        self.assertEqual(resumed['scheduler_states'], reference['scheduler_states'])
        for key, tensor in reference['model_state_dict'].items():
            torch.testing.assert_close(resumed['model_state_dict'][key], tensor, rtol=0, atol=0)

    def test_missing_checkpoint_does_not_start_a_fresh_run(self):
        with self.assertRaises(FileNotFoundError):
            self.run_training('missing', str(self.root / 'absent.pt'))

    def test_legacy_weights_can_start_a_new_run(self):
        original = self.run_training('original')
        del original['checkpoint_version']
        legacy = self.root / 'legacy.pt'
        torch.save(original, legacy)
        with self.assertRaisesRegex(ValueError, 'Legacy checkpoint'):
            self.run_training('legacy-resume', str(legacy))
        self.config.max_steps = 1
        fresh = self.run_training('legacy-weights', str(legacy), load_optimizer_state=False)
        self.assertEqual(fresh['step'], 1)
        self.assertEqual(fresh['total_tokens_seen'], 2 * 15)

    def test_short_and_zero_warmup_schedules_are_finite(self):
        for name in ['cosine', 'linear', 'polynomial', 'wsd', 'constant', 'adaptive']:
            for warmup in [0, 1, 1000]:
                with self.subTest(scheduler=name, warmup=warmup):
                    parameter = torch.nn.Parameter(torch.ones(1))
                    optimizer = torch.optim.SGD([parameter], lr=0.1)
                    config = TrainingConfig(max_steps=1, warmup_steps=warmup, scheduler=name)
                    scheduler, = trainer.setup_schedulers([optimizer], config)
                    optimizer.step()
                    scheduler.step()
                    self.assertTrue(math.isfinite(optimizer.param_groups[0]['lr']))

    def test_invalid_training_config_is_rejected(self):
        for name in ['max_steps', 'batch_size', 'gradient_accumulation_steps', 'eval_every', 'eval_steps']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                TrainingConfig(**{name: 0})


if __name__ == '__main__':
    unittest.main()
