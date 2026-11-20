"""Offline graphical service contract: executable graph through independent HF reload."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import test_run_tracking as tracking


class GraphicalAcceptanceTests(tracking.RunApiFixture):
    def test_workshop_train_stop_resume_generate_export_reload_and_mock_hub(self):
        import torch
        from config import ModelConfig
        from model.factory import build_model
        tokenizer = self.client.post('/api/assembly/tokenizer', json={'name': str(self.tokenizer)})
        self.assertEqual(tokenizer.status_code, 200, tokenizer.text)
        self.assertEqual(tokenizer.json()['vocab_size'], 8)
        self.assertEqual(tokenizer.json()['eos_token_id'], 1)
        template = self.client.get('/api/assembly/template').json()
        model = {**template, 'tokenizer_name': str(self.tokenizer), 'vocab_size': 8, 'max_seq_len': 16}
        inspect = lambda cfg: self.client.post('/api/assembly/inspect', json={'model_cfg': cfg}).json()
        before = inspect(model)
        self.assertTrue(before['valid'], before)
        changed = deepcopy(model)
        changed['assembly']['nodes'][2]['params']['count'] = 3
        self.assertGreater(inspect(changed)['parameters'], before['parameters'])
        invalid = deepcopy(model)
        invalid['assembly']['edges'][0]['target'] = 'head'
        report = inspect(invalid)
        self.assertFalse(report['valid'])
        self.assertTrue(report['errors'][0]['hint'])
        self.payload['model_cfg'] = model
        original = self.api.MetricsCallback.on_update
        def stop(callback, step):
            original(callback, step)
            if step == 2:
                asyncio.run(self.api.stop_training(self.api.training_state['run_id']))
        with patch.object(self.api.MetricsCallback, 'on_update', stop):
            start = self.client.post('/api/training/start', json=self.payload)
            self.assertEqual(start.status_code, 200, start.text)
            self.wait_for_run()
        old = self.store.get(start.json()['run_id'])
        self.assertEqual(old['status'], 'stopped')
        resumed = self.client.post(f"/api/training/runs/{old['run_id']}/resume", json={'additional_steps': 2})
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.wait_for_run()
        run = self.store.get(resumed.json()['run_id'])
        checkpoint = str(Path(run['output_dir']) / 'final_model.pt')
        self.assertEqual(run['current_step'], 4)
        generated = self.client.post('/api/inference/generate', json={'checkpoint_path': checkpoint,
            'prompt': 'a little', 'max_tokens': 3, 'strategy': 'greedy'})
        self.assertEqual(generated.status_code, 200, generated.text)
        output = self.root / 'export'
        exported = self.client.post('/api/export/local', json={'checkpoint_path': checkpoint, 'output_dir': str(output)})
        self.assertTrue(exported.json()['success'], exported.text)
        self.assertTrue((output / 'README.md').is_file())
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        reference = build_model(saved['model_config']).eval()
        reference.load_state_dict(saved['model_state_dict'])
        ids = torch.tensor([[3, 4]])
        with torch.no_grad(): logits = reference(ids)[0]
        torch.save({'ids': ids, 'logits': logits}, self.root / 'reference.pt')
        code = '''
import json, pathlib, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
root=pathlib.Path(sys.argv[1]); torch.set_num_threads(1)
m=AutoModelForCausalLM.from_pretrained(root/'export',trust_remote_code=True,local_files_only=True).eval()
t=AutoTokenizer.from_pretrained(root/'export',local_files_only=True)
r=torch.load(root/'reference.pt',weights_only=True)
with torch.no_grad():
 torch.testing.assert_close(m(r['ids']).logits,r['logits'],rtol=0,atol=0)
 out=m.generate(r['ids'],max_new_tokens=3,do_sample=False,pad_token_id=t.pad_token_id,eos_token_id=t.eos_token_id)
(root/'reload.json').write_text(json.dumps({'text':t.decode(out[0,2:],skip_special_tokens=True)}))
'''
        result = subprocess.run([sys.executable, '-I', '-c', code, str(self.root)], cwd=self.root,
            env={**os.environ, 'HF_HUB_OFFLINE': '1'}, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((self.root/'reload.json').read_text())['text'], generated.json()['generated_text'])
        with patch('huggingface_hub.create_repo') as create, patch('huggingface_hub.HfApi') as hub:
            pushed = self.client.post('/api/export/hub', json={'checkpoint_path': checkpoint,
                'repo_id': 'offline-fixture/model', 'private': True})
            self.assertTrue(pushed.json()['success'], pushed.text)
            self.assertTrue(create.call_args.kwargs['private'])
            self.assertEqual(hub.return_value.upload_folder.call_args.kwargs['repo_id'], 'offline-fixture/model')
        # Reject invalid source/tokenizer input through the same service boundary.
        self.assertEqual(self.client.post('/api/assembly/tokenizer', json={'name': ''}).status_code, 422)

    def test_inference_reloads_a_checkpoint_replaced_at_the_same_path(self):
        from unittest.mock import Mock
        import api.inference as inference
        import torch
        path=self.root/'replaced.pt'
        torch.save({'step':1},path)
        tokenizer=Mock();tokenizer.encode.return_value=[1]
        cache=dict(checkpoint_path=None,signature=None,model=None,tokenizer=None,device=None,is_chat_model=False)
        with patch.object(inference,'_model_cache',cache), patch.object(inference,'load_model_for_inference',return_value=(torch.nn.Linear(1,1),tokenizer,torch.device('cpu'))) as load, patch.object(inference,'generate_text',return_value='text'):
            request={'checkpoint_path':str(path),'prompt':'a','max_tokens':1}
            for _ in range(2):self.assertEqual(self.client.post('/api/inference/generate',json=request).status_code,200)
            self.assertEqual(load.call_count,1)
            stat=path.stat();os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1))
            self.assertEqual(self.client.post('/api/inference/generate',json=request).status_code,200)
            self.assertEqual(load.call_count,2)

    def test_frontend_routes_do_not_serve_files_outside_the_distribution(self):
        import main
        frontend=self.root/'frontend';frontend.mkdir();(frontend/'index.html').write_text('UI')
        (self.root/'forbidden.txt').write_text('private fixture')
        with patch.object(main,'frontend_dist',frontend):
            response=self.client.get('/%2e%2e%2fforbidden.txt')
            self.assertEqual(response.status_code,404,response.text)
            self.assertEqual(self.client.get('/api/unknown').status_code,404)
            self.assertEqual(self.client.get('/').text,'UI')
