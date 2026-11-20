"""Architecture templates are real, resizable graphs with reference model parity."""
import contextlib
from copy import deepcopy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'gui/backend')]
import torch
from config import ModelConfig, TrainingConfig
from model.factory import build_model
from model.assembly_templates import family_template, template_catalog, resize_assembly
from model.assembly_runtime import assembly_signature


class ArchitectureTemplatesTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.device = os.environ.get('LLMLAB_TEST_DEVICE', 'cpu')
        if self.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA required')

    def config(self, family):
        return ModelConfig(**{**family_template(family), 'vocab_size': 32, 'max_seq_len': 32})

    def test_every_catalog_size_compiles_and_counts_without_allocating_weights(self):
        for family in template_catalog():
            for size in family['sizes']:
                cfg = ModelConfig(**family_template(family['id'], size), vocab_size=32)
                with torch.device('meta'): m = build_model(cfg)
                self.assertGreater(m.count_parameters(), 0)
                self.assertEqual(cfg.count_params(), m.count_parameters())

    def test_templates_match_transformers_reference_logits_with_identical_weights(self):
        from transformers import LlamaConfig, LlamaForCausalLM, GemmaConfig, GemmaForCausalLM, Qwen2Config, Qwen2ForCausalLM
        definitions = {'llama3': (LlamaConfig, LlamaForCausalLM), 'gemma1': (GemmaConfig, GemmaForCausalLM), 'qwen25': (Qwen2Config, Qwen2ForCausalLM)}
        for family, (config_cls, model_cls) in definitions.items():
            with self.subTest(family=family):
                cfg = self.config(family)
                ours = build_model(cfg).to(self.device).eval()
                body = cfg.assembly['nodes'][2]['body']
                attn = body['nodes'][2]['params']
                reference_config = config_cls(vocab_size=32, hidden_size=cfg.d_model,
                    intermediate_size=cfg.d_ff, num_hidden_layers=cfg.n_layers,
                    num_attention_heads=cfg.n_heads, num_key_value_heads=cfg.n_kv_heads,
                    head_dim=cfg.d_model//cfg.n_heads, max_position_embeddings=32,
                    rms_norm_eps=body['nodes'][1]['params']['eps'], rope_theta=attn['rope_base'],
                    hidden_act='gelu_pytorch_tanh' if family == 'gemma1' else 'silu',
                    tie_word_embeddings=cfg.tie_word_embeddings, attention_bias=family == 'qwen25')
                reference_config._attn_implementation = 'eager'
                reference = model_cls(reference_config).to(self.device).eval()
                with torch.no_grad():
                    reference.model.embed_tokens.weight.copy_(ours.graph.bricks['embedding'].weight)
                    reference.lm_head.weight.copy_(ours.lm_head.weight)
                    reference.model.norm.weight.copy_(ours.graph.bricks['final_norm'].weight)
                    for layer, block in zip(reference.model.layers, ours.graph.bricks['block']):
                        layer.input_layernorm.weight.copy_(block.bricks['norm1'].weight)
                        layer.post_attention_layernorm.weight.copy_(block.bricks['norm2'].weight)
                        for target, source in [('q_proj','q'),('k_proj','k'),('v_proj','v'),('o_proj','out')]:
                            a, b = getattr(layer.self_attn,target), getattr(block.bricks['attn'],source)
                            a.weight.copy_(b.weight)
                            if a.bias is not None: a.bias.copy_(b.bias)
                        for target, source in [('gate_proj','gate'),('up_proj','up'),('down_proj','down')]:
                            getattr(layer.mlp,target).weight.copy_(getattr(block.bricks['ffn'],source).weight)
                    x = torch.randint(32,(2,12),device=self.device)
                    torch.testing.assert_close(ours(x)[0], reference(x).logits, rtol=2e-4, atol=2e-5)

    def test_global_resize_preserves_custom_topology_and_local_choices(self):
        cfg = self.config('qwen25')
        body = cfg.assembly['nodes'][2]['body']
        body['nodes'][5]['params']['activation'] = 'reglu'
        # Rewire norm2/FFN to post-norm, a meaningful user customization.
        for edge in body['edges']:
            if edge['target'] == 'norm2': edge['source'] = 'ffn'
            elif edge['target'] == 'ffn': edge['source'] = 'add1'
            elif edge['target'] == 'add2' and edge['targetHandle'] == 'a': edge['source'] = 'norm2'
        before = deepcopy(cfg.assembly)
        resized = resize_assembly(cfg, 224, 3, 64)
        self.assertEqual(cfg.assembly, before)
        self.assertEqual(resized.assembly['edges'], before['edges'])
        changed_body = resized.assembly['nodes'][2]['body']
        self.assertEqual(changed_body['edges'], body['edges'])
        self.assertEqual(changed_body['nodes'][5]['params']['activation'], 'reglu')
        self.assertEqual(changed_body['nodes'][5]['params']['hidden'], cfg.d_ff * 2)
        self.assertEqual(resized.assembly['nodes'][2]['params']['count'], 3)
        self.assertNotEqual(assembly_signature(cfg), assembly_signature(resized))
        model = build_model(resized).to(self.device)
        model(torch.randint(32, (1, 8), device=self.device))[0].square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_invalid_size_is_actionable_and_does_not_mutate_graph(self):
        cfg = self.config('llama3'); before = deepcopy(cfg.assembly)
        for args in [(63,2,32),(64,0,32),(64,2,100000)]:
            with self.assertRaises(ValueError): resize_assembly(cfg,*args)
        self.assertEqual(cfg.assembly, before)
        with self.assertRaises(ValueError): family_template('unknown')
        with self.assertRaises(ValueError): family_template('gemma1','huge')

    def test_family_exports_reload_independently_and_keep_template_origin(self):
        from export import export_to_local
        for family in ['llama3','gemma1','qwen25']:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); cfg = self.config(family)
                model = build_model(cfg).to(self.device).eval()
                x = torch.randint(32, (1,8),device=self.device)
                with torch.no_grad(): logits = model(x)[0].cpu()
                torch.save({'model_config':cfg,'model_state_dict':model.state_dict(),'train_config':TrainingConfig()},root/'model.pt')
                with contextlib.redirect_stdout(io.StringIO()):
                    export_to_local(str(root/'model.pt'),str(root/'export'),include_tokenizer=False)
                torch.save({'x':x.cpu(),'logits':logits},root/'reference.pt')
                code = '''
import sys, torch
from transformers import AutoModelForCausalLM
root, device, family = sys.argv[1:]
m=AutoModelForCausalLM.from_pretrained(root+'/export',trust_remote_code=True,local_files_only=True).to(device).eval()
assert m.config.assembly_template == family
r=torch.load(root+'/reference.pt',weights_only=True)
with torch.no_grad(): torch.testing.assert_close(m(r['x'].to(device)).logits.cpu(),r['logits'],rtol=0,atol=0)
'''
                result = subprocess.run([sys.executable,'-I','-c',code,tmp,self.device,family],cwd=tmp,
                    env={**os.environ,'HF_HOME':tmp+'/cache','HF_HUB_OFFLINE':'1'},capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == '__main__': unittest.main()
