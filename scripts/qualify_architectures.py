"""Reproducible, offline capability and short performance checks for every template.

Synthetic repeated batches measure mechanical learning only, never language quality.
Results describe this exact environment/shape; experimental families stay experimental.
"""
import argparse
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
import platform
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from torch.nn import functional as F
from config import ModelConfig, TrainingConfig
from model.factory import build_model
from model.assembly_templates import family_template, template_catalog
from export import export_to_local


def synchronize(device):
    if device == 'cuda': torch.cuda.synchronize()


def qualify(family, device, folder, updates):
    cfg = ModelConfig(**{**family_template(family['id']), 'vocab_size': 32, 'max_seq_len': 32})
    torch.manual_seed(123)
    model = build_model(cfg).to(device).eval()
    x = torch.randint(32, (1, 8), device=device)
    y = torch.roll(x, -1, dims=1)
    doc = torch.tensor([[0]*4 + [1]*4], device=device)
    with torch.no_grad():
        logits = model(x)[0]
        assert logits.shape == (1, 8, 32) and torch.isfinite(logits).all()
        mutated = x.clone(); mutated[:, :4] = (mutated[:, :4] + 7) % 32
        a, b = model(x, doc_ids=doc)[0], model(mutated, doc_ids=doc)[0]
        torch.testing.assert_close(a[:, 4:], b[:, 4:], rtol=1e-4, atol=1e-5)
        _, _, cache = model(x[:, :4], use_cache=True)
        cached = model(x[:, 4:], past_key_values=cache, use_cache=True)[0]
        torch.testing.assert_close(cached, logits[:, 4:], rtol=1e-4, atol=1e-5)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    def update():
        optimizer.zero_grad(set_to_none=True)
        out, auxiliary = model(x)
        loss = F.cross_entropy(out.float().reshape(-1, 32), y.reshape(-1))
        objective = loss + (auxiliary if auxiliary is not None else 0)
        objective.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        return loss.item()
    warmup = update()
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    synchronize(device); start = time.perf_counter()
    losses = [update() for _ in range(updates)]
    synchronize(device); elapsed = time.perf_counter() - start
    assert losses[-1] < warmup, (warmup, losses)
    checkpoint = folder / (family['id'] + '.pt')
    torch.save({'model_config': cfg, 'train_config': TrainingConfig(), 'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, checkpoint)
    restored = build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    restored.load_state_dict(state['model_state_dict'])
    second_optimizer = torch.optim.AdamW(restored.parameters(), lr=.002)
    second_optimizer.load_state_dict(state['optimizer'])
    model.eval(); restored.eval()
    with torch.no_grad(): torch.testing.assert_close(model(x)[0], restored(x)[0], rtol=0, atol=0)
    # Resume one identical update, including optimizer moments.
    model.train(); restored.train(); update()
    second_optimizer.zero_grad(set_to_none=True)
    out, aux = restored(x)
    (F.cross_entropy(out.float().reshape(-1, 32), y.reshape(-1)) + (aux if aux is not None else 0)).backward()
    second_optimizer.step()
    for p, q in zip(model.parameters(), restored.parameters()): torch.testing.assert_close(p, q, rtol=0, atol=0)
    model.eval()
    torch.save({'model_config': cfg, 'train_config': TrainingConfig(), 'model_state_dict': model.state_dict()}, checkpoint)
    destination = folder / family['id']
    with redirect_stdout(io.StringIO()): export_to_local(str(checkpoint), str(destination), include_tokenizer=False)
    with torch.no_grad(): expected = model(x)[0].cpu()
    torch.save({'ids': x.cpu(), 'logits': expected}, folder / (family['id'] + '-reference.pt'))
    # Profile one actual optimizer update; operator timings include dispatch/host work.
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == 'cuda': activities.append(torch.profiler.ProfilerActivity.CUDA)
    model.train()
    ranges = {}
    hooks = []
    def before(name):
        def start(module, inputs):
            context = torch.profiler.record_function('brick::' + name)
            context.__enter__(); ranges.setdefault(name, []).append(context)
        return start
    def after(name):
        def end(module, inputs, output): ranges[name].pop().__exit__(None, None, None)
        return end
    for name, module in model.named_modules():
        if name and type(module).__module__.startswith('model') and type(module).__name__ != 'AssemblyRuntime':
            hooks += [module.register_forward_pre_hook(before(name)), module.register_forward_hook(after(name))]
    try:
        with torch.profiler.profile(activities=activities) as profiler: update()
    finally:
        for hook in hooks: hook.remove()
    operators = sorted(profiler.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
    bricks = sorted({type(module).__name__ for module in model.modules() if type(module).__module__.startswith('model')})
    return {'family': family['id'], 'status': 'pass', 'support': 'stable-candidate' if family['id'] in {'llama3','gemma1','qwen25'} else 'experimental',
        'reference': family['source'], 'parameters': model.count_parameters(), 'bricks': bricks,
        'checks': ['forward', 'finite_gradients', 'document_isolation', 'cached_generation', 'checkpoint_optimizer_resume', 'synthetic_learning'],
        'initial_ce': warmup, 'final_ce': losses[-1], 'timed_updates': updates, 'warmup_updates': 1,
        'batch_size': 1, 'sequence_length': 8, 'precision': 'float32', 'tokens_per_second': 8*updates/elapsed,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        'brick_profile': [{'name': e.key, 'cpu_us': e.cpu_time_total, 'device_us': getattr(e, 'device_time_total', 0)} for e in profiler.key_averages() if e.key.startswith('brick::')],
        'operator_profile': [{'name': e.key, 'self_cpu_us': e.self_cpu_time_total,
            'self_device_us': getattr(e, 'self_device_time_total', 0)} for e in operators[:8]],
        'attention_dispatch': sorted({e.key for e in profiler.key_averages() if 'scaled_dot' in e.key or 'flash_att' in e.key})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=4)
    args = parser.parse_args()
    if args.updates < 2: parser.error('Use at least 2 timed updates.')
    if args.device == 'cuda' and not torch.cuda.is_available(): parser.error('CUDA required but unavailable')
    torch.set_num_threads(1)
    import transformers
    report = {'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__,
        'device': torch.cuda.get_device_name() if args.device == 'cuda' else platform.processor(), 'device_type': args.device,
        'scope': 'Tiny synthetic correctness/learning checks; not natural-language quality or full-size throughput.', 'results': []}
    with tempfile.TemporaryDirectory(prefix='llmlab-qualification-') as tmp:
        folder = Path(tmp)
        for family in template_catalog():
            try: result = qualify(family, args.device, folder, args.updates)
            except Exception as error:
                import traceback
                traceback.print_exc()
                result = {'family': family['id'], 'status': 'fail', 'error': f'{type(error).__name__}: {error}'}
            report['results'].append(result)
            print(f"{family['id']}: {result['status']}", flush=True)
        # Export code must also work without this repository on sys.path.
        import subprocess
        code = '''
import pathlib, sys, torch
from transformers import AutoModelForCausalLM
root=pathlib.Path(sys.argv[1]); device=sys.argv[2]; torch.set_num_threads(1)
for directory in sorted(root.iterdir()):
 if not directory.is_dir(): continue
 model=AutoModelForCausalLM.from_pretrained(directory,trust_remote_code=True,local_files_only=True).to(device).eval()
 r=torch.load(root/(directory.name+'-reference.pt'),weights_only=True)
 with torch.no_grad():
  actual=model(r['ids'].to(device)).logits.cpu()
  stable=directory.name in {'llama3','gemma1','qwen25'}
  torch.testing.assert_close(actual,r['logits'],rtol=0 if stable else 2e-4,atol=0 if stable else 2e-5)
 print(directory.name, 'isolated export parity: pass; max absolute error', (actual-r['logits']).abs().max().item())
'''
        import os
        verified = subprocess.run([sys.executable,'-I','-c',code,tmp,args.device],cwd=tmp,
            env={**os.environ,'HF_HUB_OFFLINE':'1'},capture_output=True,text=True,timeout=180)
        report['isolated_export_reload'] = {'tolerance': 'Dense stable candidates: bitwise. Experimental: rtol=2e-4, atol=2e-5.', 'passed': verified.returncode == 0, 'output': verified.stdout + verified.stderr}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    return int(any(r['status']!='pass' for r in report['results']) or not report['isolated_export_reload']['passed'])

if __name__ == '__main__': raise SystemExit(main())
