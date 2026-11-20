"""Short architecture-specific batch/accumulation and compiled-FFN benchmark.

No dataset download; synthetic next-token batches. Includes backward, optimizer,
precision and synchronization; excludes validation, I/O and checkpoint writes.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import platform
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.nn import functional as F
from config import ModelConfig
from model.factory import build_model
from model.assembly_templates import family_template

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--family',choices=['llama3','qwen4_preview'],default='llama3')
 p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--compile-ffn',action='store_true')
 args=p.parse_args();torch.set_num_threads(1)
 if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA required')
 cfg=ModelConfig(**{**family_template(args.family,'small'),'vocab_size':1024,'max_seq_len':128})
 rows=[]
 for batch in [1,2,4]:
  accumulation=4//batch
  torch.manual_seed(42)
  model=build_model(cfg).to(args.device).train()
  compiled=0
  compiled_parity=None
  if args.compile_ffn:
   from model.assembly_runtime import AssemblyFFN
   for module in model.modules():
    if isinstance(module,AssemblyFFN):
     if compiled == 0:
      sample=torch.randn(1,8,cfg.d_model,device=args.device,requires_grad=True)
      expected=module(sample);expected.sum().backward()
      grads=[parameter.grad.detach().clone() for parameter in module.parameters()]
      input_grad=sample.grad.detach().clone();module.zero_grad(set_to_none=True);sample.grad=None
      module.compile();actual=module(sample);actual.sum().backward()
      torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-5)
      torch.testing.assert_close(sample.grad,input_grad,rtol=3e-3,atol=3e-5)
      for parameter,gradient in zip(module.parameters(),grads):torch.testing.assert_close(parameter.grad,gradient,rtol=3e-3,atol=3e-5)
      module.zero_grad(set_to_none=True);compiled_parity=True
     else: module.compile()
     compiled+=1
  optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
  ids=torch.randint(1024,(4,128),device=args.device)
  def step():
   optimizer.zero_grad(set_to_none=True)
   for offset in range(0,4,batch):
    x=ids[offset:offset+batch]
    with torch.autocast('cuda',dtype=torch.bfloat16) if args.device=='cuda' else nullcontext():
     logits,aux=model(x)
     loss=F.cross_entropy(logits.float().reshape(-1,1024),torch.roll(x,-1,1).reshape(-1))
     if aux is not None:loss=loss+aux
    (loss/accumulation).backward()
   optimizer.step()
  sync=lambda:torch.cuda.synchronize() if args.device=='cuda' else None
  start=time.perf_counter();step();sync();warmup=time.perf_counter()-start
  if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
  sync();start=time.perf_counter()
  for _ in range(3):step()
  sync();elapsed=time.perf_counter()-start
  if args.family == 'llama3':
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:step()
   dispatch=sorted({event.key for event in profile.key_averages() if 'scaled_dot' in event.key or 'flash_att' in event.key})
  else: dispatch=['Reference eager attention; recurrent/convolutional PyTorch fallbacks (see Nano brick profile)']
  rows.append(dict(attention_dispatch=dispatch,microbatch=batch,accumulation=accumulation,effective_batch=4,updates=3,
   tokens_per_second=4*128*3/elapsed,elapsed_seconds=elapsed,warmup_seconds=warmup,
   peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,compiled_ffns=compiled,compiled_forward_gradient_parity=compiled_parity))
  del optimizer,model
  if args.device=='cuda':torch.cuda.empty_cache()
 report=dict(family=args.family,torch=torch.__version__,python=platform.python_version(),
  device=torch.cuda.get_device_name() if args.device=='cuda' else platform.processor(),
  precision='bfloat16 autocast / float32 weights and AdamW states' if args.device=='cuda' else 'float32',
  parameters=cfg.count_params(),context=128,vocabulary=1024,scope=__doc__,results=rows,
  fastest_microbatch=max(rows,key=lambda r:r['tokens_per_second'])['microbatch'])
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
