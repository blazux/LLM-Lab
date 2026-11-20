"""SDPA optimized dispatch must preserve dense attention outputs and gradients."""
from contextlib import nullcontext
from copy import deepcopy
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from model.assembly_runtime import AssemblyAttention

class AttentionBackendTests(unittest.TestCase):
    def test_math_and_auto_outputs_and_gradients_with_causal_document_and_alibi_masks(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        device = os.environ.get('LLMLAB_TEST_DEVICE','cpu')
        if device == 'cuda' and not torch.cuda.is_available(): self.fail('CUDA required')
        for position in ['rope','alibi']:
            for packed in [False,True]:
                with self.subTest(position=position,packed=packed):
                    a = AssemblyAttention(32,4,2,16,1,{'backend':'auto','position':position}).to(device)
                    b = deepcopy(a); b.backend='math'
                    x = torch.randn(2,8,32,device=device,requires_grad=True)
                    other = x.detach().clone().requires_grad_()
                    docs = torch.tensor([[0]*4+[1]*4]*2,device=device) if packed else None
                    with torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else nullcontext():
                        out, ref = a(x,docs), b(other,docs)
                    rtol,atol=(.05,.003) if device=='cuda' else (3e-3,3e-5)
                    torch.testing.assert_close(out,ref,rtol=rtol,atol=atol)
                    out.square().sum().backward(); ref.square().sum().backward()
                    gradients = [(x.grad,other.grad)] + [(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters())]
                    for actual,reference in gradients:
                        if device == 'cuda':
                            # BF16 cancellation makes relative errors at almost-zero
                            # elements meaningless. Bound the whole gradient error
                            # and the largest absolute error relative to its scale.
                            delta=(actual-reference).float()
                            self.assertLessEqual((delta.norm()/reference.float().norm().clamp_min(1e-7)).item(),.02)
                            self.assertLessEqual(delta.abs().max().item(),.02*reference.abs().max().item()+1e-7)
                        else:
                            torch.testing.assert_close(actual,reference,rtol=rtol,atol=atol)
