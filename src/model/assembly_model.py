"""Training/inference adapter for executable visual assemblies."""
import torch
from torch import nn
from .assembly_runtime import AssemblyRuntime


class AssemblyLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graph = AssemblyRuntime(config.assembly, config.d_model, config.vocab_size,
                                     config.max_seq_len, config.tie_word_embeddings)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def lm_head(self):
        return self.graph.bricks[self.graph.plan['output']]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x=None, input_ids=None, use_checkpoint=False, past_key_values=None,
                use_cache=False, return_hidden=False, doc_ids=None, **kwargs):
        x = input_ids if input_ids is not None else x
        if x is None:
            raise ValueError('Provide input_ids')
        length = x.shape[1]
        if past_key_values is not None:
            x = torch.cat([past_key_values[0][0], x], dim=1)
        if x.shape[1] > self.config.max_seq_len:
            raise ValueError('Sequence exceeds the assembly context length')
        def execute(ids):
            logits = self.graph(ids, doc_ids, return_hidden)
            return logits, self.graph.aux_loss
        if use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            out, aux = checkpoint(execute, x, use_reentrant=False)
        else:
            out, aux = execute(x)
        out = out[:, -length:]
        # Correctness-first decode: retain token IDs and recompute the assembled
        # DAG. A per-attention KV cache will need a separate graph-aware design.
        return (out, aux, [(x.detach(),)]) if use_cache else (out, aux)
