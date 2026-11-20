import torch
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_assembly import LLMLabAssemblyConfig
from .assembly_runtime import AssemblyRuntime


class LLMLabAssemblyForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LLMLabAssemblyConfig
    _supports_cache_class = False
    _tied_weights_keys = {}

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    def __init__(self, config):
        super().__init__(config)
        self.graph = AssemblyRuntime(config.assembly, config.d_model, config.vocab_size,
                                     config.max_seq_len, config.tie_word_embeddings)
        if config.tie_word_embeddings:
            source = next(n['id'] for n in config.assembly['nodes'] if n['kind'] == 'embedding')
            self._tied_weights_keys = {f'graph.bricks.{self.graph.plan["output"]}.weight': f'graph.bricks.{source}.weight'}
        self.post_init()

    def tie_weights(self, *args, **kwargs):
        # AssemblyRuntime ties the actual graph endpoints; no assumed lm_head name.
        if self.config.tie_word_embeddings:
            source = next(n['id'] for n in self.config.assembly['nodes'] if n['kind'] == 'embedding')
            self.graph.bricks[self.graph.plan['output']].weight = self.graph.bricks[source].weight

    def get_input_embeddings(self):
        source = next(n['id'] for n in self.config.assembly['nodes'] if n['kind'] == 'embedding')
        return self.graph.bricks[source]

    def get_output_embeddings(self):
        return self.graph.bricks[self.graph.plan['output']]

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_ids': input_ids, 'use_cache': False}

    def forward(self, input_ids=None, labels=None, attention_mask=None, use_cache=False, **kwargs):
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError('This first assembly runtime supports unpadded sequences only')
        logits = self.graph(input_ids)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                                               labels[:, 1:].reshape(-1), ignore_index=-100)
        if loss is not None and self.graph.aux_loss is not None:
            loss = loss + self.graph.aux_loss
        return CausalLMOutputWithPast(loss=loss, logits=logits)
