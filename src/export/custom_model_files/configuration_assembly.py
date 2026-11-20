from transformers import PretrainedConfig


class LLMLabAssemblyConfig(PretrainedConfig):
    model_type = 'llm-lab-assembly'
    def __init__(self, assembly=None, d_model=64, vocab_size=8192, max_seq_len=256,
                 tie_word_embeddings=True, use_cache=False, **kwargs):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.use_cache = False
        self.assembly = assembly
        self.d_model = d_model
        self.hidden_size = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.max_position_embeddings = max_seq_len
