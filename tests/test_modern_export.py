"""HF empty-weight loading must reconstruct deterministic, nonpersistent buffers."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from config import ModelConfig, TrainingConfig
from model.assembly_templates import family_template
from model.factory import build_model
from export import export_to_local
from transformers import AutoModelForCausalLM

class ModernExportTests(unittest.TestCase):
    def test_rotary_and_per_layer_embedding_scales_survive_empty_weight_loading(self):
        torch.set_num_threads(1)
        for family in ['gemma4_e2b','qwen35_dense']:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                torch.manual_seed(123)
                cfg=ModelConfig(**{**family_template(family),'vocab_size':16,'max_seq_len':16})
                model=build_model(cfg).eval(); ids=torch.randint(16,(1,8))
                with torch.no_grad(): expected=model(ids)[0]
                torch.save({'model_config':cfg,'train_config':TrainingConfig(),'model_state_dict':model.state_dict()},root/'model.pt')
                with contextlib.redirect_stdout(io.StringIO()):export_to_local(str(root/'model.pt'),str(root/'export'),include_tokenizer=False)
                loaded=AutoModelForCausalLM.from_pretrained(root/'export',trust_remote_code=True,local_files_only=True).eval()
                with torch.no_grad(): actual=loaded(ids).logits
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
