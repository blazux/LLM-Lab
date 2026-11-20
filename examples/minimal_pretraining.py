"""Local demonstration: create, train, stop, resume, generate and export a tiny LM.

python examples/minimal_pretraining.py --output-dir /tmp/llm-lab-demo
Requires the project's Python dependencies; no model, dataset or Hub upload.
"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from config import ModelConfig, TrainingConfig
from export import export_to_local
from training.train import request_stop, train_model


class StopAfterTwoUpdates:
    def on_eval(self, step, *_):
        if step == 2:
            request_stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path, help='New directory for all demo artifacts')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    # Keep the existing tokenizer cache and dataset caches out of the repository.
    os.chdir(root)
    os.environ.setdefault('HF_HOME', str(root / 'hf-cache'))
    os.environ.setdefault('HF_DATASETS_CACHE', str(root / 'dataset-cache'))
    torch.set_num_threads(1)
    torch.manual_seed(42)

    words = ['[UNK]', '[EOS]', '[PAD]', 'bonjour', 'le', 'petit', 'modele', 'apprend']
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                        eos_token='[EOS]', pad_token='[PAD]',
                                        model_input_names=['input_ids', 'attention_mask'])
    tokenizer.save_pretrained(root / 'tokenizer')
    corpus = root / 'corpus.txt'
    corpus.write_text('bonjour le petit modele apprend\n' * 400)
    model = ModelConfig(tokenizer_name=str(root / 'tokenizer'), vocab_size=len(tokenizer),
                        d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=32,
                        max_seq_len=16, dropout=0.1)
    training = TrainingConfig(max_steps=4, batch_size=1, gradient_accumulation_steps=2,
                              warmup_steps=0, eval_every=2, eval_steps=1, save_best_only=False,
                              datasets=[{'name': 'text', 'data_files': str(corpus)}])
    train_model(model, training, output_dir=str(root / 'stopped'), callback=StopAfterTwoUpdates())
    train_model(model, training, output_dir=str(root / 'resumed'),
                checkpoint_path=str(root / 'stopped/final_model.pt'))
    export_dir = export_to_local(str(root / 'resumed/final_model.pt'), str(root / 'export'))
    reloaded = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(export_dir, local_files_only=True)
    generated = reloaded.generate(**tokenizer('bonjour', return_tensors='pt'), max_new_tokens=4, do_sample=False)
    print('\nGenerated:', tokenizer.decode(generated[0], skip_special_tokens=True))
    print('Local export:', export_dir)
    print('This tiny run verifies the lifecycle; four updates do not produce a useful language model.')


if __name__ == '__main__':
    main()
