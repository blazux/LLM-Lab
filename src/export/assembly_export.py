"""Package the identical graph runtime with a Hugging Face adapter."""
import json
from pathlib import Path
import shutil
import torch
from .converter import FormatInfo, ExportFormat


def export_assembly(checkpoint_path, output_dir, config, name, description, license, include_tokenizer, use_safetensors,
                    author=None, tags=None, datasets=None, metrics=None, training_config=None):
    root = Path(output_dir)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    # Clone tensors so tied endpoint weights can both be stored in safetensors.
    weights = {key: value.detach().contiguous().clone() for key, value in saved['model_state_dict'].items()}
    if use_safetensors:
        from safetensors.torch import save_file
        save_file(weights, str(root / 'model.safetensors'), metadata={'format': 'pt'})
    else:
        torch.save(weights, root / 'pytorch_model.bin')
    hf = config.to_hf_config()
    hf.update(architectures=['LLMLabAssemblyForCausalLM'], torch_dtype=str(next(iter(weights.values())).dtype).replace('torch.', ''),
              auto_map={'AutoConfig': 'configuration_assembly.LLMLabAssemblyConfig',
                        'AutoModelForCausalLM': 'modeling_assembly.LLMLabAssemblyForCausalLM'})
    from transformers import AutoTokenizer
    if include_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        tokenizer.save_pretrained(root)
        hf.update(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    (root / 'config.json').write_text(json.dumps(hf, indent=2, ensure_ascii=False) + '\n')
    (root / 'assembly.json').write_text(json.dumps(config.assembly, indent=2, ensure_ascii=False) + '\n')
    here = Path(__file__).parent
    for file in ['configuration_assembly.py', 'modeling_assembly.py']:
        shutil.copy2(here / 'custom_model_files' / file, root / file)
    for runtime_file in ['assembly_runtime.py', 'assembly_schema.py', 'modern_bricks.py']:
        shutil.copy2(here.parent / 'model' / runtime_file, root / runtime_file)
    (root / 'requirements.txt').write_text('torch>=2.7.0\ntransformers==5.20.1\nsafetensors>=0.4.0\n')
    metadata = {'assembly_template': getattr(config, 'assembly_template', None), 'author': author, 'tags': tags or [], 'datasets': datasets or [], 'metrics': metrics or {}}
    (root / 'experiment_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    if training_config is not None:
        from dataclasses import asdict, is_dataclass
        values = asdict(training_config) if is_dataclass(training_config) else training_config
        (root / 'training_config.json').write_text(json.dumps(values, indent=2) + '\n')
    (root / 'README.md').write_text(f'''---
license: {json.dumps(license)}
library_name: transformers
pipeline_tag: text-generation
tags: {json.dumps(tags or [])}
datasets: {json.dumps(datasets or [])}
---
# {name or 'LLM-Lab visual assembly'}

{description or 'A model assembled and trained in LLM-Lab.'}

{('Author: ' + author) if author else ''}

Starting architecture: {getattr(config, 'assembly_template', None) or 'custom graph'}. The graph may have been modified after choosing this starting point.

The executable assembly is preserved in `assembly.json` and `config.json`.
Load with `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`.
This first runtime supports unpadded sequences and recomputes the graph during decoding.
''')
    return FormatInfo(ExportFormat.CUSTOM, 'Executable visual assembly; identical exported runtime', False,
                      'llm-lab-assembly', ['LLMLabAssemblyForCausalLM'])
