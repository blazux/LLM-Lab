"""Reload exports in an isolated interpreter that cannot import project modules."""
import json
import os
from pathlib import Path
import subprocess
import sys

WORKER = '''
import json, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
root, reference, device, count, result_path = sys.argv[1:]
model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True, local_files_only=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
ref = torch.load(reference, weights_only=True, map_location='cpu')
with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device=='cuda' and torch.cuda.is_bf16_supported()):
    actual = model(ref['input_ids'].to(device)).logits.cpu()
    torch.testing.assert_close(actual, ref['logits'], rtol=1e-5, atol=1e-5)
    tokens = model.generate(ref['input_ids'].to(device), max_new_tokens=int(count), do_sample=False,
                            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
text = tokenizer.decode(tokens[0, ref['input_ids'].shape[1]:], skip_special_tokens=True)
assert text == ref['continuation'], (text, ref['continuation'])
result = {'verified':True, 'max_logit_difference': (actual.float()-ref['logits'].float()).abs().max().item(),
          'greedy_generation_matches':True, 'device':device}
Path(result_path).write_text(json.dumps(result))
print(json.dumps(result))
'''


def export_and_verify(root, checkpoint, recipe, output=None):
    import torch
    from transformers import AutoTokenizer
    from model.factory import build_model
    from training.train import precision_context
    from inference.inference import generate_text
    from export import export_to_local
    root = Path(root)
    output = Path(output).resolve() if output else root / 'export'
    if output.exists():
        raise ValueError(f'Export directory already exists: {output}; choose a new --output-dir')
    device = recipe['experiment']['device']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build_model(saved['model_config']).to(device).eval()
    model.load_state_dict(saved['model_state_dict'])
    tokenizer = AutoTokenizer.from_pretrained(saved['model_config'].tokenizer_name, local_files_only=True)
    prompt = recipe['evaluation']['prompts'][0]
    x = tokenizer.encode(prompt, return_tensors='pt').to(device)
    count = min(16, saved['model_config'].max_seq_len - x.shape[1] - 1)
    if x.shape[1] < 1 or count < 1:
        raise ValueError('Verification prompt must fit within the model context')
    with torch.no_grad(), precision_context(torch.device(device)):
        logits = model(x)[0].cpu()
    continuation = generate_text(model, tokenizer, torch.device(device), prompt, max_tokens=count, strategy='greedy')
    reference = root / 'verification-reference.pt'
    torch.save({'input_ids': x.cpu(), 'logits': logits, 'continuation': continuation}, reference)
    options = recipe['export']
    export_to_local(str(checkpoint), str(output), model_name=options['model_name'],
                    model_description=options['description'], license_type=options['license'])
    result_path = output / 'verification.json'
    process = subprocess.run([sys.executable, '-I', '-c', WORKER, str(output), str(reference), device, str(count), str(result_path)],
                              cwd=output, env={**os.environ, 'HF_HUB_OFFLINE': '1'},
                              capture_output=True, text=True, timeout=120)
    (output / 'verification.log').write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f'Export reload verification failed: {output / "verification.log"}')
    # Library progress bars and diagnostics may write after our JSON on stdout.
    # Read the dedicated result artifact rather than treating logs as a protocol.
    try:
        result = json.loads(result_path.read_text())
        if result.get('verified') is not True or result.get('greedy_generation_matches') is not True:
            raise ValueError('Missing verification success flags')
    except (OSError, ValueError, AttributeError) as error:
        raise RuntimeError(f'Export verification did not produce a valid result: {output / "verification.log"}') from error
    result['path'] = str(output)
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
