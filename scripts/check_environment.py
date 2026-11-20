"""Check declared runtime dependencies and modern architecture configuration APIs."""
import argparse
import importlib
import importlib.metadata
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    from packaging.requirements import Requirement
    root = Path(__file__).resolve().parents[1]
    errors = []
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        errors.append("Supported Python versions are 3.11, 3.12 and 3.13.")
    for source in [root / 'requirements.txt', root / 'gui/backend/requirements.txt']:
        for line in source.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            required = Requirement(line)
            try:
                version = importlib.metadata.version(required.name)
                print(f'{required.name}: {version}')
                if version not in required.specifier:
                    errors.append(f'{required.name} {version} does not satisfy {required.specifier}')
            except importlib.metadata.PackageNotFoundError:
                errors.append(f'Missing dependency: {required.name}')
    # Fail early on missing private reference APIs instead of failing halfway
    # through model construction or pretending that a template is supported.
    for module, name in [
        ('qwen3_5', 'Qwen3_5TextConfig'), ('gemma4', 'Gemma4TextConfig'),
        ('llama4', 'Llama4TextConfig'), ('qwen4_exp', 'Qwen4ExpTextConfig'),
    ]:
        try:
            getattr(importlib.import_module(f'transformers.models.{module}.configuration_{module}'), name)
        except (ImportError, AttributeError) as error:
            errors.append(f'Required template API unavailable: {module}.{name}: {error}')
    try:
        import torch
        print(f'CUDA available: {torch.cuda.is_available()}')
        if args.device == 'cuda' and not torch.cuda.is_available():
            errors.append('CUDA requested but unavailable; check the driver and the selected Torch wheel.')
    except ImportError as error:
        errors.append(f'Torch import failed: {error}')
    if errors:
        print('\nEnvironment check failed:\n- ' + '\n- '.join(errors), file=sys.stderr)
        print('Run: bash scripts/bootstrap.sh --device ' + args.device, file=sys.stderr)
        return 1
    print('\nEnvironment check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
