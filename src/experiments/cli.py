"""Noninteractive experiment commands; the legacy menu remains available without arguments."""
import argparse
import json
from pathlib import Path
import sys


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(prog='llm-lab', description='Create, train, resume and export a model from a TOML recipe.')
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='Validate a recipe offline without allocating a model')
    validate.add_argument('recipe', type=Path)
    validate.add_argument('--check-data', action='store_true', help='Also scan and deduplicate source documents in a temporary directory')
    run = commands.add_parser('run', help='Start a new experiment')
    run.add_argument('recipe', type=Path)
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--stop-after', type=positive, help='Stop cleanly after this absolute optimizer update, preserving the original schedule')
    resume = commands.add_parser('resume', help='Resume saved data, tokenizer, optimizer, scheduler and RNG')
    resume.add_argument('run_dir', type=Path)
    resume.add_argument('--additional-steps', type=positive, default=0)
    resume.add_argument('--stop-after', type=positive)
    status = commands.add_parser('status', help='Read saved run state as JSON')
    status.add_argument('run_dir', type=Path)
    export = commands.add_parser('export', help='Export and verify standalone reload and generation')
    export.add_argument('run_dir', type=Path)
    export.add_argument('--output-dir', type=Path)
    generate = commands.add_parser('generate', help='Generate from a run checkpoint')
    generate.add_argument('run_dir', type=Path)
    generate.add_argument('--prompt', required=True)
    generate.add_argument('--max-tokens', type=positive, default=80)
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate':
            from .recipe import load_recipe, describe
            recipe = load_recipe(args.recipe)
            result = describe(recipe)
            if args.check_data:
                import tempfile
                from .data import prepare_data
                with tempfile.TemporaryDirectory() as tmp:
                    result['data'] = prepare_data(recipe['data'], Path(tmp) / 'data')
                    for split in result['data'].values():
                        split.pop('path')
        elif args.command == 'run':
            from .runner import run
            result = run(args.recipe, args.output_dir, args.stop_after)
        elif args.command == 'resume':
            from .runner import resume
            result = resume(args.run_dir, args.additional_steps, args.stop_after)
        else:
            from .runner import read_json, write_json, assert_snapshot, choose_checkpoint, lock, check_device, working_directory
            root = args.run_dir.resolve()
            if args.command == 'status':
                result = read_json(root / 'state.json')
            else:
                export_output = args.output_dir.resolve() if args.command == 'export' and args.output_dir else None
                with lock(root), working_directory(root):
                    assert_snapshot(root)
                    recipe = read_json(root / 'recipe.resolved.json')
                    check_device(recipe)
                    checkpoint = choose_checkpoint(root)
                    if args.command == 'export':
                        from .verification import export_and_verify
                        result = export_and_verify(root, checkpoint, recipe, export_output)
                        state = read_json(root / 'state.json')
                        state['export'] = result
                        if state.get('step', 0) >= state.get('target_steps', float('inf')):
                            state.update(phase='completed', error=None)
                        write_json(root / 'state.json', state)
                    else:
                        from inference.inference import load_model_for_inference, generate_text
                        model, tokenizer, device = load_model_for_inference(str(checkpoint), device=recipe['experiment']['device'])
                        count = min(args.max_tokens, model.config.max_seq_len - len(tokenizer.encode(args.prompt)) - 1)
                        if not args.prompt.strip() or count < 1:
                            raise ValueError('Prompt must be nonempty and leave space in the context')
                        result = {'prompt': args.prompt, 'continuation': generate_text(
                            model, tokenizer, device, args.prompt, max_tokens=count, strategy='greedy')}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, AssertionError, ArithmeticError) as error:
        print(f'llm-lab: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
