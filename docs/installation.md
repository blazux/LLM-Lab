# Install LLM-Lab

The runtime targets Linux with Python 3.11–3.13, Torch 2.13.0 and
Transformers 5.20.1. Python package versions are constrained in `constraints.txt`;
Node dependencies are installed from `gui/frontend/package-lock.json` with
`npm ci`. Node 20 or newer, Python's venv support, and curl are required.

GPU installation uses the CUDA 13.0 Torch wheel and requires a compatible NVIDIA
driver. CPU installation does not require a GPU. Keep a checkpoint's original
code, dependencies and device when exact numerical resume matters; changing the
runtime does not carry a bitwise-resume guarantee.

## Start the interface

From the repository root:

```bash
# A clean CPU installation is useful for learning, smoke tests and tiny models.
LLMLAB_DEVICE=cpu bash gui/start.sh

# On a machine with a compatible NVIDIA driver:
LLMLAB_DEVICE=cuda bash gui/start.sh
```

The launcher creates `.venv`, installs both model and API dependencies, checks
versions and architecture configuration APIs, starts the API, waits for health,
then starts the frontend. An existing backend on the selected port is reused;
it is not stopped when this launcher exits. Other servers started by the
launcher are stopped on exit. Application startup failure prevents frontend
startup instead of leaving a misleading ready message.

Open `http://127.0.0.1:5174`. The default API is `http://127.0.0.1:8000`.
Override ports with `LLMLAB_API_PORT` and `LLMLAB_UI_PORT`; the frontend proxy
uses the selected API port. Local servers bind to loopback.

`LLMLAB_DEVICE=auto` is the default: it selects CUDA when `nvidia-smi` detects a
working NVIDIA GPU, otherwise CPU. Use an explicit device for reproducible
setup. `LLMLAB_PYTHON` chooses the Python executable for a new venv;
`LLMLAB_VENV` chooses its directory. Your old `gui/backend/venv` is not modified
unless you explicitly select it.

## Install or check the runtime separately

```bash
bash scripts/bootstrap.sh --device cpu
.venv/bin/python scripts/check_environment.py --device cpu
```

The bootstrap uses uv if available, otherwise pip. It installs Torch from the
selected CPU/CUDA index first, then the constrained application dependencies.
The diagnostic reports incompatible versions and missing template APIs before
a training run is started. A successful diagnostic verifies environment APIs,
not model learning, gradients, optimized kernels or architecture fidelity.

Optional 8-bit optimizers are separate from the standard AdamW installation:

```bash
.venv/bin/python -m pip install -c constraints.txt -r requirements-gpu.txt
```

Flash Linear Attention, causal-conv1d and other optimized kernels are not
installed implicitly. Reference fallbacks remain available; use measured
benchmarks and numerical comparisons before enabling optimized execution.

## Validate locally

Install the test dependency into the same environment:

```bash
.venv/bin/python -m pip install -c constraints.txt -r requirements-dev.txt
HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
HF_HUB_OFFLINE=1 .venv/bin/python tests/smoke_test.py
HF_HUB_OFFLINE=1 .venv/bin/python tests/export_fidelity_test.py
npm --prefix gui/frontend ci
npm --prefix gui/frontend test
npm --prefix gui/frontend run build
```

The fixtures create local tokenizers and data; no model weights or datasets are
needed. Package installation itself needs network access. Use `HF_HOME`,
`HF_DATASETS_CACHE` and `MPLCONFIGDIR` to choose writable cache directories.
GPU acceptance checks are separate; the lifecycle suite supports
`LLMLAB_TEST_DEVICE=cuda` and fails when CUDA is unavailable.

The CI workflow runs the CPU suites, export checks, and frontend checks. A green
CPU workflow does not certify experimental architectures, GPU throughput or a
published Hub round trip.

## Container

The Dockerfile uses the same Torch and constrained dependency baseline and
includes scripts, examples and recipes. Build the GPU image with:

```bash
docker build -t llm-lab:local .
```

A CPU build uses the same Python base and selects the CPU wheel:

```bash
docker build --build-arg TORCH_CHANNEL=cpu -t llm-lab:cpu .
```

Run the resulting image with `-p 127.0.0.1:8000:8000 -v "$PWD/data:/app/data"`, adding
`--gpus all` for the GPU image. The container serves the built UI on port 8000.
Keep the mounted data directory when replacing the container. Follow the
[first-run guide](first-run.md) after opening the interface, or choose the
[supplied-data learning experiment](learning-path.md).

## Browser and frontend tooling

The frontend uses Node 20+ and Tailwind 4. The custom theme is loaded explicitly
through `@config`; PostCSS uses `@tailwindcss/postcss`. Browser requirements follow
[the official Tailwind upgrade guide](https://tailwindcss.com/docs/upgrade-guide):
Safari 16.4+, Chrome 111+ or Firefox 128+.
