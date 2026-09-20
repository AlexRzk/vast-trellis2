# vast-trellis2

Hardware-aware installer, recovery wizard and launcher for [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2), designed especially for ephemeral GPU machines such as Vast.ai.

The goal is simple: **do not blindly run one install command on every GPU**. The wizard detects the machine first, chooses conservative build parallelism, selects an appropriate PyTorch/CUDA wheel stack, handles Hugging Face authentication, installs TRELLIS.2 in resumable stages, and explains failures instead of leaving a half-broken environment.

## Quick start

On a fresh Ubuntu/Vast.ai instance:

```bash
git clone https://github.com/AlexRzk/vast-trellis2.git
cd vast-trellis2
chmod +x install.sh vast-trellis2
./install.sh
vast-trellis2 wizard
```

You can also run the manager directly from the repository:

```bash
./vast-trellis2 wizard
```

## Commands

```text
vast-trellis2 wizard       Interactive first-run wizard
vast-trellis2 detect       Show GPU, VRAM, RAM, CPU, driver, CUDA and chosen defaults
vast-trellis2 show         Show saved configuration
vast-trellis2 install      Clone/update TRELLIS.2 and install all required components
vast-trellis2 download     Authenticate to Hugging Face and download TRELLIS.2-4B
vast-trellis2 launch       Start the TRELLIS.2 Gradio application
vast-trellis2 status       Show process/GPU state
vast-trellis2 logs         Follow the application log
vast-trellis2 doctor       Diagnose the environment and extension imports
vast-trellis2 repair       Apply safe recovery actions after a failed build
vast-trellis2 reset-build  Remove extension build caches, preserving model downloads
```

## What is detected automatically

- NVIDIA GPU model and count
- per-GPU VRAM and compute capability
- host driver and CUDA compatibility reported by `nvidia-smi`
- installed CUDA toolkit (`nvcc`) when present
- system RAM and currently available RAM
- CPU thread count
- free disk space
- Blackwell (`sm_120`) vs Ada/Hopper/Ampere and older architectures

From those values the wizard computes `MAX_JOBS`, `CMAKE_BUILD_PARALLEL_LEVEL`, `NINJA_NUM_JOBS`, `TORCH_CUDA_ARCH_LIST`, the PyTorch CUDA wheel channel and the default attention backend.

### Why build jobs are deliberately conservative

CUDA extensions can consume several gigabytes of RAM **per compiler process**. A command such as:

```bash
MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="12.0" pip install flash-attn==2.7.3 --no-build-isolation
```

can exhaust RAM or a container memory limit even when the GPU itself has 32 GB of VRAM. `vast-trellis2` therefore estimates build concurrency from **available system RAM**, not GPU VRAM, and caps Blackwell compilation more aggressively.

Typical automatic limits:

| Available system RAM | General CUDA builds | Blackwell / flash-attn |
|---:|---:|---:|
| < 16 GB | 1 | 1 |
| 16–31 GB | 2 | 1 |
| 32–63 GB | 4 | 2 |
| 64+ GB | up to 6 | up to 2 |

You can override the result in the wizard.

## RTX 50 / Blackwell behavior

For compute capability `12.0` the wizard prefers a **cu128 PyTorch stack** and defaults to the `xformers` attention backend. This avoids making the official `flash-attn==2.7.3` source build a mandatory step on a fresh RTX 5090.

If you explicitly choose `flash_attn`, the wizard compiles it **sequentially with a reduced job count** and preserves the full log. A failure does not erase the rest of the TRELLIS installation; `vast-trellis2 repair` can switch back to `xformers` and continue.

For older supported NVIDIA GPUs the wizard can use flash-attn when appropriate. TRELLIS.2 itself currently requires at least 24 GB GPU memory for its documented workflow; the wizard warns below that threshold rather than pretending the machine is supported.

## Hugging Face token

`vast-trellis2 wizard` asks whether you want to paste a Hugging Face token. Input is hidden. The token is stored with mode `0600` under the state directory and is never printed by `show`, `doctor`, or logs.

You can add/change it later:

```bash
vast-trellis2 download
```

The downloader uses `huggingface_hub` and resumes existing files. Model data is kept outside the Python environment so rebuilding the environment does not force a re-download.

## Persistent paths

Defaults are selected for Vast.ai-style machines but can be overridden:

```text
State:       /workspace/.vast-trellis2
TRELLIS:     /workspace/TRELLIS.2
Virtualenv:  /workspace/.venvs/trellis2
HF cache:    /workspace/huggingface
Model cache: /workspace/huggingface/hub
Logs:        /workspace/.vast-trellis2/logs
```

If `/workspace` is unavailable, `$HOME` is used.

## Installation stages and resume

`install` checkpoints every successful stage:

1. system packages
2. Python environment + PyTorch
3. TRELLIS basic Python dependencies
4. `nvdiffrast`
5. `nvdiffrec/renderutils`
6. `CuMesh`
7. `o-voxel`
8. `FlexGEMM`
9. attention backend
10. import/GPU smoke test

If stage 6 fails, rerunning `vast-trellis2 install` resumes at stage 6 instead of reinstalling PyTorch and every previous extension.

Use `--force` to redo all Python stages:

```bash
vast-trellis2 install --force
```

## Error handling

Every build writes a dedicated file under `logs/build-*.log`. On failure the wizard prints a short diagnosis and the exact log path.

Recognized cases include:

- `Killed`, exit 137, OOM / out-of-memory: reduce build jobs to 1 and clear the failed build cache
- `unsupported gpu architecture`, `sm_120`: CUDA/toolchain or package is too old for Blackwell
- `CUDA driver version is insufficient`: container runtime/toolkit is newer than the host driver can support
- `nvcc not found` / `CUDA_HOME`: install or select a CUDA **toolkit**, not only a runtime image
- flash-attn compilation failure: fall back to `xformers` without throwing away TRELLIS
- no GPU visible: verify the Vast.ai instance/container was started with NVIDIA GPU access
- less than 24 GB VRAM: warn that the machine is outside TRELLIS.2's documented minimum

`vast-trellis2 repair` reads the most recent failed-build marker and applies the non-destructive fix it can safely infer.

## Useful environment overrides

```bash
VAST_TRELLIS_STATE_DIR=/workspace/.vast-trellis2
TRELLIS_DIR=/workspace/TRELLIS.2
TRELLIS_VENV=/workspace/.venvs/trellis2
HF_HOME=/workspace/huggingface
MAX_JOBS=2
TORCH_CUDA_ARCH_LIST=12.0
ATTN_BACKEND=xformers
SPARSE_ATTN_BACKEND=xformers
```

Explicit environment values take precedence over auto-detection.

## Suggested Vast.ai base images

For RTX 50-series machines, use an image that includes a recent CUDA toolkit, ideally CUDA 12.8 or newer. `nvidia-smi` showing `CUDA Version: 12.8` describes the maximum CUDA runtime supported by the driver; `nvcc --version` confirms the actual toolkit available for compiling extensions.

The wizard checks both values separately.

## Security

- Hugging Face tokens are read with hidden input and stored `0600`.
- No token is embedded in Git remotes, command-line output, generated config, or logs.
- The Gradio server binds to `0.0.0.0` only when launched explicitly.

## Upstream

This project does not replace TRELLIS.2. It automates installation and recovery around the upstream repository. The upstream code, weights and licenses remain under their respective projects.