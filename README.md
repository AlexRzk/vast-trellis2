# vast-trellis2

Adaptive installer, repair tool and launcher for [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2) on Vast.ai.

The goal is to turn a fresh Vast GPU instance into a reproducible TRELLIS.2 image-to-3D machine without manually guessing CUDA, build concurrency, RAM limits, attention backends, Hugging Face authentication or Gradio launch commands.

It is built around the failure modes that commonly appear on rented GPU containers:

- `free -h` showing the host's RAM while the container has a much smaller **cgroup limit**
- Linux terminating `python app.py` with a bare **`Killed`** because system RAM was exhausted
- RTX 50 / Blackwell requiring a CUDA toolkit new enough to compile `sm_120`
- FlashAttention source builds consuming too much RAM
- xFormers wheels built for a different PyTorch/Python combination
- stale `/tmp/extensions/*` directories breaking extension retries
- Hugging Face `401` vs `403 GatedRepoError`
- Facebook DINOv3 access still pending while an official Meta checkpoint is available locally
- `DINOv3ViTModel object has no attribute layer` after Transformers layout changes
- missing/unreadable `assets/hdri/*.exr`
- running `app.py` in Vast's default `(main)` environment instead of the TRELLIS environment
- SSH tunnels reporting `Connection refused` simply because Gradio is not listening yet

## One-command start

On a fresh Vast.ai instance:

```bash
git clone https://github.com/AlexRzk/vast-trellis2.git
cd vast-trellis2
bash install.sh
vast-trellis2 wizard
```

The wizard inspects the machine first, proposes an adaptive profile, then installs TRELLIS.2 in resumable stages.

## What the wizard detects

`vast-trellis2 detect` reports and uses:

- every visible NVIDIA GPU
- selectable GPU index on multi-GPU instances
- VRAM per GPU
- compute capability
- Blackwell / RTX 50 detection
- NVIDIA driver visibility and driver CUDA ceiling
- actual CUDA Toolkit / `nvcc`
- CPU thread count
- host RAM
- **effective cgroup RAM limit**
- current effective RAM headroom
- swap state
- free workspace disk

The cgroup check matters on Vast/Docker: a machine may show hundreds of GiB in `free -h` while the container is allowed to consume much less.

## Adaptive build concurrency

Extension builds are **serialized** so that nvdiffrast, nvdiffrec, CuMesh, O-Voxel and FlexGEMM never compile simultaneously. Each individual stage may still use several compiler workers.

Default `MAX_JOBS` based on **currently available effective RAM**:

| Available effective RAM | `MAX_JOBS` |
|---:|---:|
| < 64 GiB | 1 |
| 64–95 GiB | 2 |
| 96–127 GiB | 3 |
| 128–191 GiB | 4 |
| 192–255 GiB | 6 |
| 256+ GiB | 8 |

The value is capped by CPU thread count. `NVCC_THREADS=1` is used to avoid multiplying memory pressure inside every compiler worker.

If an extension fails, the wizard classifies common errors, cleans the failed `/tmp/extensions/...` checkout and automatically retries the stage once with one compiler job.

## CUDA / PyTorch profiles

Current tested profiles:

| CUDA Toolkit | PyTorch profile | Notes |
|---|---|---|
| 12.8 | PyTorch 2.11 + `cu128` | preferred for RTX 50 / Blackwell |
| 12.4–12.7 | PyTorch 2.6 + `cu124` | profile for older supported GPUs |

Blackwell requires CUDA Toolkit 12.8 in this wizard. A runtime-only CUDA image is not enough: TRELLIS.2 compiles custom CUDA extensions.

The wizard defaults to **xFormers** rather than compiling FlashAttention. This avoids one of the heaviest and most failure-prone source builds on rented machines while TRELLIS.2 still gets a CUDA attention backend.

## RAM recommendations

System RAM matters because TRELLIS.2 low-VRAM mode moves submodels between CPU and GPU.

Practical guidance used by the wizard:

- **<48 GiB effective RAM:** high risk of a bare `Killed`
- **48–63 GiB:** tight; use 512 and low-VRAM mode
- **64+ GiB:** recommended baseline
- **96/128+ GiB:** comfortable for builds, caching and larger generation modes

Check the effective number with:

```bash
vast-trellis2 detect
vast-trellis2 status
```

The manager also shows `/sys/fs/cgroup/memory.events` when available. A non-zero `oom_kill` counter is strong evidence that a bare `Killed` came from system/cgroup RAM exhaustion rather than GPU VRAM.

## Hugging Face token

The wizard explicitly offers authentication, or run:

```bash
vast-trellis2 auth
```

Paste a Hugging Face **read token** when prompted. Input is hidden. The token is passed to `huggingface_hub.login()` and is **not written to `config.env`, logs, Git remotes or this repository**.

TRELLIS.2 currently needs gated auxiliary models including:

```text
facebook/dinov3-vitl16-pretrain-lvd1689m
briaai/RMBG-2.0
```

A valid token does not bypass gating; the Hugging Face account must also have accepted/received access.

### 401 vs 403

- `401`: authentication/token problem
- `403 ... awaiting a review`: authentication works, but the account is not approved for that gated repo yet

The wizard explains the distinction instead of trying to repair unrelated Python packages.

## Official Meta DINOv3 local fallback

If Hugging Face DINOv3 access is still pending but you legitimately have Meta's official checkpoint:

```text
dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

copy it to the Vast instance and run:

```bash
vast-trellis2 dino-local /workspace/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

The included converter creates a local Transformers-compatible model and configures TRELLIS.2 through `DINOV3_MODEL_PATH`.

This is an **official-checkpoint conversion path**, not a gated-repository bypass. You still need legitimate access to the Meta checkpoint/license.

## Installation stages

The wizard performs installation in this order:

1. system build packages
2. Microsoft TRELLIS.2 recursive clone
3. dedicated Python 3.10 Conda environment
4. GPU-appropriate PyTorch CUDA wheel
5. TRELLIS basic Python dependencies
6. OpenCV EXR-compatible pin
7. xFormers from the selected PyTorch CUDA wheel index
8. `nvdiffrast`
9. `nvdiffrec`
10. `CuMesh`
11. `o-voxel`
12. `FlexGEMM`
13. TRELLIS/DINO compatibility patches
14. HDRI/OpenEXR validation
15. import/GPU smoke test
16. Hugging Face authentication/access checks
17. optional model pre-download
18. doctor check

Every build stage logs under:

```text
/workspace/.vast-trellis2/logs
```

Successful stages receive markers under:

```text
/workspace/.vast-trellis2/stages
```

so rerunning:

```bash
vast-trellis2 install
```

resumes instead of reinstalling PyTorch and rebuilding every CUDA extension.

Force all stages again:

```bash
vast-trellis2 install --force
```

Or clear only build caches/stage markers while preserving downloaded model caches:

```bash
vast-trellis2 reset-build
```

## Model download / cache

Run:

```bash
vast-trellis2 download
```

The downloader uses Hugging Face/Xet and enables:

```text
HF_XET_HIGH_PERFORMANCE=1
```

Default cache:

```text
/workspace/huggingface
```

It pre-caches `microsoft/TRELLIS.2-4B` and attempts DINOv3/RMBG only when your account can access them. A configured local DINOv3 is left local.

## Running TRELLIS.2

Foreground, best for initial debugging:

```bash
vast-trellis2 run
```

Background lifecycle:

```bash
vast-trellis2 start
vast-trellis2 status
vast-trellis2 logs
vast-trellis2 stop
vast-trellis2 restart
```

The manager always activates the dedicated `trellis2` environment and exports the runtime settings, so you do not accidentally start `app.py` from Vast's default `(main)` environment.

Runtime defaults include:

```text
ATTN_BACKEND=xformers
SPARSE_ATTN_BACKEND=xformers
OPENCV_IO_ENABLE_OPENEXR=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The background wrapper also records the Python exit status. Exit `137` is surfaced as a likely RAM/cgroup OOM instead of only looking like a dead Gradio page.

## Adaptive runtime configuration

The wizard selects a recommended UI resolution and low-VRAM mode based on detected resources. On multi-GPU machines it asks which GPU TRELLIS should use through `CUDA_VISIBLE_DEVICES`.

Typical choices:

- 24 GB VRAM / tight RAM: 512, low-VRAM on
- 32 GB VRAM + 64+ GB effective RAM: 1024 offered as default
- 64/80 GB VRAM: low-VRAM may be disabled for speed

You can override the wizard values. They persist in:

```text
/workspace/.vast-trellis2/config.env
```

The patcher makes these settings available to the upstream app through environment variables without hardcoding a machine-specific configuration into TRELLIS itself.

## SSH tunnel

Print the exact template with:

```bash
vast-trellis2 tunnel
```

On your **local computer**, insert the host and SSH port shown by **Vast.ai → Connect**:

```powershell
ssh -N -L 7860:127.0.0.1:7860 -p YOUR_VAST_SSH_PORT root@YOUR_VAST_HOST
```

Then open:

```text
http://127.0.0.1:7860
```

If SSH repeatedly prints:

```text
channel 3: open failed: connect failed: Connection refused
```

that normally means the SSH tunnel itself is alive but nothing is listening on remote port 7860 yet. Check:

```bash
vast-trellis2 status
vast-trellis2 logs
```

## Doctor / automatic repair

Check only:

```bash
vast-trellis2 doctor
```

Apply safe repairs:

```bash
vast-trellis2 doctor --fix
```

Deep DINO/TRELLIS feature-extractor test:

```bash
vast-trellis2 doctor --deep
```

The doctor checks:

- NVIDIA runtime
- CUDA Toolkit
- effective cgroup RAM
- recorded cgroup OOM kills
- workspace disk
- Conda environment
- PyTorch CUDA
- xFormers
- O-Voxel
- CuMesh
- FlexGEMM
- nvdiffrast
- TRELLIS imports
- OpenEXR/HDRIs
- DINOv3 access/local override
- RMBG access
- source compatibility patches
- recent runtime crash signatures

`--fix` can reapply source patches, restore HDRIs, pin OpenCV and reinstall xFormers from the saved CUDA wheel index. It intentionally does **not** attempt to bypass Hugging Face licensing/gating.

## Crash explanations

For the latest runtime log:

```bash
vast-trellis2 explain
```

Or another log:

```bash
vast-trellis2 explain /path/to/log
```

Recognized cases include:

### Bare `Killed` / exit 137

Usually Linux/cgroup system-RAM OOM, not VRAM.

```text
* Running on local URL: http://127.0.0.1:7860
Killed
```

Use more effective RAM, lower resolution, keep low-VRAM enabled and stop competing workloads.

### `CUDA out of memory`

GPU VRAM exhaustion. Lower resolution or stop competing GPU processes.

### DINOv3 `403 GatedRepoError`

The token may be valid but the HF account is not approved yet. Request access or use the official local Meta DINO converter.

### RMBG-2.0 `403`

Accept/request the model's Hugging Face access and authenticate again.

### `DINOv3ViTModel object has no attribute layer`

Transformers layout mismatch. Run:

```bash
vast-trellis2 doctor --fix
```

### `cv2.cvtColor ... !_src.empty()`

HDRI/OpenEXR loading failed. Run:

```bash
vast-trellis2 doctor --fix
```

### `ModuleNotFoundError: gradio` or another dependency

Most often `app.py` was run directly from Vast's default Python environment. Use:

```bash
vast-trellis2 run
```

instead.

### `unsupported gpu architecture 'compute_120'` / `sm_120`

The CUDA toolkit is too old for RTX 50 / Blackwell. Use a CUDA 12.8 **devel/toolkit** Vast image.

## Commands

```text
vast-trellis2                    interactive menu
vast-trellis2 wizard             complete setup wizard
vast-trellis2 detect             hardware + cgroup RAM + adaptive recommendations
vast-trellis2 install            install/resume with detected defaults
vast-trellis2 install --force    rerun all install stages
vast-trellis2 auth               Hugging Face login/token paste
vast-trellis2 download           pre-cache model weights
vast-trellis2 dino-local PATH    convert/configure official local Meta DINOv3
vast-trellis2 run                foreground Gradio
vast-trellis2 start              background Gradio
vast-trellis2 launch             alias for background start
vast-trellis2 stop               stop background process
vast-trellis2 restart            restart background process
vast-trellis2 status             GPU/RAM/cgroup/process status
vast-trellis2 logs               follow runtime log
vast-trellis2 explain            diagnose latest crash
vast-trellis2 doctor             installation/runtime checks
vast-trellis2 doctor --fix       safe automatic repairs
vast-trellis2 doctor --deep      deep DINO/TRELLIS load test
vast-trellis2 repair             alias for doctor --fix
vast-trellis2 reset-build        clear extension caches/stage markers
vast-trellis2 tunnel             SSH tunnel template
vast-trellis2 config             show saved adaptive config
```

## Persistent paths

Defaults:

```text
State/config:  /workspace/.vast-trellis2
Logs:          /workspace/.vast-trellis2/logs
Stages:        /workspace/.vast-trellis2/stages
TRELLIS repo:  /workspace/TRELLIS.2
HF cache:      /workspace/huggingface
```

The local converted DINOv3 model is stored under the state directory. Override the workspace with `VAST_TRELLIS_BASE` when required.

## Design choices

- Extension stages are not launched simultaneously; only compiler workers inside one stage are parallelized.
- Failed parallel builds are automatically retried once with a single compiler job.
- Effective cgroup RAM, not just host RAM, drives concurrency decisions.
- Hugging Face tokens are never committed or stored in `config.env`.
- Gated repositories are not bypassed.
- Gradio binds to `127.0.0.1` by default; use SSH forwarding rather than exposing the UI publicly.
- Upstream TRELLIS.2 remains a separate checkout under `/workspace/TRELLIS.2`; this project manages, patches and launches it.

## Upstream

TRELLIS.2 is developed by Microsoft and distributed under its upstream license. This repository is an installation/operations helper and is not an official Microsoft project.
