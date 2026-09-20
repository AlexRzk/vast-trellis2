# shellcheck shell=bash
# Installation, build-stage recovery, Hugging Face and DINOv3 helpers.

ensure_env_and_torch() {
  heading "Python / PyTorch"
  ensure_conda
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.10
  fi
  conda activate "$ENV_NAME"
  python -m pip install -U pip setuptools wheel packaging ninja >/dev/null
  info "Installing PyTorch $TORCH_VERSION from $TORCH_CUDA_TAG ..."
  python -m pip install -U "torch==$TORCH_VERSION" torchvision --index-url "$TORCH_INDEX"
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
PY
  ok "PyTorch CUDA runtime works."
}

ensure_xformers() {
  heading "xFormers attention backend"
  info "Installing xFormers from the same $TORCH_CUDA_TAG wheel index ..."
  python -m pip install -U xformers --index-url "$TORCH_INDEX"
  if ! python -m xformers.info >"$LOG_DIR/xformers-info.log" 2>&1; then
    cat "$LOG_DIR/xformers-info.log" >&2
    die "xFormers installed but its extensions do not load. See $LOG_DIR/xformers-info.log"
  fi
  if grep -qiE "can't load|not built with CUDA|build info.*unavailable" "$LOG_DIR/xformers-info.log"; then
    cat "$LOG_DIR/xformers-info.log" >&2
    die "xFormers/PyTorch CUDA mismatch. See $LOG_DIR/xformers-info.log"
  fi
  python - <<'PY'
import torch, xformers
print("Torch:", torch.__version__)
print("xFormers:", xformers.__version__)
PY
  ok "xFormers backend is usable."
}

configure_build_env() {
  export CUDA_HOME="$CUDA_HOME_SELECTED"
  export PATH="$CUDA_HOME/bin:$PATH"
  export MAX_JOBS="$BUILD_JOBS"
  export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
  export NVCC_THREADS="$NVCC_THREADS"
  if [[ -n "$ARCH_LIST" ]]; then export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"; else unset TORCH_CUDA_ARCH_LIST 2>/dev/null || true; fi
}

stage_tmp_dir() {
  case "$1" in
    nvdiffrast) printf '/tmp/extensions/nvdiffrast' ;;
    nvdiffrec) printf '/tmp/extensions/nvdiffrec' ;;
    cumesh) printf '/tmp/extensions/CuMesh' ;;
    o-voxel) printf '/tmp/extensions/o-voxel' ;;
    flexgemm) printf '/tmp/extensions/FlexGEMM' ;;
    *) printf '' ;;
  esac
}

explain_build_failure() {
  local stage="$1" log="$2" rc="${3:-1}"
  warn "Build stage '$stage' failed (exit $rc)."
  if grep -qiE "Killed|out of memory|cannot allocate memory|cc1plus.*killed|exit code 137" "$log" || [[ "$rc" == 137 ]]; then
    warn "Likely system/cgroup RAM OOM. This stage will retry with one compiler job."
  elif grep -qiE "unsupported gpu architecture|sm_120|compute_120" "$log"; then
    warn "CUDA toolkit/package does not support this GPU architecture. RTX 50/Blackwell needs CUDA 12.8+."
  elif grep -qiE "ModuleNotFoundError: No module named ['\"]torch|No module named torch" "$log"; then
    warn "Build ran outside the TRELLIS conda environment or PyTorch is missing."
  elif grep -qiE "already exists and is not an empty directory|destination path .* already exists" "$log"; then
    warn "A stale /tmp/extensions checkout exists. The retry removes the stage checkout."
  elif grep -qiE "CUDA_HOME|nvcc.*not found|No such file or directory.*nvcc|No CUDA runtime is found" "$log"; then
    warn "CUDA Toolkit/devel files are missing or CUDA_HOME is wrong."
  elif grep -qiE "CUDA driver version is insufficient|forward compatibility" "$log"; then
    warn "Host driver and container CUDA toolkit are incompatible."
  else
    warn "Unclassified build failure; the final log tail follows."
  fi
  tail -n 40 "$log" >&2 || true
}

run_setup_stage_once() {
  local stage="$1" jobs="$2" log="$3" tmp rc
  tmp="$(stage_tmp_dir "$stage")"
  [[ -z "$tmp" ]] || rm -rf "$tmp"
  set +e
  (
    cd "$REPO_DIR"
    export MAX_JOBS="$jobs" CMAKE_BUILD_PARALLEL_LEVEL="$jobs" NVCC_THREADS=1
    if [[ -n "$ARCH_LIST" ]]; then export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"; fi
    bash ./setup.sh "--$stage"
  ) 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

run_setup_stage() {
  local stage="$1" log="$LOG_DIR/build-$stage.log" rc
  heading "Build: $stage"
  if run_setup_stage_once "$stage" "$BUILD_JOBS" "$log"; then ok "$stage installed."; return 0; fi
  rc=$?
  explain_build_failure "$stage" "$log" "$rc"
  warn "Automatic recovery: cleaning the stage and retrying with MAX_JOBS=1 ..."
  if run_setup_stage_once "$stage" 1 "$log.retry1"; then ok "$stage installed on conservative retry."; return 0; fi
  rc=$?
  explain_build_failure "$stage" "$log.retry1" "$rc"
  die "$stage failed after automatic retry. Run 'vast-trellis2 doctor' and inspect $log*"
}

install_trellis_dependencies() {
  activate_env
  configure_build_env
  if ! stage_done basic; then
    heading "TRELLIS Python dependencies"
    set +e
    (cd "$REPO_DIR" && bash ./setup.sh --basic) 2>&1 | tee "$LOG_DIR/build-basic.log"
    local basic_rc=${PIPESTATUS[0]}
    set -e
    (( basic_rc == 0 )) || { explain_build_failure basic "$LOG_DIR/build-basic.log" "$basic_rc"; die "Basic dependency stage failed."; }

    # OpenCV 5 builds can fail to decode TRELLIS' EXR HDRIs on common Vast images.
    # Pin the known-good 4.12 wheel, then ensure current DINOv3 Transformers support.
    python -m pip install -U "opencv-python-headless==4.12.0.88" transformers huggingface_hub hf_xet
    python - <<'PY'
from transformers import DINOv3ViTModel
print("Transformers DINOv3 support: OK")
PY
    mark_stage basic
  else
    info "[basic] already complete; skipping."
  fi

  if ! stage_done xformers; then ensure_xformers; mark_stage xformers; else info "[xformers] already complete; skipping."; fi

  # Deliberately SERIAL across extensions. Running several CUDA extension builds at
  # once is a common cause of Vast OOM kills. Each one can still use BUILD_JOBS.
  local stage
  for stage in nvdiffrast nvdiffrec cumesh o-voxel flexgemm; do
    if stage_done "$stage"; then
      info "[$stage] already complete; skipping."
    else
      run_setup_stage "$stage"
      mark_stage "$stage"
    fi
  done
}

apply_patches() {
  [[ -f "$PATCH_TOOL" ]] || die "Patch utility missing: $PATCH_TOOL"
  activate_env
  python "$PATCH_TOOL" "$REPO_DIR"
}

ensure_hdri() {
  heading "HDRI / OpenEXR"
  local f base="https://raw.githubusercontent.com/microsoft/TRELLIS.2/main/assets/hdri"
  for f in forest sunset courtyard; do
    if [[ ! -s "$REPO_DIR/assets/hdri/$f.exr" || $(stat -c%s "$REPO_DIR/assets/hdri/$f.exr" 2>/dev/null || echo 0) -lt 10000 ]]; then
      mkdir -p "$REPO_DIR/assets/hdri"
      curl -fL "$base/$f.exr" -o "$REPO_DIR/assets/hdri/$f.exr"
    fi
  done
  activate_env
  OPENCV_IO_ENABLE_OPENEXR=1 python - "$REPO_DIR" <<'PY'
import os, sys, cv2
repo=sys.argv[1]
for name in ("forest", "sunset", "courtyard"):
    p=os.path.join(repo,"assets","hdri",name+".exr")
    im=cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"OpenCV cannot read {p}")
    print(name, im.shape)
PY
  ok "OpenEXR assets are readable."
}

smoke_test() {
  activate_env; configure_build_env
  heading "TRELLIS import smoke test"
  (cd "$REPO_DIR" && ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable in PyTorch"
print("Torch", torch.__version__, "CUDA", torch.version.cuda)
print("GPU", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
import xformers
import nvdiffrast.torch
import o_voxel
import cumesh
import flex_gemm
import trellis2
print("TRELLIS core imports: OK")
PY
  )
  ok "TRELLIS.2 core imports succeed."
}

cmd_auth() {
  load_config
  activate_env
  python -m pip install -q -U huggingface_hub hf_xet
  export HF_HOME="$HF_HOME_DIR"
  heading "Hugging Face authentication"
  if command -v hf >/dev/null 2>&1 && hf auth whoami >/dev/null 2>&1; then
    info "Current Hugging Face identity:"
    hf auth whoami || true
    if ! prompt_yes_no "Replace the saved token?" n; then return 0; fi
  fi
  local token
  printf 'Paste a Hugging Face READ token (input hidden): '
  IFS= read -r -s token || true
  printf '\n'
  [[ -n "$token" ]] || die "No token entered."
  HF_TOKEN="$token" python - <<'PY'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("Hugging Face token saved in the local Hugging Face cache.")
PY
  unset token
  command -v hf >/dev/null 2>&1 && hf auth whoami || true
}

hf_probe() {
  local repo="$1" filename="$2"
  activate_env
  REPO_ID="$repo" FILE_NAME="$filename" python - <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
try:
    print(hf_hub_download(os.environ['REPO_ID'], os.environ['FILE_NAME']))
except Exception as e:
    print(f"{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PY
}

cmd_dino_local() {
  load_config
  local checkpoint="${1:-}"
  [[ -n "$checkpoint" ]] || checkpoint="$(prompt 'Path to official Meta dinov3_vitl16...pth checkpoint')"
  [[ -f "$checkpoint" ]] || die "Checkpoint not found: $checkpoint"
  activate_env
  [[ -f "$DINO_CONVERTER" ]] || die "DINO converter missing: $DINO_CONVERTER"
  local out="$STATE_DIR/models/dinov3_hf/vitl16_lvd1689m"
  mkdir -p "$(dirname "$out")"
  python "$DINO_CONVERTER" "$checkpoint" "$out"
  DINOV3_MODEL_PATH="$out"
  save_config
  ok "Local DINOv3 configured: $DINOV3_MODEL_PATH"
}

check_gated_access() {
  heading "Hugging Face gated model access"
  local dino_ok=0 rembg_ok=0
  if hf_probe facebook/dinov3-vitl16-pretrain-lvd1689m config.json >/dev/null 2>"$LOG_DIR/hf-dino.err"; then
    dino_ok=1; ok "DINOv3 Hugging Face access: OK"
  elif [[ -n "$DINOV3_MODEL_PATH" && -f "$DINOV3_MODEL_PATH/config.json" ]]; then
    dino_ok=1; ok "DINOv3: using local converted model at $DINOV3_MODEL_PATH"
  else
    warn "DINOv3 Hugging Face access is unavailable."
    sed -n '1,8p' "$LOG_DIR/hf-dino.err" >&2 || true
    warn "401 = bad/missing login; 403 awaiting review = token works but access is not approved."
    warn "Use official Meta ViT-L/16 checkpoint with: vast-trellis2 dino-local /path/checkpoint.pth"
  fi

  if hf_probe briaai/RMBG-2.0 config.json >/dev/null 2>"$LOG_DIR/hf-rmbg.err"; then
    rembg_ok=1; ok "RMBG-2.0 Hugging Face access: OK"
  else
    warn "RMBG-2.0 access is unavailable. Accept/request access on Hugging Face and rerun auth/download."
    sed -n '1,8p' "$LOG_DIR/hf-rmbg.err" >&2 || true
  fi
  (( dino_ok == 1 && rembg_ok == 1 ))
}

cmd_download() {
  load_config; activate_env
  export HF_HOME="$HF_HOME_DIR" HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DOWNLOAD_TIMEOUT=120
  local free_kb
  free_kb="$(df -Pk "$WORKSPACE" | awk 'NR==2 {print $4}')"
  if [[ "$free_kb" =~ ^[0-9]+$ ]] && (( free_kb < 31457280 )); then warn "Less than 30 GiB free; model cache may exhaust disk."; fi

  check_gated_access || warn "A gated dependency is unavailable; the main TRELLIS snapshot can still be cached."
  heading "Download TRELLIS.2-4B"
  python - <<'PY'
from huggingface_hub import snapshot_download
print("Downloading/caching microsoft/TRELLIS.2-4B ...")
print("Cached at:", snapshot_download("microsoft/TRELLIS.2-4B"))
PY

  heading "Download auxiliary models"
  if [[ -z "$DINOV3_MODEL_PATH" ]]; then
    set +e
    python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("facebook/dinov3-vitl16-pretrain-lvd1689m")
print("DINOv3 cached")
PY
    local drc=$?
    set -e
    (( drc == 0 )) || warn "DINOv3 not downloaded; obtain HF approval or use 'vast-trellis2 dino-local'."
  fi
  set +e
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("briaai/RMBG-2.0")
print("RMBG-2.0 cached")
PY
  local rrc=$?
  set -e
  (( rrc == 0 )) || warn "RMBG-2.0 not downloaded. Accept/request its Hugging Face access first."
}

cmd_install() {
  local force=0
  [[ "${1:-}" == "--force" ]] && force=1
  detect_gpus; detect_cuda; recommend_settings
  load_config
  (( force == 1 )) && { rm -rf "$STAGE_DIR"; mkdir -p "$STAGE_DIR"; }
  save_config

  ensure_system_deps
  ensure_repo
  if ! stage_done torch; then ensure_env_and_torch; mark_stage torch; else activate_env; info "[torch] already complete; skipping."; fi
  install_trellis_dependencies
  apply_patches
  ensure_hdri
  smoke_test
  mark_stage smoke
  save_config
  ok "Core TRELLIS.2 installation completed."
}

cmd_reset_build() {
  rm -rf /tmp/extensions "$HOME/.cache/torch_extensions" 2>/dev/null || true
  rm -rf "$STAGE_DIR"; mkdir -p "$STAGE_DIR"
  ok "Build caches and stage markers cleared; model/Hugging Face caches were preserved."
}
