# shellcheck shell=bash
# Shared hardware detection, state, configuration and environment helpers.

APP_NAME="vast-trellis2"
VERSION="0.2.0"

WORKSPACE="${VAST_TRELLIS_BASE:-/workspace}"
[[ -d "$WORKSPACE" && -w "$WORKSPACE" ]] || WORKSPACE="$HOME"
STATE_DIR="${VAST_TRELLIS_STATE_DIR:-$WORKSPACE/.vast-trellis2}"
CONFIG_FILE="$STATE_DIR/config.env"
LOG_DIR="$STATE_DIR/logs"
PID_FILE="$STATE_DIR/trellis.pid"
STAGE_DIR="$STATE_DIR/stages"
HF_HOME_DIR="${HF_HOME:-$WORKSPACE/huggingface}"
REPO_DIR="${VAST_TRELLIS_REPO_DIR:-$WORKSPACE/TRELLIS.2}"
ENV_NAME="${VAST_TRELLIS_ENV_NAME:-trellis2}"
PORT="${VAST_TRELLIS_PORT:-7860}"
SERVER_NAME="${VAST_TRELLIS_SERVER_NAME:-127.0.0.1}"
GPU_ID="${VAST_TRELLIS_GPU_ID:-0}"

PATCH_TOOL="$TOOL_ROOT/scripts/patch_trellis.py"
DINO_CONVERTER="$TOOL_ROOT/scripts/convert_dinov3_local.py"
DOCTOR_TOOL="${VAST_TRELLIS_DOCTOR:-$TOOL_ROOT/vast-trellis-doctor}"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$STAGE_DIR" "$HF_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

CUDA_HOME_SELECTED="${CUDA_HOME:-}"
CUDA_VERSION=""
TORCH_VERSION=""
TORCH_INDEX=""
TORCH_CUDA_TAG=""
BUILD_JOBS="${MAX_JOBS:-1}"
NVCC_THREADS="${NVCC_THREADS:-1}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"
ATTN_BACKEND="${ATTN_BACKEND:-xformers}"
SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-xformers}"
DEFAULT_RESOLUTION="${TRELLIS_DEFAULT_RESOLUTION:-512}"
LOW_VRAM="${TRELLIS_LOW_VRAM:-1}"
DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-}"

c_reset='\033[0m'; c_bold='\033[1m'; c_green='\033[32m'; c_yellow='\033[33m'; c_red='\033[31m'; c_cyan='\033[36m'; c_dim='\033[2m'
if [[ -n "${NO_COLOR:-}" ]]; then c_reset=''; c_bold=''; c_green=''; c_yellow=''; c_red=''; c_cyan=''; c_dim=''; fi
info() { printf '%b%s%b\n' "$c_cyan" "$*" "$c_reset"; }
ok() { printf '%b%s%b\n' "$c_green" "$*" "$c_reset"; }
warn() { printf '%b%s%b\n' "$c_yellow" "$*" "$c_reset" >&2; }
fail() { printf '%bERROR: %s%b\n' "$c_red" "$*" "$c_reset" >&2; }
die() { fail "$*"; exit 1; }
heading() { printf '\n%b== %s ==%b\n' "$c_bold" "$*" "$c_reset"; }
stage_done() { [[ -f "$STAGE_DIR/$1.ok" ]]; }
mark_stage() { touch "$STAGE_DIR/$1.ok"; }

prompt() {
  local label="$1" default="${2:-}" value
  if [[ -n "$default" ]]; then
    read -r -p "$label [$default]: " value || true
    printf '%s' "${value:-$default}"
  else
    read -r -p "$label: " value || true
    printf '%s' "$value"
  fi
}

prompt_yes_no() {
  local label="$1" default="${2:-y}" answer hint='y/N'
  [[ "$default" == y ]] && hint='Y/n'
  read -r -p "$label [$hint]: " answer || true
  answer="${answer:-$default}"
  [[ "$answer" =~ ^[Yy]$ ]]
}

human_gib() { awk -v b="$1" 'BEGIN { printf "%.1f GiB", b/1073741824 }'; }
human_mib() { awk -v m="$1" 'BEGIN { printf "%.1f GiB", m/1024 }'; }

meminfo_bytes() {
  local key="$1"
  awk -v k="$key" '$1==k":" {printf "%.0f", $2*1024}' /proc/meminfo
}

# Vast/Docker can expose host MemTotal while enforcing a much smaller cgroup limit.
# Always use the smaller number for installation/runtime decisions.
cgroup_limit_bytes() {
  local v total
  total="$(meminfo_bytes MemTotal)"
  if [[ -r /sys/fs/cgroup/memory.max ]]; then
    v="$(cat /sys/fs/cgroup/memory.max)"
    [[ "$v" == max ]] && { printf '%s' "$total"; return; }
    if [[ "$v" =~ ^[0-9]+$ ]]; then
      (( v < total )) && printf '%s' "$v" || printf '%s' "$total"
      return
    fi
  fi
  if [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    v="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
    if [[ "$v" =~ ^[0-9]+$ ]]; then
      (( v < total )) && printf '%s' "$v" || printf '%s' "$total"
      return
    fi
  fi
  printf '%s' "$total"
}

cgroup_current_bytes() {
  if [[ -r /sys/fs/cgroup/memory.current ]]; then cat /sys/fs/cgroup/memory.current; return; fi
  if [[ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then cat /sys/fs/cgroup/memory/memory.usage_in_bytes; return; fi
  printf '0'
}

effective_available_bytes() {
  local host_avail limit current cg_avail
  host_avail="$(meminfo_bytes MemAvailable)"
  limit="$(cgroup_limit_bytes)"
  current="$(cgroup_current_bytes)"
  if (( current > 0 && limit > current )); then
    cg_avail=$((limit-current))
    (( cg_avail < host_avail )) && printf '%s' "$cg_avail" || printf '%s' "$host_avail"
  else
    printf '%s' "$host_avail"
  fi
}

cpu_threads() { nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1'; }
driver_version() { nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | xargs; }
driver_cuda_max() { nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n1; }
version_ge() { awk -v a="$1" -v b="$2" 'BEGIN{split(a,A,"."); split(b,B,"."); if ((A[1]+0)>(B[1]+0) || ((A[1]+0)==(B[1]+0) && (A[2]+0)>=(B[2]+0))) exit 0; exit 1}'; }

GPU_NAMES=()
GPU_VRAM_MIB=()
GPU_CAPS=()
GPU_COUNT=0
MIN_VRAM_MIB=0
BLACKWELL=0

fallback_cap_for_name() {
  case "$1" in
    *RTX\ 50*|*RTX\ PRO\ 6000*Blackwell*) printf '12.0' ;;
    *B100*|*B200*|*GB200*) printf '10.0' ;;
    *H100*|*H200*) printf '9.0' ;;
    *RTX\ 40*|*L40*|*L4*) printf '8.9' ;;
    *RTX\ 30*|*A40*|*A6000*) printf '8.6' ;;
    *A100*) printf '8.0' ;;
    *) printf '' ;;
  esac
}

detect_gpus() {
  GPU_NAMES=(); GPU_VRAM_MIB=(); GPU_CAPS=(); GPU_COUNT=0; MIN_VRAM_MIB=0; BLACKWELL=0
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Start a Vast.ai NVIDIA GPU instance."
  local rows line idx name mem cap _driver cap_major
  rows="$(nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "$rows" ]]; then
    rows="$(nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null || true)"
  fi
  [[ -n "$rows" ]] || die "NVIDIA runtime is present but no GPU is visible."
  while IFS= read -r line; do
    IFS=',' read -r idx name mem cap _driver <<< "$line"
    idx="$(xargs <<<"${idx:-}")"; name="$(xargs <<<"${name:-unknown}")"; mem="$(xargs <<<"${mem:-0}")"; cap="$(xargs <<<"${cap:-}")"
    if [[ ! "$cap" =~ ^[0-9]+\.[0-9]+$ ]]; then cap="$(fallback_cap_for_name "$name")"; fi
    GPU_NAMES+=("$name"); GPU_VRAM_MIB+=("${mem%.*}"); GPU_CAPS+=("$cap"); ((GPU_COUNT+=1))
    if (( MIN_VRAM_MIB == 0 || ${mem%.*} < MIN_VRAM_MIB )); then MIN_VRAM_MIB="${mem%.*}"; fi
    cap_major="${cap%%.*}"
    if [[ "$cap_major" =~ ^[0-9]+$ ]] && (( cap_major >= 10 )); then BLACKWELL=1; fi
    [[ "$name" == *"RTX 50"* || "$name" == *"B200"* || "$name" == *"B100"* || "$name" == *"GB200"* ]] && BLACKWELL=1
  done <<< "$rows"
}

find_cuda_home() {
  local candidate version
  # Prefer a tested 12.8 toolkit whenever present, even if /usr/local/cuda points to 13.x.
  for candidate in /usr/local/cuda-12.8 "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-12.4; do
    [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]] || continue
    version="$($candidate/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1)"
    if (( BLACKWELL == 1 )); then
      [[ "$version" == 12.8* ]] && { printf '%s' "$candidate"; return 0; }
    elif [[ "$version" == 12.8* || "$version" == 12.7* || "$version" == 12.6* || "$version" == 12.5* || "$version" == 12.4* ]]; then
      printf '%s' "$candidate"; return 0
    fi
  done
  if command -v nvcc >/dev/null 2>&1; then
    candidate="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
    printf '%s' "$candidate"; return 0
  fi
  return 1
}

detect_cuda() {
  CUDA_HOME_SELECTED="$(find_cuda_home || true)"
  [[ -n "$CUDA_HOME_SELECTED" ]] || die "CUDA Toolkit (nvcc) not found. Use a Vast image with CUDA Toolkit/devel, not runtime-only."
  CUDA_VERSION="$($CUDA_HOME_SELECTED/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1)"
  [[ -n "$CUDA_VERSION" ]] || die "Could not parse nvcc version from $CUDA_HOME_SELECTED/bin/nvcc"
  local major minor hostmax
  major="${CUDA_VERSION%%.*}"; minor="${CUDA_VERSION#*.}"; minor="${minor%%.*}"
  hostmax="$(driver_cuda_max || true)"
  if [[ -n "$hostmax" ]] && ! version_ge "$hostmax" "$CUDA_VERSION"; then
    die "Host NVIDIA driver advertises CUDA $hostmax but selected nvcc is $CUDA_VERSION. Choose a compatible Vast image/driver-toolkit pair."
  fi
  if (( BLACKWELL == 1 )) && ! (( major == 12 && minor >= 8 )); then
    die "Blackwell GPU detected but CUDA Toolkit $CUDA_VERSION is selected. Use a Vast CUDA 12.8 devel image."
  fi
  if (( major == 12 && minor >= 8 )); then
    TORCH_VERSION="2.11.0"; TORCH_INDEX="https://download.pytorch.org/whl/cu128"; TORCH_CUDA_TAG="cu128"
  elif (( major == 12 && minor >= 4 )); then
    TORCH_VERSION="2.6.0"; TORCH_INDEX="https://download.pytorch.org/whl/cu124"; TORCH_CUDA_TAG="cu124"
  else
    die "CUDA Toolkit $CUDA_VERSION is outside the tested profile (12.4–12.8). Install/select CUDA 12.8; Blackwell requires it."
  fi
}

recommend_settings() {
  local limit avail limit_gib avail_gib threads jobs cap joined=''
  limit="$(cgroup_limit_bytes)"; avail="$(effective_available_bytes)"
  limit_gib=$((limit/1073741824)); avail_gib=$((avail/1073741824)); threads="$(cpu_threads)"

  if (( avail_gib < 64 )); then jobs=1
  elif (( avail_gib < 96 )); then jobs=2
  elif (( avail_gib < 128 )); then jobs=3
  elif (( avail_gib < 192 )); then jobs=4
  elif (( avail_gib < 256 )); then jobs=6
  else jobs=8
  fi
  (( jobs > threads )) && jobs="$threads"; (( jobs < 1 )) && jobs=1
  BUILD_JOBS="$jobs"; NVCC_THREADS=1

  for cap in "${GPU_CAPS[@]}"; do
    [[ -n "$cap" ]] || continue
    if [[ ";$joined;" != *";$cap;"* ]]; then joined="${joined:+$joined;}$cap"; fi
  done
  ARCH_LIST="$joined"
  ATTN_BACKEND="xformers"; SPARSE_ATTN_BACKEND="xformers"

  if (( MIN_VRAM_MIB >= 65536 )); then LOW_VRAM=0; else LOW_VRAM=1; fi
  if (( MIN_VRAM_MIB >= 32768 && limit_gib >= 64 )); then DEFAULT_RESOLUTION=1024; else DEFAULT_RESOLUTION=512; fi
}

load_config() {
  [[ -f "$CONFIG_FILE" ]] || return 0
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  REPO_DIR="${REPO_DIR:-$WORKSPACE/TRELLIS.2}"; ENV_NAME="${ENV_NAME:-trellis2}"
  PORT="${PORT:-7860}"; SERVER_NAME="${SERVER_NAME:-127.0.0.1}"; GPU_ID="${GPU_ID:-0}"
}

save_config() {
  umask 077
  cat > "$CONFIG_FILE" <<EOF
REPO_DIR=$(printf '%q' "$REPO_DIR")
ENV_NAME=$(printf '%q' "$ENV_NAME")
HF_HOME_DIR=$(printf '%q' "$HF_HOME_DIR")
CUDA_HOME_SELECTED=$(printf '%q' "$CUDA_HOME_SELECTED")
CUDA_VERSION=$(printf '%q' "$CUDA_VERSION")
TORCH_VERSION=$(printf '%q' "$TORCH_VERSION")
TORCH_INDEX=$(printf '%q' "$TORCH_INDEX")
TORCH_CUDA_TAG=$(printf '%q' "$TORCH_CUDA_TAG")
BUILD_JOBS=$(printf '%q' "$BUILD_JOBS")
NVCC_THREADS=$(printf '%q' "$NVCC_THREADS")
ARCH_LIST=$(printf '%q' "$ARCH_LIST")
ATTN_BACKEND=$(printf '%q' "$ATTN_BACKEND")
SPARSE_ATTN_BACKEND=$(printf '%q' "$SPARSE_ATTN_BACKEND")
DEFAULT_RESOLUTION=$(printf '%q' "$DEFAULT_RESOLUTION")
LOW_VRAM=$(printf '%q' "$LOW_VRAM")
DINOV3_MODEL_PATH=$(printf '%q' "$DINOV3_MODEL_PATH")
PORT=$(printf '%q' "$PORT")
SERVER_NAME=$(printf '%q' "$SERVER_NAME")
GPU_ID=$(printf '%q' "$GPU_ID")
EOF
}

print_memory_advice() {
  local limit avail lg
  limit="$(cgroup_limit_bytes)"; avail="$(effective_available_bytes)"; lg=$((limit/1073741824))
  printf 'Effective RAM limit: %s\n' "$(human_gib "$limit")"
  printf 'RAM available now:   %s\n' "$(human_gib "$avail")"
  if (( lg < 48 )); then warn "<48 GiB effective RAM: high risk of a bare Linux 'Killed'. Choose a 64 GiB+ instance.";
  elif (( lg < 64 )); then warn "RAM is tight. Prefer 512 and keep low-VRAM mode enabled.";
  elif (( lg < 96 )); then info "64+ GiB RAM: suitable baseline for TRELLIS.2.";
  else ok "RAM headroom is comfortable for TRELLIS.2."; fi
}

cmd_detect() {
  detect_gpus; detect_cuda; recommend_settings
  heading "Hardware"
  local i
  for ((i=0;i<GPU_COUNT;i++)); do
    printf 'GPU %d: %-34s VRAM %-10s compute %s\n' "$i" "${GPU_NAMES[$i]}" "$(human_mib "${GPU_VRAM_MIB[$i]}")" "${GPU_CAPS[$i]:-unknown}"
  done
  printf 'CPU threads:          %s\n' "$(cpu_threads)"
  print_memory_advice
  printf 'Host MemTotal:        %s\n' "$(human_gib "$(meminfo_bytes MemTotal)")"
  printf 'Swap:                 %s\n' "$(free -h 2>/dev/null | awk '/^Swap:/ {print $2 " total, " $3 " used"}' || true)"
  printf 'Workspace disk free:  %s\n' "$(df -h "$WORKSPACE" 2>/dev/null | awk 'NR==2 {print $4}')"
  printf 'NVIDIA driver:        %s (CUDA max %s)\n' "$(driver_version)" "$(driver_cuda_max)"
  printf 'CUDA Toolkit:         %s (%s)\n' "$CUDA_VERSION" "$CUDA_HOME_SELECTED"
  heading "Recommended profile"
  printf 'PyTorch:              %s / %s\nAttention:            %s\nBuild parallel jobs:  %s\nCUDA arch list:       %s\nLow-VRAM mode:        %s\nDefault resolution:   %s\n' \
    "$TORCH_VERSION" "$TORCH_CUDA_TAG" "$ATTN_BACKEND" "$BUILD_JOBS" "${ARCH_LIST:-auto}" "$LOW_VRAM" "$DEFAULT_RESOLUTION"
  if (( MIN_VRAM_MIB < 24576 )); then warn "TRELLIS.2 documents >=24 GB VRAM. This GPU is below the upstream minimum."; fi
  if [[ -r /sys/fs/cgroup/memory.events ]] && grep -qE '^oom_kill[[:space:]]+[1-9][0-9]*$' /sys/fs/cgroup/memory.events; then
    warn "cgroup memory.events already records OOM kills on this instance."
  fi
}

ensure_system_deps() {
  heading "System packages"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      sudo git git-lfs curl wget ca-certificates build-essential cmake ninja-build pkg-config libjpeg-dev ffmpeg >/dev/null
    git lfs install >/dev/null 2>&1 || true
    ok "System build dependencies installed."
  else
    die "Automatic install currently targets Debian/Ubuntu Vast images (apt-get)."
  fi
}

source_conda() {
  if command -v conda >/dev/null 2>&1; then
    local base; base="$(conda info --base 2>/dev/null || true)"
    [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]] && source "$base/etc/profile.d/conda.sh"
    return 0
  fi
  local candidate
  for candidate in /opt/conda /opt/miniforge3 /venv/miniforge3 "$STATE_DIR/miniforge3"; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then source "$candidate/etc/profile.d/conda.sh"; return 0; fi
  done
  return 1
}

ensure_conda() {
  source_conda && return 0
  heading "Miniforge"
  local installer="$STATE_DIR/Miniforge3.sh" dest="$STATE_DIR/miniforge3"
  info "Conda not found; installing Miniforge under $dest ..."
  curl -fsSL -o "$installer" https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$installer" -b -p "$dest"
  source "$dest/etc/profile.d/conda.sh"
  ok "Miniforge installed."
}

ensure_repo() {
  heading "TRELLIS.2 source"
  if [[ -d "$REPO_DIR/.git" ]]; then
    info "Using existing checkout: $REPO_DIR"
    git -C "$REPO_DIR" submodule update --init --recursive
  else
    git clone --recursive https://github.com/microsoft/TRELLIS.2.git "$REPO_DIR"
  fi
  ok "TRELLIS.2 source ready."
}

activate_env() {
  source_conda || die "Conda is unavailable. Run: vast-trellis2 wizard"
  conda activate "$ENV_NAME" 2>/dev/null || die "Conda environment '$ENV_NAME' does not exist. Run: vast-trellis2 install"
}
