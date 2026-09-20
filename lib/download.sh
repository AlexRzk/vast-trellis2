# shellcheck shell=bash
# Hugging Face download transport overrides. Sourced after lib/install.sh so this
# file intentionally replaces cmd_download with a transport-aware implementation.

_hf_mem_gib() {
  awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || printf '0'
}

configure_hf_download_transport() {
  local mode="${1:-auto}" mem_gib ranges workers
  mem_gib="$(_hf_mem_gib)"

  export HF_HOME="$HF_HOME_DIR"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

  if (( mem_gib >= 128 )); then
    ranges=64; workers=16
  elif (( mem_gib >= 64 )); then
    ranges=32; workers=12
  else
    ranges=16; workers=8
  fi

  export VAST_TRELLIS_HF_WORKERS="${VAST_TRELLIS_HF_WORKERS:-$workers}"

  case "$mode" in
    auto|xet)
      unset HF_HUB_DISABLE_XET 2>/dev/null || true
      export HF_XET_HIGH_PERFORMANCE=1
      export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-$ranges}"
      info "Hugging Face transport: Xet high-performance (${HF_XET_NUM_CONCURRENT_RANGE_GETS} range requests/file, ${VAST_TRELLIS_HF_WORKERS} snapshot workers)."
      ;;
    direct|http)
      export HF_HUB_DISABLE_XET=1
      unset HF_XET_HIGH_PERFORMANCE HF_XET_NUM_CONCURRENT_RANGE_GETS 2>/dev/null || true
      info "Hugging Face transport: direct HTTP/CDN (Xet disabled, ${VAST_TRELLIS_HF_WORKERS} snapshot workers)."
      ;;
    *)
      die "Unknown download transport '$mode'. Use --xet or --direct."
      ;;
  esac
}

_hf_snapshot() {
  local repo="$1"
  HF_REPO_ID="$repo" python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo = os.environ["HF_REPO_ID"]
workers = int(os.environ.get("VAST_TRELLIS_HF_WORKERS", "8"))
print(f"Downloading/caching {repo} with max_workers={workers} ...")
path = snapshot_download(repo, max_workers=workers)
print("Cached at:", path)
PY
}

cmd_hf_speedtest() {
  load_config; activate_env
  export HF_HOME="$HF_HOME_DIR"
  heading "Hugging Face transfer diagnostics"
  python -m pip install -q -U huggingface_hub hf_xet
  echo "hf_xet / Hub versions:"
  python - <<'PY'
import huggingface_hub
try:
    import hf_xet
    xv = getattr(hf_xet, "__version__", "installed")
except Exception as e:
    xv = f"unavailable ({e})"
print("huggingface_hub:", huggingface_hub.__version__)
print("hf_xet:", xv)
PY
  echo
  echo "Network hints:"
  command -v curl >/dev/null 2>&1 && curl -L -sS -o /dev/null -w 'huggingface.co: %{http_code} connect=%{time_connect}s start=%{time_starttransfer}s total=%{time_total}s\n' https://huggingface.co/ || true
  echo
  echo "Current recommendation for this machine:"
  configure_hf_download_transport xet
  echo "If Xet remains slow, retry the cached download with: vast-trellis2 download --direct"
}

cmd_download() {
  load_config; activate_env
  local mode=auto
  case "${1:-}" in
    "") ;;
    --xet) mode=xet ;;
    --direct|--http) mode=direct ;;
    --help|-h)
      cat <<'EOF'
Usage:
  vast-trellis2 download            Xet high-performance (default)
  vast-trellis2 download --xet      Force Xet high-performance
  vast-trellis2 download --direct   Disable Xet; use HTTP/CDN fallback
  vast-trellis2 hf-speedtest        Show Hub/Xet diagnostics

Downloads are resumable through the Hugging Face cache. If one route is slow,
Ctrl+C and retry with the other transport; cached completed files are retained.
EOF
      return 0
      ;;
    *) die "Unknown download option: ${1:-}" ;;
  esac

  configure_hf_download_transport "$mode"

  local free_kb
  free_kb="$(df -Pk "$WORKSPACE" | awk 'NR==2 {print $4}')"
  if [[ "$free_kb" =~ ^[0-9]+$ ]] && (( free_kb < 31457280 )); then
    warn "Less than 30 GiB free; model cache may exhaust disk."
  fi

  check_gated_access || warn "A gated dependency is unavailable; the main TRELLIS snapshot can still be cached."

  heading "Download TRELLIS.2-4B"
  _hf_snapshot microsoft/TRELLIS.2-4B

  heading "Download auxiliary models"
  if [[ -z "$DINOV3_MODEL_PATH" ]]; then
    set +e
    _hf_snapshot facebook/dinov3-vitl16-pretrain-lvd1689m
    local drc=$?
    set -e
    (( drc == 0 )) || warn "DINOv3 not downloaded; obtain HF approval or use 'vast-trellis2 dino-local'."
  fi

  set +e
  _hf_snapshot briaai/RMBG-2.0
  local rrc=$?
  set -e
  (( rrc == 0 )) || warn "RMBG-2.0 not downloaded. Accept/request its Hugging Face access first."
}
