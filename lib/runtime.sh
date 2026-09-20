# shellcheck shell=bash
# Runtime lifecycle, crash explanation, tunnel helper and interactive wizard/menu.

runtime_exports() {
  export HF_HOME="$HF_HOME_DIR"
  export HF_XET_HIGH_PERFORMANCE=1
  export HF_HUB_DOWNLOAD_TIMEOUT=120
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  export CUDA_HOME="$CUDA_HOME_SELECTED"
  export PATH="$CUDA_HOME/bin:$PATH"
  export ATTN_BACKEND="$ATTN_BACKEND"
  export SPARSE_ATTN_BACKEND="$SPARSE_ATTN_BACKEND"
  export OPENCV_IO_ENABLE_OPENEXR=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export TRELLIS_LOW_VRAM="$LOW_VRAM"
  export TRELLIS_DEFAULT_RESOLUTION="$DEFAULT_RESOLUTION"
  export TRELLIS_SERVER_NAME="$SERVER_NAME"
  export TRELLIS_PORT="$PORT"
  if [[ -n "$DINOV3_MODEL_PATH" ]]; then export DINOV3_MODEL_PATH; else unset DINOV3_MODEL_PATH 2>/dev/null || true; fi
}

runtime_preflight() {
  load_config; detect_gpus
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] && (( GPU_ID < GPU_COUNT )) || die "Configured GPU_ID=$GPU_ID is invalid; visible GPU count is $GPU_COUNT."
  local avail_gib=$(( $(effective_available_bytes) / 1073741824 ))
  if (( avail_gib < 24 )); then
    die "Only ${avail_gib} GiB effective RAM is available now. Stop other workloads or use a larger instance."
  elif (( avail_gib < 40 )); then
    warn "Only ${avail_gib} GiB effective RAM is available. OOM risk is high; use resolution 512."
  fi
  [[ -f "$REPO_DIR/app.py" ]] || die "TRELLIS.2 not installed at $REPO_DIR. Run: vast-trellis2 wizard"
  if [[ -n "$DINOV3_MODEL_PATH" && ! -f "$DINOV3_MODEL_PATH/config.json" ]]; then die "Configured DINOV3_MODEL_PATH is invalid: $DINOV3_MODEL_PATH"; fi
  if [[ ! -f "$REPO_DIR/trellis2/modules/image_feature_extractor.py" ]]; then die "TRELLIS source checkout is incomplete."; fi
}

run_foreground() {
  runtime_preflight; activate_env; runtime_exports
  cd "$REPO_DIR"
  info "Starting TRELLIS.2 at http://$SERVER_NAME:$PORT"
  info "GPU=$GPU_ID | default resolution=$DEFAULT_RESOLUTION | low_vram=$LOW_VRAM | attention=$ATTN_BACKEND"
  exec python app.py
}

is_running() {
  [[ -s "$PID_FILE" ]] || return 1
  local pid; pid="$(cat "$PID_FILE")"
  kill -0 "$pid" 2>/dev/null
}

port_open() {
  python3 - "$PORT" <<'PY' >/dev/null 2>&1
import socket,sys
s=socket.socket(); s.settimeout(.2)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
}

cmd_start() {
  runtime_preflight
  if is_running; then warn "TRELLIS.2 already running as PID $(cat "$PID_FILE")."; return 0; fi
  activate_env; runtime_exports
  local log="$LOG_DIR/runtime.log" py
  py="$(command -v python)"
  : > "$log"
  heading "Starting TRELLIS.2 in background"

  # A wrapper records the child exit code. If Linux OOM-kills Python, exit 137 is
  # preserved in the log instead of presenting only a mysterious disconnected UI.
  nohup bash -c '
    cd "$1" || exit 120
    "$2" app.py
    rc=$?
    printf "\n__VAST_TRELLIS_EXIT_CODE=%s\n" "$rc"
    exit "$rc"
  ' _ "$REPO_DIR" "$py" >>"$log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  info "PID $pid; log: $log"

  local i
  for ((i=0;i<300;i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      fail "TRELLIS.2 exited during startup."
      rm -f "$PID_FILE"
      cmd_explain "$log"
      return 1
    fi
    if port_open; then
      ok "TRELLIS.2 is listening on 127.0.0.1:$PORT"
      printf 'Tunnel from your local PC:\n  ssh -N -L %s:127.0.0.1:%s -p <VAST_SSH_PORT> root@<VAST_HOST>\n' "$PORT" "$PORT"
      return 0
    fi
    sleep 1
  done
  warn "Process is alive but port $PORT did not open within 5 minutes. Follow logs with: vast-trellis2 logs"
}

cmd_stop() {
  if ! is_running; then warn "TRELLIS.2 is not running."; rm -f "$PID_FILE"; return 0; fi
  local pid; pid="$(cat "$PID_FILE")"
  kill "$pid" 2>/dev/null || true
  local i
  for ((i=0;i<20;i++)); do kill -0 "$pid" 2>/dev/null || break; sleep .5; done
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  ok "Stopped TRELLIS.2."
}

cmd_status() {
  load_config
  heading "TRELLIS.2 status"
  if is_running; then ok "Managed process running: PID $(cat "$PID_FILE")"; else warn "Managed app is not running."; fi
  if port_open; then ok "127.0.0.1:$PORT is listening."; else warn "Nothing is listening on 127.0.0.1:$PORT."; fi
  print_memory_advice
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv 2>/dev/null || true
  if [[ -r /sys/fs/cgroup/memory.events ]]; then
    echo; echo "cgroup memory.events:"; cat /sys/fs/cgroup/memory.events
  fi
}

cmd_logs() {
  local log="$LOG_DIR/runtime.log"
  [[ -f "$log" ]] || die "No runtime log yet."
  local lines="${1:-120}"
  tail -n "$lines" -f "$log"
}

cmd_explain() {
  local log="${1:-$LOG_DIR/runtime.log}"
  [[ -f "$log" ]] || die "Log not found: $log"
  heading "Error diagnosis"
  if grep -qiE '__VAST_TRELLIS_EXIT_CODE=137|^Killed$|Killed[[:space:]]*$|exit code 137' "$log"; then
    fail "Linux killed TRELLIS without a Python exception: system/cgroup RAM OOM is the primary suspect."
    print_memory_advice
    [[ -r /sys/fs/cgroup/memory.events ]] && cat /sys/fs/cgroup/memory.events
    warn "Fix: use >=64 GiB effective RAM, resolution 512, low_vram=1, and stop competing processes."
  elif grep -qi "CUDA out of memory" "$log"; then
    fail "GPU VRAM OOM."
    warn "Fix: use 512, low_vram=1, stop other GPU processes, then restart."
  elif grep -qiE "dinov3-vitl16.*GatedRepoError|facebook/dinov3.*403|facebook/dinov3.*gated" "$log"; then
    fail "DINOv3 Hugging Face access is gated."
    warn "Run 'vast-trellis2 auth' then request HF access, or convert Meta's official checkpoint with 'vast-trellis2 dino-local PATH'."
  elif grep -qiE "briaai/RMBG-2.0.*(403|gated)|RMBG-2.0.*GatedRepoError" "$log"; then
    fail "RMBG-2.0 Hugging Face access is gated."
    warn "Accept/request access to briaai/RMBG-2.0, then run 'vast-trellis2 auth' and 'vast-trellis2 download'."
  elif grep -qi "object has no attribute 'layer'" "$log"; then
    fail "Transformers DINOv3 layout mismatch."
    warn "Run 'vast-trellis2 doctor --fix' to reapply the compatibility patch."
  elif grep -qiE "cvtColor.*!_src.empty|forest.exr.*None|OpenEXR" "$log"; then
    fail "HDRI/OpenEXR loading failed."
    warn "Run 'vast-trellis2 doctor --fix' to restore HDRIs and OpenCV EXR support."
  elif grep -qi "ModuleNotFoundError" "$log"; then
    fail "Python dependency missing or TRELLIS was launched from the wrong environment."
    grep -i "ModuleNotFoundError" "$log" | tail -n 5 >&2 || true
    warn "Launch via 'vast-trellis2 run/start'; do not run app.py from Vast's default (main) environment."
  elif grep -qiE "xFormers can't load|xformers.*C\+\+/CUDA" "$log"; then
    fail "xFormers wheel / PyTorch CUDA mismatch."
    warn "Run 'vast-trellis2 doctor --fix'."
  elif grep -qiE "unsupported gpu architecture|sm_120|compute_120" "$log"; then
    fail "CUDA compiler/package cannot target this GPU architecture."
    warn "For RTX 50/Blackwell use a CUDA 12.8 toolkit/devel Vast image."
  elif grep -qiE "403 Client Error|401 Client Error" "$log"; then
    fail "A Hugging Face authentication/access error occurred."
    warn "401 usually means login/token; 403 gated means the account still needs model approval."
  else
    if [[ -r /sys/fs/cgroup/memory.events ]] && grep -qE '^oom_kill[[:space:]]+[1-9][0-9]*$' /sys/fs/cgroup/memory.events; then
      warn "No log signature matched, but cgroup memory.events records OOM kills. A bare process death may still be RAM OOM."
    else
      warn "No known signature matched. Last 60 log lines:"
    fi
    tail -n 60 "$log"
  fi
}

cmd_tunnel() {
  load_config
  cat <<EOF
TRELLIS binds to localhost by default.

On your LOCAL computer, use the exact SSH host/port shown by Vast.ai -> Connect:

  ssh -N -L $PORT:127.0.0.1:$PORT -p <VAST_SSH_PORT> root@<VAST_HOST>

Then open:
  http://127.0.0.1:$PORT

If SSH prints "channel ... connect failed: Connection refused", the tunnel itself is
alive but TRELLIS is not listening yet. Check:
  vast-trellis2 status
  vast-trellis2 logs
EOF
}

cmd_config() {
  load_config
  [[ -f "$CONFIG_FILE" ]] || die "No saved config. Run: vast-trellis2 wizard"
  cat "$CONFIG_FILE"
}

cmd_doctor() {
  [[ -x "$DOCTOR_TOOL" ]] || die "Doctor executable not installed/found: $DOCTOR_TOOL"
  exec "$DOCTOR_TOOL" "$@"
}

cmd_wizard() {
  heading "Vast TRELLIS.2 setup wizard"
  cmd_detect
  echo
  if ! prompt_yes_no "Use these detected build/runtime settings?" y; then
    BUILD_JOBS="$(prompt 'Parallel compiler jobs (extensions still build one stage at a time)' "$BUILD_JOBS")"
    DEFAULT_RESOLUTION="$(prompt 'Default resolution (512/1024/1536)' "$DEFAULT_RESOLUTION")"
    LOW_VRAM="$(prompt 'Low-VRAM mode (1=on, 0=off)' "$LOW_VRAM")"
  fi
  REPO_DIR="$(prompt 'TRELLIS.2 checkout directory' "$REPO_DIR")"
  PORT="$(prompt 'Gradio port' "$PORT")"
  if (( GPU_COUNT > 1 )); then GPU_ID="$(prompt 'GPU index to use for TRELLIS' "$GPU_ID")"; fi
  save_config

  if prompt_yes_no "Install/update TRELLIS.2 and CUDA extensions now?" y; then
    cmd_install
  else
    warn "Installation skipped. HF auth/download and launch need the trellis2 environment."
    return 0
  fi

  if prompt_yes_no "Authenticate to Hugging Face now?" y; then cmd_auth; fi

  set +e
  check_gated_access
  local gated_rc=$?
  set -e
  if (( gated_rc != 0 )) && [[ -z "$DINOV3_MODEL_PATH" ]]; then
    if prompt_yes_no "Do you have Meta's official DINOv3 ViT-L/16 .pth checkpoint on this VM?" n; then
      local cp; cp="$(prompt 'Checkpoint path')"; cmd_dino_local "$cp"
    fi
  fi

  if prompt_yes_no "Pre-download TRELLIS.2-4B and accessible auxiliary weights now?" y; then cmd_download; fi
  if [[ -x "$DOCTOR_TOOL" ]]; then "$DOCTOR_TOOL" --check || true; fi

  echo; ok "Wizard finished."
  echo "Foreground: vast-trellis2 run"
  echo "Background: vast-trellis2 start"
  echo "Tunnel:     vast-trellis2 tunnel"
}

show_splash() {
  [[ -t 1 && "${TERM:-dumb}" != dumb && "${VAST_TRELLIS_NO_SPLASH:-0}" != 1 ]] || return 0
  clear 2>/dev/null || true
  printf '%b' "$c_cyan"
  cat <<'EOF'
████████╗██████╗ ███████╗██╗     ██╗     ██╗███████╗    ██████╗
╚══██╔══╝██╔══██╗██╔════╝██║     ██║     ██║██╔════╝    ╚════██╗
   ██║   ██████╔╝█████╗  ██║     ██║     ██║███████╗     █████╔╝
   ██║   ██╔══██╗██╔══╝  ██║     ██║     ██║╚════██║    ██╔═══╝
   ██║   ██║  ██║███████╗███████╗███████╗██║███████║    ███████╗
   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝╚══════╝    ╚══════╝
EOF
  printf '%b\n' "$c_reset"
  printf '%bVast.ai adaptive installer • cgroup RAM/GPU detection • repair • launcher%b\n' "$c_dim" "$c_reset"
}

main_menu() {
  local choice
  while true; do
    echo
    cat <<'EOF'
  1) Full setup wizard
  2) Detect hardware / recommendations
  3) Install / resume installation
  4) Hugging Face login
  5) Download model weights
  6) Configure local Meta DINOv3 checkpoint
  7) Doctor
  8) Doctor + safe auto-fix
  9) Run TRELLIS foreground
 10) Start TRELLIS background
 11) Stop TRELLIS
 12) Status
 13) Logs
 14) Explain latest crash
 15) SSH tunnel instructions
  0) Exit
EOF
    read -r -p 'Choose: ' choice || return
    case "$choice" in
      1) cmd_wizard ;;
      2) cmd_detect ;;
      3) cmd_install ;;
      4) cmd_auth ;;
      5) cmd_download ;;
      6) cmd_dino_local ;;
      7) "$DOCTOR_TOOL" --check || true ;;
      8) "$DOCTOR_TOOL" --fix || true ;;
      9) run_foreground ;;
      10) cmd_start ;;
      11) cmd_stop ;;
      12) cmd_status ;;
      13) cmd_logs ;;
      14) cmd_explain ;;
      15) cmd_tunnel ;;
      0) return ;;
      *) warn "Unknown choice." ;;
    esac
  done
}

usage() {
  cat <<EOF
Vast TRELLIS.2 Wizard / Manager v$VERSION

Usage:
  vast-trellis2                    Interactive menu
  vast-trellis2 wizard             Complete adaptive install wizard
  vast-trellis2 detect             GPU/VRAM/RAM/cgroup/CUDA recommendations
  vast-trellis2 install [--force]  Install/resume; --force reruns stages
  vast-trellis2 auth               Secure Hugging Face token paste/login
  vast-trellis2 download           Cache TRELLIS.2 + accessible auxiliary weights
  vast-trellis2 dino-local PATH    Convert official Meta DINOv3 ViT-L/16 checkpoint
  vast-trellis2 run                Run Gradio in foreground
  vast-trellis2 start|launch       Start Gradio in background
  vast-trellis2 stop|restart       Stop/restart managed background process
  vast-trellis2 status             Process + RAM/cgroup + GPU status
  vast-trellis2 logs [lines]       Follow runtime log
  vast-trellis2 explain [log]      Explain known install/runtime failure
  vast-trellis2 doctor [--fix]     Diagnose or auto-fix common problems
  vast-trellis2 repair             Alias for doctor --fix
  vast-trellis2 reset-build        Clear extension build caches/stage markers
  vast-trellis2 tunnel             SSH tunnel instructions
  vast-trellis2 config             Show saved adaptive config

State: $STATE_DIR
Repo:  $REPO_DIR
HF:    $HF_HOME_DIR
EOF
}
