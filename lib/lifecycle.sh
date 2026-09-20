# shellcheck shell=bash
# Robust TRELLIS runtime process discovery and lifecycle overrides.
# Loaded after runtime.sh so these functions replace the basic PID-file-only versions.

_trellis_repo_realpath() {
  readlink -f "$REPO_DIR" 2>/dev/null || printf '%s' "$REPO_DIR"
}

_trellis_process_matches() {
  local pid="$1" cwd cmd repo
  [[ "$pid" =~ ^[0-9]+$ && -d "/proc/$pid" ]] || return 1
  cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  repo="$(_trellis_repo_realpath)"
  [[ -n "$cwd" && "$cwd" == "$repo" ]] || return 1
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$cmd" == *"app.py"* ]] || return 1
  [[ "$cmd" == *"python"* ]] || return 1
}

_trellis_discover_pids() {
  local p pid
  for p in /proc/[0-9]*; do
    pid="${p##*/}"
    _trellis_process_matches "$pid" && printf '%s\n' "$pid"
  done
}

_trellis_saved_pid_valid() {
  [[ -s "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  _trellis_process_matches "$pid"
}

_trellis_reconcile_pid() {
  # Keep a valid saved PID. Remove stale records. If exactly one matching orphan exists,
  # adopt it so stop/status work even when app.py was started manually or by an old wizard.
  if _trellis_saved_pid_valid; then
    return 0
  fi
  rm -f "$PID_FILE"

  local -a pids=()
  mapfile -t pids < <(_trellis_discover_pids)
  if (( ${#pids[@]} == 1 )); then
    printf '%s\n' "${pids[0]}" > "$PID_FILE"
    return 0
  fi
  return 1
}

is_running() {
  _trellis_reconcile_pid || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  kill -0 "$pid" 2>/dev/null
}

cmd_start() {
  runtime_preflight

  local -a existing=()
  mapfile -t existing < <(_trellis_discover_pids)
  if (( ${#existing[@]} > 0 )); then
    if (( ${#existing[@]} == 1 )); then
      printf '%s\n' "${existing[0]}" > "$PID_FILE"
      warn "TRELLIS.2 is already running as PID ${existing[0]}; adopted it into the manager."
    else
      warn "Multiple TRELLIS.2 app.py processes already exist: ${existing[*]}"
      warn "Run 'vast-trellis2 stop' to clean them before starting another instance."
    fi
    return 0
  fi

  rm -f "$PID_FILE"
  activate_env; runtime_exports
  local log="$LOG_DIR/runtime.log" py
  py="$(command -v python)"
  : > "$log"
  heading "Starting TRELLIS.2 in background"

  # Start Python directly so the PID file always contains the real GPU-owning process.
  (
    cd "$REPO_DIR"
    nohup "$py" app.py >>"$log" 2>&1 &
    printf '%s\n' "$!" > "$PID_FILE"
  )

  local pid
  pid="$(cat "$PID_FILE")"
  info "Python PID $pid; log: $log"

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
  load_config
  local -a pids=()
  mapfile -t pids < <(_trellis_discover_pids)

  # Include a valid managed PID even if discovery changed during the scan.
  if _trellis_saved_pid_valid; then
    local managed seen=0 p
    managed="$(cat "$PID_FILE")"
    for p in "${pids[@]}"; do [[ "$p" == "$managed" ]] && seen=1; done
    (( seen == 1 )) || pids+=("$managed")
  fi

  if (( ${#pids[@]} == 0 )); then
    rm -f "$PID_FILE"
    warn "No TRELLIS.2 app.py process is running."
    return 0
  fi

  info "Stopping TRELLIS.2 process(es): ${pids[*]}"
  local pid i alive
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done

  for ((i=0;i<30;i++)); do
    alive=0
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
    (( alive == 0 )) && break
    sleep .5
  done

  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      warn "PID $pid did not exit after SIGTERM; sending SIGKILL."
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  rm -f "$PID_FILE"
  ok "Stopped TRELLIS.2 and cleaned its PID state."
}

cmd_status() {
  load_config
  heading "TRELLIS.2 status"

  local -a pids=()
  mapfile -t pids < <(_trellis_discover_pids)
  if (( ${#pids[@]} == 1 )); then
    printf '%s\n' "${pids[0]}" > "$PID_FILE"
    ok "TRELLIS process running: PID ${pids[0]}"
  elif (( ${#pids[@]} > 1 )); then
    warn "Multiple TRELLIS app.py processes detected: ${pids[*]}"
    warn "Run 'vast-trellis2 stop' to terminate all TRELLIS instances safely."
  else
    rm -f "$PID_FILE"
    warn "No TRELLIS app.py process is running."
  fi

  if port_open; then ok "127.0.0.1:$PORT is listening."; else warn "Nothing is listening on 127.0.0.1:$PORT."; fi
  print_memory_advice
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv 2>/dev/null || true
  if [[ -r /sys/fs/cgroup/memory.events ]]; then
    echo; echo "cgroup memory.events:"; cat /sys/fs/cgroup/memory.events
  fi
}
