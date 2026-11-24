#!/usr/bin/env bash
# Shared process-tree & IPC resource cleanup helper for Docker simulation scripts.
#
# Why this exists:
#   1. `ros2 launch` spawns child processes (`gzserver`, `gzclient`, `rviz2`, `rtabmap`, Nav2 servers)
#      in their own sessions/process groups and does NOT install a `SIGHUP` handler. Closing a terminal
#      tab, pressing Ctrl+C twice while `rviz2` shuts down, or killing the wrapper shell strands the
#      entire simulation/SLAM/Nav2 stack running in the container (~650% CPU + ~2-4 GB RAM + VRAM).
#   2. `ros2` CLI checks (`ros2 node list`, `ros2 topic echo`) spawn a detached `ros2-daemon` that never
#      exits on its own.
#   3. Nodes that crash or get force-killed on shutdown (e.g. `nav2_collision_monitor`) leave orphaned
#      FastDDS shared-memory ring buffers (`/dev/shm/fastrtps_*` and `sem.fastrtps_*`).
#
# Tagging the environment with `UGV_LAUNCH_TAG` lets us find and reap 100% of descendants via
# `/proc/<pid>/environ` regardless of `setsid()` or reparenting to PID 1, without ever touching
# processes belonging to the other terminal or another ROS_DOMAIN_ID.

_UGV_CLEANUP_DONE=0
_UGV_GOT_SIGINT=0
UGV_LAUNCH_PID=""
UGV_WATCHDOG_PIDFILE=""
UGV_SESSION_FILE=""

_ugv_pids_for_tag() {
  local tag="$1" self_pid="${2:-$$}"
  python3 - "$tag" "$self_pid" <<'PY' 2>/dev/null || true
import glob, os, sys
needle = f"UGV_LAUNCH_TAG={sys.argv[1]}".encode()
skip = {int(sys.argv[2]), os.getpid(), os.getppid()}
found = []
for env_path in glob.glob("/proc/[0-9]*/environ"):
    try:
        pid = int(env_path.split("/")[2])
        if pid in skip:
            continue
        data = open(env_path, "rb").read()
        if needle in data.split(b"\x00"):
            found.append(pid)
    except OSError:
        pass
print(" ".join(str(p) for p in found))
PY
}

_ugv_sweep_idle_resources() {
  local caller_pid="${1:-$$}"
  # If NO Gazebo / RViz / SLAM / Nav2 / Survey node binaries remain anywhere in this container,
  # stop detached ros2-daemon instances and unlink orphaned FastDDS /dev/shm files & /tmp params.
  python3 - "$caller_pid" <<'PY' 2>/dev/null || true
import glob, os, signal, sys, time

ACTIVE_BINARIES = {
    "gzserver", "gzclient", "rviz2", "rtabmap", "rgbd_sync", "obstacles_detection",
    "controller_server", "planner_server", "bt_navigator", "behavior_server",
    "smoother_server", "velocity_smoother", "waypoint_follower", "collision_monitor",
    "coqui_survey", "coqui_sound_emitter", "mic_array_localizer",
    "traversability_node", "ekf_node", "imu_filter_madgwick_node",
}
skip = {int(sys.argv[1]), os.getpid(), os.getppid()}
daemons = []
busy = False

for cmd_path in glob.glob("/proc/[0-9]*/cmdline"):
    try:
        pid = int(cmd_path.split("/")[2])
        if pid in skip:
            continue
        raw = open(cmd_path, "rb").read()
        if not raw:
            continue
        if b"ros2-daemon" in raw:
            daemons.append(pid)
            continue
        args = [a.decode("utf-8", "ignore") for a in raw.split(b"\x00") if a]
        if not args:
            continue
        # Check executable basenames (skipping python3 interpreter wrapper in args[0])
        exes = {os.path.basename(args[0])}
        if exes & {"python3", "python"} and len(args) > 1 and not args[1].startswith("-"):
            exes.add(os.path.basename(args[1]))
        if exes & ACTIVE_BINARIES:
            busy = True
    except OSError:
        pass

if not busy:
    for pid in daemons:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    if daemons:
        time.sleep(0.3)
        for pid in daemons:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    for pattern in ("/dev/shm/fastrtps_*", "/dev/shm/sem.fastrtps_*",
                    "/tmp/coqui_*_*.yaml", "/tmp/coqui_rand_world_*.world",
                    "/tmp/launch_params_*"):
        for path in glob.glob(pattern):
            try:
                os.unlink(path)
            except OSError:
                pass
PY
}

_ugv_reap_tag() {
  local tag="$1" grace_sigint="${2:-1}" self_pid="${3:-$$}"
  local pids i

  if [ "$grace_sigint" = "1" ]; then
    pids="$(_ugv_pids_for_tag "$tag" "$self_pid")"
    if [ -n "$pids" ]; then
      kill -INT $pids 2>/dev/null || true
    fi
  fi

  # Wait up to 3.5s for graceful ROS 2 launch / Gazebo teardown
  for i in $(seq 1 35); do
    pids="$(_ugv_pids_for_tag "$tag" "$self_pid")"
    [ -z "$pids" ] && break
    sleep 0.1
  done

  # Escalate survivors to SIGTERM
  pids="$(_ugv_pids_for_tag "$tag" "$self_pid")"
  if [ -n "$pids" ]; then
    kill -TERM $pids 2>/dev/null || true
    for i in $(seq 1 15); do
      pids="$(_ugv_pids_for_tag "$tag" "$self_pid")"
      [ -z "$pids" ] && break
      sleep 0.1
    done
  fi

  # Final SIGKILL guarantee so nothing can ever stay wedged in the container
  pids="$(_ugv_pids_for_tag "$tag" "$self_pid")"
  if [ -n "$pids" ]; then
    kill -KILL $pids 2>/dev/null || true
    sleep 0.1
  fi
}

_ugv_run_cleanup() {
  [ "$_UGV_CLEANUP_DONE" -eq 1 ] && return 0
  _UGV_CLEANUP_DONE=1
  trap '' INT HUP TERM

  if [ -n "$UGV_WATCHDOG_PIDFILE" ] && [ -f "$UGV_WATCHDOG_PIDFILE" ]; then
    local wpid
    wpid="$(cat "$UGV_WATCHDOG_PIDFILE" 2>/dev/null || true)"
    [ -n "$wpid" ] && kill -KILL "$wpid" 2>/dev/null || true
    rm -f "$UGV_WATCHDOG_PIDFILE" 2>/dev/null || true
  fi

  if [ -n "$UGV_SESSION_FILE" ]; then
    rm -f "$UGV_SESSION_FILE" 2>/dev/null || true
  fi

  if [ -n "${UGV_LAUNCH_TAG:-}" ]; then
    if [ "$_UGV_GOT_SIGINT" -eq 1 ]; then
      _ugv_reap_tag "$UGV_LAUNCH_TAG" 0 "$$"
    else
      _ugv_reap_tag "$UGV_LAUNCH_TAG" 1 "$$"
    fi
  fi

  _ugv_sweep_idle_resources "$$"
}

ugv_init_launch_cleanup() {
  local prefix="$1" domain="${2:-0}" session_file="${3:-}" require_session_file="${4:-}"
  UGV_SESSION_FILE="$session_file"
  export UGV_LAUNCH_TAG="${prefix}_${domain}_$$_$(date +%s)"
  UGV_WATCHDOG_PIDFILE="/tmp/ugv_wd_${domain}_$$.pid"

  # Double-forked daemon watchdog (without UGV_LAUNCH_TAG, detached from pts stdin/stdout so it
  # never prints bash job-control messages and never holds pts open). Detects:
  #   a) parent shell death (even `kill -9`),
  #   b) terminal PTY hangup (`POLLHUP`), AND
  #   c) simulation peer shutdown (`require_session_file` removed when Terminal 1 stops, defeating
  #      Docker's moby/moby#9098 where closing a `docker exec -it` host tab never sends SIGHUP)!
  local parent_pid="$$" tag="$UGV_LAUNCH_TAG" sfile="$UGV_SESSION_FILE" lib_abs
  lib_abs="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  local tty_path=""
  tty_path="$(readlink "/proc/$$/fd/0" 2>/dev/null || true)"
  local wd_code
  wd_code='
import os, select, signal, subprocess, sys, time

parent_pid = int(sys.argv[1])
tag, sfile, lib, pidfile, tty_path, req_sfile = sys.argv[2:8]
with open(pidfile, "w") as pf:
    pf.write(str(os.getpid()))

tty_fd = None
poller = None
if tty_path.startswith("/dev/pts/"):
    try:
        tty_fd = os.open(tty_path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY)
        poller = select.poll()
        poller.register(tty_fd, select.POLLHUP | select.POLLERR)
    except OSError:
        tty_fd = None

try:
    while True:
        try:
            os.kill(parent_pid, 0)
        except OSError:
            break
        if req_sfile and not os.path.exists(req_sfile):
            break
        if poller is not None:
            ev = poller.poll(0)
            if any(mask & (select.POLLHUP | select.POLLERR) for _, mask in ev):
                break
        time.sleep(0.5)
finally:
    if tty_fd is not None:
        try:
            os.close(tty_fd)
        except OSError:
            pass
    try:
        os.unlink(pidfile)
    except OSError:
        pass

if sfile:
    try:
        os.unlink(sfile)
    except OSError:
        pass

subprocess.run(
    ["bash", "-c", "source \"$1\"; _ugv_reap_tag \"$2\" 1 \"$3\"", "_", lib, tag, str(parent_pid)],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
)
try:
    os.kill(parent_pid, signal.SIGKILL)
except OSError:
    pass
time.sleep(0.2)
subprocess.run(
    ["bash", "-c", "source \"$1\"; _ugv_sweep_idle_resources \"$2\"", "_", lib, str(parent_pid)],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
)
'
  (
    env -u UGV_LAUNCH_TAG python3 -c "$wd_code" "$parent_pid" "$tag" "$sfile" "$lib_abs" "$UGV_WATCHDOG_PIDFILE" "$tty_path" "$require_session_file" </dev/null >/dev/null 2>&1 &
  )

  trap '_UGV_GOT_SIGINT=1; exit 130' INT
  trap 'exit 129' HUP
  trap 'exit 143' TERM
  trap '_ugv_run_cleanup' EXIT
}

ugv_run_foreground() {
  "$@" <&0 &
  UGV_LAUNCH_PID=$!
  local rc=0
  wait "$UGV_LAUNCH_PID" || rc=$?
  return "$rc"
}
