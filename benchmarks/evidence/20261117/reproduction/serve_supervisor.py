"""Own one isolated validation server; never signal unrelated processes."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def qualified_exit(receipt):
    return bool(receipt.get("was_ready") and receipt.get("stop_observed")
                and receipt.get("server_returncode") == 0
                and not receipt.get("forced_shutdown")
                and not receipt.get("startup_timeout")
                and not receipt.get("startup_engine_failed"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--argv-file", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready-timeout", type=float, default=1200)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    invocation = json.loads(Path(args.argv_file).read_text())
    supervisor_source = Path(__file__).read_bytes()
    (run / "supervisor-source.py").write_bytes(supervisor_source)
    argv = invocation["argv"]
    if not isinstance(argv, list) or not argv or not all(isinstance(value, str) for value in argv):
        raise ValueError("argv must be a nonempty list of strings")
    # Refuse to launch over an existing listener; no stop-by-port behavior.
    with socket.socket() as probe:
        # Permit the previous owned listener's TIME_WAIT sockets, as the API
        # server itself does. listen() still rejects an active listener; never
        # use SO_REUSEPORT or stop any process found on this port.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
        probe.listen(1)
    env = os.environ.copy()
    env.pop("VLLM_VERSION", None)
    overrides = dict(invocation.get("env", {}))
    python_prefix = overrides.pop("PYTHONPATH", None)
    env.update(overrides)
    if python_prefix:
        # Preserve CANN's Python bindings (notably acl); source selection is
        # a prefix, not a replacement for the accelerator environment.
        env["PYTHONPATH"] = os.pathsep.join(value for value in (python_prefix, env.get("PYTHONPATH", "")) if value)
    reservation = 0
    if "--kv-transfer-config" in argv:
        transfer = json.loads(argv[argv.index("--kv-transfer-config") + 1])
        reservation = int(transfer.get("kv_connector_extra_config", {}).get("cpu_bytes_to_use", 0))
    shm = os.statvfs("/dev/shm")
    available = shm.f_bavail * shm.f_frsize
    preflight = {"shm_available_bytes": available, "cpu_reservation_bytes": reservation,
                 "ipc_headroom_bytes": 1024**3}
    (run / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if available < reservation + preflight["ipc_headroom_bytes"]:
        failure = {"was_ready": False, "startup_preflight_failed": True,
                   "reason": "insufficient shared-memory headroom", **preflight}
        (run / "exit.json").write_text(json.dumps(failure, indent=2) + "\n")
        print("PREFLIGHT_FAILED", json.dumps(failure), flush=True)
        return 1
    stop = {"requested": False}
    def requested(signum, frame):
        stop["requested"] = True
    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)
    signal.signal(signal.SIGHUP, requested)
    started = time.monotonic()
    metadata = {"started_at": now(), "supervisor_sha256": hashlib.sha256(supervisor_source).hexdigest(), "argv": argv, "cwd": str(Path(args.cwd).resolve()),
                "supervisor_pid": os.getpid(), "port": args.port,
                "env_overrides": invocation.get("env", {}),
                "inherited_runtime_env": {name: env.get(name) for name in (
                    "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "PYTHONPATH", "VLLM_VERSION",
                    "LD_PRELOAD", "LD_LIBRARY_PATH", "VLLM_PLUGINS",
                    "PYTORCH_NPU_ALLOC_CONF", "VLLM_WORKER_MULTIPROC_METHOD")}}
    git_env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    metadata["plugin_commit"] = subprocess.check_output(["git", "-C", args.cwd, "rev-parse", "HEAD"], text=True, env=git_env).strip()
    metadata["plugin_worktree_status"] = subprocess.check_output(["git", "-C", args.cwd, "status", "--short"], text=True, env=git_env).strip()
    source_root = Path(args.cwd) / "src" / "vllm_hust_kv_tiering"
    metadata["plugin_source_sha256"] = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_root.rglob("*")) if path.is_file() and path.suffix in (".py", ".json")}
    metadata["diagnostic_sources_sha256"] = {}
    for index, source_name in enumerate(invocation.get("additional_sources", [])):
        source = Path(source_name).resolve(strict=True)
        content = source.read_bytes()
        metadata["diagnostic_sources_sha256"][str(source)] = hashlib.sha256(content).hexdigest()
        (run / f"diagnostic-source-{index}-{source.name}").write_bytes(content)
    with (run / "server.log").open("wb") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=args.cwd, env=env, start_new_session=True)
        metadata.update(server_pid=process.pid, server_pgid=process.pid)
        (run / "invocation.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("SERVER_STARTED", json.dumps(metadata), flush=True)
        ready = False
        timeout = False
        startup_failed = False
        forced_shutdown = False
        stop_observed = False
        try:
            while process.poll() is None:
                if stop["requested"] or (run / "stop.request").exists():
                    stop_observed = True
                    break
                if not ready:
                    with (run / "server.log").open("rb") as current_log:
                        current_log.seek(0, 2)
                        current_log.seek(max(0, current_log.tell() - 1024**2))
                        tail = current_log.read().decode("utf-8", errors="replace")
                    if "EngineCore failed to start" in tail:
                        startup_failed = True
                        failure = {"reason": "EngineCore failed to start", "observed_at": now()}
                        (run / "startup-failure.json").write_text(json.dumps(failure, indent=2) + "\n")
                        print("STARTUP_ENGINE_FAILURE", json.dumps(failure), flush=True)
                        break
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                            ready = response.status == 200
                    except (OSError, urllib.error.URLError):
                        pass
                    if ready:
                        receipt = {"ready_at": now(), "startup_seconds": time.monotonic() - started}
                        (run / "ready.json").write_text(json.dumps(receipt, indent=2) + "\n")
                        print("MODEL_READY", json.dumps(receipt), flush=True)
                    elif time.monotonic() - started > args.ready_timeout:
                        timeout = True
                        print("STARTUP_TIMEOUT", flush=True)
                        break
                time.sleep(1)
        finally:
            if process.poll() is None:
                # The Popen child owns this new session/process group. Verify
                # that relationship immediately before sending any signal.
                if os.getpgid(process.pid) != process.pid:
                    raise RuntimeError("server no longer owns its expected process group")
                # Ask the API leader to shut down its engine/workers first.
                # Signalling the entire group immediately bypasses the host's
                # graceful connector teardown and can strand mmap names.
                process.terminate()
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    if os.getpgid(process.pid) != process.pid:
                        raise RuntimeError("cannot verify owned process group")
                    forced_shutdown = True
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
        receipt = {"finished_at": now(), "server_returncode": process.returncode,
                   "was_ready": ready, "startup_timeout": timeout, "startup_engine_failed": startup_failed,
                   "stop_requested": stop["requested"] or (run / "stop.request").exists(),
                   "stop_observed": stop_observed, "forced_shutdown": forced_shutdown,
                   "elapsed_seconds": time.monotonic() - started}
        (run / "exit.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print("SERVER_EXIT", json.dumps(receipt), flush=True)
        return 0 if qualified_exit(receipt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
