"""Frozen full-HBM diagnostics. Own only explicitly launched supervisors."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

ROOT = Path('/root/CGCL/kv-tiering-performance-recheck-20261116')
PREVIOUS = Path('/root/CGCL/kv-tiering-working-set-20261116')
PLUGIN = PREVIOUS / 'revision-v2/plugin'
PY = '/root/CGCL/kv-tiering-benefit-search-20261108/runtime-venv-v1/bin/python'
OLD = Path('/root/CGCL/kv-tiering-recovery-validation-20261112')


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def event(**data):
    data['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save(ROOT / 'STATUS.json', data)
    with (ROOT / 'events.jsonl').open('a') as handle:
        handle.write(json.dumps(data) + '\n')
    print(json.dumps(data), flush=True)


def metric_monitor(stop):
    with (ROOT / 'monitor.jsonl').open('x') as out:
        while not stop.is_set():
            record = {'epoch': time.time()}
            try:
                record['status'] = json.loads((ROOT / 'STATUS.json').read_text())
                with urllib.request.urlopen('http://127.0.0.1:18236/metrics', timeout=3) as response:
                    record['metrics'] = [line for line in response.read().decode().splitlines()
                        if not line.startswith('#') and any(word in line for word in
                        ('kv_tiering', 'kv_offload', 'prefix_cache', 'preempt',
                         'kv_cache_usage', 'num_requests_running', 'num_requests_waiting',
                         'prompt_tokens_by_source', 'request_prefill_kv_computed_tokens'))]
            except (OSError, ValueError) as exc:
                record['error'] = str(exc)
            out.write(json.dumps(record) + '\n')
            out.flush()
            stop.wait(5)


def main():
    with (ROOT / 'controller-start.json').open('x') as handle:
        json.dump({'pid': os.getpid()}, handle)
    hardware = subprocess.check_output(['npu-smi', 'info'], text=True)
    (ROOT / 'npu-before.txt').write_text(hardware)
    if hardware.count('No running processes found') != 2:
        raise RuntimeError('Both NPUs must be free before starting')
    spec = importlib.util.spec_from_file_location('validation', PREVIOUS / 'run_validation.py')
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    validation.ROOT = ROOT
    validation.PLUGIN = PLUGIN
    manifest = json.loads((PREVIOUS / 'revision-v2/local-source-manifest.json').read_text())
    save(ROOT / 'local-source-manifest.json', manifest)
    validation.check_sources()
    plan = json.loads((OLD / 'quality-plan.json').read_text())
    gold = plan['phases'][0]['requests']
    plan['phases'] = [
        {'name': 'gold-cold', 'requests': gold},
        {'name': 'gold-immediate-revisit', 'requests': gold},
        {'name': 'gold-short-churn', 'requests': [{'prefix_id': i} for i in range(500, 504)]},
        {'name': 'gold-short-revisit', 'requests': gold},
    ]
    plan['purpose'] = 'full-HBM cache eligibility diagnostic; not official or general performance score'
    save(ROOT / 'eligibility-plan.json', plan)
    save(ROOT / 'plan.json', {
        'scope': plan['purpose'], 'arms': ['native', 'default', 'forced-admission'],
        'hbm': 'unrestricted automatic allocation gpu-memory-utilization=0.92',
        'cpu_bytes_total': 14 * 1024**3, 'plugin_source_unchanged': True,
        'forced_admission': 'diagnostic only: disable HBM-pressure admission gate; all safety guards unchanged',
        'requests_each': 16, 'gold_each': 12,
    })
    original_config = validation.config
    def config(label, limited, mode):
        cfg, src = original_config(label, limited, mode)
        cfg['additional_sources'] = [str(Path(__file__)), str(ROOT / 'eligibility-plan.json')]
        if label == 'eligibility-forced-admission':
            index = cfg['argv'].index('--kv-transfer-config') + 1
            kv = json.loads(cfg['argv'][index])
            kv['kv_connector_extra_config']['tiering_hbm_pressure_aware'] = False
            cfg['argv'][index] = json.dumps(kv)
        assert '--kv-cache-memory-bytes' not in cfg['argv']
        assert '--num-gpu-blocks-override' not in cfg['argv']
        return cfg, src
    validation.config = config
    validation.event = event
    stop = threading.Event()
    event(stage='starting')
    monitor = threading.Thread(target=metric_monitor, args=(stop,), daemon=True)
    monitor.start()
    try:
        for name in ('native', 'default', 'forced-admission'):
            validation.arm('eligibility-' + name, False,
                'native' if name == 'native' else 'working-set', ROOT / 'eligibility-plan.json')
        event(stage='complete')
    finally:
        stop.set()
        monitor.join(timeout=10)
        (ROOT / 'npu-after.txt').write_text(subprocess.check_output(['npu-smi', 'info'], text=True))
        validation.check_sources()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        event(stage='failed', error=repr(exc))
        raise
