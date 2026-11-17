"""Server-local validation; owned supervisors, immutable inputs, no Git writes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/root/CGCL/kv-tiering-working-set-20261116')
OLD = Path('/root/CGCL/kv-tiering-recovery-validation-20261112')
PY = '/root/CGCL/kv-tiering-benefit-search-20261108/runtime-venv-v1/bin/python'
PLUGIN = ROOT / 'plugin'
BASE = Path('/root/CGCL/vllm-hust-kv-tiering-engineering-20261105')
PORT = 18236


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def event(**kw):
    obj = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **kw}
    save(ROOT / 'STATUS.json', obj)
    with (ROOT / 'events.jsonl').open('a') as f:
        f.write(json.dumps(obj) + '\n')
    print(json.dumps(obj), flush=True)


def check_sources():
    manifest = json.loads((ROOT / 'local-source-manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert hashlib.sha256((PLUGIN / name).read_bytes()).hexdigest() == expected, name


def config(label, limited, mode):
    cfg = json.loads((OLD / 'hybrid-cpu-qual.argv.json').read_text())
    a = cfg['argv']
    i = a.index('--speculative-config')
    del a[i:i + 2]
    if not limited:
        i = a.index('--kv-cache-memory-bytes')
        del a[i:i + 2]
        a += ['--gpu-memory-utilization', '0.92']
        a[a.index('--max-model-len') + 1] = '65536'
        a[a.index('--max-num-seqs') + 1] = '128'
    i = a.index('--kv-transfer-config')
    if mode == 'native':
        del a[i:i + 2]
    else:
        kv = json.loads(a[i + 1])
        extra = kv['kv_connector_extra_config']
        extra['cpu_bytes_to_use'] = (2 if limited else 14) * 1024**3
        if mode == 'working-set':
            extra.update(tiering_working_set_aware=True,
                         tiering_working_set_window_sec=600.0,
                         tiering_working_set_max_entries=64000,
                         tiering_working_set_max_new_bytes=64 * 1024**2)
        a[i + 1] = json.dumps(kv)
    src = BASE if mode == 'original' else PLUGIN
    paths = cfg['env']['PYTHONPATH'].split(':')
    cfg['env']['PYTHONPATH'] = ':'.join([str(src / 'src'), *paths[1:]])
    cfg['label'] = label
    cfg['additional_sources'] = [str(Path(__file__)), str(ROOT / 'performance-plan.json')]
    return cfg, src


def wait_for(predicate, timeout, description):
    start = time.monotonic()
    while not predicate():
        if time.monotonic() - start > timeout:
            raise TimeoutError(description)
        time.sleep(2)


def arm(label, limited, mode, plan):
    check_sources()
    cfg, src = config(label, limited, mode)
    invocation = ROOT / (label + '.argv.json')
    save(invocation, cfg)
    run = ROOT / (label + '-server')
    with (ROOT / (label + '-supervisor.log')).open('x') as log:
        supervisor = subprocess.Popen([PY, '-u', str(OLD / 'serve_supervisor.py'),
            '--run-dir', str(run), '--argv-file', str(invocation), '--cwd', str(src),
            '--port', str(PORT)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(ROOT / (label + '-pid.json'), {'pid': supervisor.pid})
    event(stage='server_starting', arm=label, pid=supervisor.pid)
    try:
        wait_for(lambda: (run / 'ready.json').exists() or supervisor.poll() is not None, 1250, 'startup')
        if not (run / 'ready.json').exists():
            raise RuntimeError(label + ' startup failed')
        event(stage='client_running', arm=label)
        with (ROOT / (label + '-client.log')).open('x') as log:
            client = subprocess.run([PY, '-u', str(OLD / 'run_quality.py'), '--root', str(ROOT),
                '--label', label, '--plan', str(plan)], stdout=log, stderr=subprocess.STDOUT, timeout=2400)
        if client.returncode:
            raise RuntimeError(label + ' client failed')
        records = [json.loads(line) for line in (ROOT / (label + '-client/requests.jsonl')).read_text().splitlines()]
        assert all(r['valid'] for r in records)
        gold = [r for r in records if 'expected' in r]
        assert all(r['output_text'].strip().splitlines()[0].strip() == r['expected'] for r in gold), 'golden answer mismatch'
        save(ROOT / (label + '-audit.json'), {'requests': len(records), 'valid': len(records),
             'golden_cases': len(gold), 'golden_first_line_passed': len(gold), 'scope': 'retrieval diagnostic only'})
        event(stage='arm_passed', arm=label, requests=len(records), golden=len(gold))
    finally:
        if supervisor.poll() is None:
            (run / 'stop.request').touch()
            supervisor.wait(timeout=130)
        if supervisor.returncode:
            raise RuntimeError(label + ' supervisor did not exit cleanly')
    check_sources()


def main():
    # Exclusive create: a second launch cannot start another controller.
    with (ROOT / 'controller-start.json').open('x') as f:
        json.dump({'pid': os.getpid(), 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, f)
    check_sources()
    identities = {}
    for label, path in [('core', Path('/root/CGCL/kv-tiering-bugfix-perf-20261106/core-mtp-offload-safe')),
                        ('ascend', Path('/root/CGCL/kv-tiering-bugfix-perf-20261106/ascend-feedback-identity-safe')),
                        ('original_plugin', BASE), ('modified_plugin', PLUGIN)]:
        identities[label] = {'root': str(path), 'commit': subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip(),
                            'status': subprocess.check_output(['git', '-C', str(path), 'status', '--short'], text=True)}
    save(ROOT / 'runtime-git-identity.json', identities)
    shutil.copyfile(OLD / 'quality-plan.json', ROOT / 'quality-plan.json')
    plan = {'seed': 20261116, 'prefix_len': 32768, 'suffix_len': 256, 'output_len': 32,
        'purpose': 'synthetic complete-workload engineering ablation; NOT official dataset or agent quality; full HBM in all arms',
        'phases': [
            {'name': 'cold-pressure', 'concurrency': 64, 'requests': [{'prefix_id': i} for i in range(80)]},
            {'name': 'revisit', 'concurrency': 8, 'requests': [{'prefix_id': i} for i in range(8)]},
            {'name': 'new-pressure', 'concurrency': 64, 'requests': [{'prefix_id': i} for i in range(80, 112)]},
        ]}
    save(ROOT / 'performance-plan.json', plan)
    save(ROOT / 'plan.json', {'scope': 'real-host and NPU qualification then full-HBM synthetic ablation',
        'performance_order': ['native', 'original', 'working-set'], 'repetitions': 1,
        'no_general_speedup_claim': True, 'limited_hbm_scope': 'quality diagnostic only',
        'full_hbm_utilization': .92, 'no_mtp': True, 'performance_requests_each_arm': 120,
        'cpu_bytes_full_hbm': 14 * 1024**3, 'working_set_window_sec': 600,
        'working_set_request_new_bytes': 64 * 1024**2, 'excludes_startup_from_client_phase_timing': True})
    for mode in ('native', 'working-set'):
        arm('qual-' + mode, True, mode, ROOT / 'quality-plan.json')
    for mode in ('native', 'original', 'working-set'):
        arm('full-' + mode, False, mode, ROOT / 'performance-plan.json')
    event(stage='complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        event(stage='failed', error=repr(exc))
        raise
