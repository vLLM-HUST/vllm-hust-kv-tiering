"""User-requested fixed first nine rounds,1142/2215 requests,128 task pool."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

B = Path('/root/CGCL/kv-tiering-performance-recheck-20261116')
ROOT = B / 'half-confirmation-v1'
ORIGINAL = B / 'pressure-confirmation-v3'
PRESSURE = B / 'pressure-measurements-v1'
PREVIOUS = Path('/root/CGCL/kv-tiering-working-set-20261116')
PY = '/root/CGCL/kv-tiering-benefit-search-20261108/runtime-venv-v1/bin/python'


def load(p):
    return json.loads(p.read_text())


def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    with (ROOT / 'controller-start.json').open('x') as f:
        json.dump({'pid': os.getpid()}, f)
    assert subprocess.check_output(['npu-smi', 'info'], text=True).count('No running processes found') == 2
    assert load(ORIGINAL / 'USER_REQUESTED_SHORTENING.json')['requests'] == 1142
    assert load(ORIGINAL / 'r2-native-server/exit.json')['server_returncode'] == 0
    plan = load(ORIGINAL / 'swe-plan.json')
    plan['phases'] = plan['phases'][:9]
    assert sum(len(x['requests']) for x in plan['phases']) == 1142
    save(ROOT / 'swe-plan.json', plan)
    save(ROOT / 'plan.json', {'scope': 'User shortened confirmation; fixed first9 complete rounds,not64 complete tasks',
        'requests_each': 1142, 'original_requests': 2215, 'source_task_pool': 128,
        'preserve_cold_start': True, 'selection': 'first9 already completed Native rounds; no output/performance ranking',
        'native_scope': 'Matched completed prefix from user-stopped original Native; later/inflight requests excluded',
        'timing': 'phase wall comparison only; Native whole client wall unavailable for prefix',
        'candidate': 'default pressure gate,working set enabled,14GiB CPU,fullHBM0.92'})
    shutil.copyfile(PRESSURE / 'local-source-manifest.json', ROOT / 'local-source-manifest.json')
    native = ROOT / 'half-native-client'
    native.mkdir()
    phases = {x['name'] for x in plan['phases']}
    rows = [load_line for line in (ORIGINAL / 'r2-native-client/requests.jsonl').read_text().splitlines()
        if (load_line := json.loads(line))['phase'] in phases]
    assert len(rows) == 1142 and all(x['valid'] for x in rows)
    (native / 'requests.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    save(native / 'plan.json', plan)
    save(native / 'complete.json', {'scope': 'complete selected1142-request prefix; original2215-request run intentionally stopped',
        'user_shortened': True, 'requests': 1142})
    for name in phases:
        shutil.copyfile(ORIGINAL / 'r2-native-client' / (name + '-summary.json'), native / (name + '-summary.json'))
    shutil.copytree(ORIGINAL / 'r2-native-server', ROOT / 'half-native-server')
    shutil.copyfile(ORIGINAL / 'r2-native.argv.json', ROOT / 'half-native.argv.json')
    campaign = module('half_campaign', B / 'swe32-measurements-v1/run_swe_campaign.py')
    campaign.ROOT = ROOT
    validation = module('half_validation', PRESSURE / 'validation-source.py')
    validation.ROOT, validation.PLUGIN, validation.event = ROOT, PREVIOUS / 'revision-v2/plugin', campaign.event
    validation.OLD = PRESSURE / 'fixtures'
    original_config = validation.config
    def config(label, limited, mode):
        cfg, src = original_config(label, limited, mode)
        cfg['argv'][cfg['argv'].index('--max-model-len') + 1] = '262144'
        cfg['additional_sources'] = [str(Path(__file__)), str(ROOT / 'swe-plan.json'), str(ROOT / 'plan.json')]
        return cfg, src
    validation.config = config
    monitor = module('half_monitor', B / 'run_recheck.py')
    monitor.ROOT = ROOT
    stop = threading.Event()
    thread = threading.Thread(target=monitor.metric_monitor, args=(stop,), daemon=True)
    thread.start()
    try:
        validation.arm('half-working-set', False, 'working-set', ROOT / 'swe-plan.json')
    finally:
        stop.set()
        thread.join(timeout=10)
    analysis = module('half_analysis', B / 'analyze_recheck.py').analyze(ROOT)
    assert analysis['all_complete_inputs_identical']
    for a in analysis['arms'].values():
        assert a['requests'] == 1142 and a['completed_all'] and not a['invalid'] and not a['usage_prompt_mismatches']
        assert sum(a['prompt_by_source_tokens'].values()) == a['logical_prompt_tokens']
        assert a['server_exit']['server_returncode'] == 0 and not a['server_exit']['forced_shutdown']
    n, c = analysis['arms']['half-native'], analysis['arms']['half-working-set']
    assert n['reported_kv_gib_per_card'] == c['reported_kv_gib_per_card']
    assert n['reported_cache_tokens'] == c['reported_cache_tokens']
    argv = [load(ROOT / (name + '.argv.json'))['argv'] for name in ['half-native','half-working-set']]
    for a in argv:
        assert '--kv-cache-memory-bytes' not in a and '--num-gpu-blocks-override' not in a
        assert a[a.index('--gpu-memory-utilization') + 1] == '0.92'
        if '--kv-transfer-config' in a:
            i = a.index('--kv-transfer-config'); del a[i:i+2]
    assert argv[0] == argv[1]
    comparison = {k + '_change_pct': 100 * (c[k] / n[k] - 1) for k in ['output_tok_s','ttft_mean_s','ttft_p95_s','tpot_mean_s']}
    comparison['prefill_saved_tokens'] = n['counters']['vllm:request_prefill_kv_computed_tokens_sum'] - c['counters']['vllm:request_prefill_kv_computed_tokens_sum']
    save(ROOT / 'analysis.json', analysis)
    save(ROOT / 'comparison.json', comparison)
    subprocess.run([PY, str(B / 'capture_identity.py'), 'half-final'], check=True)
    first, last = load(B / 'campaign-start-identity.json'), load(B / 'half-final-identity.json')
    for key in ['plugin_sha256','repositories','model_files','weight_stat_only_not_full_byte_hash','client_sha256','supervisor_sha256']:
        assert first[key] == last[key], key
    hardware = subprocess.check_output(['npu-smi','info'], text=True)
    (ROOT / 'npu-final.txt').write_text(hardware)
    assert hardware.count('No running processes found') == 2
    save(ROOT / 'final-audit.json', {'all_selected_requests_valid': True,'same_inputs_and_shared_cli':True,
        'source_runtime_model_unchanged':True,'two_npus_idle':True,'scope':'half-prefix confirmation only','comparison':comparison})
    campaign.event(stage='complete', requests_each=1142, comparison=comparison)
    save(B / 'GLOBAL_STATUS.json', {'stage':'complete','confirmation_scope':'user-shortened1142-request fixed prefix',
        'audit':str(ROOT/'final-audit.json'),'npus_idle':True})


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save(ROOT / 'STATUS.json', {'stage':'failed','error':repr(exc)})
        raise
