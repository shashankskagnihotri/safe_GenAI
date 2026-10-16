"""Shared records, ownership and artifact checks for the fixed benchmark."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
import time

ROOT = Path('/ceph/sagnihot/projects/safety_genAI')
CAMPAIGN = ROOT / 'outputs/FINALIZED_WORKING_METHOD_16_OCT'
STATE = CAMPAIGN / 'campaign_state.json'
REGISTRY = CAMPAIGN / 'specification/cohort_registry.jsonl'
APPROVAL = CAMPAIGN / 'specification/dispatch_approval.json'
USER = 'sagnihot'
ARMS = ('Baseline', 'Riddikulus')
MODEL_KEYS = ('flux1_dev', 'flux2_dev', 'cosmos3_super_text2image', 'sd35_large', 'flux3_image_klein')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def jsonlines(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.partial')
    with temporary.open('x') as stream:
        for value in values:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


@contextmanager
def campaign_lock():
    lock = CAMPAIGN / '.controller_lock'; deadline = time.monotonic() + 180
    while True:
        try:
            lock.mkdir(); break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise RuntimeError('Campaign lock still owned: ' + str(lock))
            time.sleep(2)
    atomic(lock / 'owner.json', {'pid': os.getpid(), 'host': os.uname().nodename, 'utc': utc()})
    try:
        yield
    finally:
        (lock / 'owner.json').unlink(); lock.rmdir()


def save(current):
    current['updated_utc'] = utc(); atomic(STATE, current)


def check_stop():
    if (CAMPAIGN / 'STOP').exists() or (CAMPAIGN / 'REPAIR_HOLD').exists():
        raise RuntimeError('Benchmark STOP or REPAIR_HOLD is present')


def command(args, timeout=45):
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {args!r}\n{result.stdout}\n{result.stderr}')
    return result.stdout.strip()


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('Unsafe or missing stable identity: ' + repr(value))
    return value


def registry():
    rows = [json.loads(line) for line in REGISTRY.read_text().splitlines() if line.strip()]
    if len(rows) != 30 or len({r['prompt_id'] for r in rows}) != 30:
        raise RuntimeError('Registry must preserve exactly30 unique original prompts')
    for category in ('nudity', 'violence'):
        if sum(r['dataset_category'].lower() == category for r in rows) != 15:
            raise RuntimeError('Registry must preserve15 prompts in each category')
    for row in rows:
        safe_id(row['prompt_id'])
        if not isinstance(row['original_prompt'], str) or not row['original_prompt']:
            raise RuntimeError('Registry prompt missing')
        if row['admission']['status'] not in {'APPROVED', 'BLOCKED_CONTENT_GENERATION'}:
            raise RuntimeError('Unresolved prompt admission: ' + row['prompt_id'])
    return rows


def check_dispatch(current, model_key):
    check_stop(); approval = read(APPROVAL)
    model = current['models'][model_key]
    expected_source = approval.get('model_source_manifest_sha256', {}).get(model_key, approval.get('source_manifest_sha256'))
    if (approval.get('approved') is not True
            or approval.get('cohort_registry_sha256') != current['cohort_registry_sha256']
            or digest(REGISTRY) != current['cohort_registry_sha256']
            or expected_source != model['source_manifest_sha256']
            or digest(model['source_manifest']) != model['source_manifest_sha256']
            or model_key not in approval.get('approved_model_keys', [])):
        raise RuntimeError('Root dispatch approval does not authorize these exact inputs/source/model')


def verify_sources(current, model_key):
    model = current['models'][model_key]
    manifest = read(model['source_manifest'])
    if digest(model['source_manifest']) != model['source_manifest_sha256']:
        raise RuntimeError('Source manifest changed')
    for relative, expected in manifest['source_sha256'].items():
        if digest(Path(model['source_root']) / relative) != expected:
            raise RuntimeError('Sealed source changed: ' + relative)
    framework = manifest.get('framework_provenance', {}).get(model['environment_python'])
    if framework is None: raise RuntimeError('Pinned native framework provenance is missing')
    for filename, expected in framework['native_files_sha256'].items():
        if digest(filename) != expected: raise RuntimeError('Pinned native framework file changed: ' + filename)


def verify_arm(arm, *, full=False):
    if digest(arm['spec']) != arm['spec_sha256']:
        raise RuntimeError('Immutable arm specification changed')
    attempt = Path(arm['attempt']); receipt_path = attempt / '_GENERATION_SUCCESS'
    receipt = read(receipt_path)
    if receipt.get('spec_sha256') != arm['spec_sha256']:
        raise RuntimeError('Generation receipt differs from the registered specification')
    files = receipt.get('artifact_sha256')
    if not isinstance(files, dict) or not {'image.png', 'timing.json', 'numerical_audit.json'} <= files.keys():
        raise RuntimeError('Generation receipt omits required evidence')
    verified_before = arm.get('verification', {}).get('generation_receipt_sha256') == digest(receipt_path)
    verify_files = files if full or not verified_before else {name: files[name] for name in ('image.png', 'timing.json', 'numerical_audit.json')}
    for relative, expected in verify_files.items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts or digest(attempt / relative) != expected:
            raise RuntimeError('Generated artifact mismatch: ' + relative)
    audit = read(attempt / 'numerical_audit.json')
    if audit.get('passed') is not True or audit.get('native_updates') != read(arm['spec'])['native_steps']:
        raise RuntimeError('Numerical generation audit failed or updates incomplete')
    return {'generation_receipt_sha256': digest(receipt_path), 'image_sha256': files['image.png'],
            'timing_sha256': files['timing.json'], 'numerical_audit_sha256': files['numerical_audit.json'],
            'verified_utc': utc()}


def queue():
    result = {}
    for line in command(['squeue', '-u', USER, '-h', '-o', '%i|%u|%k|%T|%R']).splitlines():
        fields = line.split('|', 4)
        if len(fields) != 5:
            raise RuntimeError('Unrecognized scheduler queue row')
        result[fields[0]] = dict(zip(('job_id', 'user', 'comment', 'state', 'node_or_reason'), fields))
    return result


def accounting(job_id):
    rows = []
    raw = command(['sacct', '-X', '-n', '-P', '-j', str(job_id),
                   '--format=JobIDRaw,User,State,ExitCode,ElapsedRaw,Start,End,Submit,NodeList,AllocTRES%256'])
    for line in raw.splitlines():
        fields = line.split('|')
        if len(fields) >= 10 and fields[0] == str(job_id):
            rows.append(dict(zip(('job_id', 'user', 'state', 'exit_code', 'elapsed_seconds',
                                  'start', 'end', 'submit', 'nodes', 'alloc_tres'), fields[:10])))
    if len(rows) != 1 or rows[0]['user'] != USER:
        raise RuntimeError('Missing or ambiguous owned root scheduler accounting')
    rows[0]['state'] = rows[0]['state'].split()[0].rstrip('+')
    return rows[0]


def verify_live(record, job):
    if record['user'] != USER or record['job_id'] != job['job_id'] or record['comment'] != job['comment']:
        raise RuntimeError('Live scheduler ownership differs from the registered job')


def cancel_owned(job):
    uid = pwd.getpwnam(USER).pw_uid
    text = command(['scontrol', 'show', 'job', job['job_id'], '-o'])
    values = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)', text))
    if (values.get('JobId') != job['job_id'] or values.get('UserId') != f'{USER}({uid})'
            or values.get('Comment') != job['comment']):
        raise RuntimeError('STOP refuses cancellation of an unowned allocation')
    command(['scancel', job['job_id']])


def export_manifests(current):
    expected, completed = [], []
    for cell in sorted(current['cells'].values(), key=lambda c: (c['cohort_index'], c['model_key'])):
        for name, arm in cell['arms'].items():
            row = {k: cell[k] for k in ('cell_id', 'model_key', 'prompt_id', 'category', 'original_prompt', 'cohort_index')}
            row.update(image_id=cell['cell_id'] + '/' + name, arm=name,
                       status=arm.get('status', cell['status']), reason=cell.get('reason'),
                       spec_path=arm.get('spec'), spec_sha256=arm.get('spec_sha256'),
                       attempt=arm.get('attempt'), provenance=cell.get('provenance', {}))
            expected.append(row)
            if arm.get('status') == 'GENERATION_COMPLETE':
                verification = arm['verification']; attempt = Path(arm['attempt'])
                completed.append(dict(row, image_path=str(attempt / 'image.png'),
                    image_sha256=verification['image_sha256'],
                    generation_receipt_path=str(attempt / '_GENERATION_SUCCESS'),
                    generation_receipt_sha256=verification['generation_receipt_sha256'],
                    timing_path=str(attempt / 'timing.json'), timing_sha256=verification['timing_sha256']))
    jsonlines(CAMPAIGN / 'expected_image_manifest.jsonl', expected)
    jsonlines(CAMPAIGN / 'image_manifest.jsonl', completed)
    current['coverage'] = {'expected_pairs': 150, 'expected_images': 300,
        'completed_images': len(completed), 'admitted_prompts': sum(r['admission']['status'] == 'APPROVED' for r in registry()),
        'blocked_prompts': sum(r['admission']['status'] != 'APPROVED' for r in registry()),
        'status_counts': {status: sum(r['status'] == status for r in expected) for status in sorted({r['status'] for r in expected})}}
