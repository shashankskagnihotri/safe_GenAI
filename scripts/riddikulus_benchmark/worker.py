"""Load one model once and execute its admitted matched pairs in one chunk."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import inspect
import subprocess
import sys
import time
import traceback

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE)]
from scripts.riddikulus_benchmark.common import (
    ARMS, CAMPAIGN, STATE, append, atomic, campaign_lock, check_dispatch, check_stop,
    digest, export_manifests, read, save, utc, verify_arm, verify_sources,
)


def update_arm(cell_id, name, attempt, **values):
    with campaign_lock():
        current = read(STATE); arm = current['cells'][cell_id]['arms'][name]
        if arm.get('attempt') != str(attempt):
            raise RuntimeError('Worker lost exact arm-attempt ownership')
        arm.update(values, updated_utc=utc())
        cell = current['cells'][cell_id]
        cell['status'] = ('GENERATION_COMPLETE' if all(a['status'] == 'GENERATION_COMPLETE' for a in cell['arms'].values())
                          else 'RUNNING' if values.get('status') == 'RUNNING' else cell['status'])
        export_manifests(current); save(current)


def record_flux3_runtime(execution_path, execution, torch):
    """Observe the actual allocated host without additional model predictions."""
    record = {'schema': 'riddikulus-benchmark.allocated-gpu-runtime.v1', 'utc': utc(),
        'job_id': os.environ['SLURM_JOB_ID'], 'host': os.uname().nodename,
        'model_key': execution['model_key'], 'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda, 'gpu_count': torch.cuda.device_count(),
        'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'additional_model_calls': 0,
        'interpretation': 'Driver and CUDA metadata are observations. Successful native prediction parity establishes actual workload execution.'}
    filename = execution_path.parent / 'allocated_gpu_runtime.json'
    record['queries'] = []
    for arguments in (['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'],
                      ['nvidia-smi', '--version']):
        query = {'command': arguments, 'timeout_seconds': 15}
        try:
            result = subprocess.run(arguments, text=True, capture_output=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            query.update(status='FAILED', error=str(exc)); record['queries'].append(query)
            atomic(filename, record)
            raise RuntimeError('Could not inspect the actual FLUX3 allocation driver') from exc
        query.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        record['queries'].append(query); atomic(filename, record)
        if result.returncode:
            raise RuntimeError('Actual FLUX3 allocation driver inspection failed')
    record['driver_versions'] = sorted(set(line.strip() for line in record['queries'][0]['stdout'].splitlines() if line.strip()))
    atomic(filename, record)
    return record


def record_flux3_loaded_sources(execution_path, execution, adapter):
    """Bind loaded native Python classes to their real implementation files."""
    runtime = execution['model_config']['model']; reference = Path(runtime['reference_source_path']).resolve()
    observed = {}
    for role, value in (('pipeline', adapter.pipeline), ('transformer', adapter.pipeline.transformer),
                        ('autoencoder', adapter.pipeline.vae), ('text_encoder', adapter.pipeline.text_encoder),
                        ('scheduler', adapter.pipeline.scheduler)):
        filename = Path(inspect.getfile(type(value))).resolve()
        observed[role] = {'class': type(value).__module__ + '.' + type(value).__qualname__,
                          'source_path': str(filename), 'source_sha256': digest(filename)}
        if filename.is_relative_to(reference):
            relative = str(filename.relative_to(reference))
            if runtime['reference_source_sha256'].get(relative) != observed[role]['source_sha256']:
                raise RuntimeError('Loaded FLUX3 source differs from its official pinned identity')
    expected_pipeline = reference / 'flux3/pipeline.py'
    if observed['pipeline']['source_path'] != str(expected_pipeline):
        raise RuntimeError('Loaded FLUX3 pipeline is not the pinned official implementation')
    atomic(execution_path.parent / 'loaded_native_sources.json', {
        'schema': 'riddikulus-benchmark.loaded-native-sources.v1', 'utc': utc(),
        'job_id': os.environ['SLURM_JOB_ID'], 'sources': observed,
        'reference_source_commit': runtime['reference_source_commit'], 'additional_model_calls': 0})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--execution', required=True)
    args = parser.parse_args(); execution_path = Path(args.execution); execution = read(execution_path)
    job_id = os.environ.get('SLURM_JOB_ID', '')
    if not job_id.isdigit():
        raise RuntimeError('Worker requires its registered Slurm allocation')
    deadline = time.monotonic() + 30
    while True:
        current = read(STATE); chunk = current['chunks'][execution['chunk_id']]
        if str(chunk.get('job', {}).get('job_id')) == job_id:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Scheduler job never registered as this chunk owner')
        time.sleep(2)
    if digest(execution_path) != chunk['execution_sha256']:
        raise RuntimeError('Chunk execution specification changed')
    check_dispatch(current, execution['model_key']); verify_sources(current, execution['model_key'])
    if Path(execution['source_root']).resolve() != SOURCE.resolve():
        raise RuntimeError('Worker executable differs from its sealed generation source')
    import torch
    from riddikulus.loading import load_model
    from riddikulus.runner import run_arm, reset_adapter_for_arm
    if torch.cuda.device_count() != execution['gpu_count']:
        raise RuntimeError('Visible GPU count differs from the approved native placement')
    for item in execution['items']:
        if item['admission_status'] != 'APPROVED':
            raise RuntimeError('An unapproved prompt entered a generation execution')
    if execution['model_key'] == 'flux3_image_klein':
        record_flux3_runtime(execution_path, execution, torch)
    atomic(execution_path.parent / 'worker_progress.json', {'status': 'LOADING_MODEL', 'utc': utc(), 'job_id': job_id})
    started = time.monotonic()
    adapter = load_model(execution['model_config'], device='cuda', dtype=torch.bfloat16)
    load_seconds = time.monotonic() - started
    if execution['model_key'] == 'flux3_image_klein':
        record_flux3_loaded_sources(execution_path, execution, adapter)
    atomic(execution_path.parent / 'model_loading.json', {'seconds': load_seconds, 'job_id': job_id,
        'utc': utc(), 'gpu_count': torch.cuda.device_count(),
        'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    load_charged = False
    active = None
    try:
        for item in execution['items']:
            cell_id = item['cell_id']
            for name in ARMS:
                check_stop(); current = read(STATE); check_dispatch(current, execution['model_key'])
                arm = current['cells'][cell_id]['arms'][name]
                if arm.get('status') == 'GENERATION_COMPLETE':
                    verify_arm(arm); continue
                registered = item['arms'][name]; attempt = Path(registered['attempt'])
                if (arm.get('attempt') != str(attempt) or digest(arm['spec']) != registered['spec_sha256']
                        or arm['spec_sha256'] != registered['spec_sha256']):
                    raise RuntimeError('Registered scientific configuration or attempt changed')
                active = (cell_id, name, attempt)
                with (attempt / 'worker_owner.json').open('x') as stream:
                    import json
                    json.dump({'job_id': job_id, 'pid': os.getpid(), 'host': os.uname().nodename,
                               'utc': utc(), 'execution_sha256': chunk['execution_sha256']}, stream)
                spec = copy.deepcopy(read(arm['spec']))
                if name == 'Riddikulus':
                    baseline = current['cells'][cell_id]['arms']['Baseline']; verify_arm(baseline)
                    spec['_baseline_attempt'] = baseline['attempt']
                spec['_shared_model_load_seconds'] = load_seconds
                reset_adapter_for_arm(adapter)
                update_arm(cell_id, name, attempt, status='RUNNING', scheduler_job_id=job_id)
                atomic(execution_path.parent / 'worker_progress.json', {'status': 'GENERATING', 'utc': utc(),
                    'job_id': job_id, 'cell_id': cell_id, 'arm': name, 'attempt': str(attempt)})
                run_arm(adapter, spec, arm['spec_sha256'], attempt,
                        load_seconds=0.0 if load_charged else load_seconds,
                        resume=Path(registered['resume']) if registered.get('resume') else None)
                load_charged = True
                verification = verify_arm(arm)
                update_arm(cell_id, name, attempt, status='GENERATION_COMPLETE', verification=verification)
                active = None
        atomic(execution_path.parent / '_CHUNK_SUCCESS', {'utc': utc(), 'job_id': job_id,
            'execution_sha256': chunk['execution_sha256'], 'completed_admitted_pairs': len(execution['items']),
            'planned_slots': execution['planned_slots'], 'blocked_slots': execution['blocked_slots']})
    except BaseException as exc:
        failure = {'utc': utc(), 'job_id': job_id, 'type': type(exc).__name__,
            'message': str(exc), 'traceback': traceback.format_exc(),
            'active': [str(v) for v in active] if active else None}
        atomic(execution_path.parent / 'worker_failure.json', failure)
        if active is not None:
            update_arm(*active, status='FAILED', failure=failure)
        raise


if __name__ == '__main__':
    main()
