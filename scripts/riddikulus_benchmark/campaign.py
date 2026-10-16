"""Prepare, dispatch and durably supervise immutable matched benchmark chunks."""
from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
import runpy
import shutil
import sys
import time
import traceback

SOURCE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(SOURCE), str(SOURCE / 'src')]
from scripts.riddikulus_benchmark.common import (
    APPROVAL, ARMS, CAMPAIGN, MODEL_KEYS, REGISTRY, ROOT, STATE, accounting, append,
    atomic, campaign_lock, cancel_owned, canonical_digest, check_dispatch, check_stop,
    command, digest, export_manifests, queue, read, registry, safe_id, save, utc,
    verify_arm, verify_live, verify_sources,
)

TRANSIENT = {'NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED'}
TERMINAL = {'COMPLETE_APPROVED_ITEMS', 'BLOCKED_CONTENT_GENERATION', 'BLOCKED_MODEL',
            'NEEDS_DIAGNOSIS', 'INFRASTRUCTURE_RETRIES_EXHAUSTED', 'CONTINUATIONS_EXHAUSTED'}


def reference_cell(model_key, model, row):
    """Preserve model-native initial-noise contracts in every matched arm."""
    cfg = model['model_config']
    result = {'model': model_key, 'model_revision': cfg['model'].get('revision'),
              'original_prompt': row['original_prompt'],
              'original_prompt_sha256': __import__('hashlib').sha256(row['original_prompt'].encode()).hexdigest(),
              'prompt_id': row['prompt_id'], 'seed': row.get('seed', 0)}
    if model_key == 'flux2_dev':
        contract = model.get('initial_latent_contract')
        if not isinstance(contract, dict) or set(contract) != {'initial_latent_reference', 'expected_initial_latent_sha256'}:
            raise RuntimeError('FLUX.2 requires its pinned canonical initial-latent contract')
        pin = contract['initial_latent_reference']
        if (pin['model'], pin['model_revision'], pin['seed']) != (model_key, result['model_revision'], result['seed']):
            raise RuntimeError('Canonical initial-latent identity differs from the requested model/seed')
        artifact = (ROOT / pin['path']).resolve()
        if not artifact.is_relative_to((ROOT / 'outputs').resolve()) or digest(artifact) != pin['file_sha256']:
            raise RuntimeError('Canonical initial-latent artifact is absent or changed')
        result.update(copy.deepcopy(contract))
    return result


def snapshot(destination):
    files = []
    for directory in ('src/riddikulus', 'scripts/riddikulus_benchmark', 'configs/riddikulus'):
        base = ROOT / directory
        if not base.is_dir() and directory == 'configs/riddikulus': continue
        if not base.is_dir():
            raise RuntimeError('Core or benchmark source is not ready: ' + directory)
        files.extend(p for p in sorted(base.rglob('*')) if p.is_file() and p.suffix in {'.py', '.sbatch', '.json', '.yaml', '.yml'} and '__pycache__' not in p.parts)
    hashes = {}
    for path in files:
        if path.suffix == '.py': ast.parse(path.read_text(), str(path))
        relative = path.relative_to(ROOT); target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        hashes[str(relative)] = digest(target)
    return hashes


def framework_provenance(models):
    """Fingerprint installed native pipeline/scheduler code without loading weights."""
    result = {}
    program = "import importlib.metadata as m,json;d=m.distribution('diffusers');print(json.dumps({'packages':{p:m.version(p) for p in ('torch','diffusers','transformers','accelerate','safetensors')},'diffusers_root':str(d.locate_file('diffusers'))}))"
    for model in models.values():
        if model['status'] != 'READY': continue
        environment = model['environment_python']
        if environment not in result:
            observed = json.loads(command([environment, '-c', program]))
            base = Path(observed['diffusers_root'])
            relative = ('__init__.py', 'pipelines/flux/pipeline_flux.py', 'pipelines/flux2/pipeline_flux2.py',
                        'pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py',
                        'schedulers/scheduling_flow_match_euler_discrete.py', 'schedulers/scheduling_unipc_multistep.py',
                        'models/transformers/transformer_flux.py', 'models/transformers/transformer_flux2.py',
                        'models/transformers/transformer_sd3.py')
            observed['native_files_sha256'] = {str(base / name): digest(base / name) for name in relative if (base / name).is_file()}
            result[environment] = observed
        observed = result[environment]
        runtime = model['model_config']['model']
        source_hashes = runtime.get('reference_source_sha256')
        if runtime.get('adapter') == 'flux3' and not source_hashes:
            raise RuntimeError('FLUX3 native provenance requires the pinned official source hashes')
        if source_hashes:
            source_root = Path(runtime['reference_source_path']).resolve()
            authenticated = {}
            for relative, expected in source_hashes.items():
                if Path(relative).is_absolute() or '..' in Path(relative).parts:
                    raise RuntimeError('External native source path escapes its pinned root')
                filename = source_root / relative
                if not filename.is_file() or digest(filename) != expected:
                    raise RuntimeError('Pinned official native source changed: ' + str(filename))
                authenticated[str(filename)] = expected
            observed['native_files_sha256'].update(authenticated)
            observed.setdefault('external_official_sources', {})[model['model_key']] = {
                'reference_source_path': str(source_root),
                'reference_source_commit': runtime['reference_source_commit'],
                'reference_source_sha256': copy.deepcopy(source_hashes),
                'actual_files_sha256': authenticated,
                'pipeline_source_path': str(source_root / 'flux3/pipeline.py') if runtime.get('adapter') == 'flux3' else None,
            }
    return result


def initialize(model_registry_path):
    with campaign_lock():
        check_stop(); rows = registry()
        if STATE.exists():
            existing = read(STATE)
            if existing['cohort_registry_sha256'] != digest(REGISTRY):
                raise RuntimeError('Existing benchmark cohort differs')
            return existing
        model_registry_path = Path(model_registry_path)
        model_records = read(model_registry_path)
        if isinstance(model_records, dict) and 'models' in model_records: model_records = model_records['models']
        if isinstance(model_records, list): model_records = {row['model_key']: row for row in model_records}
        if set(model_records) != set(MODEL_KEYS):
            raise RuntimeError('Exactly the five registered model identities are required')
        method_module = runpy.run_path(str(ROOT / 'src/riddikulus/selection.py'))
        hierarchy_source = ROOT / 'src/riddikulus/data/hierarchy_h4_atomic.json'
        specification = CAMPAIGN / 'specification'; hierarchy = specification / 'hierarchy_h4_atomic.json'
        if hierarchy.exists():
            if digest(hierarchy) != digest(hierarchy_source): raise RuntimeError('Existing hierarchy differs')
        else: shutil.copyfile(hierarchy_source, hierarchy)
        controller = read(ROOT / 'src/riddikulus/data/controller_config.json')
        source = specification / 'generation_source/SOURCE_0001'
        if source.exists(): raise RuntimeError('Unregistered generation source already exists')
        hashes = snapshot(source); source_manifest = specification / 'generation_source/SOURCE_0001_manifest.json'
        atomic(source_manifest, {'schema': 'riddikulus-benchmark.generation-source.v1', 'utc': utc(), 'source_sha256': hashes,
                                'framework_provenance': framework_provenance(model_records)})
        current = {'schema': 'riddikulus-benchmark.campaign.v1', 'status': 'PREPARED_AWAITING_ROOT_DISPATCH',
            'created_utc': utc(), 'cohort_registry_sha256': digest(REGISTRY),
            'model_registry_path': str(model_registry_path.resolve()), 'model_registry_sha256': digest(model_registry_path),
            'source_root': str(source), 'source_manifest': str(source_manifest), 'source_manifest_sha256': digest(source_manifest),
            'models': {}, 'cells': {}, 'chunks': {}, 'monitor': None,
            'infrastructure_retry_limit': 3, 'planned_continuation_limit': 4,
            'scientific_configuration_policy': 'One fixed method per model; scientific failures are retained, never regenerated for success.'}
        by_category = {c: sorted((r for r in rows if r['dataset_category'].lower() == c), key=lambda r: r['cohort_index'])
                       for c in ('nudity', 'violence')}
        for key in MODEL_KEYS:
            model = copy.deepcopy(model_records[key]); model['model_key'] = key
            if model.get('status') not in {'READY', 'BLOCKED_MODEL'}: raise RuntimeError('Model needs an explicit READY/BLOCKED_MODEL status')
            model.update(source_root=str(source), source_manifest=str(source_manifest),
                         source_manifest_sha256=current['source_manifest_sha256'])
            current['models'][key] = model
            for row in rows:
                category = row['dataset_category'].upper(); prompt_id = safe_id(row['prompt_id']); cell_id = key + '/' + prompt_id
                content_blocked = row['admission']['status'] != 'APPROVED'
                status = ('BLOCKED_CONTENT_GENERATION' if content_blocked else
                          'BLOCKED_MODEL' if model['status'] != 'READY' else 'READY')
                reason = row['admission']['reason'] if content_blocked else model.get('reason') if model['status'] != 'READY' else None
                cell = {'cell_id': cell_id, 'model_key': key, 'prompt_id': prompt_id, 'category': category,
                    'cohort_index': row['cohort_index'], 'original_prompt': row['original_prompt'],
                    'seed': row.get('seed', 0), 'admission': copy.deepcopy(row['admission']),
                    'status': status, 'reason': reason, 'arms': {}}
                cfg = model.get('model_config')
                if cfg:
                    cell['provenance'] = {'model_id': cfg['model']['model_id'], 'model_revision': cfg['model'].get('revision'),
                        'seed': row.get('seed', 0), 'generation': cfg['generation'], 'source_manifest_sha256': current['source_manifest_sha256']}
                for name in ARMS:
                    run = CAMPAIGN / category / prompt_id / key / name / 'RUN_0001'
                    arm = {'status': status, 'attempt': None, 'attempt_history': [], 'run_root': str(run)}
                    if cfg and model['status'] == 'READY':
                        steps = cfg['generation']['num_inference_steps']; warm = 3 if key == 'flux3_image_klein' else 10
                        method = (method_module['default_baseline']() if name == 'Baseline'
                                  else method_module['default_method'](num_inference_steps=steps, warmup_steps=warm))
                        spec = {'schema': 'riddikulus-benchmark.run.v1', 'run_id': 'RUN_0001', 'arm_name': name,
                            'model_key': key, 'method_enabled': name != 'Baseline', 'method': method,
                            'model_config': copy.deepcopy(cfg), 'controller_config': copy.deepcopy(controller),
                            'hierarchy_path': str(hierarchy), 'hierarchy_sha256': digest(hierarchy),
                            'reference_cell': reference_cell(key, model, row),
                            'native_steps': steps, 'seed': row.get('seed', 0),
                            'preview_steps': [2,3,5,10] if steps == 16 else [5,9,10,14,15,28,29,30],
                            'checkpoint_steps': [0,2,3,5,10,15] if steps == 16 else [0,5,9,10,14,15,25,28,29,30,40,49],
                            'project_root': str(ROOT), 'stop_paths': [str(CAMPAIGN / 'STOP'), str(CAMPAIGN / 'REPAIR_HOLD')],
                            'source_root': str(source), 'source_sha256': hashes,
                            'source_manifest': str(source_manifest), 'source_manifest_sha256': current['source_manifest_sha256'],
                            'cohort_registry_sha256': current['cohort_registry_sha256'], 'content_admission': copy.deepcopy(row['admission']),
                            'environment_python': model['environment_python'], 'expected_gpu_count': model['gpu_count']}
                        path = run / 'specification/run.json'; atomic(path, spec)
                        arm.update(spec=str(path), spec_sha256=digest(path))
                    cell['arms'][name] = arm
                current['cells'][cell_id] = cell
            for index in range(3):
                slots = by_category['nudity'][index*5:(index+1)*5] + by_category['violence'][index*5:(index+1)*5]
                chunk_id = key + f'_CHUNK_{index+1:04d}'
                ids = [key + '/' + r['prompt_id'] for r in slots]
                runnable = [i for i in ids if current['cells'][i]['status'] == 'READY']
                current['chunks'][chunk_id] = {'chunk_id': chunk_id, 'model_key': key, 'cell_ids': ids,
                    'status': 'READY' if runnable else 'BLOCKED_MODEL' if model['status'] != 'READY' else 'BLOCKED_CONTENT_GENERATION',
                    'job': None, 'job_history': [], 'retry_count': 0, 'continuation_count': 0}
        export_manifests(current); save(current)
        return current


def launch(chunk_id):
    with campaign_lock():
        check_stop(); current = read(STATE); chunk = current['chunks'][chunk_id]; model = current['models'][chunk['model_key']]
        check_dispatch(current, chunk['model_key']); verify_sources(current, chunk['model_key'])
        if model['status'] != 'READY' or chunk['status'] not in {'READY', 'RETRY_READY', 'CONTINUATION_READY'}:
            raise RuntimeError('Chunk is not ready for an authorized launch')
        visible = queue()
        if any(job['job_id'] in visible for job in chunk['job_history']): raise RuntimeError('An earlier chunk allocation still owns these outputs')
        directory = CAMPAIGN / 'CHUNKS' / chunk_id
        histories = sorted(directory.glob('EXECUTION_*')); number = len(histories) + 1
        execution_dir = directory / f'EXECUTION_{number:04d}'; execution_dir.mkdir(parents=True, exist_ok=False)
        items, blocked = [], []
        for cell_id in chunk['cell_ids']:
            cell = current['cells'][cell_id]
            if cell['admission']['status'] != 'APPROVED': blocked.append(cell_id); continue
            if all(arm['status'] == 'GENERATION_COMPLETE' for arm in cell['arms'].values()):
                for arm in cell['arms'].values(): verify_arm(arm)
                continue
            if cell['status'] == 'BLOCKED_MODEL': raise RuntimeError('Model remains blocked')
            registered = {}
            for name in ARMS:
                arm = cell['arms'][name]
                if arm['status'] == 'GENERATION_COMPLETE':
                    verify_arm(arm); registered[name] = dict(arm, reuse_completed=True); continue
                previous = arm.get('attempt'); resume = None
                if previous and (Path(previous) / 'checkpoints/latest.json').exists(): resume = str(Path(previous) / 'checkpoints/latest.json')
                count = len(arm['attempt_history']) + 1
                attempt = Path(arm['run_root']) / 'ATTEMPTS' / f'ATTEMPT_{count:04d}'
                attempt.mkdir(parents=True, exist_ok=False)
                arm['attempt_history'].append(str(attempt))
                arm.update(attempt=str(attempt), status='SUBMITTED', resume=resume)
                registered[name] = {'spec': arm['spec'], 'spec_sha256': arm['spec_sha256'], 'attempt': str(attempt), 'resume': resume}
            items.append({'cell_id': cell_id, 'admission_status': cell['admission']['status'], 'arms': registered})
        if not items:
            chunk['status'] = 'COMPLETE_APPROVED_ITEMS'; export_manifests(current); save(current)
            return chunk
        execution = {'schema': 'riddikulus-benchmark.chunk-execution.v1', 'chunk_id': chunk_id,
            'model_key': chunk['model_key'], 'model_config': model['model_config'], 'gpu_count': model['gpu_count'],
            'source_root': model['source_root'], 'planned_slots': len(chunk['cell_ids']), 'blocked_slots': blocked,
            'items': items, 'created_utc': utc()}
        execution_path = execution_dir / 'execution.json'; atomic(execution_path, execution)
        comment = 'riddikulus-benchmark:' + chunk_id + f':{number}'
        args = ['sbatch', '--parsable', '--no-requeue', '--account=ml-staff', '--qos=walltime5d',
            '--partition=gpu-vram-94gb', '--nodes=1', '--ntasks=1', '--cpus-per-task=16',
            '--gpus-per-task=nvidia_h100_nvl:' + str(model['gpu_count']), '--mem-per-gpu=128G',
            '--time=' + ('12:00:00' if model['gpu_count'] == 2 else '08:00:00'),
            '--chdir=' + str(ROOT), '--job-name=rid-' + chunk_id, '--comment=' + comment,
            '--output=' + str(execution_dir / 'generation_%j.out'), '--error=' + str(execution_dir / 'generation_%j.err'),
            str(Path(model['source_root']) / 'scripts/riddikulus_benchmark/generate.sbatch'),
            str(execution_path), model['source_root'], model['environment_python'], str(model['gpu_count'])]
        journal = {'status': 'SUBMITTING', 'command': args, 'comment': comment, 'created_utc': utc()}
        submission_path = execution_dir / 'submission.json'; atomic(submission_path, journal)
        chunk.update(status='SUBMITTING', execution_path=str(execution_path), execution_sha256=digest(execution_path)); save(current)
        result = command(args); job_id = result.split(';')[0]
        if not job_id.isdigit(): raise RuntimeError('Ambiguous scheduler submission: ' + result)
        job = {'job_id': job_id, 'comment': comment, 'submitted_utc': utc()}
        journal.update(status='SUBMITTED', **job); atomic(submission_path, journal)
        chunk.update(status='SUBMITTED', job=job); chunk['job_history'].append(job)
        export_manifests(current); save(current)
        return chunk


def enable_model(model_key, model_registry_path):
    """Bind a previously blocked model to a new source seal, leaving others fixed."""
    with campaign_lock():
        check_stop(); current = read(STATE); old = current['models'][model_key]
        records = read(model_registry_path); records = records.get('models', records)
        model = copy.deepcopy(records[model_key])
        if old['status'] != 'BLOCKED_MODEL' or model['status'] != 'READY':
            raise RuntimeError('Late model registration requires BLOCKED_MODEL to explicitly READY')
        old_config = copy.deepcopy(old['model_config']); ready_config = copy.deepcopy(model['model_config'])
        added_source_provenance = (model_key == 'flux3_image_klein'
            and 'reference_source_sha256' not in old_config['model']
            and bool(ready_config['model'].get('reference_source_sha256')))
        if added_source_provenance:
            ready_config['model'].pop('reference_source_sha256')
        if (old_config != ready_config
                or any(old[name] != model[name] for name in ('gpu_count', 'environment_python'))):
            raise RuntimeError('Late readiness cannot change the pinned scientific model configuration')
        if any(c['job_history'] for c in current['chunks'].values() if c['model_key'] == model_key):
            raise RuntimeError('A previously dispatched model cannot be rebound')
        base = CAMPAIGN / 'specification/generation_source'
        numbers = [int(p.name.rsplit('_', 1)[1]) for p in base.glob('SOURCE_*') if p.is_dir()]
        source = base / f'SOURCE_{max(numbers, default=0)+1:04d}'
        hashes = snapshot(source); manifest = source.with_name(source.name + '_manifest.json')
        atomic(manifest, {'schema': 'riddikulus-benchmark.generation-source.v1', 'utc': utc(),
                         'model_registration': model_key, 'source_sha256': hashes,
                         'framework_provenance': framework_provenance({model_key: model})})
        model.update(source_root=str(source), source_manifest=str(manifest), source_manifest_sha256=digest(manifest),
                     model_registry_path=str(Path(model_registry_path).resolve()), model_registry_sha256=digest(model_registry_path))
        current.setdefault('model_registration_history', []).append({'utc': utc(), 'model_key': model_key,
            'previous': old, 'new': model, 'added_official_source_hash_provenance': added_source_provenance})
        current['models'][model_key] = model
        method_module = runpy.run_path(str(source / 'src/riddikulus/selection.py'))
        cfg = model['model_config']; steps = cfg['generation']['num_inference_steps']; warmup = 3 if steps == 16 else 10
        templates = next(c['arms'] for c in current['cells'].values() if c['model_key'] != model_key and all('spec' in a for a in c['arms'].values()))
        for cell in current['cells'].values():
            if cell['model_key'] != model_key: continue
            admitted = cell['admission']['status'] == 'APPROVED'
            cell.update(status='READY' if admitted else 'BLOCKED_CONTENT_GENERATION',
                        reason=None if admitted else cell['admission']['reason'])
            cell['provenance'] = {'model_id': cfg['model']['model_id'], 'model_revision': cfg['model'].get('revision'),
                'seed': cell['seed'], 'generation': cfg['generation'], 'source_manifest_sha256': digest(manifest)}
            for name, arm in cell['arms'].items():
                if arm.get('spec') or arm['attempt_history']: raise RuntimeError('Late model arm already has execution evidence')
                spec = copy.deepcopy(read(templates[name]['spec']))
                spec.update(model_key=model_key, model_config=copy.deepcopy(cfg), native_steps=steps, seed=cell['seed'],
                    method=(method_module['default_baseline']() if name == 'Baseline' else
                            method_module['default_method'](num_inference_steps=steps, warmup_steps=warmup)),
                    reference_cell={'model': model_key, 'model_revision': cfg['model'].get('revision'),
                        'original_prompt': cell['original_prompt'],
                        'original_prompt_sha256': __import__('hashlib').sha256(cell['original_prompt'].encode()).hexdigest(),
                        'prompt_id': cell['prompt_id'], 'seed': cell['seed']},
                    preview_steps=[2,3,5,10] if steps == 16 else [5,9,10,14,15,28,29,30],
                    checkpoint_steps=[0,2,3,5,10,15] if steps == 16 else [0,5,9,10,14,15,25,28,29,30,40,49],
                    source_root=str(source), source_sha256=hashes, source_manifest=str(manifest),
                    source_manifest_sha256=digest(manifest), content_admission=copy.deepcopy(cell['admission']),
                    environment_python=model['environment_python'], expected_gpu_count=model['gpu_count'])
                path = Path(arm['run_root']) / 'specification/run.json'
                if path.exists(): raise RuntimeError('Late model specification already exists')
                atomic(path, spec); arm.update(spec=str(path), spec_sha256=digest(path), status=cell['status'])
        for chunk in current['chunks'].values():
            if chunk['model_key'] == model_key:
                chunk['status'] = 'READY' if any(current['cells'][c]['status'] == 'READY' for c in chunk['cell_ids']) else 'BLOCKED_CONTENT_GENERATION'
        current['status'] = 'ACTIVE_GENERATION'; export_manifests(current); save(current)
        return model


def reconcile():
    visible = queue()
    with campaign_lock():
        current = read(STATE); stopped = (CAMPAIGN / 'STOP').exists()
        for chunk in current['chunks'].values():
            if chunk['status'] == 'SUBMITTING':
                raise RuntimeError('Ambiguous SUBMITTING state requires exact scheduler reconciliation before retry')
            job = chunk.get('job')
            if not job or chunk['status'] not in {'SUBMITTED', 'RUNNING', 'STOPPING'}: continue
            if job['job_id'] in visible:
                record = visible[job['job_id']]; verify_live(record, job); chunk['scheduler'] = record
                if stopped: cancel_owned(job); chunk['status'] = 'STOPPING'
                elif record['state'] == 'RUNNING': chunk['status'] = 'RUNNING'
                continue
            record = accounting(job['job_id']); chunk.setdefault('accounting', []).append(record)
            for cell_id in chunk['cell_ids']:
                cell = current['cells'][cell_id]
                if cell['admission']['status'] != 'APPROVED': continue
                for arm in cell['arms'].values():
                    if arm.get('attempt') and (Path(arm['attempt']) / '_GENERATION_SUCCESS').exists():
                        arm.update(status='GENERATION_COMPLETE', verification=verify_arm(arm))
                if all(arm['status'] == 'GENERATION_COMPLETE' for arm in cell['arms'].values()): cell['status'] = 'GENERATION_COMPLETE'
            complete = all(current['cells'][c]['admission']['status'] != 'APPROVED'
                           or current['cells'][c]['status'] == 'GENERATION_COMPLETE' for c in chunk['cell_ids'])
            if stopped: chunk['status'] = 'STOPPED'
            elif record['state'] == 'COMPLETED' and record['exit_code'] == '0:0' and complete:
                chunk['status'] = 'COMPLETE_APPROVED_ITEMS'
            elif record['state'] == 'TIMEOUT':
                if chunk['continuation_count'] < current['planned_continuation_limit']:
                    chunk['continuation_count'] += 1; chunk['status'] = 'CONTINUATION_READY'
                else: chunk['status'] = 'CONTINUATIONS_EXHAUSTED'
            elif record['state'] in TRANSIENT:
                if chunk['retry_count'] < current['infrastructure_retry_limit']:
                    chunk['retry_count'] += 1; chunk['status'] = 'RETRY_READY'
                else: chunk['status'] = 'INFRASTRUCTURE_RETRIES_EXHAUSTED'
            else: chunk['status'] = 'NEEDS_DIAGNOSIS'
            append(CAMPAIGN / 'events.jsonl', {'utc': utc(), 'chunk_id': chunk['chunk_id'], 'scheduler': record,
                'status': chunk['status'], 'retry_count': chunk['retry_count'], 'continuation_count': chunk['continuation_count']})
        export_manifests(current)
        if stopped: current['status'] = 'STOPPED'
        elif all(c['status'] in TERMINAL for c in current['chunks'].values()):
            current['status'] = ('GENERATION_RECONCILED_WITH_BLOCKERS' if any(c['status'] not in {'COMPLETE_APPROVED_ITEMS'} for c in current['chunks'].values())
                                 else 'GENERATION_COMPLETE_AWAITING_EVALUATION')
        else: current['status'] = 'ACTIVE_GENERATION'
        save(current); return current


def stop():
    atomic(CAMPAIGN / 'STOP', {'requested_utc': utc(), 'source': 'explicit campaign stop command'})
    return reconcile()


def start_monitor():
    with campaign_lock():
        check_stop(); current = read(STATE); visible = queue()
        prior = current.get('monitor')
        if prior:
            if prior['job_id'] in visible: return prior
            previous = accounting(prior['job_id'])
            current.setdefault('monitor_history', []).append(dict(prior, accounting=previous))
            current['monitor'] = None
        path = CAMPAIGN / 'MONITOR'; path.mkdir(exist_ok=True)
        control_source = current.get('control_source_root', current['source_root'])
        args = ['sbatch', '--parsable', '--requeue', '--account=ml-staff', '--qos=walltime5d', '--partition=cpu',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=4G', '--time=01:00:00',
            '--comment=riddikulus-benchmark:monitor', '--job-name=riddikulus-monitor',
            '--output=' + str(path / 'monitor_%j.out'), '--error=' + str(path / 'monitor_%j.err'),
            str(Path(control_source) / 'scripts/riddikulus_benchmark/monitor.sbatch'), control_source]
        atomic(path / 'submission.json', {'status': 'SUBMITTING', 'utc': utc(), 'command': args})
        job = command(args).split(';')[0]
        if not job.isdigit(): raise RuntimeError('Ambiguous monitor submission')
        current['monitor'] = {'job_id': job, 'submitted_utc': utc(), 'control_source_root': control_source,
                              'planned_requeue_after_seconds': 2700, 'maximum_planned_requeues': 120}
        atomic(path / 'submission.json', {'status': 'SUBMITTED', 'utc': utc(), 'command': args, 'job_id': job})
        save(current); return current['monitor']


def monitor():
    import os
    job_id = os.environ.get('SLURM_JOB_ID', '')
    segment_started = time.monotonic()
    restart_count = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
    deadline = time.monotonic() + 30
    while (read(STATE).get('monitor') or {}).get('job_id') != job_id:
        if time.monotonic() >= deadline: raise RuntimeError('Monitor lacks registered ownership')
        time.sleep(2)
    registered_state = read(STATE)
    if registered_state.get('control_source_manifest'):
        source = Path(registered_state['control_source_root']).resolve()
        manifest_path = Path(registered_state['control_source_manifest'])
        if (Path(__file__).resolve().parents[2] != source
                or digest(manifest_path) != registered_state['control_source_manifest_sha256']):
            raise RuntimeError('Monitor source differs from its registered control seal')
        for relative, expected in read(manifest_path)['source_sha256'].items():
            if digest(source / relative) != expected:
                raise RuntimeError('Sealed monitor source changed: ' + relative)
    while True:
        try:
            if (read(STATE).get('monitor') or {}).get('job_id') != job_id: raise RuntimeError('Monitor ownership changed')
            current = reconcile()
            for chunk_id, chunk in current['chunks'].items():
                if chunk['status'] in {'RETRY_READY', 'CONTINUATION_READY'}: launch(chunk_id)
            current = read(STATE)
            finished = current['status'] in {'STOPPED', 'GENERATION_RECONCILED_WITH_BLOCKERS', 'GENERATION_COMPLETE_AWAITING_EVALUATION'}
            atomic(CAMPAIGN / 'monitor_heartbeat.json', {'utc': utc(), 'job_id': job_id,
                'observer_running': not finished, 'status': current['status'], 'coverage': current.get('coverage'),
                'planned_restart_count': restart_count, 'segment_elapsed_seconds': time.monotonic() - segment_started,
                'chunk_states': {k: c['status'] for k, c in current['chunks'].items()},
                'capabilities': 'Owned STOP, exact artifact reconciliation and bounded infrastructure retry/resume. No image judgment, scientific retuning or autonomous code repair.'})
            if finished: return
            if time.monotonic() - segment_started >= 2700:
                # This is a planned controller continuation, not a generation retry.
                # Same scheduler job identity and persisted campaign state survive.
                if restart_count >= 120:
                    raise RuntimeError('Bounded monitor planned-continuation limit reached')
                registered = current['monitor']
                visible = queue()
                record = visible.get(job_id)
                if (registered['job_id'] != job_id or record is None
                        or record['user'] != 'sagnihot'
                        or record['comment'] != 'riddikulus-benchmark:monitor'):
                    raise RuntimeError('Refusing to requeue a monitor without exact scheduler ownership')
                check_stop()
                append(CAMPAIGN / 'events.jsonl', {'utc': utc(), 'action': 'planned_monitor_requeue',
                       'job_id': job_id, 'restart_count': restart_count})
                command(['scontrol', 'requeue', job_id])
                return
        except Exception as exc:
            failure = {'utc': utc(), 'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
            with campaign_lock():
                current = read(STATE)
                if (current.get('monitor') or {}).get('job_id') != job_id: raise
                current['monitor_error_count'] = current.get('monitor_error_count', 0) + 1
                current['last_monitor_failure'] = failure; save(current)
                append(CAMPAIGN / 'monitor_failures.jsonl', failure)
                atomic(CAMPAIGN / 'monitor_heartbeat.json', {'utc': utc(), 'job_id': job_id,
                    'observer_running': current['monitor_error_count'] < 3,
                    'status': 'NEEDS_DIAGNOSIS', 'failure': failure})
                if current['monitor_error_count'] >= 3: raise
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    init = sub.add_parser('initialize'); init.add_argument('--model-registry', required=True)
    start = sub.add_parser('launch'); start.add_argument('chunk_id')
    enable = sub.add_parser('enable-model'); enable.add_argument('model_key', choices=MODEL_KEYS); enable.add_argument('--model-registry', required=True)
    for name in ('status', 'stop', 'start-monitor', 'monitor'): sub.add_parser(name)
    args = parser.parse_args()
    if args.action == 'initialize': value = initialize(args.model_registry)
    elif args.action == 'enable-model': value = enable_model(args.model_key, args.model_registry)
    elif args.action == 'launch': value = launch(args.chunk_id)
    elif args.action == 'status': value = reconcile()
    elif args.action == 'stop': value = stop()
    elif args.action == 'start-monitor': value = start_monitor()
    else: monitor(); return
    print(json.dumps(value, indent=2))


if __name__ == '__main__': main()
