"""Native Riddikulus generation, diagnostics and exact checkpoint continuation."""
from __future__ import annotations
import copy
from contextlib import nullcontext
from dataclasses import asdict
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import torch
from safetensors.torch import load_file, save_file
from .artifacts import atomic, append, digest, read, utc, check_stop, execution_scope
from .checkpoints import checkpoint, restore
from . import native_utils as legacy
from .native_utils import zero_controller
from .numerics import tensor_sha256
from .controller import ControllerConfig, ExposureState, apply_correction, additive_leaf_effects
from .steering import CorrectedHierarchySteerer
from .predictions import _NativePredictor
from .selection import validate_method, activation_scores_from_joint, select_pool_and_active, is_baseline
from .hierarchy import MirroredConceptHierarchy
from .model_state import setup_native_state_policy, reset_arm_cache, prefix_mode_counters, scheduler_hash

POOL_SIZE = 25
ACTIVATION_MARGINS = ('0.10', '0.25', '0.40', '0.50', '0.60')

def progress(attempt, **values):
    atomic(Path(attempt) / "progress.json", dict(values, updated_utc=utc(), job_id=os.environ.get("SLURM_JOB_ID")))

def _profile(spec):
    steps = int(spec['model_config']['generation']['num_inference_steps'])
    adapter_name = spec['model_config']['model']['adapter']
    expected = (16, 3) if adapter_name == 'flux3' else (50, 10)
    method = validate_method(spec['method'])
    warmup = method.get('warmup_steps', expected[1])
    if (steps, warmup) != expected:
        raise RuntimeError('Model/native-step/acquisition profile is not qualified')
    return steps, warmup

def default_preview_steps(steps=50):
    if steps == 50: return [5, 9, 10, 14, 15, 28, 29, 30]
    if steps == 16: return [2, 3, 5, 10]
    raise ValueError('Unqualified native-step count')

def default_checkpoint_steps(steps=50):
    if steps == 50: return [0, 5, 9, 10, 14, 15, 25, 28, 29, 30, 40, 49]
    if steps == 16: return [0, 2, 3, 5, 10, 15]
    raise ValueError('Unqualified native-step count')

def progress(attempt, **values):
    atomic(attempt / "progress.json", dict(values, updated_utc=utc(), job_id=os.environ.get("SLURM_JOB_ID")))


def runtime_metadata():
    import importlib
    from importlib.metadata import version
    packages = {}
    for name in ('torch', 'diffusers', 'transformers', 'accelerate', 'safetensors'):
        module = importlib.import_module(name)
        packages[name] = {'version': version(name), 'module_file': str(Path(module.__file__).resolve())}
    return {
        'packages': packages, 'python': sys.version, 'cuda_version': torch.version.cuda,
        'torch_git_version': torch.version.git_version,
        'gpus': [{'index': i, 'name': torch.cuda.get_device_name(i),
                  'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory,
                  'compute_capability': list(torch.cuda.get_device_capability(i))}
                 for i in range(torch.cuda.device_count())],
    }


class LocalPredictionProvider:
    """Native exact-text cache with same-state unconditional reuse where supported."""

    def __init__(self, *, adapter, **unused):
        self.adapter = adapter
        self.predictor = _NativePredictor(adapter)
        self.last_diagnostics = {}
        self.conditioning_evidence = {}
        self.scope_stats = None
        self.attempt = Path(unused['rendezvous_dir']).parent if 'rendezvous_dir' in unused else None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __call__(self, *, adapter, latents, timestep, state, requests):
        if adapter is not self.adapter:
            raise RuntimeError('Prediction provider attached to another model')
        begun = time.perf_counter()
        predictions, evidence = {}, {}
        scope_factory = getattr(adapter, 'shared_prediction_scope', None)
        scope = (nullcontext(self.scope_stats) if self.scope_stats is not None
                 else scope_factory() if callable(scope_factory) else nullcontext({}))
        with scope as reuse:
            with torch.inference_mode():
                for completed, (text, role) in enumerate(requests.items(), 1):
                    check_stop()
                    predictions[text] = self.predictor.predict(text, role, latents, timestep, state)
                    evidence[text] = dict(self.predictor.last_conditioning_evidence, replica_rank=0)
                    if self.attempt is not None and (completed == 1 or completed % 25 == 0 or completed == len(requests)):
                        observed = read(self.attempt / 'progress.json')
                        observed.update(concept_requests_completed=completed,
                                        concept_requests_total=len(requests),
                                        concept_phase_seconds=time.perf_counter() - begun,
                                        updated_utc=utc())
                        atomic(self.attempt / 'progress.json', observed)
                        print(json.dumps({'event': 'CONCEPT_PROGRESS', 'step': observed.get('current_step'),
                                          'completed': completed, 'total': len(requests),
                                          'seconds': observed['concept_phase_seconds']}), flush=True)
        self.conditioning_evidence = evidence
        self.last_diagnostics = {
            'unique_requests': len(requests), 'seconds': time.perf_counter() - begun,
            'replicas': 1, 'status': 'COMPLETE',
            'cross_replica_actual_prediction_checked': False,
            'unconditional_cache_hits': int(reuse.get('unconditional_cache_hits', 0)),
            'unconditional_cache_misses': int(reuse.get('unconditional_cache_misses', 0)),
            'scope': 'conditioning, current-state native predictions, and validation; no communication',
        }
        return predictions


class ForwardCounter:
    """Count actual denoiser module invocations; CFG/cache semantics remain native."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.count = 0
        self.handle = None

    def __enter__(self):
        def record(module, args):
            self.count += 1
        self.handle = self.transformer.register_forward_pre_hook(record)
        return self

    def __exit__(self, *args):
        self.handle.remove()
        return False


def native_step_with_counterfactual(adapter, before, v_base, prediction, timestep,
                                    state, generator, index, feature_dim):
    """Compare two next states from the same complete native solver history.

    The base counterfactual shares the actual past trajectory, including prior
    steering. It is a one-update effect, not an unsteered whole-image baseline.
    The genuine pipeline scheduler is advanced exactly once. The two isolated
    native solver clones perform arithmetic only and make no model calls.
    """
    scheduler = adapter.pipeline.scheduler
    if scheduler.step_index is not None and scheduler.step_index != index:
        raise RuntimeError('Native scheduler position differs from update index')
    snapshots = [copy.deepcopy(scheduler), copy.deepcopy(scheduler)]
    generator_before = generator.get_state().clone()
    torch_before = torch.get_rng_state().clone()
    cuda_before = [v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    outputs = []
    try:
        for clone, field in zip(snapshots, (prediction, v_base)):
            adapter.pipeline.scheduler = clone
            outputs.append(adapter.scheduler_step(field, timestep, before, copy.deepcopy(state), generator=generator))
    finally:
        adapter.pipeline.scheduler = scheduler
    if (not torch.equal(generator_before, generator.get_state())
            or not torch.equal(torch_before, torch.get_rng_state())
            or any(not torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))):
        raise RuntimeError('Approved deterministic native solver shadow consumed RNG')
    actual = adapter.scheduler_step(prediction, timestep, before, state, generator=generator)
    if scheduler.step_index != index + 1:
        raise RuntimeError('The genuine scheduler did not advance exactly once')
    if not torch.equal(outputs[0].latents, actual.latents):
        raise RuntimeError('Complete native solver shadow differs from actual next latent')
    difference = actual.latents.float() - outputs[1].latents.float()
    euler = type(scheduler).__name__ == 'FlowMatchEulerDiscreteScheduler'
    dt = scheduler.sigmas[index + 1] - scheduler.sigmas[index]
    ideal = (dt.float().to(prediction.device) * (prediction.float() - v_base.float())
             if euler else None)
    record = {
        'step': index, 'native_shadow_bitwise_equal': True, 'scheduler_updates': 1,
        'isolated_solver_shadow_updates': 2, 'additional_model_forwards': 0,
        'scheduler_class': type(scheduler).__name__,
        'actual_next_latent_sha256': tensor_sha256(actual.latents),
        'base_counterfactual_next_latent_sha256': tensor_sha256(outputs[1].latents),
        'native_difference_rms': _rms(difference),
        'ideal_difference_rms': _rms(ideal) if euler else None,
        'rounding_difference_rms': _rms(difference - ideal) if euler else None,
        'native_against_ideal': _projection(difference, ideal) if euler else None,
        'ideal_euler_diagnostic_status': 'OBSERVED' if euler else 'UNOBSERVED_NON_EULER_NATIVE_SOLVER',
        'nonzero_feature_fraction': float((difference != 0).float().mean()),
        'nonzero_token_fraction': float((difference != 0).any(dim=feature_dim).float().mean()),
        'feature_dim': feature_dim, 'sample_dtype': str(before.dtype),
        'prediction_dtype': str(prediction.dtype), 'output_dtype': str(actual.latents.dtype),
        'finite': bool(torch.isfinite(actual.latents).all() and torch.isfinite(difference).all()),
        'definition': 'Applied versus ordinary native next state at identical pre-update latent and complete solver history. Aggregate association, not per-leaf causal attribution.',
    }
    if not record['finite']:
        raise RuntimeError('Nonfinite actual/counterfactual native latent')
    return actual, record, difference, ideal


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _rms(value):
    return float(value.float().square().mean().sqrt())


def _projection(value, reference):
    """Observational alignment with the aggregate result, not causal attribution."""
    a, b = value.float().reshape(-1), reference.float().reshape(-1)
    dot = float(torch.dot(a, b))
    aa, bb = float(torch.dot(a, a)), float(torch.dot(b, b))
    return {'dot': dot, 'projection_coefficient': dot / bb if bb > 0 else None,
            'cosine': dot / math.sqrt(aa * bb) if aa > 0 and bb > 0 else None}


def _save_tensors(path, tensors, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.partial')
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(temporary),
              metadata={k: str(v) for k, v in metadata.items()})
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _remember_artifacts(selection, attempt, paths):
    for path in paths:
        relative = str(path.relative_to(attempt))
        current = digest(path)
        prior = selection['committed_artifacts'].get(relative)
        if prior is not None and prior != current:
            raise RuntimeError(f'Immutable diagnostic artifact changed: {relative}')
        selection['committed_artifacts'][relative] = current


def restore_artifacts(receipt, attempt, selection):
    old_attempt = Path(read(receipt)['path']).parents[1]
    for relative, expected in selection['committed_artifacts'].items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] not in {'activations', 'previews'}:
            raise RuntimeError('Checkpoint artifact path escapes its allowed diagnostic directories')
        source, target = old_attempt / rel, attempt / rel
        if digest(source) != expected:
            raise RuntimeError(f'Checkpoint diagnostic artifact corrupted: {relative}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
            dst.flush(); os.fsync(dst.fileno())
        if digest(target) != expected:
            raise RuntimeError(f'Copied checkpoint diagnostic artifact corrupted: {relative}')


def _restore_decode_artifacts(receipt, attempt, selection, spec):
    restore_artifacts(receipt, attempt, selection)
    repair = spec.get('infrastructure_resume_repair', {})
    if (repair.get('kind') != 'sd35-native-vae-shift-decode-only.v1'
            or read(receipt)['spec_sha256'] != repair['source_spec_sha256']):
        return
    moved = []
    for relative, expected in list(selection['committed_artifacts'].items()):
        rel = Path(relative)
        if rel.parts[0] != 'previews' or rel.parts[1] == 'historical_decoder_invalid':
            continue
        historical = Path('previews/historical_decoder_invalid') / Path(*rel.parts[1:])
        target = attempt / historical
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError('Historical decoder-invalid preview destination already exists')
        (attempt / rel).rename(target)
        del selection['committed_artifacts'][relative]
        selection['committed_artifacts'][str(historical)] = expected
        moved.append({'original_relative_path': relative, 'historical_relative_path': str(historical),
                      'sha256': expected, 'status': 'HISTORICAL_DECODER_INVALID'})
    if moved:
        path = attempt / 'previews/historical_decoder_invalid/manifest.json'
        atomic(path, {'schema': 'sd35-historical-decoder-invalid-previews.v1',
                     'status': 'HISTORICAL_DECODER_INVALID', 'source_checkpoint_receipt': str(receipt),
                     'reason': 'Original SD3.5 decoder omitted the native VAE shift factor.',
                     'reconstruction_limit': 'These predicted-clean estimates used pre-update latents and predictions, '
                         'which cannot be exactly recovered from post-update checkpoints. Original bytes are preserved; '
                         'these are not corrected previews and must not be used for corrected visual evaluation.',
                     'artifacts': moved})
        _remember_artifacts(selection, attempt, [path])


def _sd35_decode_image(adapter, latents, state, attempt, index, terminal):
    if adapter.adapter_name != 'sd35':
        return legacy._decoded_to_pil(adapter.decode_latents(latents, state))
    before_sha = tensor_sha256(latents)
    adapter._sd35_native_decode_evidence = None
    with ForwardCounter(adapter.pipeline.transformer) as forwards:
        image = legacy._decoded_to_pil(adapter.decode_latents(latents, state))
    proof = adapter._sd35_native_decode_evidence
    if (forwards.count != 0 or tensor_sha256(latents) != before_sha or not isinstance(proof, dict)
            or proof.get('latent_sha256') != before_sha or proof.get('formula_contract') != 'SOURCE_AUDITED_NATIVE_AFFINE'
            or proof.get('finite') is not True):
        raise RuntimeError('SD3.5 decode changed latents, called a denoiser, or lacks native decode proof')
    append(attempt / 'traces/native_decode.jsonl', dict(proof, step=index,
           kind='final' if terminal else 'predicted_clean_estimate', fed_back=False,
           additional_denoiser_forwards=forwards.count, input_latent_unchanged=True, utc=utc()))
    return image


def _sd35_repair_runner_ast(path):
    """Remove only the named decode/diagnostic handoff additions for comparison."""
    import ast
    tree = ast.parse(Path(path).read_text())
    helpers = {'_restore_decode_artifacts', '_sd35_decode_image',
               '_sd35_repair_runner_ast', '_validate_sd35_decode_repair'}
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in helpers]
    substitutions = {
        "_restore_decode_artifacts(resume, attempt, selection, spec)":
            "restore_artifacts(resume, attempt, selection)",
        "_sd35_decode_image(adapter, preview.detach(), state, attempt, index, index == steps - 1)":
            "legacy._decoded_to_pil(adapter.decode_latents(preview.detach(), state))",
        "_sd35_decode_image(adapter, latents, state, attempt, steps - 1, True)":
            "legacy._decoded_to_pil(adapter.decode_latents(latents, state))",
    }
    replacements = {ast.dump(ast.parse(old, mode='eval').body): ast.parse(new, mode='eval').body
                    for old, new in substitutions.items()}
    branch = ast.dump(ast.parse(
        "if repair.get('kind') == 'sd35-native-vae-shift-decode-only.v1':\n"
        "    return _validate_sd35_decode_repair(spec, spec_sha, receipt_path, receipt, repair, old)\n").body[0])
    artifact_branch = ast.dump(ast.parse(
        "if adapter.adapter_name == 'sd35':\n"
        "    files += ['traces/native_decode.jsonl']\n").body[0])
    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'resume_spec_sha':
                node.body = [child for child in node.body if ast.dump(child) != branch]
            if node.name == 'generate_latent':
                node.body = [child for child in node.body if ast.dump(child) != artifact_branch]
            return self.generic_visit(node)
        def visit_Call(self, node):
            return copy.deepcopy(replacements.get(ast.dump(node), node))
    return ast.dump(Normalize().visit(tree))


def _validate_sd35_decode_repair(spec, spec_sha, receipt_path, receipt, repair, old):
    """Authenticate the exact two-file SD3.5 decoder repair and unchanged science."""
    import ast
    kind = 'sd35-native-vae-shift-decode-only.v1'
    proof_path = Path(repair['repair_receipt'])
    if digest(proof_path) != repair['repair_receipt_sha256']:
        raise RuntimeError('SD3.5 decode repair receipt hash differs')
    proof = read(proof_path)
    identity_keys = ('source_spec', 'source_spec_sha256', 'checkpoint_receipt',
                     'checkpoint_receipt_sha256', 'checkpoint_sha256', 'changed_source_files')
    if (proof.get('schema') != kind or proof.get('scientific_retuning') is not False
            or any(proof.get(key) != repair[key] for key in identity_keys)):
        raise RuntimeError('SD3.5 decode repair receipt identity differs')
    if (digest(receipt['path']) != repair['checkpoint_sha256']
            or spec['model_config']['model']['adapter'] != 'sd35'
            or spec['model_config']['model']['model_id'] != 'stabilityai/stable-diffusion-3.5-large'
            or spec['model_config']['model']['revision'] != old['reference_cell']['model_revision']):
        raise RuntimeError('SD3.5 decode repair checkpoint/model identity differs')
    allowed = ['src/riddikulus/adapters/sd35_adapter.py', 'src/riddikulus/runner.py']
    if (set(old['source_sha256']) != set(spec['source_sha256'])
            or sorted(k for k in old['source_sha256'] if old['source_sha256'][k] != spec['source_sha256'][k]) != allowed
            or repair['changed_source_files'] != allowed
            or proof.get('source_file_changes') != {
                relative: {'old_sha256': old['source_sha256'][relative],
                           'new_sha256': spec['source_sha256'][relative]} for relative in allowed}):
        raise RuntimeError('SD3.5 decode repair changed files beyond the explicit two-file receipt')
    for label, source_spec in (('old_source', old), ('new_source', spec)):
        expected = {'root': source_spec['source_root'], 'manifest': source_spec['source_manifest'],
                    'manifest_sha256': source_spec['source_manifest_sha256'],
                    'source_sha256': source_spec['source_sha256']}
        if proof.get(label) != expected or digest(expected['manifest']) != expected['manifest_sha256']:
            raise RuntimeError('SD3.5 decode repair source manifest identity differs: ' + label)
        manifest = read(expected['manifest'])
        if (manifest.get('schema') != 'riddikulus-benchmark.generation-source.v1'
                or manifest['source_sha256'] != expected['source_sha256']):
            raise RuntimeError('SD3.5 decode repair manifest contents differ: ' + label)
        for relative, sha in expected['source_sha256'].items():
            rel = Path(relative)
            if rel.is_absolute() or '..' in rel.parts or digest(Path(expected['root']) / rel) != sha:
                raise RuntimeError('SD3.5 decode repair sealed source changed: ' + relative)
    # Each source field was individually authenticated above, before normalization.
    mutable = {'created_utc', 'source_root', 'source_sha256', 'source_manifest', 'source_manifest_sha256',
               'infrastructure_resume_repair', '_shared_model_load_seconds', '_baseline_attempt'}
    if ({k: v for k, v in old.items() if k not in mutable}
            != {k: v for k, v in spec.items() if k not in mutable}):
        raise RuntimeError('SD3.5 decode repair changed the scientific specification')
    old_root, new_root = Path(old['source_root']), Path(spec['source_root'])
    adapter_path = Path('src/riddikulus/adapters/sd35_adapter.py')
    before, after = (ast.parse((root / adapter_path).read_text()) for root in (old_root, new_root))
    classes = [node for node in after.body if isinstance(node, ast.ClassDef) and node.name == 'StableDiffusion35Adapter']
    if len(classes) != 1:
        raise RuntimeError('SD3.5 repair does not expose its exact native adapter class')
    decode_methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == 'decode_latents']
    if len(decode_methods) != 1:
        raise RuntimeError('SD3.5 repair must add exactly one native decode override')
    classes[0].body.remove(decode_methods[0])
    if (ast.dump(before) != ast.dump(after)
            or _sd35_repair_runner_ast(old_root / 'src/riddikulus/runner.py')
            != _sd35_repair_runner_ast(new_root / 'src/riddikulus/runner.py')):
        raise RuntimeError('SD3.5 decode repair changed native predictions, scheduler, controller or selection code')
    native = proof['native_decode']
    framework = read(old['source_manifest'])['framework_provenance'][old['environment_python']]
    if (digest(native['pipeline_source_path']) != native['pipeline_source_sha256']
            or framework['native_files_sha256'].get(native['pipeline_source_path']) != native['pipeline_source_sha256']
            or digest(native['vae_config_path']) != native['vae_config_sha256']
            or Path(native['vae_config_path']).resolve() !=
                (Path(spec['model_config']['model']['local_snapshot_path']) / 'vae/config.json').resolve()
            or (native['scaling_factor'], native['shift_factor']) != (1.5305, 0.0609)):
        raise RuntimeError('SD3.5 repair native pipeline/VAE provenance differs')
    vae_config = read(native['vae_config_path'])
    if (vae_config['scaling_factor'], vae_config['shift_factor']) != (native['scaling_factor'], native['shift_factor']):
        raise RuntimeError('SD3.5 repair VAE inverse-normalization constants differ')
    new_framework = read(spec['source_manifest'])['framework_provenance'][spec['environment_python']]
    if new_framework != framework:
        raise RuntimeError('SD3.5 decode repair changed its pinned native framework')
    return repair['source_spec_sha256']


def resume_spec_sha(spec, spec_sha, receipt_path):
    """Authorize one recorded infrastructure-only source handoff, never retuning."""
    receipt = read(receipt_path)
    if receipt['spec_sha256'] == spec_sha:
        return spec_sha
    repair = spec.get('infrastructure_resume_repair')
    if not isinstance(repair, dict) or repair.get('scientific_retuning') is not False:
        raise RuntimeError('Cross-source checkpoint requires an explicit infrastructure repair')
    old_path = Path(repair['source_spec'])
    if (digest(old_path) != repair['source_spec_sha256']
            or receipt['spec_sha256'] != repair['source_spec_sha256']
            or str(Path(receipt_path).resolve()) != str(Path(repair['checkpoint_receipt']).resolve())
            or digest(receipt_path) != repair['checkpoint_receipt_sha256']
            or receipt['sha256'] != repair['checkpoint_sha256']):
        raise RuntimeError('Infrastructure repair source/checkpoint identity differs')
    old = read(old_path)
    if repair.get('kind') == 'sd35-native-vae-shift-decode-only.v1':
        return _validate_sd35_decode_repair(spec, spec_sha, receipt_path, receipt, repair, old)
    mutable = {'created_utc', 'source_root', 'source_sha256',
               'scientific_fingerprint', 'infrastructure_resume_repair',
               '_shared_model_load_seconds', '_baseline_attempt'}
    if ({k: v for k, v in old.items() if k not in mutable}
            != {k: v for k, v in spec.items() if k not in mutable}):
        raise RuntimeError('Infrastructure resume changed the scientific specification')
    if set(old['source_sha256']) != set(spec['source_sha256']):
        raise RuntimeError('Infrastructure resume added or removed source files')
    changed = sorted(k for k in old['source_sha256']
                     if old['source_sha256'][k] != spec['source_sha256'][k])
    if changed != ['src/riddikulus/runner.py'] or changed != repair['changed_source_files']:
        raise RuntimeError('Infrastructure repair changed files beyond runner initialization/resume')
    for relative, expected in old['source_sha256'].items():
        if digest(Path(old['source_root']) / relative) != expected:
            raise RuntimeError('Original immutable checkpoint source changed: ' + relative)
    return repair['source_spec_sha256']


def validate_selection_state(selection, method, leaves, next_index):
    warmup_steps = method.get('warmup_steps', 10)
    acquisition_index = warmup_steps
    if selection.get('method_sha256') != canonical_sha(method):
        raise RuntimeError('Checkpoint belongs to another method')
    guided = [] if is_baseline(method) else list(range(warmup_steps, next_index))
    history = selection.get('selection_history')
    if not isinstance(history, list) or [e['step'] for e in history] != guided:
        raise RuntimeError('Committed fixed-pool selections differ from completed guided updates')
    expected_scans = [i for i in method['full_scan_steps'] if i < next_index]
    scans = selection.get('scan_history')
    if not isinstance(scans, list) or [e['step'] for e in scans] != expected_scans:
        raise RuntimeError('Committed full scan differs from the acquisition boundary')
    pool = selection.get('candidate_pool_ids')
    if is_baseline(method):
        if pool != [] or history or scans:
            raise RuntimeError('Baseline acquired a candidate pool')
    elif next_index <= acquisition_index:
        if pool is not None or scans or history:
            raise RuntimeError('Pool acquired during native warmup')
    else:
        if (pool != sorted(set(pool)) or len(pool) != POOL_SIZE or not set(pool) <= set(leaves)
                or selection['candidate_pool_sha256'] != canonical_sha(pool)
                or scans[0]['candidate_pool_ids'] != pool):
            raise RuntimeError('Frozen pool identity differs from the acquisition event')
    for event in history:
        ids = event['selected_leaf_ids']
        if (ids != sorted(set(ids)) or len(ids) != POOL_SIZE or ids != pool
                or event['selection_sha256'] != canonical_sha(ids)
                or event['candidate_pool_sha256'] != selection['candidate_pool_sha256']):
            raise RuntimeError('A correction identity differs from the complete fixed 25-leaf pool')
    ids = history[-1]['selected_leaf_ids'] if history else []
    if selection.get('selected_leaf_ids') != ids or selection.get('selection_sha256') != canonical_sha(ids):
        raise RuntimeError('Checkpoint latest correction pool differs from committed history')
    if not isinstance(selection.get('committed_artifacts'), dict):
        raise RuntimeError('Checkpoint lacks authenticated diagnostics')


def save_scan(attempt, index, routed, selection, method, spec_sha, before):
    acquisition_index = method['warmup_steps']
    rows = list(routed.activation_rows); ids = [r['node_id'] for r in rows]
    joint = routed.activation_joint
    if index != acquisition_index or len(rows) != 287 or ids != sorted(set(ids)) or not isinstance(joint, dict):
        raise RuntimeError('Acquisition must export every leaf once at index10')
    if set(joint) != {'unsafe_cosine', 'cosine_gap', 'joint_reliability'}:
        raise RuntimeError('Incomplete selector joint evidence')
    positions = rows[0]['position_count']
    if any(tuple(v.shape) != (287, positions) for v in joint.values()):
        raise RuntimeError('Full-scan joint tensor shape differs from native positions')
    for row_index, row in enumerate(rows):
        computed = activation_scores_from_joint(**{k: v[row_index] for k,v in joint.items()})
        for margin, expected in computed.items():
            for key, actual in row['activation_scores'][margin].items():
                if not math.isclose(actual, expected[key], rel_tol=2e-6, abs_tol=1e-8):
                    raise RuntimeError('Saved joint fields do not reproduce activation')
    replay = select_pool_and_active(rows, candidate_pool_ids=None, pool_size=POOL_SIZE,
                                   active_k=POOL_SIZE, selection_margin=.10)
    if (list(replay['candidate_pool_ids']) != list(routed.candidate_pool_ids)
            or list(replay['selected_leaf_ids']) != list(routed.selected_leaf_ids)):
        raise RuntimeError('Saved acquisition does not reproduce the complete correction pool')
    pool = sorted(routed.candidate_pool_ids)
    event = dict(routed.pool_event or {}, step=index, spec_sha256=spec_sha,
                 method_sha256=canonical_sha(method), candidate_pool_ids=pool,
                 candidate_pool_sha256=canonical_sha(pool), preupdate_latent_sha256=tensor_sha256(before))
    if len(pool) != POOL_SIZE or sorted(routed.selected_leaf_ids) != pool:
        raise RuntimeError('Pure acquisition must select exactly25 candidate identities')
    target = attempt / f'activations/scan_{index:02d}_joint.safetensors'
    _save_tensors(target, joint, {'spec_sha256': spec_sha, 'step': index,
                                 'row_ids_json': json.dumps(ids)})
    meta = target.with_suffix('.json')
    atomic(meta, {'row_ids': ids, 'joint_shape': [287,positions],
                 'pool_event': event, 'tensor_sha256': digest(target), 'spec_sha256':spec_sha})
    csv_path = attempt / f'activations/scan_{index:02d}.csv'
    columns = ['node_id','unsafe_concept','safe_concept','alias_representative_id',
               'is_exact_alias_representative','rank','selected','position_count',
               'reliable_positions','reliable_fraction','raw_direction_rms','cosine_distance',
               'sigmoid_weight','direction_usable']
    scores = [f'{label}_{margin}' for margin in ACTIVATION_MARGINS
              for label in ('A','confidence_mass','support_positions','support_fraction')]
    temporary = csv_path.with_suffix('.partial')
    with temporary.open('x', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns+['in_candidate_pool']+scores)
        writer.writeheader()
        for row in rows:
            record={k:row.get(k) for k in columns};record['in_candidate_pool']=row['node_id'] in pool
            for margin in ACTIVATION_MARGINS:
                score=row['activation_scores'][margin]
                for label,key in [('A','mean_confidence'),('confidence_mass','confidence_mass'),
                                  ('support_positions','support_positions'),('support_fraction','support_fraction')]:
                    record[f'{label}_{margin}']=score[key]
            writer.writerow(record)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,csv_path)
    _remember_artifacts(selection,attempt,[target,meta,csv_path])
    selection.update(candidate_pool_ids=pool,candidate_pool_sha256=canonical_sha(pool),last_scan_step=index)
    selection['scan_history'].append(event)
    append(attempt/'traces/pool_events.jsonl',event)


def save_field_maps(attempt, index, routed, candidate, delta, actual_difference, ideal, selection, spec_sha):
    terms = list(routed.selected_terms) if routed else []
    raw_sum = torch.zeros_like(candidate, dtype=torch.float32)
    for term in terms:
        raw_sum = raw_sum + term.confidence.float() * term.direction.float()
    tensors = {'aggregate_raw_additive': raw_sum, 'aggregate_post_conflict': candidate,
               'aggregate_actual_prediction_delta': delta,
               'aggregate_native_latent_difference': actual_difference}
    if ideal is not None:
        tensors['aggregate_ideal_euler_latent_difference'] = ideal
    ids = []
    for number, term in enumerate(terms):
        ids.append(term.concept_id)
        tensors[f'leaf_{number:02d}_raw_safe_minus_unsafe'] = routed.selected_raw_directions[term.concept_id]
        tensors[f'leaf_{number:02d}_weighted_direction'] = term.direction
        tensors[f'leaf_{number:02d}_steering_confidence'] = term.confidence
    target = attempt / f'activations/selected_fields_{index:02d}.safetensors'
    _save_tensors(target, tensors, {'spec_sha256': spec_sha, 'step': index, 'leaf_ids_json': json.dumps(ids),
                  'ideal_euler_field_status': 'OBSERVED' if ideal is not None else 'UNOBSERVED_NON_EULER_NATIVE_SOLVER',
                  'attribution': 'Shared aggregate projections are observational, not causal per-leaf contributions.'})
    _remember_artifacts(selection, attempt, [target])


def validate_trajectory(spec, records, selection, leaves, attempt=None):
    steps, warmup_steps = _profile(spec)
    acquisition_index = warmup_steps
    method=validate_method(spec['method']);validate_selection_state(selection,method,leaves,steps)
    if [r['step'] for r in records]!=list(range(steps)):
        raise RuntimeError('Expected exactly50 ordered native updates')
    hierarchy=MirroredConceptHierarchy.from_yaml_file(Path(spec['hierarchy_path']))
    if sorted(k for k,n in hierarchy.nodes.items() if n.steerable)!=list(leaves):
        raise RuntimeError('Native trajectory inventory differs from H4')
    sigmas=[r['sigma'] for r in records]+[records[-1]['next_sigma']]
    if any(not math.isfinite(v) for v in sigmas) or any(a<=b for a,b in zip(sigmas,sigmas[1:])):
        raise RuntimeError('Nonfinite or non-descending native sigma schedule')
    checks=[];pool=selection['candidate_pool_ids']
    for i,r in enumerate(records):
        baseline=is_baseline(method);guided=not baseline and i>=warmup_steps;scan=guided and i==acquisition_index
        expected=leaves if scan else pool if guided else []
        if r['evaluated_leaf_ids']!=expected or r['full_scan']!=scan:
            raise RuntimeError('Warmup/full-scan/pool evaluation inventory mismatch')
        selected=selection['selection_history'][i-warmup_steps]['selected_leaf_ids'] if guided else []
        if r['selected_leaf_ids']!=selected or not set(r['active_leaf_ids'])<=set(selected):
            raise RuntimeError('An unelected candidate contributed a correction')
        calls=len({hierarchy.neutral_concept}|{getattr(hierarchy.nodes[k],role)
                   for k in expected for role in ('unsafe_concept','safe_concept')}) if expected else 0
        if r['pool']['unique_requests']!=calls:
            raise RuntimeError('Model request count differs from exact text deduplication')
        if guided and (len(selected)!=POOL_SIZE or selected!=pool or r['direction_source_step']!=i):
            raise RuntimeError('Fresh complete fixed-pool invariant failed')
        if not guided and (calls or r['controller']['applied_delta_rms'] or r['latent_effect']['native_difference_rms']):
            raise RuntimeError('Native-only update performed a safety calculation/intervention')
        if not baseline and i<warmup_steps and r['warmup_baseline_match'].get('passed') is not True:
            raise RuntimeError('METHOD warmup differs from native Baseline')
        if (not r['latent_effect']['native_shadow_bitwise_equal']
                or not r['latent_effect']['finite'] or r['latent_effect']['scheduler_updates']!=1):
            raise RuntimeError('Native solver effect audit failed')
        control=r['controller'];limit=spec['controller_config']['global_relative_rms']*control['base_rms']
        tol=spec['controller_config']['budget_atol']+spec['controller_config']['budget_rtol']*limit
        if not math.isfinite(control['applied_delta_rms']) or control['applied_delta_rms']>limit+tol:
            raise RuntimeError('Actual native prediction exceeds the global RMS bound')
        checks.append({'step':i,'full_scan':scan,'evaluated_leaves':len(expected),
                       'selected_leaves':len(selected),'candidate_pool_leaves':POOL_SIZE if guided else 0,
                       'concept_calls':calls,'passed':True})
    if attempt is not None:
        for rel,expected in selection['committed_artifacts'].items():
            if digest(attempt/rel)!=expected:raise RuntimeError('Committed diagnostic changed:'+rel)
        if not is_baseline(method):
            meta=read(attempt/f'activations/scan_{acquisition_index:02d}_joint.json')
            if meta['row_ids']!=list(leaves):raise RuntimeError('Incomplete saved acquisition')
    return {'schema':'nailing-method.pool25-warm10.numerical-audit.v1','passed':True,'native_updates':steps,
            'method_sha256':canonical_sha(method),'full_scan_steps':method['full_scan_steps'],
            'candidate_pool_sha256':selection['candidate_pool_sha256'],
            'selected_leaf_count':len(selection['selected_leaf_ids']),'checks':checks,
            'warmup_native_matches':warmup_steps if not is_baseline(method) else None,
            'guided_updates':steps-warmup_steps if not is_baseline(method) else 0,
            'native_shadow_equal_all_updates':True,
            'local_bound_verification':'Enforced on actual BF16 correction on each guided update.'}


def commit_selection(attempt,index,routed,selection,spec_sha):
    ids=sorted(routed.selected_leaf_ids);pool=selection['candidate_pool_ids']
    if len(ids)!=POOL_SIZE or list(routed.candidate_pool_ids)!=pool or ids!=pool:
        raise RuntimeError('Fresh correction identities differ from the complete immutable25-pool')
    event=dict(routed.selection_event or {},step=index,spec_sha256=spec_sha,
               selected_leaf_ids=ids,selection_sha256=canonical_sha(ids),
               candidate_pool_ids=pool,candidate_pool_sha256=selection['candidate_pool_sha256'])
    selection.update(selected_leaf_ids=ids,selection_sha256=canonical_sha(ids))
    selection['selection_history'].append(event)
    append(attempt/'traces/selection_events.jsonl',event)


def warmup_match(before,v_base,after,scheduler_before,scheduler_after,baseline,index):
    found={'preupdate_latent_sha256':tensor_sha256(before),
           'ordinary_prediction_sha256':tensor_sha256(v_base),
           'latent_sha256':tensor_sha256(after),
           'scheduler_before_sha256':scheduler_before,'scheduler_after_sha256':scheduler_after}
    for key,actual in found.items():
        if baseline[index][key]!=actual:
            raise RuntimeError(f'METHOD native warmup differs from Baseline at{index}: {key}')
    return dict(found,passed=True,step=index,additional_model_forwards=0)


def generate_latent(adapter, spec, spec_sha, attempt, world_size, load_seconds, resume):
    steps, warmup_steps = _profile(spec)
    acquisition_index = warmup_steps
    native_dtype = torch.float32 if adapter.adapter_name == "flux3" else torch.bfloat16
    started = time.perf_counter()
    method = validate_method(spec['method']); row = spec['reference_cell']
    if (spec.get('preview_steps') != default_preview_steps(steps)
            or spec.get('checkpoint_steps') != default_checkpoint_steps(steps)):
        raise RuntimeError('Diagnostic boundaries differ from the qualified native profile')
    cfg = ControllerConfig(**spec['controller_config'])
    if world_size != 1 or cfg.aggregation_mode != 'additive_leaf_v1' or cfg.group_redundancy:
        raise RuntimeError('Expected one native model replica and unnormalized additive leaf steering')
    if not is_baseline(method) and (cfg.unsafe_gap_threshold != method['steering_margin'] or cfg.unsafe_abs_threshold != .25
            or cfg.unsafe_abs_ramp != .25 or cfg.unsafe_gap_ramp != .25):
        raise RuntimeError('Detector/controller confidence differs from the approved method')
    generator = torch.Generator(device='cuda').manual_seed(int(row['seed']))
    generation = dict(spec['model_config']['generation']); steps = int(generation.pop('num_inference_steps'))
    generation.pop('task', None)
    latents, state = legacy._prepare_initial_latents(adapter, row, generation, generator)
    initial_sha = tensor_sha256(latents)
    if row.get('expected_initial_latent_sha256') is not None and initial_sha != row['expected_initial_latent_sha256']:
        raise RuntimeError('Canonical initial latent identity mismatch')
    if latents.shape[0] != 1 or latents.dtype != native_dtype:
        raise RuntimeError('Expected one BF16 native latent sample')
    feature_dim = int(adapter.latent_layout(latents).feature_dim)
    _save_tensors(attempt / 'initial_latent.safetensors', {'prepared_latents': latents},
                  {'spec_sha256': spec_sha, 'model_revision': row['model_revision'], 'seed': row['seed']})
    timesteps = adapter.set_timesteps(steps, latents=latents, state=state); scheduler = adapter.pipeline.scheduler
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers import UniPCMultistepScheduler
    qualified = {FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler}
    if adapter.adapter_name == 'flux3':
        declared = getattr(adapter, 'native_scheduler_class', None)
        if declared is None or type(scheduler) is not declared:
            raise RuntimeError('FLUX3 native scheduler class is not explicitly qualified')
        qualified = {declared}
    if (type(scheduler) not in qualified
            or scheduler.config.get('stochastic_sampling', False)
            or scheduler.config.get('solver_p') is not None
            or len(timesteps) != steps or latents.dtype != native_dtype):
        raise RuntimeError('Expected approved deterministic native 50-step BF16 flow sampler')
    sigmas = scheduler.sigmas.cpu().tolist()
    if len(sigmas) != steps + 1 or any(not math.isfinite(s) for s in sigmas) or any(a <= b for a, b in zip(sigmas, sigmas[1:])):
        raise RuntimeError('Native schedule must expose 51 finite strictly descending sigmas')
    intervals = legacy._sigma_intervals(adapter, steps)
    adapter.begin_conditioning_provenance_scope()
    base = adapter.prepare_prompt_for_state(row['original_prompt'], state, prompt_view='execution_prompt', call_role='base')
    condition_sha = legacy.encoding_fingerprint(base.data)
    setup_native_state_policy(adapter,row['original_prompt'],state)
    hierarchy = MirroredConceptHierarchy.from_yaml_file(Path(spec['hierarchy_path']))
    leaves = sorted(k for k, node in hierarchy.nodes.items() if node.steerable)
    if len(hierarchy.nodes) != 343 or len(leaves) != 287:
        raise RuntimeError('The approved full 343-node, 287-leaf inventory is required')
    steerer = CorrectedHierarchySteerer(hierarchy=hierarchy, config=cfg, namespace=f"{spec.get('model_key', row['model'])}/{spec['run_id']}", alias_groups={}, num_inference_steps=steps, warmup_steps=warmup_steps)
    exposure = ExposureState(); start_index = 0
    selection = {'method_sha256': canonical_sha(method), 'initial_latent_sha256': initial_sha,
                 'selected_leaf_ids': [], 'selection_sha256': canonical_sha([]),
                 'candidate_pool_ids': [] if is_baseline(method) else None,
                 'candidate_pool_sha256': canonical_sha([]) if is_baseline(method) else None,
                 'selection_history': [],
                 'scan_history': [], 'last_scan_step': None, 'committed_artifacts': {}}
    timers = {'prior_generation_seconds': 0.0, 'preview_decode_seconds': 0.0, 'checkpoint_seconds': 0.0}
    for name in ('traces', 'checkpoints', 'previews', 'activations'):
        (attempt / name).mkdir(exist_ok=True)
    if resume:
        latents, state, start_index = restore(resume, attempt, resume_spec_sha(spec, spec_sha, resume), adapter, generator, exposure,
                                             condition_sha, selection, timers)
        _restore_decode_artifacts(resume, attempt, selection, spec)
        if selection['initial_latent_sha256'] != initial_sha:
            raise RuntimeError('Resume native seed draw differs from the original initial latent')
    else:
        for name in ('selection_events','pool_events'):
            (attempt / f'traces/{name}.jsonl').touch(exist_ok=True)
    validate_selection_state(selection, method, leaves, start_index)
    previous_elapsed = timers['prior_generation_seconds']
    baseline_records=None
    if not is_baseline(method):
        baseline_path=Path(spec['_baseline_attempt'])
        baseline_records=[json.loads(l) for l in (baseline_path/'traces/steering.jsonl').read_text().splitlines()]
        if [r['step'] for r in baseline_records]!=list(range(steps)):
            raise RuntimeError('METHOD requires its complete matched Baseline')
        baseline_meta=read(baseline_path/'metadata.json')
    metadata = {'spec_sha256': spec_sha, 'run_id': spec['run_id'], 'attempt': str(attempt),
                'method': method, 'hierarchy_sha256': digest(spec['hierarchy_path']),
                'model_revision': row['model_revision'], 'controller_config': asdict(cfg),
                'controller_enabled': not is_baseline(method), 'arm_name': spec['arm_name'],
                'shared_model_load_seconds': spec.get('_shared_model_load_seconds', load_seconds),
                'initial_latent_sha256': initial_sha, 'base_condition_sha256': condition_sha,
                'scheduler_config': dict(scheduler.config), 'scheduler_step_source_sha256':
                    hashlib.sha256(inspect.getsource(type(scheduler).step).encode()).hexdigest(),
                'runner_source_sha256': digest(Path(__file__)),
                'sigmas': sigmas, 'actual_runtime': runtime_metadata(),
                'placement': adapter.config.get('_eccv_placement'), 'resume_receipt': str(resume) if resume else None,
                'study_placement': adapter.config.get('_study_placement'),
                'guardrail_proof': adapter.config.get('_study_guardrail_proof'),
                'start_step': start_index, 'replicas': world_size, 'generation_status': 'DENOISING', 'start_utc': utc()}
    if resume:
        prior = read(Path(read(resume)['path']).parents[1] / 'metadata.preparing.json')
        for key in ('native_prediction_parity', 'native_prediction_parity_status'):
            if key not in prior:
                raise RuntimeError('Resume checkpoint lacks native first-step parity evidence')
            metadata[key] = prior[key]
    if not is_baseline(method):
        metadata['native_prediction_parity']=dict(baseline_meta['native_prediction_parity'],
            verification_scope=f'Inherited verified native Baseline, supplemented by{warmup_steps} actual warmup hash comparisons',
            extra_transformer_forwards=0)
        metadata['native_prediction_parity_status']=baseline_meta['native_prediction_parity_status']
        metadata['baseline_reference']={'attempt':str(baseline_path),'image_sha256':digest(baseline_path/'image.png')}
    atomic(attempt / 'metadata.preparing.json', metadata)
    preview_steps = set(spec['preview_steps'])
    checkpoint_steps = set(spec['checkpoint_steps'])
    identity = {'run_id': spec['run_id'], 'attempt_id': attempt.name, 'spec_sha256': spec_sha}
    denoise_start = time.perf_counter()
    with LocalPredictionProvider(adapter=adapter, rendezvous_dir=attempt / 'rendezvous', identity=identity,
                                  world_size=world_size, stop_paths=spec.get('stop_paths', []),
                                  startup_timeout_s=1800, request_timeout_s=1800) as pool, ForwardCounter(adapter.pipeline.transformer) as forwards:
        with torch.inference_mode():
            for index in range(start_index, steps):
                check_stop(); begin = time.perf_counter(); timestep = timesteps[index]
                sigma, following, h, _ = intervals[index]; before = latents
                guided=not is_baseline(method) and index>=warmup_steps
                scan=guided and index==acquisition_index
                evaluated=leaves if scan else selection['candidate_pool_ids'] if guided else []
                if evaluated is None:raise RuntimeError('Pool refresh began before acquisition')
                stage='native_baseline' if is_baseline(method) else 'native_warmup' if not guided else 'full_scan' if scan else 'pool_refresh'
                scheduler_before=scheduler_hash(scheduler) if index<warmup_steps else None
                cache_before=prefix_mode_counters(adapter)
                progress(attempt, status='DENOISING', completed_steps=index, total_steps=steps,
                         current_step=index, step_phase='STARTING_FULL_SCAN' if scan else 'STARTING_SELECTED_REFRESH',
                         stage=stage, evaluated_leaves=len(evaluated),
                         selected_leaf_ids=selection['selected_leaf_ids'], step_started_utc=utc())
                forward_start = forwards.count
                scope_factory = getattr(adapter, 'shared_prediction_scope', None)
                step_scope = scope_factory() if callable(scope_factory) else nullcontext({})
                with step_scope as reuse:
                    pool.scope_stats = reuse
                    legacy._sync(); phase = time.perf_counter()
                    v_base = adapter.predict_vector_field(before, timestep, base, state); legacy._sync()
                    base_seconds = time.perf_counter() - phase; parity_seconds = 0.0
                    if not torch.isfinite(v_base).all() or v_base.dtype != torch.bfloat16:
                        raise RuntimeError('Nonfinite or non-BF16 native base prediction')
                    if index == 0 and is_baseline(method):
                        phase = time.perf_counter()
                        from .native_checks import verify_first_prediction
                        try:
                            parity = verify_first_prediction(adapter, before, timestep, base, state, v_base, spec['model_config']['generation'])
                        except RuntimeError as exc:
                            if hasattr(exc, 'diagnostics'):
                                atomic(attempt / 'metadata.parity.json', exc.diagnostics)
                            raise
                        metadata['native_prediction_parity'] = parity
                        metadata['native_prediction_parity_status'] = 'PASS'
                        atomic(attempt / 'metadata.parity.json', parity)
                        legacy._sync(); parity_seconds = time.perf_counter() - phase
                        atomic(attempt / 'metadata.preparing.json', metadata)
                    pool.last_diagnostics = {}; provider_seconds = 0.0; controller_seconds = 0.0; routed = None
                    if evaluated:
                        phase = time.perf_counter()
                        routed = steerer.calculate(adapter=adapter, latents=before, timestep=timestep, state=state,
                            v_base=v_base, effective_weight=1.0, step_index=index, frozen=None, prediction_provider=pool,
                            evaluated_leaf_ids=None if scan else evaluated,
                            candidate_pool_ids=None if scan else selection['candidate_pool_ids'],
                            selection_margin=method['selection_margin'], active_k=POOL_SIZE,pool_size=POOL_SIZE,
                            sigmoid_temperature=method['sigmoid_temperature'], export_activation_joint=scan)
                        provider_seconds = pool.last_diagnostics['seconds']
                        if sorted(r['node_id'] for r in routed.node_records) != evaluated:
                            raise RuntimeError('Actual evaluated leaf inventory differs from the scan or current bank')
                        if any(r['direction_source_step'] != index for r in routed.node_records):
                            raise RuntimeError('A selected direction was frozen across native timesteps')
                        if routed.v_neutral is None:
                            raise RuntimeError('Current concept fields lack a current neutral prediction')
                        ranked = select_pool_and_active(
                            routed.activation_rows,
                            candidate_pool_ids=None if scan else selection['candidate_pool_ids'],
                            pool_size=POOL_SIZE, active_k=POOL_SIZE, selection_margin=.10)
                        if (ranked['candidate_pool_ids'] != list(routed.candidate_pool_ids)
                                or ranked['selected_leaf_ids'] != list(routed.selected_leaf_ids)):
                            raise RuntimeError('Current-state activation replay disagrees with the fixed correction pool')
                        if scan:
                            save_scan(attempt, index, routed, selection, method, spec_sha, before)
                        elif list(routed.candidate_pool_ids)!=evaluated or routed.pool_event is not None:
                            raise RuntimeError('Pool changed outside acquisition')
                        commit_selection(attempt,index,routed,selection,spec_sha)
                        applied = apply_correction(v_base, routed.v_neutral, routed.candidate, feature_dim=feature_dim,
                                                   delta_sigma=h, exposure_state=exposure, config=cfg)
                        prediction, delta, control = applied.prediction, applied.delta_fp32, applied.diagnostics
                        candidate = routed.candidate
                        effects = additive_leaf_effects(routed.selected_terms, candidate=candidate, applied_delta=delta)
                        legacy._sync(); controller_seconds = max(0., time.perf_counter() - phase - provider_seconds)
                        pool_record = dict(pool.last_diagnostics)
                    else:
                        prediction = v_base; delta = torch.zeros_like(v_base, dtype=torch.float32); candidate = delta
                        effects = []
                        control = zero_controller(v_base, exposure, 'native_baseline' if is_baseline(method) else 'native_warmup')
                        pool_record = {'unique_requests': 0, 'seconds': 0.0, 'replicas': world_size,
                                       'status': 'UNOBSERVED_NO_CONCEPT_CALLS', 'cross_replica_actual_prediction_checked': False}
                pool.scope_stats = None
                forward_end = forwards.count
                pool_record['native_transformer_forwards'] = forward_end - forward_start
                pool_record['forward_count_scope'] = 'Whole current-state base, parity, and concept work; separate from concept-only tensor requests'
                # Deep-copy the complete native solver history before any step.
                # UniPC corrections depend on history and cannot use Euler algebra.
                phase = time.perf_counter()
                result, effect, native_difference, ideal = native_step_with_counterfactual(
                    adapter, before, v_base, prediction, timestep, state, generator,
                    index, feature_dim,
                )
                latents, state = result.latents, result.state
                scheduler_after=scheduler_hash(scheduler) if index<warmup_steps else None
                matched_warmup={}
                if not is_baseline(method) and index<warmup_steps:
                    matched_warmup=warmup_match(before,v_base,latents,scheduler_before,scheduler_after,baseline_records,index)
                legacy._sync(); scheduler_seconds = time.perf_counter() - phase
                phase = time.perf_counter()
                effect.update(spec_sha256=spec_sha, selection_sha256=selection['selection_sha256'])
                append(attempt / 'traces/latent_effect.jsonl', effect)
                if routed:
                    terms = {term.concept_id: term for term in routed.selected_terms}
                    for item in effects:
                        node_id = item.get('concept_id', item.get('node_id'))
                        term = terms[node_id]
                        requested = term.confidence.float() * term.direction.float()
                        requested_latent = (scheduler.sigmas[index + 1] - scheduler.sigmas[index]).float().to(requested.device) * requested
                        item['aggregate_native_latent_projection'] = _projection(native_difference, requested_latent)
                        item['aggregate_native_latent_projection_scope'] = (
                            'association_with_signed_Euler_leaf_increment_not_causal'
                            if type(scheduler).__name__ == 'FlowMatchEulerDiscreteScheduler'
                            else 'association_with_signed_sigma_scaled_leaf_field_proxy_not_native_UniPC_increment_not_causal')
                if index in checkpoint_steps:
                    save_field_maps(attempt, index, routed, candidate, delta, native_difference, ideal, selection, spec_sha)
                legacy._sync(); diagnostic_seconds = time.perf_counter() - phase
                decode_seconds = 0.0
                if index in preview_steps or index == steps - 1:
                    phase = time.perf_counter()
                    preview = latents if index == steps - 1 else (before.float() - sigma * prediction.float()).to(before.dtype)
                    image = _sd35_decode_image(adapter, preview.detach(), state, attempt, index, index == steps - 1)
                    path = attempt / ('image.png' if index == steps - 1 else f'previews/step_{index:02d}_predicted_clean.png')
                    legacy._atomic_image(path, image)
                    atomic(path.with_suffix('.json'), {'step': index, 'kind': 'final' if index == steps - 1 else 'predicted_clean_estimate',
                           'sha256': digest(path), 'fed_back': False, 'preupdate_latent_sha256': tensor_sha256(before), 'utc': utc()})
                    if index != steps - 1:
                        _remember_artifacts(selection, attempt, [path, path.with_suffix('.json')])
                    legacy._sync(); decode_seconds = time.perf_counter() - phase
                    timers['preview_decode_seconds'] += decode_seconds
                record = {'step': index, 'sigma': sigma, 'next_sigma': following, 'delta_sigma': h,
                          'stage':stage, 'full_scan': scan,
                          'effective_weight':1.0 if guided else 0.0, 'direction_policy': method['direction_policy'],
                          'direction_source_step': index if evaluated else None,
                          'selected_leaf_ids': selection['selected_leaf_ids'], 'selection_sha256': selection['selection_sha256'],
                          'evaluated_leaf_ids': evaluated, 'active_leaf_ids': list(routed.active_leaf_ids) if routed else [],
                          'controller': control, 'hierarchy': routed.diagnostics if routed else {'status': 'UNOBSERVED_NO_CONCEPT_CALLS'},
                          'pool': pool_record, 'leaf_effects': effects, 'latent_effect': effect,
                          'candidate_bank_count':len(selection['candidate_pool_ids'] or []),
                          'candidate_pool_ids':selection['candidate_pool_ids'],
                          'candidate_pool_sha256':selection['candidate_pool_sha256'],
                          'admission_theta':None,'top_k_used':scan,
                          'ordinary_prediction_sha256':tensor_sha256(v_base) if index<warmup_steps else None,
                          'preupdate_latent_sha256':tensor_sha256(before) if index<warmup_steps else None,
                          'scheduler_before_sha256':scheduler_before,'scheduler_after_sha256':scheduler_after,
                          'warmup_baseline_match':matched_warmup,
                          'prefix_cache_before':cache_before,'prefix_cache_after':prefix_mode_counters(adapter),
                          'positive_confidence_leaf_count': len(routed.active_leaf_ids) if routed else 0,
                          'nonzero_requested_leaf_count': sum(item['requested_rms'] > 0 for item in effects),
                          'aggregate_correction_survives_native_update': effect['native_difference_rms'] > 0,
                          'cumulative_exposure': exposure.cumulative_exposure,
                          'applied_delta_sha256': tensor_sha256(delta), 'latent_sha256': tensor_sha256(latents),
                          'timing': {'base_prediction_seconds': base_seconds, 'parity_seconds': parity_seconds,
                                     'concept_provider_seconds': provider_seconds, 'controller_and_scan_export_seconds': controller_seconds,
                                     'scheduler_seconds': scheduler_seconds, 'numerical_diagnostic_seconds': diagnostic_seconds,
                                     'preview_decode_seconds': decode_seconds}, 'step_seconds': time.perf_counter() - begin}
                if routed:
                    with (attempt / 'traces/hierarchy.jsonl').open('a') as trace:
                        for node in routed.node_records:
                            trace.write(json.dumps(dict(node, step=index), sort_keys=True, allow_nan=False) + '\n')
                        trace.flush(); os.fsync(trace.fileno())
                else:
                    (attempt / 'traces/hierarchy.jsonl').touch(exist_ok=True)
                append(attempt / 'traces/leaf_activation.jsonl', {'step': index, 'full_scan': scan,
                       'selected_leaf_ids': selection['selected_leaf_ids'], 'evaluated_leaf_ids': evaluated,
                       'unevaluated_status': 'UNOBSERVED', 'rows': list(routed.activation_rows) if routed else [],
                       'leaf_effects': effects})
                append(attempt / 'traces/steering.jsonl', record)
                atomic(attempt / 'selection.json', dict(selection, spec_sha256=spec_sha))
                if index in checkpoint_steps:
                    phase = time.perf_counter(); timers['prior_generation_seconds'] = previous_elapsed + time.perf_counter() - started
                    receipt = checkpoint(attempt, index, latents, state, scheduler, exposure, generator,
                                         spec_sha, condition_sha, selection, timers,adapter=adapter)
                    elapsed = time.perf_counter() - phase; timers['checkpoint_seconds'] += elapsed
                    append(attempt / 'traces/checkpoint_timing.jsonl', {'step': index, 'seconds': elapsed, 'attempt': str(attempt), 'utc': utc()})
                    timers['prior_generation_seconds'] = previous_elapsed + time.perf_counter() - started
                    path = attempt / 'traces/checkpoint_timing.jsonl'
                    receipt.update(state_boundary='after_native_update', next_step=index + 1,
                        timing_state_after_commit=dict(timers), checkpoint_timing_commit={
                            'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)})
                    atomic(attempt / 'checkpoints/latest.json', receipt)
                    atomic(attempt / f'checkpoints/step_{index:02d}.json', receipt)
                progress(attempt, status='DENOISING', completed_steps=index + 1, total_steps=steps,
                         stage=record['stage'], selected_leaves=len(selection['selected_leaf_ids']),
                         selected_leaf_ids=selection['selected_leaf_ids'], active_leaves=len(record['active_leaf_ids']),
                         step_seconds=record['step_seconds'])
                print(json.dumps({'event': 'METHOD_LATENT_STEP', 'step': index, 'full_scan': scan,
                                  'seconds': record['step_seconds'], 'concept_calls': pool_record['unique_requests'],
                                  'selected_leaf_ids': selection['selected_leaf_ids'],
                                  'native_latent_difference_rms': effect['native_difference_rms']}), flush=True)
    denoise_seconds = time.perf_counter() - denoise_start
    if start_index == steps:
        phase = time.perf_counter()
        with torch.inference_mode():
            image = _sd35_decode_image(adapter, latents, state, attempt, steps - 1, True)
        legacy._atomic_image(attempt / 'image.png', image)
        atomic(attempt / 'image.json', {'step': steps - 1, 'kind': 'final', 'sha256': digest(attempt / 'image.png'),
                                      'restored_from_committed_terminal_latent': True, 'utc': utc()})
        timers['preview_decode_seconds'] += time.perf_counter() - phase
        atomic(attempt / 'selection.json', dict(selection, spec_sha256=spec_sha))
    records = [json.loads(line) for line in (attempt / 'traces/steering.jsonl').read_text().splitlines()]
    audit = validate_trajectory(spec, records, selection, leaves, attempt)
    atomic(attempt / 'numerical_audit.json', audit)
    if metadata.get('native_prediction_parity_status') != 'PASS':
        raise RuntimeError('Native first-step parity is not verified')
    from .loading import runtime_guardrail_proof
    guardrail_proof = runtime_guardrail_proof(adapter)
    atomic(attempt / 'guardrail_proof.json', dict(guardrail_proof, spec_sha256=spec_sha, utc=utc()))
    metadata.update(generation_status='COMPLETE_AWAITING_VISUAL_REVIEW', end_utc=utc(), selection=selection,
                    guardrail_proof=guardrail_proof)
    atomic(attempt / 'metadata.json', metadata)
    segment_seconds = time.perf_counter() - started
    totals = {key: sum(r['timing'][key] for r in records) for key in records[0]['timing']}
    concept_calls = sum(r['pool']['unique_requests'] for r in records)
    replica_audit = sum((r['pool']['replicas'] - 1) for r in records if r['pool']['cross_replica_actual_prediction_checked'])
    atomic(attempt / 'timing.json', {'model_load_seconds': load_seconds, 'generation_segment_seconds': segment_seconds,
        'generation_trajectory_seconds_including_committed_prior_segments': previous_elapsed + segment_seconds,
        'denoising_segment_seconds_including_previews': denoise_seconds,
        'stage_seconds': {stage: sum(r['step_seconds'] for r in records if r['stage'] == stage)
                         for stage in ('native_warmup','full_scan','pool_refresh','native_baseline')},
        'component_seconds_nested_within_stage': totals,
        'preview_and_final_decode_seconds': timers['preview_decode_seconds'], 'checkpoint_seconds': timers['checkpoint_seconds'],
        'shared_model_load_seconds': spec.get('_shared_model_load_seconds', load_seconds),
        'loading_attribution': 'Charged once to first newly executed arm in paired allocation; shared load reported separately, never sum it twice.',
        'arm_name': spec['arm_name'], 'replicas': world_size, 'gpu_count': torch.cuda.device_count(), 'gpu_hours_segment': torch.cuda.device_count() * (load_seconds + segment_seconds) / 3600,
        'resumed': bool(resume), 'full_scan_steps': method['full_scan_steps'],
        'selected_leaf_count': len(selection['selected_leaf_ids']),
        'candidate_pool_leaf_count':len(selection['candidate_pool_ids'] or []),
        'distinct_selected_leaf_count':len({node for event in selection['selection_history'] for node in event['selected_leaf_ids']}),
        'model_call_counts': {'base_prediction_tensors': steps,
                             'native_parity_ordinary_predictions':1 if is_baseline(method) else 0,
                             'native_parity_transformer_forwards': metadata['native_prediction_parity'].get('extra_transformer_forwards', 1),
                             'concept_and_neutral_unique_prediction_tensors': concept_calls,
                             'cross_replica_duplicate_audit': replica_audit, 'total_prediction_tensor_requests':steps+int(is_baseline(method))+concept_calls+replica_audit},
        'model_call_count_scope': 'Committed native base/concept prediction tensor requests. Native CFG may require two transformer forwards per request; measured forward counts are recorded separately. No replica duplicate audit is used. Failed/replayed work belongs to scheduler accounting.',
        'actual_native_transformer_forwards': sum(r['pool']['native_transformer_forwards'] for r in records),
        'nonzero_steering_updates': sum(r['controller']['applied_delta_rms'] > 0 for r in records),
        'nonzero_native_latent_updates': sum(r['latent_effect']['native_difference_rms'] > 0 for r in records),
        'timing_scope': 'Stage timers include nested components and previews, exclude checkpoint and trace/progress I/O. Provider includes conditioning/prediction/communication/validation; component separation not instrumented is not inferred. Queue/allocation accounting belongs to coordinator.',
        'resume_timing_limitation': 'Includes authenticated committed prior segment time. Failed/replayed work and interrupted checkpoint journal commits require scheduler accounting.'})
    files = ['initial_latent.safetensors', 'guardrail_proof.json', 'image.png', 'image.json', 'metadata.json', 'timing.json', 'selection.json', 'numerical_audit.json',
             'traces/steering.jsonl', 'traces/hierarchy.jsonl', 'traces/leaf_activation.jsonl',
             'traces/selection_events.jsonl','traces/pool_events.jsonl', 'traces/latent_effect.jsonl', 'traces/checkpoint_timing.jsonl']
    if adapter.adapter_name == 'sd35':
        files += ['traces/native_decode.jsonl']
    files += sorted(selection['committed_artifacts'])
    atomic(attempt / '_GENERATION_SUCCESS', {'spec_sha256': spec_sha, 'artifact_sha256': {p: digest(attempt / p) for p in files},
           'visual_success': None, 'utc': utc()})
    progress(attempt, status='AWAITING_AGENT_REVIEW', completed_steps=steps, total_steps=steps,
             selected_leaves=len(selection['selected_leaf_ids']))


def verify_completed_arm(arm):
    attempt = Path(arm['attempt'])
    receipt = read(attempt / '_GENERATION_SUCCESS')
    if receipt['spec_sha256'] != arm['spec_sha256']:
        raise RuntimeError('Completed arm belongs to another specification')
    for relative, expected in receipt['artifact_sha256'].items():
        if digest(attempt / relative) != expected:
            raise RuntimeError(f'Completed arm artifact changed: {relative}')
    return read(attempt / 'metadata.json')

def reset_adapter_for_arm(adapter):
    """Use a fresh complete native scheduler and reset declared generation caches."""
    scheduler = adapter.pipeline.scheduler
    adapter.pipeline.scheduler = type(scheduler).from_config(dict(scheduler.config))
    reset_arm_cache(adapter)

def run_arm(adapter, spec, spec_sha, attempt, *, load_seconds=0.0, resume=None):
    """Execute one immutable Baseline or METHOD attempt using already loaded weights.

    METHOD receives ``_baseline_attempt`` as runtime metadata, not as a mutation
    of its sealed scientific specification. A matching completed arm is reused.
    ``resume`` explicitly names its authenticated checkpoint receipt.
    """
    attempt = Path(attempt)
    attempt.mkdir(parents=True, exist_ok=True)
    arm = {'attempt': str(attempt), 'spec_sha256': spec_sha}
    if (attempt / '_GENERATION_SUCCESS').exists():
        return verify_completed_arm(arm)
    if resume is None and ((attempt / 'image.png').exists() or (attempt / 'traces/steering.jsonl').exists()):
        raise RuntimeError('Existing partial generation requires an explicit checkpoint or new infrastructure attempt')
    _profile(spec)
    if str(adapter.model_id) != spec['model_config']['model']['model_id']:
        raise RuntimeError('Loaded model differs from the immutable arm specification')
    if not is_baseline(spec['method']) and '_baseline_attempt' not in spec:
        raise RuntimeError('METHOD requires its authenticated matched Baseline attempt')
    root_token = legacy._PROJECT_ROOT.set(Path(spec.get('project_root', Path.cwd())).resolve())
    try:
        reset_adapter_for_arm(adapter)
        with execution_scope(attempt, spec.get('stop_paths', ())):
            generate_latent(adapter, spec, spec_sha, attempt, 1, load_seconds, Path(resume) if resume else None)
        return verify_completed_arm(arm)
    except Exception as exc:
        atomic(attempt / 'failure.json', {'type': type(exc).__name__, 'error': str(exc),
               'traceback': traceback.format_exc(), 'utc': utc(), 'spec_sha256': spec_sha})
        progress(attempt, status='FAILED', error=str(exc))
        raise
    finally:
        legacy._PROJECT_ROOT.reset(root_token)
