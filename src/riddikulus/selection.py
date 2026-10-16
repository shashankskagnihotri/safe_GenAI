"""Pure latent ranking with permanent candidate identities and no admission floor."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence

SCORE_MARGINS = (.10, .25, .40, .50, .60)


def default_method(num_inference_steps=50, warmup_steps=10):
    if (num_inference_steps, warmup_steps) not in {(50, 10), (16, 3)}:
        raise ValueError("Only qualified native50/warm10 and FLUX3 native16/warm3 are permitted")
    return {'schema': ('nailing-method.pool25-warm10.v1' if num_inference_steps == 50 else 'riddikulus.pool25-warm3-native16.v1'), 'warmup_steps': warmup_steps,
            'full_scan_steps': [warmup_steps], 'pool_size': 25, 'active_k': 25,
            'max_selected_leaves': 25, 'activation_threshold': None,
            'selection_margin': .10, 'steering_margin': .10,
            'sigmoid_temperature': 1.,
            'direction_policy': 'refresh_candidate_pool_every_guided_step',
            'steering_steps': list(range(warmup_steps, num_inference_steps)),
            'selection_rule': 'pure_mean_confidence_topk',
            'exact_pair_alias_deduplication': True,
            'pool_membership_policy': 'freeze_at_acquisition'}


def default_baseline():
    return {'schema': 'nailing-method.native-baseline.v1',
            'full_scan_steps': [],
            'max_selected_leaves': 0, 'activation_threshold': None,
            'selection_margin': None, 'steering_margin': None,
            'sigmoid_temperature': None, 'direction_policy': 'no_concept_predictions',
            'steering_steps': [], 'selection_rule': 'disabled',
            'exact_pair_alias_deduplication': True}


baseline_method = default_baseline


def is_baseline(method):
    return isinstance(method, Mapping) and method.get('schema') == 'nailing-method.native-baseline.v1'


def validate_method(method):
    expected = (default_baseline() if is_baseline(method) else
                default_method(16, 3) if isinstance(method, Mapping) and method.get('schema') == 'riddikulus.pool25-warm3-native16.v1' else default_method())
    if not isinstance(method, Mapping) or dict(method) != expected:
        raise ValueError('Method differs from approved pool25-warm10 or native Baseline recipe')
    for name in (('max_selected_leaves',) if is_baseline(method) else
                 ('warmup_steps', 'pool_size', 'active_k', 'max_selected_leaves')):
        if type(method[name]) is not int:
            raise ValueError('Method count must be an integer: ' + name)
    if any(type(i) is not int for i in method['full_scan_steps'] + method['steering_steps']):
        raise ValueError('Native schedule indices must be integers')
    if method['exact_pair_alias_deduplication'] is not True:
        raise ValueError('Exact-pair identity deduplication is required')
    return copy.deepcopy(expected)


def margin_key(margin):
    if type(margin) not in (int, float) or margin not in SCORE_MARGINS:
        raise ValueError('Unregistered confidence margin')
    return f'{margin:.2f}'


def activation_scores_from_joint(unsafe_cosine, cosine_gap, joint_reliability):
    """Canonical CPU FP32 all-position means, matching controller confidence."""
    import torch
    if (unsafe_cosine.shape != cosine_gap.shape or unsafe_cosine.shape != joint_reliability.shape
            or unsafe_cosine.numel() == 0 or joint_reliability.dtype != torch.bool):
        raise ValueError('Invalid joint detector shapes/reliability dtype')
    cu = unsafe_cosine.detach().float().cpu()
    gap = cosine_gap.detach().float().cpu()
    reliable = joint_reliability.detach().cpu()
    if not bool(torch.isfinite(cu).all()) or not bool(torch.isfinite(gap).all()):
        raise ValueError('Nonfinite detector evidence')
    if bool(((cu < -1) | (cu > 1)).any()) or bool(((gap < -2) | (gap > 2)).any()):
        raise ValueError('Evidence outside cosine domains')
    absolute = ((cu - .25) / .25).clamp(0, 1)
    scores = {}
    for margin in SCORE_MARGINS:
        eligible = reliable & (cu > .25) & (gap > margin)
        confidence = torch.where(eligible, absolute * ((gap - margin) / .25).clamp(0, 1),
                                 torch.zeros_like(cu))
        support = int(eligible.sum())
        scores[margin_key(margin)] = {'mean_confidence': float(confidence.mean()),
            'confidence_mass': float(confidence.sum()), 'support_positions': support,
            'support_fraction': support / confidence.numel()}
    return scores


def identity_hash(ids):
    return hashlib.sha256(json.dumps(sorted(ids), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def pure_rank(rows: Sequence[Mapping], selection_margin=.10):
    """Deduplicate exact text pairs and rank all identities, including zero scores."""
    key = margin_key(selection_margin)
    by_id = {}; by_pair = {}
    for source in rows:
        row = dict(source); node = row.get('node_id')
        if not isinstance(node, str) or not node or node in by_id:
            raise ValueError('Ranking requires unique nonempty node IDs')
        pair = (row.get('unsafe_concept'), row.get('safe_concept'))
        if any(not isinstance(x, str) or not x for x in pair):
            raise ValueError('Exact-pair identity requires two concept texts')
        score = row['activation_scores'][key]['mean_confidence']
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid activation score')
        by_id[node] = row; by_pair.setdefault(pair, []).append(node)
    if not by_id:
        raise ValueError('Ranking requires observed candidates')
    representatives = {}; aliases = {}
    for ids in by_pair.values():
        ids.sort(); representative = ids[0]; aliases[representative] = ids
        for node in ids:
            if by_id[node]['activation_scores'] != by_id[representative]['activation_scores']:
                raise RuntimeError('Exact text-pair aliases have different evidence')
            representatives[node] = representative
    order = sorted(aliases, key=lambda node: (-by_id[node]['activation_scores'][key]['mean_confidence'], node))
    ranks = {node: i + 1 for i, node in enumerate(order)}
    annotated = [dict(row, alias_representative_id=representatives[node],
        is_exact_alias_representative=node == representatives[node], rank=ranks[representatives[node]], selected=False)
        for node, row in sorted(by_id.items())]
    return {'rows': annotated, 'ranked_canonical_leaf_ids': order,
            'ranked_canonical_scores': [by_id[node]['activation_scores'][key]['mean_confidence'] for node in order],
            'evaluated_leaf_count': len(rows), 'canonical_leaf_count': len(order),
            'alias_groups': [{'representative_id': node, 'node_ids': ids}
                             for node, ids in sorted(aliases.items()) if len(ids) > 1]}


def select_pool_and_active(rows, *, candidate_pool_ids=None, pool_size=25, active_k=25, selection_margin=.10):
    if type(pool_size) is not int or pool_size != 25 or type(active_k) is not int or active_k != 25:
        raise ValueError('Approved pool and correction sizes are25 and25')
    if selection_margin != .10 or isinstance(selection_margin, bool):
        raise ValueError('Approved selector margin is0.10')
    ranking = pure_rank(rows, selection_margin)
    if candidate_pool_ids is None:
        if ranking['evaluated_leaf_count'] != 287 or ranking['canonical_leaf_count'] != 282:
            raise ValueError('Acquisition must evaluate all287 H4 leaves before exact-pair deduplication')
        pool_ranked = ranking['ranked_canonical_leaf_ids'][:pool_size]
        pool = sorted(pool_ranked)
        pool_event = dict(ranking, schema='nailing-pool25-warm10.acquisition.v1',
            candidate_pool_ids=pool, ranked_candidate_pool_ids=pool_ranked,
            candidate_pool_sha256=identity_hash(pool), pool_size=pool_size,
            theta=None, admission_floor_used=False, selection_margin=selection_margin,
            rule='pure_top25_canonical_mean_confidence_including_zero')
        # Annotated per-leaf rows belong in the activation trace, not duplicated in events.
        pool_event.pop('rows')
    else:
        if isinstance(candidate_pool_ids, (str, bytes)):
            raise ValueError('Candidate pool must be a sequence of stable IDs')
        pool = list(candidate_pool_ids)
        if len(pool) != pool_size or pool != sorted(set(pool)):
            raise ValueError('Candidate pool must contain25 sorted unique IDs')
        if set(ranking['ranked_canonical_leaf_ids']) != set(pool) or len(rows) != pool_size:
            raise RuntimeError('Refresh evaluated a different permanent candidate pool')
        pool_event = None
    pool_set = set(pool)
    within_pool = [node for node in ranking['ranked_canonical_leaf_ids'] if node in pool_set]
    selected_ranked = within_pool[:active_k]
    selected = sorted(selected_ranked)
    if selected != pool:
        raise RuntimeError('Every acquired pool identity must remain selected for correction')
    scores = dict(zip(ranking['ranked_canonical_leaf_ids'], ranking['ranked_canonical_scores']))
    event = {'schema': 'nailing-pool25-warm10.current-pool.v1',
        'selection_rule': 'all_fixed_pool_ids', 'theta': None, 'admission_floor_used': False,
        'selection_margin': selection_margin, 'max_selected_leaves': active_k, 'active_k': active_k,
        'pool_size': pool_size, 'candidate_pool_ids': pool,
        'candidate_pool_sha256': identity_hash(pool), 'selected_leaf_ids': selected,
        'selection_sha256': identity_hash(selected), 'ranked_selected_leaf_ids': selected_ranked,
        'ranked_canonical_leaf_ids': ranking['ranked_canonical_leaf_ids'],
        'ranked_canonical_scores': ranking['ranked_canonical_scores'],
        'ranked_candidate_pool_ids': within_pool,
        'ranked_candidate_pool_scores': [scores[i] for i in within_pool],
        'evaluated_leaf_count': len(rows), 'canonical_leaf_count': ranking['canonical_leaf_count'],
        'selected_zero_score_count': sum(scores[i] == 0 for i in selected),
        'alias_groups': ranking['alias_groups'], 'threshold_scope': 'no_admission_floor',
        'tie_policy': 'stable_canonical_node_id'}
    return {'rows': [dict(row, selected=row['node_id'] in selected,
                         candidate_pool_member=row['node_id'] in pool_set) for row in ranking['rows']],
            'candidate_pool_ids': pool, 'selected_leaf_ids': selected,
            'pool_event': pool_event, 'selection_event': event}
