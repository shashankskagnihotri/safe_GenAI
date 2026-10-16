"""Fresh latent evidence from a frozen25-ID pool, with all25 independently weighted additive terms.

No admission floor, no frozen prediction directions, and no prompt-text routing.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

import torch

from .numerics import tensor_sha256
from .numerics import (
    _condition_to_cpu,
    _condition_to_device,
    tensor_stats,
)
from .conditioning_cache import (
    StateAwareConditionCache, canonical_identity_sha256, encoding_fingerprint,
)
from .hierarchy import MirroredConceptHierarchy

from .diagnostics import DIAGNOSTIC_SCHEMA
from .controller import (
    ControllerConfig, ConceptTerm, combine_terms, concept_evidence, concept_multiplier,
    sigmoid_concept_multiplier, _group_terms, _distribution, _field_alignment_diagnostics,
)
from .selection import activation_scores_from_joint, margin_key, select_pool_and_active


def prediction_guidance_metadata(adapter: Any) -> dict[str, Any]:
    """Report the scale actually read by each unchanged adapter prediction."""
    if adapter.adapter_name in {"flux", "flux2"}:
        scale = float(adapter.config["guidance_scale"])
        return {
            "mechanism": "native_distilled_guidance_tensor",
            "scale": scale, "guidance_tensor_dtype": "torch.float32",
            "separate_unconditional_prediction": False,
            "source": "adapter.config.guidance_scale",
        }
    if adapter.adapter_name in {"cosmos3_t2i", "sd35"}:
        scale = float(adapter._manual_guidance_scale())
        return {
            "mechanism": "classifier_free_guidance",
            "scale": scale, "separate_unconditional_prediction": scale != 1.0,
            "formula": "uncond + scale * (cond - uncond)",
            "source": "adapter._manual_guidance_scale()",
        }
    if adapter.adapter_name == "flux3":
        return {"mechanism": "native_distilled_flow_map", "scale": None,
                "separate_unconditional_prediction": False, "source": "official_native_adapter"}
    raise ValueError(f"Unregistered guidance semantics: {adapter.adapter_name}")


def _record_sums(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "node_count": len(records),
        "active_node_count": sum(record["active"] for record in records),
        "position_observations": sum(record["evidence"]["position_count"] for record in records),
        "support_positions_sum": sum(record["evidence"]["support_positions"] for record in records),
        "active_positions_sum": sum(record["evidence"]["active_positions"] for record in records),
        "reliable_positions_sum": sum(record["evidence"]["reliability"]["effective"]["positions"] for record in records),
        "confidence_mass": sum(record["evidence"]["confidence_mass"] for record in records),
        "unsafe_cosine_sum": sum(record["evidence"]["unsafe_cosine_sum"] for record in records),
        "safe_cosine_sum": sum(record["evidence"]["safe_cosine_sum"] for record in records),
        "cosine_gap_sum": sum(record["evidence"]["cosine_gap_sum"] for record in records),
    }


def _branch_summaries(
    *, hierarchy: MirroredConceptHierarchy, branch_by_node: Mapping[str, str],
    records: list[dict[str, Any]], terms: list[ConceptTerm],
    config: ControllerConfig, reference: torch.Tensor,
) -> dict[str, Any]:
    branches = {}
    for branch_id in hierarchy.nodes[hierarchy.root_id].child_ids:
        branch_records = [record for record in records if record["branch_id"] == branch_id]
        leaves = [record for record in branch_records if record["steerable"]]
        branch_terms = [term for term in terms if branch_by_node[term.concept_id] == branch_id]
        groups = _group_terms(branch_terms, config)
        active_groups = [group for group in groups if bool((group.confidence > 0).any())]
        raw_sum = torch.zeros_like(reference, dtype=torch.float32)
        for term in branch_terms:
            raw_sum = raw_sum + term.confidence.float() * term.direction.float()
        branches[branch_id] = {
            "branch_id": branch_id,
            "unsafe_concept": hierarchy.nodes[branch_id].unsafe_concept,
            "safe_concept": hierarchy.nodes[branch_id].safe_concept,
            "inventory_node_count": sum(value == branch_id for value in branch_by_node.values()),
            "inventory_leaf_count": sum(
                branch_by_node.get(node_id) == branch_id and node.steerable
                for node_id, node in hierarchy.nodes.items()
            ),
            "all_nodes": _record_sums(branch_records), "leaves": _record_sums(leaves),
            "observation_scope": "evaluated_leaves_only; ancestor predictions unobserved",
            "ancestor_prediction_status": "unobserved_not_required_for_leaf_steering",
            "raw_leaf_count": len(branch_terms),
            "active_raw_leaf_count": sum(record["active"] for record in leaves),
            "grouped_leaf_count": len(active_groups),
            "grouped_leaf_confidence_mass": sum(float(group.confidence.float().sum()) for group in active_groups),
            "raw_confidence_weighted_direction_sum_stats": tensor_stats(raw_sum),
            "grouping_scope": "within_branch_only; cross_branch_aliases_can_change_global_counts",
            "direction_sum_scope": "after_per_concept_cosine_multiplier_before_grouping_overlap_normalization_lambda_and_conflicts",
        }
    return branches


@dataclass
class FrozenDirections:
    source_step: int | None = None
    directions_cpu: dict[str, torch.Tensor] = field(default_factory=dict)
    direction_sha256: dict[str, str] = field(default_factory=dict)
    bank_sha256: str | None = None
    multiplier_records: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        # An acquired empty bank is a valid zero-evidence frozen state.
        return self.source_step is not None

    def capture(
        self, directions: Mapping[str, torch.Tensor], *, step: int,
        multiplier_records: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if set(directions) != set(multiplier_records):
            raise ValueError("Every frozen direction must have its acquisition multiplier")
        values = {}
        records = {}
        for node_id, direction in sorted(directions.items()):
            if direction.dtype != torch.float32 or not torch.isfinite(direction).all():
                raise RuntimeError("Only finite float32 unmasked directions may be frozen")
            values[node_id] = direction.detach().contiguous().cpu().clone()
            record = dict(multiplier_records[node_id])
            multiplier = record["multiplier"]
            if not isinstance(multiplier, (int, float)) or not 0 <= multiplier <= 2:
                raise ValueError("Frozen multiplier must be finite and lie in [0, 2]")
            if record.get("source_step") != int(step):
                raise ValueError("Frozen multiplier and direction acquisition steps differ")
            # A JSON roundtrip detaches every nested diagnostic from caller mutation.
            records[node_id] = json.loads(json.dumps(record, allow_nan=False))
        self.source_step = int(step)
        self.directions_cpu = values
        self.direction_sha256 = {key: tensor_sha256(value) for key, value in values.items()}
        self.multiplier_records = records
        payload = json.dumps({"directions": self.direction_sha256, "multipliers": records},
                             sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.bank_sha256 = hashlib.sha256(payload.encode()).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "source_step": self.source_step,
            "active_leaf_ids_at_acquisition": sorted(self.directions_cpu),
            "direction_sha256": self.direction_sha256,
            "bank_sha256": self.bank_sha256,
            "representation": "raw_unmasked_float32_safe_minus_unsafe_and_acquisition_cosine_multiplier",
            "multiplier_records": self.multiplier_records,
            "reuse_policy": "fixed_acquired_leaf_set_raw_directions_and_multipliers; current_evidence_conflicts_budgets",
        }


@dataclass
class RoutedCorrection:
    candidate: torch.Tensor
    v_neutral: torch.Tensor | None
    active_leaf_ids: tuple[str, ...]
    acquired_directions: dict[str, torch.Tensor]
    acquired_multiplier_records: dict[str, dict[str, Any]]
    node_records: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    selected_leaf_ids: tuple[str, ...] | None = None
    candidate_pool_ids: tuple[str, ...] | None = None
    pool_event: dict[str, Any] | None = None
    activation_rows: tuple[dict[str, Any], ...] = ()
    activation_joint: dict[str, torch.Tensor] = field(default_factory=dict)
    selection_event: dict[str, Any] | None = None
    selected_terms: tuple[ConceptTerm, ...] = ()
    selected_raw_directions: dict[str, torch.Tensor] = field(default_factory=dict)


class CorrectedHierarchySteerer:
    def __init__(
        self,
        *,
        hierarchy: MirroredConceptHierarchy,
        config: ControllerConfig,
        namespace: str,
        alias_groups: Mapping[str, str] | None = None,
        num_inference_steps: int = 50, warmup_steps: int = 10,
    ) -> None:
        hierarchy.validate()
        if (num_inference_steps, warmup_steps) not in {(50, 10), (16, 3)}:
            raise ValueError("Unqualified native acquisition profile")
        self.num_inference_steps = num_inference_steps
        self.acquisition_index = warmup_steps
        if len(hierarchy.nodes) != 343 or sum(node.steerable for node in hierarchy.nodes.values()) != 287:
            raise ValueError("Pool25-warm10 requires the complete343-node/287-leaf H4 inventory")
        self.hierarchy = hierarchy
        self.config = config
        self.cache = StateAwareConditionCache(namespace=namespace)
        self.alias_groups = dict(alias_groups or {})
        self._condition_data_sha256: dict[str, str] = {}
        self.branch_by_node: dict[str, str] = {}
        self.hierarchy_path_by_node: dict[str, tuple[str, ...]] = {}
        for node_id, node in hierarchy.nodes.items():
            if node_id == hierarchy.root_id:
                continue
            ancestor = node
            while ancestor.parent_id != hierarchy.root_id:
                ancestor = hierarchy.nodes[ancestor.parent_id]
            self.branch_by_node[node_id] = ancestor.id
            path = []
            parent_id = node.parent_id
            while parent_id != hierarchy.root_id:
                path.append(parent_id)
                parent_id = hierarchy.nodes[parent_id].parent_id
            self.hierarchy_path_by_node[node_id] = tuple(reversed(path))
        unknown = set(self.alias_groups) - set(hierarchy.nodes)
        if unknown:
            raise ValueError(f"Alias map contains unknown hierarchy nodes: {sorted(unknown)}")
        exact_pairs: dict[tuple[str, str], str] = {}
        self.canonical_leaf_by_id: dict[str, str] = {}
        for node_id, node in sorted(hierarchy.nodes.items()):
            if node.steerable:
                pair = (node.unsafe_concept, node.safe_concept)
                self.canonical_leaf_by_id[node_id] = exact_pairs.setdefault(pair, node_id)

    def _alias_key(self, node_id: str) -> str:
        if node_id in self.alias_groups:
            return "declared:" + self.alias_groups[node_id]
        node = self.hierarchy.nodes[node_id]
        # Exact textual pairs only; sharing a safe target never establishes aliasing.
        pair = [node.unsafe_concept, node.safe_concept]
        return "pair:" + json.dumps(pair, separators=(",", ":"))

    def calculate(
        self,
        *,
        adapter: Any,
        latents: torch.Tensor,
        timestep: Any,
        state: Any,
        v_base: torch.Tensor,
        effective_weight: float,
        step_index: int,
        frozen: FrozenDirections | None = None,
        prediction_provider: Callable[..., Mapping[str, torch.Tensor]] | None = None,
        evaluated_leaf_ids: Sequence[str] | None = None,
        candidate_pool_ids: Sequence[str] | None = None,
        selection_margin: float = 0.10,
        active_k: int = 25,
        pool_size: int = 25,
        sigmoid_temperature: float = 1.0,
        export_activation_joint: bool = False,
    ) -> RoutedCorrection:
        started = time.perf_counter()
        additive = self.config.aggregation_mode == "additive_leaf_v1"
        if not additive or frozen is not None or self.alias_groups:
            raise ValueError("Pool25-warm10 requires fresh additive fields without declared semantic aliases")
        if effective_weight != 1.0 or type(step_index) is not int or not self.acquisition_index <= step_index < self.num_inference_steps:
            raise ValueError("Pool25-warm10 steering runs only native indices10 through49 at weight1")
        if type(active_k) is not int or active_k != 25 or type(pool_size) is not int or pool_size != 25:
            raise ValueError("Pool25-warm10 requires25 evaluated candidates and all25 correction leaves")
        if selection_margin != .10 or isinstance(selection_margin, bool) or sigmoid_temperature != 1.0 or isinstance(sigmoid_temperature, bool):
            raise ValueError("Pool25-warm10 margin and sigmoid temperature must be0.10 and1")
        fixed = {
            "center_residuals": True, "threshold_gates": True, "confidence_weighting": True,
            "unsafe_abs_threshold": .25, "unsafe_gap_threshold": .10,
            "unsafe_abs_ramp": .25, "unsafe_gap_ramp": .25,
            "minimum_support_fraction": 0., "minimum_support_positions": 1,
            "allow_suppression_only": False, "lambda_max": 1.,
            "group_redundancy": False, "conflict_handling": True, "conflict_mu": 1.,
            "norm_budgets": True, "global_relative_rms": .6, "local_relative_rms": .85,
            "prompt_residual_relative_rms": None, "cumulative_exposure_limit": None,
            "quantization_backtracking": "ratio_margin",
        }
        for name, value in fixed.items():
            if getattr(self.config, name) != value:
                raise ValueError(f"Pool25-warm10 requires fixed {name}={value}")
        if step_index == self.acquisition_index:
            if evaluated_leaf_ids is not None or candidate_pool_ids is not None:
                raise ValueError("Acquisition must evaluate all leaves before creating the candidate pool")
        else:
            if evaluated_leaf_ids is None or candidate_pool_ids is None:
                raise ValueError("Guided refresh needs its exact permanent candidate pool")
            if isinstance(evaluated_leaf_ids, (str, bytes)) or isinstance(candidate_pool_ids, (str, bytes)):
                raise ValueError("Candidate identities must be ID sequences")
            evaluated_leaf_ids = tuple(evaluated_leaf_ids)
            candidate_pool_ids = tuple(candidate_pool_ids)
            if (len(evaluated_leaf_ids) != pool_size or len(set(evaluated_leaf_ids)) != pool_size
                    or candidate_pool_ids != tuple(sorted(set(candidate_pool_ids)))
                    or len(candidate_pool_ids) != pool_size
                    or set(evaluated_leaf_ids) != set(candidate_pool_ids)):
                raise RuntimeError("Refresh evaluated identities differ from the fixed25 candidate pool")
            if any(self.canonical_leaf_by_id.get(node_id) != node_id for node_id in evaluated_leaf_ids):
                raise ValueError("Refresh pool must contain known canonical H4 leaves")
        if type(export_activation_joint) is not bool or (export_activation_joint and step_index != self.acquisition_index):
            raise ValueError("Joint full-inventory export is acquisition-only")
        # Acquisition always retains its actual already-computed joint evidence.
        export_activation_joint = step_index == self.acquisition_index
        # Legacy evidence options are intentionally disabled in this new recipe.
        selection_gap_threshold = None
        selection_abs_threshold = .25
        activation_threshold = None
        mark = self.cache.mark()
        predictions: dict[str, torch.Tensor] = {}
        conditioning_by_role: dict[str, dict[str, Any]] = {}
        guidance = prediction_guidance_metadata(adapter)
        cache_hits = 0
        cache_misses = 0
        supplied_predictions: Mapping[str, torch.Tensor] | None = None
        provider_evidence: Mapping[str, Mapping[str, Any]] | None = None
        provider_condition_events: list[dict[str, Any]] = []

        def predict(text: str, role: str) -> torch.Tensor:
            nonlocal cache_hits, cache_misses
            condition_cpu = None
            if provider_evidence is None:
                condition_mark = self.cache.mark()
                condition_cpu = self.cache.get_or_prepare_one(
                    adapter=adapter, prompt=text, state=state,
                    prompt_view="registered_concept_only", call_role=role,
                    prepare=lambda: _condition_to_cpu(adapter.prepare_prompt_for_state(
                        text, state, prompt_view="registered_concept_only", call_role=role,
                    )),
                )
                event, = self.cache.evidence_since(condition_mark)
                data_sha = self._condition_data_sha256.get(event["identity_sha256"])
                if data_sha is None:
                    data_sha = encoding_fingerprint(condition_cpu.data)
                uncond_key = condition_cpu.data.get("uncond_prediction_cache_key")
            else:
                event = dict(provider_evidence[text])
                identity = adapter.conditioning_cache_identity(
                    text, state, prompt_view="registered_concept_only", call_role=role,
                )
                identity_sha = canonical_identity_sha256(identity)
                if (event.get("identity_sha256") != identity_sha
                        or canonical_identity_sha256(event.get("identity")) != identity_sha):
                    raise RuntimeError("Replica conditioning identity differs from leader's current state")
                if (event.get("prompt_sha256") != hashlib.sha256(text.encode()).hexdigest()
                        or event.get("prompt_view") != "registered_concept_only"):
                    raise RuntimeError("Replica conditioning prompt identity differs from request")
                for name in ("condition_data_sha256", "encoding_fingerprint"):
                    digest = event.get(name)
                    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                        raise RuntimeError(f"Missing or invalid replica conditioning fingerprint: {name}")
                if event.get("status") not in {"hit", "miss"}:
                    raise RuntimeError("Replica conditioning cache status is invalid")
                data_sha = event["condition_data_sha256"]
                uncond_key = event.get("unconditional_prediction_cache_key")
                event.update({
                    "representative_call_role": event.get("call_role"), "call_role": role,
                    "replica_event_status": event["status"],
                    "status": "hit" if text in predictions else event["status"],
                    "step_local_alias_reuse": text in predictions,
                    "cache_backend": "prediction_provider",
                    "leader_reencoded_condition": False,
                })
                provider_condition_events.append(event)
            key = event["identity_sha256"]
            previous_data_sha = self._condition_data_sha256.setdefault(key, data_sha)
            if previous_data_sha != data_sha:
                raise RuntimeError("Encoded condition changed while its cache identity stayed fixed")
            conditioning_by_role[role] = {
                **event,
                "condition_data_sha256": self._condition_data_sha256[key],
                "prediction_cache_key": hashlib.sha256(text.encode()).hexdigest(),
                "prediction_cache_hit": text in predictions,
                "guidance": dict(guidance),
                "unconditional_prediction_cache_key": uncond_key,
            }
            if text in predictions:
                cache_hits += 1
                return predictions[text]
            if supplied_predictions is None:
                assert condition_cpu is not None
                condition = _condition_to_device(condition_cpu, latents.device)
                value = adapter.predict_vector_field(latents, timestep, condition, state)
            else:
                value = supplied_predictions[text]
                if value.device != v_base.device or value.dtype != v_base.dtype:
                    raise RuntimeError("Provider prediction dtype/device differs from native base")
            if value.shape != v_base.shape:
                raise RuntimeError("Concept prediction shape differs from native base")
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite concept prediction: {role}")
            cache_misses += 1
            predictions[text] = value
            return value

        feature_dim = int(adapter.latent_layout(v_base).feature_dim)
        if frozen is not None and not frozen.available:
            raise RuntimeError("Frozen directions requested before acquisition")
        node_ids = (
            sorted(frozen.directions_cpu) if frozen is not None else
            [node_id for node_id in self.hierarchy.traversal_order() if self.hierarchy.nodes[node_id].steerable]
        )
        if evaluated_leaf_ids is not None:
            selected_set = set(evaluated_leaf_ids)
            node_ids = [node_id for node_id in node_ids if node_id in selected_set]
        if frozen is not None and set(frozen.directions_cpu) != set(frozen.multiplier_records):
            raise RuntimeError("Frozen raw directions and multiplier inventory differ")
        requests = {self.hierarchy.neutral_concept: "neutral"}
        for node_id in node_ids:
            node = self.hierarchy.nodes[node_id]
            requests.setdefault(node.unsafe_concept, f"unsafe_source:{node_id}")
            requests.setdefault(node.safe_concept, f"safe_target:{node_id}")
        if prediction_provider is not None:
            supplied_predictions = prediction_provider(
                adapter=adapter, latents=latents, timestep=timestep, state=state,
                requests=requests,
            )
            if not isinstance(supplied_predictions, Mapping) or set(supplied_predictions) != set(requests):
                raise RuntimeError("Provider returned a different unique concept request inventory")
            provider_evidence = getattr(prediction_provider, "conditioning_evidence", None)
            if provider_evidence is not None and (
                not isinstance(provider_evidence, Mapping) or set(provider_evidence) != set(requests)
            ):
                raise RuntimeError("Provider returned a different conditioning evidence inventory")
        node_records: list[dict[str, Any]] = []
        terms: list[ConceptTerm] = []
        acquired: dict[str, torch.Tensor] = {}
        acquired_multipliers: dict[str, dict[str, Any]] = {}
        active: list[str] = []
        selected: list[str] | None = [] if selection_gap_threshold is not None else None
        activation_rows: list[dict[str, Any]] = []
        activation_joint_by_id: dict[str, dict[str, torch.Tensor]] = {}
        selection_event: dict[str, Any] | None = None
        admitted_bank: tuple[str, ...] | None = None
        current_raw_by_id: dict[str, torch.Tensor] = {}
        scope_factory = getattr(adapter, "shared_prediction_scope", None)
        scope = (nullcontext(prediction_provider.last_diagnostics) if prediction_provider is not None
                 else scope_factory() if callable(scope_factory) else nullcontext({}))
        with scope as reuse:
            v_neutral = predict(self.hierarchy.neutral_concept, "neutral")
            for node_id in node_ids:
                node = self.hierarchy.nodes[node_id]
                unsafe = predict(node.unsafe_concept, f"unsafe_source:{node_id}")
                safe = predict(node.safe_concept, f"safe_target:{node_id}")
                evidence = concept_evidence(
                    v_base, v_neutral, unsafe, safe, feature_dim=feature_dim, config=self.config,
                    selection_gap_threshold=selection_gap_threshold,
                    selection_abs_threshold=selection_abs_threshold,
                    capture_joint=additive,
                )
                if selected is not None and evidence.diagnostics["one_time_selection"]["selected"]:
                    selected.append(node_id)
                record = {
                    "step_index": int(step_index), "node_id": node_id,
                    "diagnostic_schema_version": DIAGNOSTIC_SCHEMA["version"],
                    "branch_id": self.branch_by_node[node_id],
                    "parent_id": node.parent_id, "level": node.level,
                    "unsafe_concept": node.unsafe_concept, "safe_concept": node.safe_concept,
                    "steerable": node.steerable, "active": bool(evidence.active),
                    "ancestor_gate_used": False,
                    "evidence": evidence.diagnostics,
                    "confidence_mass": evidence.diagnostics["confidence_mass"],
                    "confidence_stats": tensor_stats(evidence.confidence),
                    "direction_source_step": frozen.source_step if frozen is not None else step_index,
                    "conditioning": {
                        "neutral": conditioning_by_role["neutral"],
                        "unsafe": conditioning_by_role[f"unsafe_source:{node_id}"],
                        "safe": conditioning_by_role[f"safe_target:{node_id}"],
                        "neutral_safe_text_equal": self.hierarchy.neutral_concept == node.safe_concept,
                        "neutral_safe_identity_equal": (
                            conditioning_by_role["neutral"]["identity_sha256"]
                            == conditioning_by_role[f"safe_target:{node_id}"]["identity_sha256"]
                        ),
                        "neutral_safe_encoded_data_equal": (
                            conditioning_by_role["neutral"]["condition_data_sha256"]
                            == conditioning_by_role[f"safe_target:{node_id}"]["condition_data_sha256"]
                        ),
                        "neutral_safe_prediction_equal": bool(torch.equal(v_neutral, safe)),
                        "neutral_unsafe_prediction_equal": bool(torch.equal(v_neutral, unsafe)),
                        "unsafe_safe_prediction_equal": bool(torch.equal(unsafe, safe)),
                    },
                }
                if node.steerable:
                    raw_direction = (
                        frozen.directions_cpu[node_id].to(device=v_base.device) if frozen is not None
                        else safe.float() - unsafe.float()
                    )
                    if raw_direction.shape != v_base.shape:
                        raise RuntimeError("Frozen direction shape differs from current prediction")
                    if frozen is None:
                        if additive:
                            multiplier, multiplier_record = sigmoid_concept_multiplier(
                                v_neutral, unsafe, safe, config=self.config,
                                temperature=sigmoid_temperature,
                            )
                        else:
                            multiplier, multiplier_record = concept_multiplier(
                                v_neutral, unsafe, safe, config=self.config,
                            )
                        multiplier_record["source_step"] = int(step_index)
                    else:
                        multiplier_record = dict(frozen.multiplier_records[node_id])
                        multiplier = multiplier_record["multiplier"]
                    direction = raw_direction * multiplier
                    terms.append(ConceptTerm(
                        concept_id=node_id, alias_key=self._alias_key(node_id),
                        direction=direction, confidence=evidence.confidence,
                        hierarchy_path=self.hierarchy_path_by_node[node_id],
                    ))
                    record["direction_stats"] = tensor_stats(direction)
                    record["raw_direction_stats"] = tensor_stats(raw_direction)
                    record["cosine_multiplier"] = multiplier_record
                    record["multiplier_is_frozen"] = frozen is not None
                    record["direction_usable"] = (
                        bool(evidence.active) and multiplier > 0
                        and record["raw_direction_stats"]["rms"] > 0
                    )
                    record["alias_key"] = self._alias_key(node_id)
                    record["aggregation_hierarchy_path"] = list(self.hierarchy_path_by_node[node_id])
                    if evidence.active:
                        active.append(node_id)
                        if frozen is None:
                            acquired[node_id] = raw_direction
                            acquired_multipliers[node_id] = multiplier_record
                    if additive:
                        current_raw_by_id[node_id] = raw_direction
                        assert evidence.joint is not None
                        scores = activation_scores_from_joint(**evidence.joint)
                        # Both computations use the same declared ramps and all-position mean.
                        if scores["0.10"]["mean_confidence"] != float(evidence.confidence.detach().float().cpu().mean()):
                            raise RuntimeError("Activation .10 score differs from applied confidence evidence")
                        joint = evidence.joint
                        reliable_count = int(joint["joint_reliability"].sum())
                        position_count = joint["unsafe_cosine"].numel()
                        activation_rows.append({
                            "node_id": node_id, "step_index": int(step_index),
                            "unsafe_concept": node.unsafe_concept, "safe_concept": node.safe_concept,
                            "alias_representative_id": self.canonical_leaf_by_id[node_id],
                            "is_exact_alias_representative": self.canonical_leaf_by_id[node_id] == node_id,
                            "rank": None, "selected": False,
                            "activation_scores": scores, "position_count": position_count,
                            "reliable_positions": reliable_count,
                            "reliable_fraction": reliable_count / position_count,
                            "raw_direction_rms": record["raw_direction_stats"]["rms"],
                            "cosine_distance": multiplier_record["cosine_distance"],
                            "sigmoid_weight": multiplier,
                            "direction_usable": record["direction_usable"],
                        })
                        record["activation_scores"] = scores
                        if export_activation_joint:
                            activation_joint_by_id[node_id] = {
                                key: tensor.detach().reshape(-1).contiguous().cpu().clone()
                                for key, tensor in joint.items()
                            }
                    if frozen is not None:
                        # Observation only: reuse predictions already evaluated for
                        # current evidence. Neither this direction nor kappa enters
                        # terms, acquisition, confidence, or the frozen bank.
                        with torch.no_grad():
                            current_raw = safe.detach().float() - unsafe.detach().float()
                            current_kappa, current_geometry = concept_multiplier(
                                v_neutral.detach(), unsafe.detach(), safe.detach(), config=self.config,
                            )
                            finite_geometry = all(
                                math.isfinite(value) for value in current_geometry.values()
                                if isinstance(value, float)
                            )
                            current_kappa_observed = current_geometry["geometry_reliable"] and finite_geometry
                            current_geometry = {
                                key: None if isinstance(value, float) and not math.isfinite(value) else value
                                for key, value in current_geometry.items()
                            }
                            comparable_kappa = (current_kappa_observed
                                                and multiplier_record["geometry_reliable"])
                            record["frozen_direction_drift"] = {
                                "schema": "frozen_direction_geometry_v1",
                                "current_step": int(step_index), "frozen_source_step": frozen.source_step,
                                "scope": "evaluated_frozen_bank_leaf; raw_unmasked_safe_minus_unsafe; no_semantic_safety_interpretation",
                                "raw_direction_alignment": _field_alignment_diagnostics(current_raw, raw_direction),
                                "current_kappa": current_kappa if current_kappa_observed else None,
                                "frozen_kappa": multiplier if multiplier_record["geometry_reliable"] else None,
                                "kappa_comparison_status": (
                                    "unobserved_nonfinite_geometry" if not finite_geometry else
                                    "observed" if comparable_kappa else "unobserved_unreliable_geometry"
                                ),
                                "kappa_difference_current_minus_frozen": current_kappa - multiplier if comparable_kappa else None,
                                "current_over_frozen_kappa_ratio": current_kappa / multiplier if comparable_kappa and multiplier > 0 else None,
                                "kappa_ratio_unobserved_reason": (
                                    "nonfinite_geometry" if not finite_geometry else
                                    "unreliable_geometry" if not comparable_kappa else
                                    "zero_frozen_kappa" if multiplier == 0 else None
                                ),
                                "current_kappa_geometry": current_geometry,
                                "additional_model_calls": 0, "feeds_steering": False,
                            }
                            del current_raw
                node_records.append(record)
        if additive:
            decision = select_pool_and_active(
                activation_rows, candidate_pool_ids=candidate_pool_ids,
                pool_size=pool_size, active_k=active_k, selection_margin=selection_margin,
            )
            admitted_bank = tuple(decision["selected_leaf_ids"])
            fixed_pool = tuple(decision["candidate_pool_ids"])
            activation_rows = decision["rows"]
            selection_event = dict(decision["selection_event"], step=int(step_index), step_index=int(step_index))
            pool_event = (dict(decision["pool_event"], step=int(step_index), step_index=int(step_index))
                          if decision["pool_event"] is not None else None)
            admitted = set(admitted_bank)
            terms = sorted((term for term in terms if term.concept_id in admitted),
                           key=lambda term: term.concept_id)
            active = sorted(node_id for node_id in active if node_id in admitted)
            acquired = {node_id: value for node_id, value in acquired.items() if node_id in admitted}
            acquired_multipliers = {node_id: value for node_id, value in acquired_multipliers.items()
                                    if node_id in admitted}
            for record in node_records:
                record["admitted_bank_member"] = record["node_id"] in admitted
                record["requested_correction_active"] = (
                    record["node_id"] in admitted and record["direction_usable"]
                )
        if len(terms) > active_k:
            raise RuntimeError("Controller received more than25 correction terms")
        candidate, combination = combine_terms(
            terms, feature_dim=feature_dim, config=self.config,
            effective_weight=effective_weight, reference=v_base,
        )
        condition_events = (tuple(provider_condition_events) if provider_evidence is not None
                            else self.cache.evidence_since(mark))
        multiplier_values = torch.tensor(
            [record["cosine_multiplier"]["multiplier"] for record in node_records], dtype=torch.float32,
        )
        unobserved_ancestors = [
            node_id for node_id, node in self.hierarchy.nodes.items()
            if node.level > 0 and not node.steerable
        ]
        diagnostics = {
            "calculation_seconds": time.perf_counter() - started,
            "nodes_evaluated": len(node_records),
            "unique_concept_predictions_requested": len(requests),
            "prediction_provider_used": prediction_provider is not None,
            "ancestor_predictions": {
                "status": "unobserved", "reason": "not_used_by_leaf_controller",
                "node_ids": sorted(unobserved_ancestors), "count": len(unobserved_ancestors),
                "topology_preserved_for_aggregation": True,
            },
            "cosine_multipliers": {
                "scope": "evaluated_leaf_inventory; acquisition geometry when frozen",
                "source_step": frozen.source_step if frozen is not None else int(step_index),
                "count": len(node_records),
                "zero_count": sum(record["cosine_multiplier"]["zero_multiplier"] for record in node_records),
                "unreliable_geometry_count": sum(not record["cosine_multiplier"]["geometry_reliable"] for record in node_records),
                "usable_active_leaf_count": sum(record["direction_usable"] for record in node_records),
                "distribution": _distribution(multiplier_values, [i / 10 for i in range(21)]),
            },
            "leaves_evaluated": sum(record["steerable"] for record in node_records),
            "active_leaves": len(active),
            "condition_cache_hits": sum(row["status"] == "hit" for row in condition_events),
            "condition_cache_misses": sum(row["status"] == "miss" for row in condition_events),
            "prediction_cache_hits": cache_hits, "prediction_cache_misses": cache_misses,
            "unconditional_cache_hits": int(reuse.get("unconditional_cache_hits", 0)),
            "unconditional_cache_misses": int(reuse.get("unconditional_cache_misses", 0)),
            "combination": combination,
            "diagnostic_schema_version": DIAGNOSTIC_SCHEMA["version"],
            "branch_by_node": dict(self.branch_by_node),
            "branches": _branch_summaries(
                hierarchy=self.hierarchy, branch_by_node=self.branch_by_node,
                records=node_records, terms=terms, config=self.config, reference=v_base,
            ) if not additive else {
                "scope": "hierarchy_is_inventory_only; no_branch_direction_aggregation",
                "evaluated_leaf_count": len(node_records), "admitted_bank_size": len(terms),
            },
            "raw_leaf_count": combination["raw_leaf_count"],
            "active_raw_leaf_count": combination["active_raw_leaf_count"],
            "grouped_leaf_count": combination["grouped_leaf_count"],
            "condition_cache_events": list(condition_events),
            "conditioning_by_role": conditioning_by_role,
            "per_branch_guidance": {role: dict(guidance) for role in ("neutral", "unsafe_source", "safe_target")},
            "current_state_evidence_recomputed": True,
            "direction_acquisition_occurred": frozen is None,
            "hierarchy_policy": (
                "fixed_selected_leaf_set_current_evidence_and_geometry" if evaluated_leaf_ids is not None else
                "exhaustive_leaves_no_ancestor_predictions" if frozen is None else
                "acquired_leaf_set_current_evidence_frozen_geometry"
            ),
            **({"evaluated_leaf_ids": list(evaluated_leaf_ids), "evaluated_inventory_size": len(evaluated_leaf_ids),
                "direction_policy": "refresh_selected", "topology_preserved_for_aggregation": True}
               if evaluated_leaf_ids is not None else {}),
            **({"one_time_selection": {"selected_leaf_ids": sorted(selected),
                                       "selected_count": len(selected),
                                       "selection_step": int(step_index),
                                       "gap_threshold": selection_gap_threshold,
                                       "absolute_threshold": selection_abs_threshold,
                                       "rule": "any_reliable_position"}}
               if selected is not None else {}),
        }
        if additive:
            diagnostics.update(
                method_schema="nailing-method.pool25-warm10.v1",
                hierarchy_policy="inventory_only_additive_selected_leaves",
                topology_preserved_for_aggregation=False,
                evaluated_leaf_ids=[row["node_id"] for row in activation_rows],
                selected_leaf_ids=list(admitted_bank), admitted_bank_size=len(admitted_bank),
                candidate_pool_ids=list(fixed_pool), candidate_pool_size=len(fixed_pool),
                pool_membership_policy="freeze_at_acquisition", pool_event=pool_event,
                selection_threshold_scope="no_admission_floor",
                activation_threshold=activation_threshold, selection_margin=selection_margin,
                steering_margin=0.10, sigmoid_temperature=sigmoid_temperature,
                direction_policy="refresh_candidate_pool_every_guided_step",
                activation_arithmetic="canonical_CPU_FP32_from_current_FP32_joint_cosines",
            )
        joint_export = ({key: torch.stack([activation_joint_by_id[row["node_id"]][key]
                                           for row in activation_rows])
                         for key in ("unsafe_cosine", "cosine_gap", "joint_reliability")}
                        if activation_joint_by_id else {})
        return RoutedCorrection(
            candidate, v_neutral, tuple(active), acquired, acquired_multipliers,
            tuple(node_records), diagnostics,
            admitted_bank if additive else (tuple(sorted(selected)) if selected is not None else None),
            activation_rows=tuple(activation_rows), activation_joint=joint_export,
            candidate_pool_ids=fixed_pool, pool_event=pool_event,
            selection_event=selection_event, selected_terms=tuple(terms) if additive else (),
            selected_raw_directions={node_id: current_raw_by_id[node_id] for node_id in admitted_bank}
            if additive else {},
        )
