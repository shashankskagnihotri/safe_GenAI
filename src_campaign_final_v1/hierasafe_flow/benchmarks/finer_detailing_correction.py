from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import traceback
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from PIL import Image
import yaml

from hierasafe_flow.adapters.hunyuan_video_adapter import (
    HUNYUAN_DUAL_VIEW_CONFIG_KEY,
    HunyuanConditioningPlan,
    hunyuan_conditioning_plan_sha256,
)
from hierasafe_flow.benchmarks.benign_park import (
    _generation_config,
    _resolve_model_config,
    _runner_config as _shared_runner_config,
    _runtime_config,
    _sanitize_generation_for_task,
    _steering_config,
    _write_job_notes,
)
from hierasafe_flow.benchmarks.slurm_tracking import (
    ENVIRONMENT_PREFLIGHT_DIGEST_FILENAME,
    ENVIRONMENT_PREFLIGHT_FILENAME,
    environment_preflight_required,
    read_environment_preflight,
    write_execution_identity,
)
from hierasafe_flow.cli.run_redteam_tri_condition import run_native_negative_prompt_baseline
from hierasafe_flow.generation.temporal_artifacts import read_temporal_evidence
from hierasafe_flow.generation.runner import GenerationRunner
from hierasafe_flow.evaluation.flux1_dual_view_jobs_v3 import (
    MODE_AUDIT as FLUX1_V3_MODE_AUDIT,
    MODE_EXECUTION as FLUX1_V3_MODE_EXECUTION,
    MODE_PREVIEW as FLUX1_V3_MODE_PREVIEW,
    SOURCE_PROTOCOL_INPUT_ROLES as FLUX1_V3_SOURCE_INPUT_ROLES,
    canonical_sha256 as flux1_canonical_sha256,
    execution_protocol_input_roles_v3,
    is_flux1_job_v3,
    project_flux1_job_v3,
    validate_flux1_job_v3,
    validate_flux1_runtime_v3,
)
from hierasafe_flow.evaluation.temporal_qualification import (
    TEMPORAL_CRITERIA_BY_MODEL,
    validate_temporal_production_gate,
)
from hierasafe_flow.steering.concept_graph import ConceptHierarchy
from hierasafe_flow.utils.io import ensure_dir, write_json, write_yaml
from hierasafe_flow.utils.config import deep_merge, load_yaml


BENCHMARK_NAME = "finer_detailing_correction_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROMPT_IDS = (
    "01_sad_young_girl",
    "02_angry_old_man",
    "03_empty_outdoor_mall",
)
VARIATION_GROUPS = (
    "01_baseline",
    "02_negative_prompt",
    "03_concept_steering",
    "04_shapley_concept_steering",
    "05_concept_steering_single_pair",
    "06_shapley_concept_steering_single_pair",
)
VARIATION_ALIASES = {
    "baseline": "01_baseline",
    "native_negative_prompt": "02_negative_prompt",
    "negative_prompt": "02_negative_prompt",
    "conceptsteer_full": "03_concept_steering",
    "shapley_full": "04_shapley_concept_steering",
    "conceptsteer_single": "05_concept_steering_single_pair",
    "shapley_single": "06_shapley_concept_steering_single_pair",
}


_VARIATION_KIND_ALIASES: dict[str, str] = {}


def _normalize_variation_group(variation_group: str) -> str:
    return VARIATION_ALIASES.get(variation_group, variation_group)


def _normalize_variant_kind(variant_kind: str) -> str:
    return _VARIATION_KIND_ALIASES.get(variant_kind, variant_kind)
MODEL_NAMES = (
    "cogvideox_5b",
    "cosmos3_super_text2image",
    "flux1_dev",
    "flux2_dev",
    "hunyuan_video",
    "ideogram4_nf4",
    "joyai_echo",
    "ltx_23",
    "qwen_image",
    "qwen_image_2512",
    "sd35_large",
    "wan22_t2v_a14b",
)
EXPECTED_MODEL_REVISIONS = {
    "cogvideox_5b": "8fc5b281006c82b82d34fd2543d2f0ebb4e7e321",
    "cosmos3_super_text2image": "c92432b9dff7eeb055d11ef5cf01c2c85bda0f28",
    "flux1_dev": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
    "flux2_dev": "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
    "hunyuan_video": "e8c2aaa66fe3742a32c11a6766aecbf07c56e773",
    "ideogram4_nf4": "1874bc70267ba2c823a7239e1d70dd308c8d64dc",
    "joyai_echo": "4187f9a53c6eff3a76c51e79bd27f70d10f7591b",
    "ltx_23": "432e0d3c2d1769aaa4d295f9243f7062bf6b47ee",
    "qwen_image": "75e0b4be04f60ec59a75f475837eced720f823b6",
    "qwen_image_2512": "25468b98e3276ca6700de15c6628e51b7de54a26",
    "sd35_large": "ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f",
    "wan22_t2v_a14b": "5be7df9619b54f4e2667b2755bc6a756675b5cd7",
}
SEGMENTED_TEMPORAL_MODELS = frozenset({"cogvideox_5b", "hunyuan_video", "joyai_echo"})
EXPECTED_CHECKPOINT_SETS: dict[str, list[dict[str, Any]]] = {
    "cogvideox_5b": [
        {
            "role": "primary",
            "model_id": "zai-org/CogVideoX-5b",
            "revision": "8fc5b281006c82b82d34fd2543d2f0ebb4e7e321",
        },
        {
            "role": "continuation",
            "model_id": "zai-org/CogVideoX-5b-I2V",
            "revision": "a6f0f4858a8395e7429d82493864ce92bf73af11",
        },
    ],
    "hunyuan_video": [
        {
            "role": "primary",
            "model_id": "hunyuanvideo-community/HunyuanVideo",
            "revision": "e8c2aaa66fe3742a32c11a6766aecbf07c56e773",
            "conversion": "community_diffusers_conversion",
        },
        {
            "role": "continuation",
            "model_id": "hunyuanvideo-community/HunyuanVideo-I2V",
            "revision": "fb9d287ef02fe6d39f2e23df6dcec1294e6c28d2",
            "conversion": "community_diffusers_conversion",
        },
    ],
    "joyai_echo": [
        {
            "role": "primary",
            "model_id": "jdopensource/JoyAI-Echo",
            "revision": "4187f9a53c6eff3a76c51e79bd27f70d10f7591b",
        },
        {
            "role": "route_source",
            "source_root": "/ceph/sagnihot/projects/JoyAI-Echo",
            "revision": "bdd3ec9ecad0bbbfc006cf5288709cb744c00b01",
            "tracked_diff_sha256": (
                "fa783e1f63be60d87014e5a0f8f4ef8cf020d27aa4583ab1cf7378d4c3129ae0"
            ),
        },
    ],
}
SEGMENTED_PROTOCOL_CONTRACT = {
    "temporal_protocol_schema_version": 2,
    "native_temporal_call_schema_version": 2,
    "segment_trace_schema_version": 1,
    "temporal_evidence_schema_version": 1,
    "segmented_temporal_audit_schema_version": 1,
}
MODEL_SPECIFIC_VIDEO_PROMPT_FIELDS = {
    "cogvideox_5b": "video_prompt_cogvideox_5b",
    "hunyuan_video": "video_prompt_hunyuan_video",
}

# The launch-disabled Flux common-seed stage reuses ordinary baseline job
# semantics while binding additional preregistration/equivalence artifacts.
# Keeping the exact role set here lets immutable snapshot publication include
# those bytes without weakening the five-role contract for ordinary jobs.
# Keep the historical public name as an exact v2 compatibility alias.  New
# common-seed protocols must never change what that name means: sealed v2
# evidence is always interpreted by the v2 reader, while v3 is routed to its
# independent validator below.
FLUX_COMMON_SEED_V2_STAGE = "flux1_common_seed_source_ladder_v2"
FLUX_COMMON_SEED_STAGE = FLUX_COMMON_SEED_V2_STAGE
FLUX_COMMON_SEED_V3_STAGE = "flux1_common_seed_source_ladder_v3"
FLUX_COMMON_SEED_STAGES = frozenset(
    {FLUX_COMMON_SEED_V2_STAGE, FLUX_COMMON_SEED_V3_STAGE}
)

# Explicit hidden construction context for byte-sealed v2 manifests.  New
# ordinary campaign builds default to the dual-view v3 preview route and can be
# promoted to execution only with the authenticated equivalence DAG.  Nothing
# except the historical v2 builder may select the legacy context.
FLUX1_ROUTE_CONTEXT_LEGACY_V2 = "legacy_v2"
FLUX1_ROUTE_CONTEXT_V3_PREVIEW = FLUX1_V3_MODE_PREVIEW
FLUX1_ROUTE_CONTEXT_V3_EXECUTION = FLUX1_V3_MODE_EXECUTION
FLUX1_ROUTE_CONTEXTS = frozenset(
    {
        FLUX1_ROUTE_CONTEXT_LEGACY_V2,
        FLUX1_ROUTE_CONTEXT_V3_PREVIEW,
        FLUX1_ROUTE_CONTEXT_V3_EXECUTION,
    }
)
FLUX_COMMON_SEED_PROTOCOL_INPUT_ROLES = frozenset(
    {
        "common_seed_config",
        "common_seed_protocol",
        "prompt_suite",
        "negative_prompt_suite",
        "model_config",
        "repaired_flux_adapter",
        "native_equivalence_result",
        "native_equivalence_receipt",
    }
)

# Calibration v2 consumes the independently sealed common-seed selection.
# These roles are admitted only when both the dedicated benchmark and stage
# identities match; the dedicated strict reader still owns the full 45-row
# scientific contract.
FLUX_CALIBRATION_V2_BENCHMARK = "flux1_native_negative_true_cfg_scale_v2"
FLUX_CALIBRATION_V2_STAGE = "flux1_native_negative_scale_calibration_v2"
FLUX_CALIBRATION_V2_PROTOCOL_INPUT_ROLES = frozenset(
    {
        "calibration_v2_config",
        "calibration_v2_config_sidecar",
        "calibration_v2_protocol",
        "calibration_v2_protocol_sidecar",
    }
)
FLUX_CALIBRATION_V2_SELECTION_INPUT_ROLES = frozenset(
    {
        "selection_receipt",
        "selection_receipt_sidecar",
        "selection_publication_commit",
        "selection_publication_commit_sidecar",
        "authentication_report",
        "reviewer_manifest",
        "private_unblinding_map",
        "normalized_manual_review_ledger",
    }
)
FLUX_CALIBRATION_V2_INPUT_ROLES = frozenset(
    FLUX_CALIBRATION_V2_PROTOCOL_INPUT_ROLES | FLUX_CALIBRATION_V2_SELECTION_INPUT_ROLES
)

PERSON_PAIR_IDS = (
    "facial_affect_negative_to_happy",
    "body_pose_sitting_to_walking",
    "clothing_color_green_to_red_blue",
    "sandwich_action_eating_to_holding",
    "composition_static_to_dynamic",
)
MALL_PAIR_IDS = (
    "sky_color_blue_to_pink",
    "vertical_circulation_escalators_to_marble_stairs",
    "horizontal_floor_marble_to_tile",
    "signage_sale_to_new_arrival",
    "merchandise_handbags_to_cars",
)
PAIR_IDS_BY_PROMPT = {
    "01_sad_young_girl": PERSON_PAIR_IDS,
    "02_angry_old_man": PERSON_PAIR_IDS,
    "03_empty_outdoor_mall": MALL_PAIR_IDS,
}
NO_COMPOSITION_PAIR_IDS = PERSON_PAIR_IDS[:-1]
EMOTION_POSE_ACTION_PAIR_IDS = (
    "facial_affect_negative_to_happy",
    "body_pose_sitting_to_walking",
    "sandwich_action_eating_to_holding",
)

VIDEO_CONTRACT = {
    "duration_seconds": 15.0,
    "num_frames": 240,
    "fps": 16,
}
PERSON_IMAGE_CONTRACT = {"height": 1216, "width": 832}
MALL_IMAGE_CONTRACT = {"height": 832, "width": 1216}

SHAPLEY_CONFIG: dict[str, Any] = {
    "estimator": "antithetic_permutation",
    "seed": 0,
    "min_permutations": 8,
    "max_permutations": 256,
    "confidence": 0.95,
    "relative_ci": 0.10,
    "absolute_ci": 1.0e-4,
    "ci_positive_only": True,
    "ci_quantile": 0.99,
    "score_tau": 0.10,
    "positive_only": True,
    "top_mass": 0.90,
    "per_coordinate_cap": 1.0,
    "trust_region_ratio": 0.25,
    "backtracking_scales": [1.0, 0.5, 0.25, 0.125, 0.0625],
    "interpolation_fraction_cap": 1.0,
    "efficiency_tolerance": 5.0e-3,
    # Final experiment manifests must fail closed when the deterministic
    # antithetic estimator exhausts its budget without meeting the CI gate.
    "require_convergence": True,
    "token_chunk_size": 65_536,
    "exact_max_work": 2_000_000,
}
SHAPLEY_PROVENANCE = {
    "protocol_version": 2,
    "trace_schema_version": 2,
    "requested_method_name": "SHAPELY_CONCEPT_STEERING",
    "canonical_mathematical_name": "Shapley concept steering",
    "inspiration_repository": "https://github.com/slds-lmu/shapleig",
    "players": "feature-channel coordinates within each flattened denoiser vector-field token",
    "attribution_scope": "denoiser vector-field feature-channel coordinates",
    "game": {
        "coalition_background": "safe-prompt denoiser vector-field prediction",
        "coalition_explicand": "current sequential denoiser vector-field prediction",
    },
    "intervention_identity": (
        "shapley_selected_attribution_weighted_current_to_safe_prediction_rollback"
    ),
    "candidate_dtype": "model_dtype",
    "score_dtype": "float32",
    "quantize_before_score": True,
    "strict_score_decrease": True,
    "safe_endpoint_cap": True,
    "post_quantization_trust_check": True,
    "interpretation": "functional_attribution_not_causal",
    "single_pair_attribution_recomputed": True,
}

# These profiles reuse only architecture-specific hyperparameters curated in the
# preceding study (composition mode, strength, stride, and applicable pair
# windows). The 2026-07-15 protocol deliberately replaces the prior active-pair
# selection with either all five prompt-specific pairs or exactly one ablated
# pair, so the prior variant name is provenance rather than an exact rerun label.
CURATED_STEERING_PROFILES: dict[str, dict[str, Any]] = {
    "cogvideox_5b": {
        "source_variant": "conceptsteer_no_composition_append_strength_1p00_mask_off_norm_off_full_window",
        "active_pair_ids": NO_COMPOSITION_PAIR_IDS,
        "prompt_composition": "append",
        "strength": 1.0,
    },
    "cosmos3_super_text2image": {
        "source_variant": "conceptsteer_full_default",
        "active_pair_ids": PERSON_PAIR_IDS,
        "prompt_composition": "concept_only",
        "strength": 1.0,
    },
    "flux1_dev": {
        "source_variant": "conceptsteer_full_default",
        "active_pair_ids": EMOTION_POSE_ACTION_PAIR_IDS,
        "prompt_composition": "concept_only",
        "strength": 1.0,
        "pair_overrides": {
            "facial_affect_negative_to_happy": {"start_fraction": 0.25, "end_fraction": 0.75},
            "body_pose_sitting_to_walking": {"start_fraction": 0.0, "end_fraction": 1.0},
            "sandwich_action_eating_to_holding": {"start_fraction": 0.0, "end_fraction": 1.0},
        },
    },
    "flux2_dev": {
        "source_variant": "conceptsteer_full_default",
        "active_pair_ids": PERSON_PAIR_IDS,
        "prompt_composition": "concept_only",
        "strength": 1.0,
    },
    "hunyuan_video": {
        "source_variant": "conceptsteer_full_default",
        "active_pair_ids": PERSON_PAIR_IDS,
        "prompt_composition": "concept_only",
        "strength": 1.0,
        "step_stride": 10,
    },
    "ideogram4_nf4": {
        "source_variant": "initial_full_default_unvalidated_for_ideogram4",
        "active_pair_ids": PERSON_PAIR_IDS,
        "prompt_composition": "concept_only",
        "strength": 1.0,
    },
    "joyai_echo": {
        "source_variant": "conceptsteer_full_default",
        "active_pair_ids": PERSON_PAIR_IDS,
        "prompt_composition": "concept_only",
        "strength": 1.0,
    },
    "ltx_23": {
        "source_variant": "conceptsteer_full_append_strength_0p10_mask_off_norm_off_full_window",
        "active_pair_ids": PERSON_PAIR_IDS,
        "prompt_composition": "append",
        "strength": 0.1,
    },
    "qwen_image": {
        "source_variant": "conceptsteer_no_composition_concept_only_strength_1p00_mask_off_norm_off_full_window",
        "active_pair_ids": NO_COMPOSITION_PAIR_IDS,
        "prompt_composition": "concept_only",
        "strength": 1.0,
    },
    "qwen_image_2512": {
        "source_variant": "conceptsteer_full_strength_1p50",
        "active_pair_ids": PERSON_PAIR_IDS,
        "prompt_composition": "concept_only",
        "strength": 1.5,
    },
    "sd35_large": {
        "source_variant": "conceptsteer_full_strength_0p50",
        "active_pair_ids": PERSON_PAIR_IDS,
        "prompt_composition": "concept_only",
        "strength": 0.5,
    },
    "wan22_t2v_a14b": {
        "source_variant": "conceptsteer_full_default",
        "active_pair_ids": PERSON_PAIR_IDS,
        "prompt_composition": "concept_only",
        "strength": 1.0,
        "step_stride": 10,
    },
}

EXPECTED_NATIVE_NEGATIVE_SUPPORT = {
    "cogvideox_5b": True,
    "cosmos3_super_text2image": False,
    "flux1_dev": True,
    "flux2_dev": False,
    "hunyuan_video": True,
    "ideogram4_nf4": False,
    "joyai_echo": False,
    "ltx_23": False,
    "qwen_image": True,
    "qwen_image_2512": True,
    "sd35_large": True,
    "wan22_t2v_a14b": True,
}

NATIVE_NEGATIVE_UNSUPPORTED_REASONS = {
    "cosmos3_super_text2image": (
        "Cosmos3's configured generation pipeline does not expose a native negative_prompt input."
    ),
    "flux2_dev": "Flux2Pipeline.__call__ does not expose a native negative_prompt input.",
    "ideogram4_nf4": (
        "Ideogram4Pipeline uses a fixed image-only unconditional transformer and does not expose negative_prompt."
    ),
    "joyai_echo": (
        "JoyAI-Echo uses its vendor standalone inference engine, which does not expose a native negative prompt."
    ),
    "ltx_23": (
        "The configured LTX-2.3 distilled checkpoint requires CFG=1; at CFG=1 LTX2Pipeline ignores "
        "negative_prompt, and the vendor distilled pipeline exposes no negative-prompt interface."
    ),
}

# These pipelines expose negative_prompt but only use the negative branch when
# true classifier-free guidance is greater than one. The value is frozen in the
# manifest so a nominally supported comparison cannot silently become a no-op.
EFFECTIVE_NATIVE_NEGATIVE_OPTIONS: dict[str, dict[str, Any]] = {
    "flux1_dev": {"true_cfg_scale": 4.0},
    "hunyuan_video": {"true_cfg_scale": 4.0},
    "qwen_image": {"true_cfg_scale": 4.0},
    "qwen_image_2512": {"true_cfg_scale": 4.0},
}


def _require_flux1_native_negative_calibration_v2_admission(root: Path) -> None:
    """Admit ordinary Flux native-negative rows only after the final v2 decision."""

    # Local import avoids a module cycle: the evaluator deliberately imports
    # this benchmark to reuse its frozen semantic contracts.
    from hierasafe_flow.evaluation import native_negative_calibration_v2

    try:
        selection = native_negative_calibration_v2.read_final_evidence(
            project_root=root,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(
            "Ordinary flux1_dev native-negative generation is blocked because the "
            "canonical calibration-v2 final evidence is absent or invalid."
        ) from exc

    if (
        selection.get("status") != "selected"
        or selection.get("normal_benchmark_update_authorized") is not True
    ):
        raise ValueError(
            "Ordinary flux1_dev native-negative generation is blocked because "
            "calibration-v2 produced no selected scale."
        )
    options = EFFECTIVE_NATIVE_NEGATIVE_OPTIONS.get("flux1_dev")
    expected_scale = options.get("true_cfg_scale") if isinstance(options, Mapping) else None
    selected_scale = selection.get("selected_true_cfg_scale")
    if (
        isinstance(expected_scale, bool)
        or not isinstance(expected_scale, (int, float))
        or isinstance(selected_scale, bool)
        or not isinstance(selected_scale, (int, float))
        or float(selected_scale) != float(expected_scale)
    ):
        raise ValueError(
            "Ordinary flux1_dev native-negative true_cfg_scale differs from the "
            "fully verified calibration-v2 selection: "
            f"configured={expected_scale!r}, selected={selected_scale!r}."
        )


# This is the complete, launch-blocking implementation surface for this
# benchmark.  Config/model/concept inputs are frozen separately per job.  The
# directories below intentionally include every adapter and every module in the
# generation, logging, steering, and utility layers, so adding a new Python
# file to one of those layers is itself detectable provenance drift.
IMPLEMENTATION_REQUIRED_FILES = (
    "src/hierasafe_flow/__init__.py",
    "src/hierasafe_flow/benchmarks/__init__.py",
    "src/hierasafe_flow/benchmarks/benign_park.py",
    "src/hierasafe_flow/benchmarks/flux1_native_negative_calibration.py",
    "src/hierasafe_flow/benchmarks/flux1_native_negative_calibration_v2.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_correction.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_phase.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_production_smoke.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_smoke_launch.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_qualification.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_qualification_launch.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_campaign_launch.py",
    "src/hierasafe_flow/benchmarks/slurm_tracking.py",
    "src/hierasafe_flow/evaluation/temporal_promotion.py",
    "src/hierasafe_flow/utils/immutable_publication.py",
    "src/hierasafe_flow/cli/run_redteam_tri_condition.py",
    "scripts/__init__.py",
    "scripts/finer_detailing_environment_dispatch.py",
    "scripts/finer_detailing_smoke_dispatch.py",
    "scripts/finer_detailing_qualification_dispatch.py",
    "scripts/finer_detailing_campaign_dispatch.py",
    "scripts/audit_finer_detailing_correction.py",
    "scripts/evaluate_finer_detailing_production_smoke_gate.py",
    "scripts/orchestrate_finer_detailing_phase.py",
    "scripts/orchestrate_finer_detailing_smoke_launch.py",
    "scripts/orchestrate_finer_detailing_qualification_launch.py",
    "scripts/orchestrate_finer_detailing_campaign_launch.py",
    "scripts/manage_finer_detailing_campaign_cohort.py",
    "scripts/plan_finer_detailing_production_smoke.py",
    "scripts/plan_finer_detailing_qualification.py",
    "scripts/prepare_finer_detailing_visual_review.py",
    "scripts/prepare_finer_detailing_temporal_promotion.py",
    "scripts/run_flux1_native_negative_calibration.py",
    "scripts/run_flux1_native_negative_calibration_dispatched.sh",
    "scripts/run_flux1_native_negative_calibration_v2.py",
    "scripts/run_flux1_native_negative_calibration_v2_dispatched.sh",
    "scripts/evaluate_finer_detailing_native_negative_calibration_v2.py",
    "scripts/run_flux1_common_seed_v2_dispatched.sh",
    "scripts/finer_detailing_environment_dispatch_v3.py",
    "scripts/run_flux1_common_seed_v3_dispatched.sh",
    "scripts/run_finer_detailing_dispatched.sh",
    "scripts/run_finer_detailing_smoke_dispatched.sh",
    "scripts/run_finer_detailing_qualification_dispatched.sh",
    "scripts/run_finer_detailing_campaign_dispatched.sh",
    "scripts/run_finer_detailing_correction.py",
    "scripts/write_finer_detailing_seed_selection.py",
    "scripts/create_conda_env.sh",
    "requirements-ltx23.txt",
    "requirements.txt",
    "environment-ltx23.yml",
    "environment.yml",
    "pyproject.toml",
    "slurm/finer_detailing_manifest_array_h100.sbatch",
    "slurm/finer_detailing_shapley_array_h100.sbatch",
    "slurm/flux1_native_negative_calibration_h100.sbatch",
    "slurm/flux1_native_negative_calibration_v2_h100.sbatch",
    "slurm/flux1_common_seed_v2_h100.sbatch",
    "slurm/flux1_common_seed_v3_h100.sbatch",
    "slurm/finer_detailing_production_smoke_h100.sbatch",
    "slurm/finer_detailing_qualification_h100.sbatch",
    "slurm/finer_detailing_campaign_h100.sbatch",
)
IMPLEMENTATION_PYTHON_DIRECTORIES = (
    "src/hierasafe_flow/adapters",
    "src/hierasafe_flow/evaluation",
    "src/hierasafe_flow/generation",
    "src/hierasafe_flow/logging_utils",
    "src/hierasafe_flow/steering",
    "src/hierasafe_flow/utils",
)
IMPLEMENTATION_GIT_PATHS = (
    "src/hierasafe_flow/__init__.py",
    "src/hierasafe_flow/adapters",
    "src/hierasafe_flow/evaluation",
    "src/hierasafe_flow/benchmarks/__init__.py",
    "src/hierasafe_flow/benchmarks/benign_park.py",
    "src/hierasafe_flow/benchmarks/flux1_native_negative_calibration.py",
    "src/hierasafe_flow/benchmarks/flux1_native_negative_calibration_v2.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_correction.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_phase.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_production_smoke.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_smoke_launch.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_qualification.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_qualification_launch.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_campaign_launch.py",
    "src/hierasafe_flow/benchmarks/slurm_tracking.py",
    "src/hierasafe_flow/evaluation/temporal_promotion.py",
    "src/hierasafe_flow/utils/immutable_publication.py",
    "src/hierasafe_flow/cli/run_redteam_tri_condition.py",
    "src/hierasafe_flow/generation",
    "src/hierasafe_flow/logging_utils",
    "src/hierasafe_flow/steering",
    "src/hierasafe_flow/utils",
    "scripts/__init__.py",
    "scripts/finer_detailing_environment_dispatch.py",
    "scripts/finer_detailing_smoke_dispatch.py",
    "scripts/finer_detailing_qualification_dispatch.py",
    "scripts/finer_detailing_campaign_dispatch.py",
    "scripts/audit_finer_detailing_correction.py",
    "scripts/evaluate_finer_detailing_production_smoke_gate.py",
    "scripts/orchestrate_finer_detailing_phase.py",
    "scripts/orchestrate_finer_detailing_smoke_launch.py",
    "scripts/orchestrate_finer_detailing_qualification_launch.py",
    "scripts/orchestrate_finer_detailing_campaign_launch.py",
    "scripts/manage_finer_detailing_campaign_cohort.py",
    "scripts/plan_finer_detailing_production_smoke.py",
    "scripts/plan_finer_detailing_qualification.py",
    "scripts/prepare_finer_detailing_visual_review.py",
    "scripts/prepare_finer_detailing_temporal_promotion.py",
    "scripts/run_flux1_native_negative_calibration.py",
    "scripts/run_flux1_native_negative_calibration_dispatched.sh",
    "scripts/run_flux1_native_negative_calibration_v2.py",
    "scripts/run_flux1_native_negative_calibration_v2_dispatched.sh",
    "scripts/evaluate_finer_detailing_native_negative_calibration_v2.py",
    "scripts/run_flux1_common_seed_v2_dispatched.sh",
    "scripts/finer_detailing_environment_dispatch_v3.py",
    "scripts/run_flux1_common_seed_v3_dispatched.sh",
    "scripts/run_finer_detailing_dispatched.sh",
    "scripts/run_finer_detailing_smoke_dispatched.sh",
    "scripts/run_finer_detailing_qualification_dispatched.sh",
    "scripts/run_finer_detailing_campaign_dispatched.sh",
    "scripts/run_finer_detailing_correction.py",
    "scripts/write_finer_detailing_seed_selection.py",
    "scripts/create_conda_env.sh",
    "requirements-ltx23.txt",
    "requirements.txt",
    "environment-ltx23.yml",
    "environment.yml",
    "pyproject.toml",
    "slurm/finer_detailing_manifest_array_h100.sbatch",
    "slurm/finer_detailing_shapley_array_h100.sbatch",
    "slurm/flux1_native_negative_calibration_h100.sbatch",
    "slurm/flux1_native_negative_calibration_v2_h100.sbatch",
    "slurm/flux1_common_seed_v2_h100.sbatch",
    "slurm/flux1_common_seed_v3_h100.sbatch",
    "slurm/finer_detailing_production_smoke_h100.sbatch",
    "slurm/finer_detailing_qualification_h100.sbatch",
    "slurm/finer_detailing_campaign_h100.sbatch",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def flux_common_seed_module_for_stage(stage: Any) -> Any:
    """Return the one scientific module authorized to interpret ``stage``.

    Imports are deliberately lazy because both versioned evaluation modules
    reuse this benchmark builder.  Most importantly, v3 is not an alias for
    v2: each stage reaches only its own validators and release machinery.
    """

    if stage == FLUX_COMMON_SEED_V2_STAGE:
        from hierasafe_flow.evaluation import flux1_common_seed_v2

        return flux1_common_seed_v2
    if stage == FLUX_COMMON_SEED_V3_STAGE:
        from hierasafe_flow.evaluation import flux1_common_seed_v3

        return flux1_common_seed_v3
    raise ValueError(f"Unknown Flux common-seed stage: {stage!r}.")


def flux_common_seed_stage_from_manifest(manifest: Mapping[str, Any]) -> str | None:
    """Classify a manifest without collapsing its versioned stage identity.

    A partial claim is still returned so the owning module can reject its
    malformed topology.  A document that mixes v2 and v3 is rejected before
    either module can reinterpret the other's evidence.
    """

    claims: set[str] = set()
    top_stage = manifest.get("stage")
    if top_stage in FLUX_COMMON_SEED_STAGES:
        claims.add(str(top_stage))
    jobs = manifest.get("jobs")
    if isinstance(jobs, list):
        claims.update(
            str(job.get("stage"))
            for job in jobs
            if isinstance(job, Mapping) and job.get("stage") in FLUX_COMMON_SEED_STAGES
        )
    if len(claims) > 1:
        raise ValueError(
            "A generation manifest cannot mix Flux common-seed protocol versions: "
            f"{sorted(claims)}."
        )
    return next(iter(claims), None)


def reject_reserved_flux_common_seed_outputs(
    job: Mapping[str, Any], root: Path | None = None
) -> None:
    """Protect every versioned common-seed output root from ordinary jobs."""

    resolved_root = (root or project_root()).resolve()
    for stage in (FLUX_COMMON_SEED_V2_STAGE, FLUX_COMMON_SEED_V3_STAGE):
        flux_common_seed_module_for_stage(stage).reject_reserved_common_seed_output_without_stage(
            job, resolved_root
        )


def _flux_common_seed_protocol_input_roles(stage: str) -> frozenset[str]:
    if stage == FLUX_COMMON_SEED_V2_STAGE:
        return FLUX_COMMON_SEED_PROTOCOL_INPUT_ROLES
    module = flux_common_seed_module_for_stage(stage)
    roles = getattr(module, "V3_REQUIRED_PROTOCOL_INPUT_ROLES", None)
    if not isinstance(roles, frozenset) or not roles or not all(
        isinstance(role, str) and role for role in roles
    ):
        raise RuntimeError("Flux common-seed v3 did not publish an exact input-role contract.")
    return roles


def _implementation_file_paths(root: Path) -> tuple[str, ...]:
    """Return the complete project-relative generation implementation surface."""

    root = root.resolve()
    relative_paths = set(IMPLEMENTATION_REQUIRED_FILES)
    for relative_directory in IMPLEMENTATION_PYTHON_DIRECTORIES:
        directory = root / relative_directory
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Required implementation directory does not exist: {directory}"
            )
        # Directory discovery is deliberate: a newly added adapter or steering
        # module must change the frozen set instead of escaping provenance.
        relative_paths.update(
            path.relative_to(root).as_posix() for path in directory.glob("*.py") if path.is_file()
        )
    missing = [relative for relative in sorted(relative_paths) if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Required implementation files do not exist: {missing}")
    return tuple(sorted(relative_paths))


def _implementation_files(root: Path) -> dict[str, str]:
    root = root.resolve()
    return {
        relative: _sha256_file(root / relative) for relative in _implementation_file_paths(root)
    }


def _implementation_files_digest(files: dict[str, str]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_command(root: Path, arguments: list[str]) -> bytes:
    completed = subprocess.run(
        ["git", "-c", "core.quotepath=false", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=120,
    )
    return completed.stdout


def _git_provenance(root: Path) -> dict[str, Any]:
    """Capture Git state scoped only to files that determine generation.

    Live output, debugging, report, manifest, test, and documentation writes
    are intentionally outside these pathspecs. Untracked source is content
    hashed independently because ``git diff`` does not include it.
    """

    root = root.resolve()
    pathspec = list(IMPLEMENTATION_GIT_PATHS)
    try:
        inside = _git_command(root, ["rev-parse", "--is-inside-work-tree"]).strip()
        if inside != b"true":
            raise RuntimeError(f"Not inside a Git work tree: {root}")
        head = _git_command(root, ["rev-parse", "HEAD"]).decode("ascii").strip()
        status_bytes = _git_command(
            root,
            ["status", "--porcelain=v1", "--untracked-files=all", "--", *pathspec],
        )
        tracked_diff = _git_command(
            root,
            ["diff", "--binary", "--no-ext-diff", "HEAD", "--", *pathspec],
        )
        untracked_output = _git_command(
            root,
            ["ls-files", "--others", "--exclude-standard", "--", *pathspec],
        )
    except (FileNotFoundError, subprocess.SubprocessError, RuntimeError) as exc:
        return {
            "repository_available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    untracked_paths = sorted(line for line in untracked_output.decode("utf-8").splitlines() if line)
    untracked_files: dict[str, str] = {}
    for relative in untracked_paths:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Untracked implementation source disappeared: {path}")
        untracked_files[relative] = _sha256_file(path)
    status_lines = status_bytes.decode("utf-8").splitlines()
    return {
        "repository_available": True,
        "git_head": head,
        "dirty": bool(status_lines),
        "dirty_status": status_lines,
        "dirty_status_sha256": hashlib.sha256(status_bytes).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "tracked_diff_size_bytes": len(tracked_diff),
        "untracked_source_files": untracked_files,
        "untracked_source_content_sha256": _implementation_files_digest(untracked_files),
        "pathspecs": pathspec,
    }


def _implementation_provenance(root: Path) -> dict[str, Any]:
    files = _implementation_files(root)
    return {
        "implementation_files": files,
        "implementation_files_sha256": _implementation_files_digest(files),
        "git_provenance": _git_provenance(root),
    }


def _provenance_copy(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "implementation_files": dict(provenance["implementation_files"]),
        "implementation_files_sha256": str(provenance["implementation_files_sha256"]),
        "git_provenance": deepcopy(provenance["git_provenance"]),
    }


def _provenance_from_container(container: dict[str, Any]) -> dict[str, Any]:
    return {
        "implementation_files": container.get("implementation_files"),
        "implementation_files_sha256": container.get("implementation_files_sha256"),
        "git_provenance": container.get("git_provenance"),
    }


def _verify_implementation_provenance(
    container: dict[str, Any],
    root: Path,
    *,
    verify_git: bool = True,
) -> None:
    """Fail closed if the executing implementation differs from the manifest."""

    frozen = _provenance_from_container(container)
    files = frozen["implementation_files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Job is missing its frozen implementation_files mapping.")
    if any(
        not isinstance(path, str) or not isinstance(digest, str) for path, digest in files.items()
    ):
        raise ValueError("Frozen implementation_files must map relative paths to SHA-256 strings.")
    actual_digest = _implementation_files_digest(files)
    expected_digest = str(frozen["implementation_files_sha256"] or "")
    if actual_digest != expected_digest:
        raise ValueError(
            "Frozen implementation_files mapping digest is inconsistent: "
            f"expected={expected_digest}, actual={actual_digest}."
        )

    root = root.resolve()
    current_paths = _implementation_file_paths(root)
    if tuple(sorted(files)) != current_paths:
        new_paths = sorted(set(current_paths) - set(files))
        removed_paths = sorted(set(files) - set(current_paths))
        raise ValueError(
            "Implementation file set changed before execution: "
            f"new_or_unfrozen={new_paths}, missing_or_removed={removed_paths}."
        )
    for relative, expected in sorted(files.items()):
        path = root / relative
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Implementation source changed before execution: {path}; "
                f"expected sha256={expected}, actual sha256={actual}. "
                "Refusing to create output or load a model."
            )

    git_provenance = frozen["git_provenance"]
    if not isinstance(git_provenance, dict):
        raise ValueError("Job is missing frozen Git provenance.")
    if verify_git and _git_provenance(root) != git_provenance:
        raise ValueError(
            "Git implementation state changed before execution even though the declared "
            "source file check completed. Refusing to create output or load a model."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the three-prompt finer-detailing correction study."
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--write-manifest", default=None)
    parser.add_argument("--output-root", default="outputs/finer_detailing_correction")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument(
        "--models",
        default="all",
        help="'all' or a comma-separated subset of canonical model names.",
    )
    parser.add_argument(
        "--prompt-ids", default="all", help="'all' or a comma-separated prompt subset."
    )
    parser.add_argument(
        "--variations",
        default="all",
        help=(
            "'all' or a comma-separated subset of the six numbered variation groups. "
            "Short aliases baseline, negative_prompt, conceptsteer_full, shapley_full, "
            "conceptsteer_single, and shapley_single are accepted. The July seven-method "
            "campaign is intentionally isolated in hierasafe_flow.campaigns."
        ),
    )
    parser.add_argument(
        "--pair-ids",
        default="all",
        help=(
            "'all' or comma-separated concept-pair IDs for the two single-pair groups. "
            "With mixed prompts, each ID is included only for prompts whose tree defines it."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-scoped-output",
        action="store_true",
        help=(
            "Place this replicate under seed_<8 digits> and include the seed in condition_id. "
            "Required when multiple seed manifests are launched into one output root."
        ),
    )
    parser.add_argument(
        "--allow-unvalidated-temporal-pilot",
        action="store_true",
        help=(
            "Permit manifest construction for explicitly pilot-only long-video protocols. "
            "Omit this flag for production; unapproved temporal protocols then fail closed."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument(
        "--cpu-offload", default=None, choices=[None, "model", "sequential", "true", "false"]
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-latents", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-traces", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = project_root()
    _validate_generation_seed(args.seed)
    if args.attempt <= 0:
        raise ValueError("--attempt must be a positive integer.")

    if args.manifest:
        manifest = read_manifest(Path(args.manifest), root)
        if args.index is None:
            results = [
                run_job(
                    _job_with_launch_manifest_binding(job, manifest, job_index=index),
                    root,
                )
                for index, job in enumerate(manifest["jobs"])
            ]
            print(json.dumps({"num_jobs": len(results), "results": results}, indent=2))
            return
        jobs = manifest["jobs"]
        if args.index < 0 or args.index >= len(jobs):
            raise IndexError(f"Job index {args.index} is outside 0..{len(jobs) - 1}.")
        print(
            json.dumps(
                run_job(
                    _job_with_launch_manifest_binding(
                        jobs[args.index], manifest, job_index=args.index
                    ),
                    root,
                ),
                indent=2,
            )
        )
        return

    manifest = build_manifest(args, root)
    if args.write_manifest:
        write_manifest_immutable(manifest, Path(args.write_manifest), root)
        print(f"Wrote immutable {len(manifest['jobs'])}-job manifest to {args.write_manifest}")
        return
    results = [
        run_job(
            _job_with_launch_manifest_binding(job, manifest, job_index=index),
            root,
        )
        for index, job in enumerate(manifest["jobs"])
    ]
    print(json.dumps({"num_jobs": len(results), "results": results}, indent=2))


def build_manifest(
    args: argparse.Namespace,
    root: Path,
    *,
    flux1_route_context: str = FLUX1_ROUTE_CONTEXT_V3_PREVIEW,
    flux1_protocol_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if flux1_route_context not in FLUX1_ROUTE_CONTEXTS:
        raise ValueError(
            f"Unknown Flux.1 route context {flux1_route_context!r}; "
            f"expected one of {sorted(FLUX1_ROUTE_CONTEXTS)}."
        )
    _validate_generation_seed(args.seed)
    implementation_provenance = _implementation_provenance(root)
    prompt_suite_path = root / "configs/experiments/finer_detailing_correction_prompts.yaml"
    negative_prompt_path = (
        root / "configs/experiments/finer_detailing_correction_negative_prompts.yaml"
    )
    base_config_path = root / "configs/default.yaml"
    prompts = _load_prompts(prompt_suite_path, root)
    negative_prompts = _load_negative_prompts(negative_prompt_path)
    models = _selection(args.models, MODEL_NAMES, "model")
    if flux1_protocol_inputs is not None and "flux1_dev" not in models:
        raise ValueError("Flux-v3 protocol inputs were supplied to a manifest with no flux1_dev jobs.")
    if (
        flux1_route_context == FLUX1_ROUTE_CONTEXT_LEGACY_V2
        and models != ("flux1_dev",)
    ):
        raise ValueError(
            "The hidden legacy_v2 route context is restricted to an all-Flux historical cohort."
        )
    prompt_ids = _selection(args.prompt_ids, PROMPT_IDS, "prompt")
    variation_groups = _variation_selection(args.variations)
    if "flux1_dev" in models and "02_negative_prompt" in variation_groups:
        _require_flux1_native_negative_calibration_v2_admission(root.resolve())
    selected_single_pairs = _single_pair_selection(
        getattr(args, "pair_ids", "all"), prompt_ids, variation_groups
    )
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = root / output_root

    provenance_cache: dict[Path, dict[str, str]] = {}
    jobs: list[dict[str, Any]] = []
    for prompt_id in prompt_ids:
        prompt_entry = prompts[prompt_id]
        prompt_pair_ids = PAIR_IDS_BY_PROMPT[prompt_id]
        for model_name in models:
            model_config_path, resolved_model_name = _resolve_model_config(root, model_name)
            if resolved_model_name != model_name:
                raise ValueError(
                    f"Canonical model mismatch: requested {model_name}, resolved {resolved_model_name}."
                )
            model_config = load_yaml(model_config_path)
            model_revision = _validated_model_revision(model_name, model_config)
            generation = _generation_config(args, model_config)
            generation = _apply_task_layout_contract(generation, prompt_id, args)
            _assert_generation_contract(generation, model_name)
            runtime = _runtime_config(args, model_config)
            task = str(generation["task"])
            temporal_protocol_snapshot = _validated_temporal_protocol_snapshot(
                model_name,
                model_config,
                allow_unvalidated_pilot=bool(args.allow_unvalidated_temporal_pilot),
            )
            artifact_manifest_provenance = _artifact_manifest_provenance(
                temporal_protocol_snapshot,
                root=root,
                cache=provenance_cache,
            )
            prompt, selected_prompt_field = _prompt_for_model(
                prompt_entry,
                task=task,
                model_name=model_name,
            )
            concept_tree = Path(str(prompt_entry["concept_trees_by_task"][task]))
            negative_prompt = str(negative_prompts[prompt_id][task])
            concept_tree_snapshot = load_yaml(concept_tree)
            prompt_snapshot = _prompt_semantic_snapshot(
                prompt_entry,
                prompt_id,
                task,
                prompt,
                selected_prompt_field=selected_prompt_field,
            )
            hunyuan_conditioning_plan = (
                _build_hunyuan_conditioning_plan(
                    prompt_id=prompt_id,
                    base_prompt=prompt,
                    negative_prompt=negative_prompt,
                    concept_tree_snapshot=concept_tree_snapshot,
                    prompt_entry=prompt_entry,
                )
                if model_name == "hunyuan_video"
                else None
            )
            input_files = {
                "base_config": _file_provenance(base_config_path, provenance_cache),
                "model_config": _file_provenance(model_config_path, provenance_cache),
                "prompt_suite": _file_provenance(prompt_suite_path, provenance_cache),
                "concept_tree": _file_provenance(concept_tree, provenance_cache),
                "negative_prompt_config": _file_provenance(negative_prompt_path, provenance_cache),
            }
            if artifact_manifest_provenance is not None:
                input_files["artifact_manifest"] = artifact_manifest_provenance
            for variation_group in variation_groups:
                for active_pair_id in _active_pair_instances(
                    variation_group,
                    prompt_pair_ids,
                    selected_single_pairs[prompt_id],
                ):
                    variant_spec = _variation_spec(
                        variation_group,
                        model_name,
                        prompt_pair_ids,
                        active_pair_id=active_pair_id,
                    )
                    path_parts = _variation_path_parts(variation_group, active_pair_id)
                    variation_dir = output_root / prompt_id / model_name
                    for part in path_parts:
                        variation_dir /= part
                    replicate_dir = variation_dir
                    if bool(args.seed_scoped_output):
                        replicate_dir /= f"seed_{int(args.seed):08d}"
                    output_dir = replicate_dir / "attempts" / f"attempt_{args.attempt:03d}"
                    variant_id = _variant_id(variation_group, active_pair_id)
                    condition_id = f"{prompt_id}__{model_name}__{variant_id}"
                    if bool(args.seed_scoped_output):
                        condition_id += f"__seed_{int(args.seed):08d}"
                    expected_media = not (
                        variant_spec["kind"] == "native_negative_prompt"
                        and variant_spec.get("capability") == "not_supported"
                    )
                    job: dict[str, Any] = {
                            "schema_version": 3,
                            "benchmark": BENCHMARK_NAME,
                            "stage": "full_and_single_pair_conceptsteer_and_shapley",
                            "attempt": args.attempt,
                            "variation": variation_group,
                            "variant": variant_id,
                            "condition_id": condition_id,
                            "variant_spec": variant_spec,
                            "prior_source_variant": variant_spec.get("prior_source_variant"),
                            "profile_adaptation": variant_spec.get("profile_adaptation"),
                            "prompt_id": prompt_id,
                            "prompt": prompt,
                            "prompt_metadata": prompt_entry,
                            "prompt_snapshot": prompt_snapshot,
                            "negative_prompt": negative_prompt,
                            **(
                                {
                                    "hunyuan_dual_view_conditioning": deepcopy(
                                        hunyuan_conditioning_plan
                                    )
                                }
                                if hunyuan_conditioning_plan is not None
                                else {}
                            ),
                            "seed": int(args.seed),
                            "seed_scoped_output": bool(args.seed_scoped_output),
                            "model_name": model_name,
                            "model_revision": model_revision,
                            **(
                                {
                                    "checkpoint_set": deepcopy(
                                        temporal_protocol_snapshot["checkpoint_set"]
                                    ),
                                    "checkpoint_set_sha256": temporal_protocol_snapshot[
                                        "checkpoint_set_sha256"
                                    ],
                                    "segmented_temporal_contract": deepcopy(
                                        SEGMENTED_PROTOCOL_CONTRACT
                                    ),
                                    "artifact_manifest": artifact_manifest_provenance["path"],
                                    "artifact_manifest_sha256": artifact_manifest_provenance[
                                        "sha256"
                                    ],
                                }
                                if artifact_manifest_provenance is not None
                                else {}
                            ),
                            **(
                                {
                                    "temporal_protocol_snapshot": temporal_protocol_snapshot,
                                    "temporal_pilot_authorized": bool(
                                        args.allow_unvalidated_temporal_pilot
                                        and temporal_protocol_snapshot["qualification"] == "pilot"
                                    ),
                                }
                                if temporal_protocol_snapshot is not None
                                else {}
                            ),
                            "model_config": str(model_config_path),
                            "base_config": str(base_config_path),
                            "concept_tree": str(concept_tree),
                            "concept_tree_snapshot": concept_tree_snapshot,
                            "prompt_suite": str(prompt_suite_path),
                            "negative_prompt_config": str(negative_prompt_path),
                            "input_files": input_files,
                            **_provenance_copy(implementation_provenance),
                            "variation_dir": str(variation_dir),
                            "output_dir": str(output_dir),
                            "expected_native_negative_support": EXPECTED_NATIVE_NEGATIVE_SUPPORT[
                                model_name
                            ],
                            "native_negative_prompt_options": dict(
                                variant_spec.get("native_negative_prompt_options", {})
                            ),
                            "expected_media": expected_media,
                            "runtime": runtime,
                            "generation": {**generation, "num_outputs_per_prompt": 1},
                            "logging": {"tensorboard": bool(args.tensorboard)},
                            "output": {
                                "decode": True,
                                "save_latents": bool(args.save_latents),
                                "save_traces": bool(args.save_traces),
                                "image_format": "png",
                                "video_format": "mp4",
                            },
                        }
                    if model_name == "flux1_dev" and flux1_route_context != (
                        FLUX1_ROUTE_CONTEXT_LEGACY_V2
                    ):
                        job = project_flux1_job_v3(
                            job,
                            project_root=root,
                            protocol_inputs=flux1_protocol_inputs,
                            mode=flux1_route_context,
                        )
                    jobs.append(job)

    output_dirs = [job["output_dir"] for job in jobs]
    if len(output_dirs) != len(set(output_dirs)):
        raise ValueError("Manifest contains duplicate output directories.")
    condition_ids = [job["condition_id"] for job in jobs]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("Manifest contains duplicate condition IDs.")
    expected_media_jobs = sum(bool(job["expected_media"]) for job in jobs)
    expected_not_supported_jobs = sum(not bool(job["expected_media"]) for job in jobs)
    unvalidated_temporal_pilot_jobs = sum(
        (job.get("temporal_protocol_snapshot") or {}).get("qualification") == "pilot"
        for job in jobs
    )
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "benchmark": BENCHMARK_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "seed_scoped_output": bool(args.seed_scoped_output),
        "attempt": args.attempt,
        "models": list(models),
        "prompt_ids": list(prompt_ids),
        "variation_groups": list(variation_groups),
        "selected_single_pair_ids_by_prompt": {
            prompt_id: list(pair_ids) for prompt_id, pair_ids in selected_single_pairs.items()
        },
        "output_root": str(output_root),
        "num_jobs": len(jobs),
        "expected_media_jobs": expected_media_jobs,
        "expected_not_supported_jobs": expected_not_supported_jobs,
        "video_contract": dict(VIDEO_CONTRACT),
        "person_image_contract": dict(PERSON_IMAGE_CONTRACT),
        "mall_image_contract": dict(MALL_IMAGE_CONTRACT),
        "shapley_config": deepcopy(SHAPLEY_CONFIG),
        "shapley_provenance": deepcopy(SHAPLEY_PROVENANCE),
        "segmented_temporal_contract": deepcopy(SEGMENTED_PROTOCOL_CONTRACT),
        **(
            {
                "flux1_route_context": flux1_route_context,
                "flux1_route_status": (
                    "historical_legacy_v2"
                    if flux1_route_context == FLUX1_ROUTE_CONTEXT_LEGACY_V2
                    else (
                        "implementation_preview_not_launchable_pending_native_equivalence"
                        if flux1_route_context == FLUX1_ROUTE_CONTEXT_V3_PREVIEW
                        else "authenticated_for_execution_after_native_equivalence"
                    )
                ),
            }
            if "flux1_dev" in models
            else {}
        ),
        "checkpoint_sets_by_model": {
            model_name: deepcopy(EXPECTED_CHECKPOINT_SETS[model_name])
            for model_name in models
            if model_name in SEGMENTED_TEMPORAL_MODELS
        },
        "checkpoint_set_sha256_by_model": {
            model_name: next(
                job["checkpoint_set_sha256"] for job in jobs if job["model_name"] == model_name
            )
            for model_name in models
            if model_name in SEGMENTED_TEMPORAL_MODELS
        },
        "allows_unvalidated_temporal_pilot": bool(
            args.allow_unvalidated_temporal_pilot and unvalidated_temporal_pilot_jobs
        ),
        "unvalidated_temporal_pilot_jobs": unvalidated_temporal_pilot_jobs,
        **_provenance_copy(implementation_provenance),
        "jobs": jobs,
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def _job_with_launch_manifest_binding(
    job: dict[str, Any],
    manifest: dict[str, Any],
    *,
    job_index: int | None = None,
) -> dict[str, Any]:
    bound = deepcopy(job)
    digest = str(manifest.get("manifest_sha256", ""))
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Launch manifest lacks its canonical SHA-256 binding.")
    bound["launch_manifest_sha256"] = digest
    if job_index is not None:
        if isinstance(job_index, bool) or not 0 <= job_index < len(manifest.get("jobs", [])):
            raise IndexError(
                f"Launch manifest job index {job_index!r} is outside the manifest job range."
            )
        if manifest["jobs"][job_index] != job:
            raise ValueError(
                "Launch manifest job index does not identify the supplied generation job."
            )
        bound["launch_manifest_job_index"] = job_index
    return bound


def _validate_generation_seed(seed: int) -> None:
    """Validate a reproducible 32-bit generation seed.

    The Shapley permutation estimator has its own separately frozen seed.  This
    value controls only model sampling and may vary across target-blind
    baseline-qualification manifests; every job within one manifest remains
    paired on the same seed.
    """

    if isinstance(seed, bool) or not 0 <= int(seed) <= 2**32 - 1:
        raise ValueError("--seed must be an integer in the inclusive range 0..2**32-1.")


def _validated_model_revision(model_name: str, model_config: dict[str, Any]) -> str:
    """Require every benchmark checkpoint to resolve from one exact Git commit."""

    expected = EXPECTED_MODEL_REVISIONS[model_name]
    model = model_config.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"Model config for {model_name} has no model mapping.")
    actual = model.get("revision")
    if (
        not isinstance(actual, str)
        or len(actual) != 40
        or any(character not in "0123456789abcdef" for character in actual)
    ):
        raise ValueError(
            f"Model config for {model_name} must pin model.revision to a lowercase "
            f"40-hex commit; found {actual!r}."
        )
    if actual != expected:
        raise ValueError(
            f"Model revision drift for {model_name}: expected={expected}, configured={actual}."
        )
    return actual


def _validated_checkpoint_set(
    model_name: str,
    model_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], str] | None:
    if model_name not in SEGMENTED_TEMPORAL_MODELS:
        return None
    model = model_config.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"Model config for {model_name} has no model mapping.")
    configured = model.get("checkpoint_set")
    expected = EXPECTED_CHECKPOINT_SETS[model_name]
    if configured != expected:
        raise ValueError(
            f"Model config checkpoint_set for {model_name} differs from the sealed ordered set."
        )
    encoded = json.dumps(configured, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    configured_digest = model.get("checkpoint_set_sha256")
    if configured_digest != digest:
        raise ValueError(f"Model config checkpoint-set digest mismatch for {model_name}.")
    return deepcopy(configured), digest


def _validated_temporal_protocol_snapshot(
    model_name: str,
    model_config: dict[str, Any],
    *,
    allow_unvalidated_pilot: bool,
) -> dict[str, Any] | None:
    """Classify and freeze each video model's temporal qualification state.

    Adapter-level gates validate the model-specific evidence schema in detail.
    This benchmark-level gate prevents an explicitly unvalidated long-video
    protocol from entering a normal production manifest merely because its
    output arithmetic is structurally valid.
    """

    model = model_config.get("model")
    generation = model_config.get("generation")
    if not isinstance(model, dict) or not isinstance(generation, dict):
        raise ValueError(f"Model config for {model_name} is missing model/generation mappings.")
    protocol_items = [
        (key, value) for key, value in model.items() if key.endswith("_temporal_protocol")
    ]
    task = str(generation.get("task"))
    if task == "text_to_image":
        if protocol_items:
            raise ValueError(
                f"Image model {model_name} unexpectedly defines temporal protocols: "
                f"{[key for key, _ in protocol_items]}."
            )
        return None
    if task != "text_to_video":
        raise ValueError(f"Model {model_name} has unsupported task {task!r}.")
    if len(protocol_items) != 1:
        raise ValueError(
            f"Video model {model_name} must define exactly one *_temporal_protocol mapping; "
            f"found {[key for key, _ in protocol_items]}."
        )
    key, raw_protocol = protocol_items[0]
    if not isinstance(raw_protocol, dict):
        raise ValueError(f"model.{key} for {model_name} must be a mapping.")
    protocol = deepcopy(raw_protocol)
    checkpoint_set = _validated_checkpoint_set(model_name, model_config)
    if model_name in SEGMENTED_TEMPORAL_MODELS:
        if protocol.get("schema_version") != 2:
            raise ValueError(
                f"{model_name} must use segmented temporal protocol schema 2; stale media "
                "cannot launch under schema 1."
            )
        for field, expected in SEGMENTED_PROTOCOL_CONTRACT.items():
            protocol_field = field.removeprefix("temporal_protocol_")
            if field == "temporal_protocol_schema_version":
                continue
            if protocol.get(protocol_field) != expected:
                raise ValueError(
                    f"model.{key}.{protocol_field} must be {expected}; "
                    f"found {protocol.get(protocol_field)!r}."
                )
        serialized = json.dumps(protocol, sort_keys=True).lower()
        forbidden_fragments = (
            "single_trajectory_riflex",
            "effective_position_fps",
            "video_rope_uses_effective_16_fps",
            "repaired_video_rope_241_frame_trajectory_terminal_crop",
        )
        present = [fragment for fragment in forbidden_fragments if fragment in serialized]
        if present:
            raise ValueError(f"model.{key} retains obsolete temporal fields/routes: {present}.")
        artifact_path = protocol.get("artifact_manifest")
        artifact_digest = protocol.get("artifact_manifest_sha256")
        if not isinstance(artifact_path, str) or not artifact_path:
            raise ValueError(f"model.{key} must bind a machine-readable artifact manifest.")
        if (
            not isinstance(artifact_digest, str)
            or len(artifact_digest) != 64
            or any(character not in "0123456789abcdef" for character in artifact_digest)
        ):
            raise ValueError(f"model.{key} artifact manifest digest is malformed.")
        manifest_path = Path(artifact_path)
        if not manifest_path.is_absolute():
            manifest_path = project_root() / manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"model.{key} artifact manifest does not exist: {manifest_path}"
            )
        actual_artifact_digest = _sha256_file(manifest_path)
        if actual_artifact_digest != artifact_digest:
            raise ValueError(
                f"model.{key} artifact manifest digest mismatch: "
                f"expected={artifact_digest}, actual={actual_artifact_digest}."
            )
        artifact_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if artifact_payload.get("schema_version") != 1:
            raise ValueError(f"model.{key} artifact manifest must use schema version 1.")
        protocol["artifact_manifest"] = manifest_path.relative_to(project_root()).as_posix()
    execution_phase = protocol.get("execution_phase")
    if execution_phase not in {"pilot", "production"}:
        raise ValueError(
            f"model.{key}.execution_phase must be 'pilot' or 'production'; "
            f"found {execution_phase!r}."
        )
    if execution_phase == "pilot":
        if protocol.get("production_gate") is not None:
            raise ValueError(
                f"Pilot-only model.{key} for {model_name} cannot carry production_gate."
            )
        qualification = "pilot"
    else:
        if not isinstance(protocol.get("production_gate"), dict):
            raise ValueError(
                f"Production model.{key} for {model_name} requires production_gate evidence."
            )
        validate_temporal_production_gate(
            protocol,
            model_name=model_name,
            model_revision=_validated_model_revision(model_name, model_config),
            criteria_names=TEMPORAL_CRITERIA_BY_MODEL[model_name],
        )
        from hierasafe_flow.evaluation.temporal_promotion import (
            require_complete_temporal_production_promotion,
        )

        require_complete_temporal_production_promotion(root=project_root())
        qualification = "production"
    if qualification == "pilot" and not allow_unvalidated_pilot:
        raise ValueError(
            f"Video model {model_name} uses an unvalidated temporal pilot. Pass "
            "--allow-unvalidated-temporal-pilot only for isolated qualification manifests; "
            "production manifests remain blocked."
        )
    snapshot = {
        "config_key": key,
        "qualification": qualification,
        "protocol": protocol,
    }
    if checkpoint_set is not None:
        snapshot.update(
            {
                "checkpoint_set": checkpoint_set[0],
                "checkpoint_set_sha256": checkpoint_set[1],
                **SEGMENTED_PROTOCOL_CONTRACT,
            }
        )
    return snapshot


def _artifact_manifest_provenance(
    temporal_snapshot: dict[str, Any] | None,
    *,
    root: Path,
    cache: dict[Path, dict[str, str]],
) -> dict[str, str] | None:
    if temporal_snapshot is None or temporal_snapshot.get("temporal_protocol_schema_version") != 2:
        return None
    protocol = temporal_snapshot["protocol"]
    path = Path(str(protocol["artifact_manifest"]))
    if not path.is_absolute():
        path = root / path
    provenance = _file_provenance(path, cache)
    if provenance["sha256"] != protocol["artifact_manifest_sha256"]:
        raise ValueError("Temporal artifact manifest changed during manifest construction.")
    return provenance


def _variation_spec(
    variation_group: str,
    model_name: str,
    prompt_pair_ids: tuple[str, ...],
    *,
    active_pair_id: str | None = None,
) -> dict[str, Any]:
    variation_group = _normalize_variation_group(variation_group)
    if variation_group == "01_baseline":
        return {"kind": "baseline"}
    if variation_group == "02_negative_prompt":
        if model_name in NATIVE_NEGATIVE_UNSUPPORTED_REASONS:
            return {
                "kind": "native_negative_prompt",
                "capability": "not_supported",
                "reason": NATIVE_NEGATIVE_UNSUPPORTED_REASONS[model_name],
            }
        return {
            "kind": "native_negative_prompt",
            "capability": "supported",
            "native_negative_prompt_options": dict(
                EFFECTIVE_NATIVE_NEGATIVE_OPTIONS.get(model_name, {})
            ),
        }

    steering_groups = {
        "03_concept_steering": ("conceptsteer", False),
        "04_shapley_concept_steering": ("shapley_concept_steering", False),
        "05_concept_steering_single_pair": ("conceptsteer", True),
        "06_shapley_concept_steering_single_pair": ("shapley_concept_steering", True),
    }
    if variation_group not in steering_groups:
        raise ValueError(f"Unknown variation group {variation_group!r}.")
    kind, is_single_pair = steering_groups[variation_group]
    if is_single_pair:
        if active_pair_id not in prompt_pair_ids:
            raise ValueError(
                f"Single-pair variation {variation_group!r} requires one pair from {prompt_pair_ids}; "
                f"got {active_pair_id!r}."
            )
        active_pair_ids = (str(active_pair_id),)
    else:
        if active_pair_id is not None:
            raise ValueError(
                f"Full variation {variation_group!r} cannot select {active_pair_id!r}."
            )
        active_pair_ids = tuple(prompt_pair_ids)
    if len(prompt_pair_ids) != 5 or len(active_pair_ids) not in {1, 5}:
        raise ValueError(
            f"Invalid pair cardinality for {variation_group}: all={prompt_pair_ids}, active={active_pair_ids}."
        )
    profile = dict(CURATED_STEERING_PROFILES[model_name])
    pair_overrides = {
        pair_id: dict(override)
        for pair_id, override in dict(profile.get("pair_overrides", {})).items()
        if pair_id in active_pair_ids
    }
    spec: dict[str, Any] = {
        "kind": kind,
        "active_pair_ids": active_pair_ids,
        "strength": float(profile["strength"]),
        "margin": 0.05,
        "tau": 0.10,
        "schedule": "full_window",
        "schedule_window": (0.0, 1.0),
        "local_mask": False,
        "normalize_directions": False,
        "prompt_composition": str(profile["prompt_composition"]),
        "step_stride": int(profile.get("step_stride", 1)),
        "pair_overrides": pair_overrides,
        "prior_source_variant": str(profile["source_variant"]),
        "profile_adaptation": (
            "architecture_hyperparameters_reused_but_active_pair_forced_to_exact_single_pair_"
            "by_2026-07-15_protocol"
            if is_single_pair
            else "architecture_hyperparameters_reused_but_active_pairs_forced_to_prompt_full_five_"
            "by_2026-07-15_protocol"
        ),
        "pair_selection": "single" if is_single_pair else "full",
    }
    if kind == "shapley_concept_steering":
        spec["shapley"] = deepcopy(SHAPLEY_CONFIG)
        spec["shapley_provenance"] = deepcopy(SHAPLEY_PROVENANCE)
    return spec


def _variation_selection(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return VARIATION_GROUPS
    requested = tuple(item.strip() for item in value.split(",") if item.strip())
    normalized = tuple(_normalize_variation_group(item) for item in requested)
    unknown = sorted(set(normalized) - set(VARIATION_GROUPS))
    if not normalized or unknown:
        raise ValueError(
            f"Invalid variation selection {value!r}; unknown={unknown}, "
            f"allowed={list(VARIATION_GROUPS)}, aliases={sorted(VARIATION_ALIASES)}"
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError(
            f"Variation selection contains duplicates after alias expansion: {value!r}"
        )
    return normalized


def _single_pair_selection(
    value: str,
    prompt_ids: tuple[str, ...],
    variation_groups: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    single_groups = {
        "05_concept_steering_single_pair",
        "06_shapley_concept_steering_single_pair",
    }
    if value.strip().lower() == "all":
        return {prompt_id: PAIR_IDS_BY_PROMPT[prompt_id] for prompt_id in prompt_ids}
    if not (set(variation_groups) & single_groups):
        raise ValueError(
            "--pair-ids is meaningful only when a single-pair variation group is selected."
        )
    requested = tuple(item.strip() for item in value.split(",") if item.strip())
    legal_union = {pair_id for prompt_id in prompt_ids for pair_id in PAIR_IDS_BY_PROMPT[prompt_id]}
    unknown = sorted(set(requested) - legal_union)
    if not requested or unknown:
        raise ValueError(
            f"Invalid --pair-ids selection {value!r}; unknown={unknown}, legal union={sorted(legal_union)}."
        )
    if len(requested) != len(set(requested)):
        raise ValueError(f"--pair-ids contains duplicates: {value!r}")
    selected = {
        prompt_id: tuple(
            pair_id for pair_id in PAIR_IDS_BY_PROMPT[prompt_id] if pair_id in requested
        )
        for prompt_id in prompt_ids
    }
    return selected


def _active_pair_instances(
    variation_group: str,
    prompt_pair_ids: tuple[str, ...],
    selected_single_pair_ids: tuple[str, ...],
) -> tuple[str | None, ...]:
    if variation_group in {
        "05_concept_steering_single_pair",
        "06_shapley_concept_steering_single_pair",
    }:
        illegal = sorted(set(selected_single_pair_ids) - set(prompt_pair_ids))
        if illegal:
            raise ValueError(f"Selected single-pair IDs are invalid for this prompt: {illegal}")
        return tuple(selected_single_pair_ids)
    return (None,)


def _variation_path_parts(variation_group: str, active_pair_id: str | None) -> tuple[str, ...]:
    if variation_group in {"03_concept_steering", "04_shapley_concept_steering"}:
        if active_pair_id is not None:
            raise ValueError(f"Full variation {variation_group} cannot have pair {active_pair_id}.")
        return (variation_group, "full")
    if variation_group in {
        "05_concept_steering_single_pair",
        "06_shapley_concept_steering_single_pair",
    }:
        if not active_pair_id:
            raise ValueError(f"Single-pair variation {variation_group} is missing its pair ID.")
        return (variation_group, active_pair_id)
    if active_pair_id is not None:
        raise ValueError(
            f"Non-steering variation {variation_group} cannot have pair {active_pair_id}."
        )
    return (variation_group,)


def _variant_id(variation_group: str, active_pair_id: str | None) -> str:
    if variation_group == "03_concept_steering":
        return "conceptsteer_full"
    if variation_group == "04_shapley_concept_steering":
        return "shapley_concept_steering_full"
    if variation_group == "05_concept_steering_single_pair":
        return f"conceptsteer_single__{active_pair_id}"
    if variation_group == "06_shapley_concept_steering_single_pair":
        return f"shapley_concept_steering_single__{active_pair_id}"
    return variation_group


def _apply_task_layout_contract(
    generation: dict[str, Any],
    prompt_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Apply modality-appropriate framing without changing pixel workload.

    The person prompts explicitly require head-to-feet framing; an equal-area
    portrait canvas gives that instruction physical room. The object-only mall
    prompt explicitly requests a wide architectural view, so it receives the
    reciprocal landscape canvas. Explicit CLI height/width overrides remain
    available for engineering diagnostics and are never silently replaced.
    """

    resolved = dict(generation)
    if str(resolved.get("task")) != "text_to_image":
        return resolved
    contract = (
        PERSON_IMAGE_CONTRACT
        if prompt_id in {"01_sad_young_girl", "02_angry_old_man"}
        else MALL_IMAGE_CONTRACT
    )
    if getattr(args, "height", None) is None:
        resolved["height"] = contract["height"]
    if getattr(args, "width", None) is None:
        resolved["width"] = contract["width"]
    return resolved


def _assert_generation_contract(generation: dict[str, Any], model_name: str) -> None:
    task = str(generation.get("task"))
    if task == "text_to_image":
        unexpected = sorted(set(generation) & {"duration_seconds", "num_frames", "fps"})
        if unexpected:
            raise ValueError(
                f"Image model {model_name} retained video-only generation keys: {unexpected}"
            )
        return
    if task != "text_to_video":
        raise ValueError(f"Model {model_name} has unsupported generation task {task!r}.")
    actual = {
        "duration_seconds": float(generation.get("duration_seconds", -1.0)),
        "num_frames": int(generation.get("num_frames", -1)),
        "fps": int(generation.get("fps", -1)),
    }
    if actual != VIDEO_CONTRACT:
        raise ValueError(
            f"Every text-to-video job must use the exact 15 s / 16 fps / 240 frame contract; "
            f"model {model_name} resolved to {actual}."
        )


def _build_hunyuan_conditioning_plan(
    *,
    prompt_id: str,
    base_prompt: str,
    negative_prompt: str,
    concept_tree_snapshot: dict[str, Any],
    prompt_entry: dict[str, Any],
) -> dict[str, Any]:
    """Freeze exact Llama strings and independent compact CLIP views.

    HunyuanVideo has a long Llama branch and a 77-token CLIP pooled branch.
    The experiment strings remain byte-exact Llama inputs; the prompt-suite
    supplies independently authored concept-first CLIP views so preservation
    details are never silently lost to the upstream ``prompt_2`` routing bug.
    Optional token fingerprints are added after the tokenizer-only provisional
    preflight and become mandatory for production manifests.
    """

    clip_views = prompt_entry.get("hunyuan_clip_views")
    if not isinstance(clip_views, dict):
        raise ValueError(f"Prompt {prompt_id} is missing hunyuan_clip_views.")
    if set(clip_views) != {"base", "native_negative", "neutral", "pairs"}:
        raise ValueError(
            f"Prompt {prompt_id} hunyuan_clip_views must contain exactly "
            "base/native_negative/neutral/pairs."
        )
    pair_clip_views = clip_views["pairs"]
    if not isinstance(pair_clip_views, dict):
        raise ValueError(f"Prompt {prompt_id} Hunyuan pair CLIP views must be a mapping.")
    raw_pairs = concept_tree_snapshot.get("pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != 5:
        raise ValueError(f"Prompt {prompt_id} Hunyuan plan requires exactly five concept pairs.")
    pair_ids = tuple(str(pair.get("id")) for pair in raw_pairs if isinstance(pair, dict))
    if pair_ids != PAIR_IDS_BY_PROMPT[prompt_id] or set(pair_clip_views) != set(pair_ids):
        raise ValueError(
            f"Prompt {prompt_id} Hunyuan CLIP pair IDs differ from the frozen concept tree."
        )

    token_fingerprints = prompt_entry.get("hunyuan_token_fingerprints", {})
    if not isinstance(token_fingerprints, dict):
        raise ValueError(f"Prompt {prompt_id} hunyuan_token_fingerprints must be a mapping.")
    entries: list[dict[str, Any]] = []

    def add_entry(
        entry_id: str,
        role: str,
        raw_prompt: Any,
        clip_prompt: Any,
        pair_id: str | None = None,
    ) -> None:
        if not isinstance(raw_prompt, str) or not raw_prompt.strip():
            raise ValueError(f"Hunyuan entry {prompt_id}/{entry_id} has an empty raw prompt.")
        if not isinstance(clip_prompt, str) or not clip_prompt.strip():
            raise ValueError(f"Hunyuan entry {prompt_id}/{entry_id} has an empty CLIP view.")
        raw_sha256 = hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()
        llama_sha256 = hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()
        clip_sha256 = hashlib.sha256(clip_prompt.encode("utf-8")).hexdigest()
        entry: dict[str, Any] = {
            "entry_id": entry_id,
            "role": role,
            "pair_id": pair_id,
            "raw_prompt": raw_prompt,
            "raw_prompt_sha256": raw_sha256,
            "llama_prompt": raw_prompt,
            "llama_prompt_sha256": llama_sha256,
            "clip_prompt": clip_prompt,
            "clip_prompt_sha256": clip_sha256,
        }
        frozen = token_fingerprints.get(entry_id)
        if frozen is not None:
            required = {
                "llama_token_count",
                "llama_token_ids_sha256",
                "clip_token_count",
                "clip_token_ids_sha256",
            }
            if not isinstance(frozen, dict) or set(frozen) != required:
                raise ValueError(
                    f"Hunyuan token fingerprint {prompt_id}/{entry_id} must contain exactly "
                    f"{sorted(required)}."
                )
            entry.update(frozen)
        entries.append(entry)

    add_entry("base", "baseline", base_prompt, clip_views["base"])
    add_entry(
        "native_negative",
        "native_negative",
        negative_prompt,
        clip_views["native_negative"],
    )
    add_entry(
        "neutral",
        "neutral",
        concept_tree_snapshot.get("neutral_concept"),
        clip_views["neutral"],
    )
    for pair in raw_pairs:
        if not isinstance(pair, dict):
            raise ValueError(f"Prompt {prompt_id} contains a malformed concept pair.")
        pair_id = str(pair["id"])
        pair_views = pair_clip_views[pair_id]
        if not isinstance(pair_views, dict) or set(pair_views) != {"unsafe", "safe"}:
            raise ValueError(
                f"Prompt {prompt_id} Hunyuan CLIP views for {pair_id} require unsafe/safe."
            )
        add_entry(
            f"{pair_id}__unsafe",
            "unsafe",
            pair.get("unsafe_concept"),
            pair_views["unsafe"],
            pair_id,
        )
        add_entry(
            f"{pair_id}__safe",
            "safe",
            pair.get("safe_sibling_concept"),
            pair_views["safe"],
            pair_id,
        )

    expected_entry_ids = {
        "base",
        "native_negative",
        "neutral",
        *(f"{pair_id}__{role}" for pair_id in pair_ids for role in ("unsafe", "safe")),
    }
    if set(token_fingerprints) != expected_entry_ids:
        raise ValueError(
            f"Prompt {prompt_id} must freeze token fingerprints for every Hunyuan entry; "
            f"missing={sorted(expected_entry_ids - set(token_fingerprints))}, "
            f"unknown={sorted(set(token_fingerprints) - expected_entry_ids)}."
        )
    plan: dict[str, Any] = {"schema_version": 1, "entries": entries}
    plan["plan_sha256"] = hunyuan_conditioning_plan_sha256(plan)
    HunyuanConditioningPlan.from_mapping(plan)
    return plan


def _prompt_semantic_snapshot(
    prompt_entry: dict[str, Any],
    prompt_id: str,
    task: str,
    prompt: str,
    *,
    selected_prompt_field: str | None = None,
) -> dict[str, Any]:
    modality = "image" if task == "text_to_image" else "video"
    snapshot: dict[str, Any] = {
        "prompt_id": prompt_id,
        "task": task,
        "selected_prompt_field": (
            selected_prompt_field or prompt_entry["prompt_fields_by_task"][task]
        ),
        "selected_concept_tree_field": prompt_entry["concept_tree_fields_by_task"][task],
        "prompt": prompt,
        "pair_ids": list(PAIR_IDS_BY_PROMPT[prompt_id]),
    }
    for key in ("source_attributes", "target_attributes", "preservation_constraints"):
        value = prompt_entry.get(f"{modality}_{key}", prompt_entry.get(key))
        if value is not None:
            snapshot[key] = value
    target_reference = prompt_entry.get(
        f"{modality}_target_reference_prompt",
        prompt_entry.get("target_reference_prompt", prompt_entry.get("target_reference")),
    )
    if target_reference is not None:
        snapshot["target_reference_prompt"] = target_reference
    return snapshot


def _prompt_for_model(
    prompt_entry: dict[str, Any],
    *,
    task: str,
    model_name: str,
) -> tuple[str, str]:
    """Select a frozen model-specific prompt only where tokenizer limits require it."""

    if task == "text_to_video" and model_name in MODEL_SPECIFIC_VIDEO_PROMPT_FIELDS:
        field = MODEL_SPECIFIC_VIDEO_PROMPT_FIELDS[model_name]
        value = prompt_entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Prompt {prompt_entry.get('prompt_id')} is missing required {field!r} "
                f"for {model_name}."
            )
        return value, field
    field = str(prompt_entry["prompt_fields_by_task"][task])
    return str(prompt_entry["prompts_by_task"][task]), field


def _file_provenance(
    path: Path,
    cache: dict[Path, dict[str, str]] | None = None,
) -> dict[str, str]:
    resolved = path.resolve()
    if cache is not None and resolved in cache:
        return dict(cache[resolved])
    if not resolved.is_file():
        raise FileNotFoundError(f"Frozen benchmark input does not exist: {resolved}")
    record = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
    }
    if cache is not None:
        cache[resolved] = dict(record)
    return record


def _verify_job_input_files(
    job: dict[str, Any],
    cache: dict[Path, str] | None = None,
) -> None:
    required = {
        "base_config",
        "model_config",
        "prompt_suite",
        "concept_tree",
        "negative_prompt_config",
    }
    if job.get("segmented_temporal_contract") is not None:
        required.add("artifact_manifest")
    flux1_v3_inputs: Mapping[str, Any] | None = None
    if is_flux1_job_v3(job):
        route = job["flux1_dual_view_route_v3"]
        route_mode = route.get("mode") if isinstance(route, Mapping) else None
        flux1_v3_roles = (
            FLUX1_V3_SOURCE_INPUT_ROLES
            if route_mode == FLUX1_V3_MODE_PREVIEW
            else execution_protocol_input_roles_v3()
        )
        raw_flux1_v3_inputs = route.get("protocol_inputs") if isinstance(route, Mapping) else None
        if (
            not isinstance(raw_flux1_v3_inputs, Mapping)
            or set(raw_flux1_v3_inputs) != set(flux1_v3_roles)
        ):
            raise ValueError("Flux-v3 job does not bind its exact route-specific input roles.")
        flux1_v3_inputs = raw_flux1_v3_inputs
        required.update(flux1_v3_roles)
    if job.get("benchmark") == "flux1_native_negative_true_cfg_scale_v1":
        required.update(
            {
                "calibration_config",
                "sealed_preregistration",
                "sealed_preregistration_sidecar",
            }
        )
    calibration_v2_identity = (
        job.get("benchmark"),
        job.get("stage"),
    )
    calibration_v2_expected_identity = (
        FLUX_CALIBRATION_V2_BENCHMARK,
        FLUX_CALIBRATION_V2_STAGE,
    )
    calibration_v2_claimed = (
        job.get("benchmark") == FLUX_CALIBRATION_V2_BENCHMARK
        or job.get("stage") == FLUX_CALIBRATION_V2_STAGE
    )
    calibration_v2_inputs: Mapping[str, Any] | None = None
    if calibration_v2_claimed:
        if calibration_v2_identity != calibration_v2_expected_identity:
            raise ValueError(
                "Flux calibration-v2 benchmark/stage identity is only valid as an "
                f"exact pair: expected={calibration_v2_expected_identity}, "
                f"found={calibration_v2_identity}."
            )
        raw_bindings = job.get("calibration_v2_bindings")
        if not isinstance(raw_bindings, Mapping) or set(raw_bindings) != {
            "selected_common_seed",
            "protocol_inputs",
            "selection_evidence",
        }:
            raise ValueError("Flux calibration-v2 job has malformed evidence bindings.")
        protocol_inputs = raw_bindings.get("protocol_inputs")
        selection_inputs = raw_bindings.get("selection_evidence")
        if (
            not isinstance(protocol_inputs, Mapping)
            or set(protocol_inputs) != set(FLUX_CALIBRATION_V2_PROTOCOL_INPUT_ROLES)
            or not isinstance(selection_inputs, Mapping)
            or set(selection_inputs) != set(FLUX_CALIBRATION_V2_SELECTION_INPUT_ROLES)
        ):
            raise ValueError(
                "Flux calibration-v2 job does not bind the exact protocol and "
                "selected-seed evidence roles."
            )
        calibration_v2_inputs = {**protocol_inputs, **selection_inputs}
        if set(calibration_v2_inputs) != set(FLUX_CALIBRATION_V2_INPUT_ROLES):
            raise ValueError("Flux calibration-v2 evidence role union is not exact.")
        required.update(FLUX_CALIBRATION_V2_INPUT_ROLES)
    common_seed_inputs: Mapping[str, Any] | None = None
    common_seed_stage = (
        str(job["stage"]) if job.get("stage") in FLUX_COMMON_SEED_STAGES else None
    )
    common_seed_roles: frozenset[str] = frozenset()
    if common_seed_stage is not None:
        common_seed_roles = _flux_common_seed_protocol_input_roles(common_seed_stage)
        raw_common_seed_inputs = job.get("common_seed_protocol_inputs")
        if not isinstance(raw_common_seed_inputs, Mapping) or set(raw_common_seed_inputs) != set(
            common_seed_roles
        ):
            found = (
                sorted(raw_common_seed_inputs)
                if isinstance(raw_common_seed_inputs, Mapping)
                else raw_common_seed_inputs
            )
            raise ValueError(
                "Flux common-seed job has invalid protocol input roles: "
                f"stage={common_seed_stage!r}, expected={sorted(common_seed_roles)}, "
                f"found={found}."
            )
        common_seed_inputs = raw_common_seed_inputs
        required.update(common_seed_roles)
    records = job.get("input_files")
    if not isinstance(records, dict) or set(records) != required:
        raise ValueError(
            f"Job {job.get('variant')} has invalid frozen input provenance roles: "
            f"expected={sorted(required)}, found={sorted(records) if isinstance(records, dict) else records}."
        )
    job_path_fields = {
        "base_config": "base_config",
        "model_config": "model_config",
        "prompt_suite": "prompt_suite",
        "concept_tree": "concept_tree",
        "negative_prompt_config": "negative_prompt_config",
    }
    if "artifact_manifest" in required:
        job_path_fields["artifact_manifest"] = "artifact_manifest"
    if job.get("benchmark") == "flux1_native_negative_true_cfg_scale_v1":
        job_path_fields.update(
            {
                "calibration_config": "calibration_config",
                "sealed_preregistration": "sealed_preregistration",
                "sealed_preregistration_sidecar": "sealed_preregistration_sidecar",
            }
        )
    for role, field_name in job_path_fields.items():
        record = records[role]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(
                f"Job {job.get('variant')} has malformed {role} provenance: {record!r}"
            )
        path = Path(str(record["path"])).resolve()
        if path != Path(str(job[field_name])).resolve():
            raise ValueError(
                f"Job {job.get('variant')} {role} path differs from frozen provenance: "
                f"job={job[field_name]}, frozen={path}."
            )
        if not path.is_file():
            raise FileNotFoundError(f"Frozen {role} input disappeared before execution: {path}")
        if cache is not None and path in cache:
            actual = cache[path]
        else:
            actual = _sha256_file(path)
            if cache is not None:
                cache[path] = actual
        expected = str(record["sha256"])
        if actual != expected:
            raise ValueError(
                f"Frozen {role} input changed before execution: {path}; expected sha256={expected}, "
                f"actual sha256={actual}. Refusing to load a model from a non-reproducible job."
            )
    if common_seed_inputs is not None:
        for role in common_seed_roles:
            record = records[role]
            protocol_record = common_seed_inputs[role]
            if record != protocol_record:
                raise ValueError(
                    f"Flux common-seed job protocol binding drifted for role {role!r}."
                )
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise ValueError(
                    f"Flux common-seed job has malformed {role} provenance: {record!r}"
                )
            path = Path(str(record["path"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Flux common-seed input disappeared before execution: {path}"
                )
            if cache is not None and path in cache:
                actual = cache[path]
            else:
                actual = _sha256_file(path)
                if cache is not None:
                    cache[path] = actual
            if actual != str(record["sha256"]):
                raise ValueError(
                    f"Flux common-seed input changed before execution: {path}; "
                    f"expected sha256={record['sha256']}, actual sha256={actual}."
                )
    if calibration_v2_inputs is not None:
        for role in FLUX_CALIBRATION_V2_INPUT_ROLES:
            record = records[role]
            bound_record = calibration_v2_inputs[role]
            if not isinstance(bound_record, Mapping):
                raise ValueError(f"Flux calibration-v2 binding for role {role!r} is malformed.")
            expected_record = {
                "path": str(bound_record.get("path", "")),
                "sha256": str(bound_record.get("sha256", "")),
            }
            if record != expected_record:
                raise ValueError(
                    f"Flux calibration-v2 job input binding drifted for role {role!r}."
                )
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                raise ValueError(
                    f"Flux calibration-v2 job has malformed {role} provenance: {record!r}"
                )
            path = Path(str(record["path"])).resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"Flux calibration-v2 input disappeared before execution: {path}"
                )
            if cache is not None and path in cache:
                actual = cache[path]
            else:
                actual = _sha256_file(path)
                if cache is not None:
                    cache[path] = actual
            if actual != str(record["sha256"]):
                raise ValueError(
                    f"Flux calibration-v2 input changed before execution: {path}; "
                    f"expected sha256={record['sha256']}, actual sha256={actual}."
                )
    if flux1_v3_inputs is not None:
        for role, bound_record in flux1_v3_inputs.items():
            record = records[role]
            if record != bound_record:
                raise ValueError(f"Flux-v3 job input binding drifted for role {role!r}.")
            if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
                raise ValueError(f"Flux-v3 job has malformed {role} provenance.")
            path = Path(str(record["path"]))
            if not path.is_absolute() or not path.is_file() or path.is_symlink():
                raise FileNotFoundError(
                    f"Flux-v3 input is absent, relative, or symlinked before execution: {path}"
                )
            resolved = path.resolve()
            if cache is not None and resolved in cache:
                actual = cache[resolved]
            else:
                actual = _sha256_file(resolved)
                if cache is not None:
                    cache[resolved] = actual
            if actual != str(record["sha256"]):
                raise ValueError(
                    f"Flux-v3 input changed before execution: {resolved}; "
                    f"expected sha256={record['sha256']}, actual sha256={actual}."
                )


def _verify_job_semantic_snapshots(job: dict[str, Any]) -> None:
    prompt_snapshot = job.get("prompt_snapshot")
    tree_snapshot = job.get("concept_tree_snapshot")
    if not isinstance(prompt_snapshot, dict) or not isinstance(tree_snapshot, dict):
        raise ValueError(
            f"Job {job.get('variant')} is missing frozen prompt/tree semantic snapshots."
        )
    prompt_id = str(job["prompt_id"])
    task = str(job["generation"]["task"])
    if prompt_snapshot.get("prompt_id") != prompt_id or prompt_snapshot.get("task") != task:
        raise ValueError(f"Job {job.get('variant')} prompt snapshot identity is inconsistent.")
    if str(prompt_snapshot.get("prompt")) != str(job["prompt"]):
        raise ValueError(
            f"Job {job.get('variant')} prompt differs from its frozen semantic snapshot."
        )
    if tuple(prompt_snapshot.get("pair_ids") or ()) != PAIR_IDS_BY_PROMPT[prompt_id]:
        raise ValueError(f"Job {job.get('variant')} pair IDs differ from its prompt snapshot.")

    model_name = str(job["model_name"])
    configured_revision = _validated_model_revision(
        model_name,
        model_config := load_yaml(Path(str(job["model_config"]))),
    )
    if job.get("model_revision") != configured_revision:
        raise ValueError(
            f"Job {job.get('variant')} model revision differs from its pinned model config: "
            f"job={job.get('model_revision')!r}, configured={configured_revision}."
        )
    expected_temporal_snapshot = _validated_temporal_protocol_snapshot(
        model_name,
        model_config,
        allow_unvalidated_pilot=True,
    )
    if expected_temporal_snapshot is None:
        if "temporal_protocol_snapshot" in job or "temporal_pilot_authorized" in job:
            raise ValueError(
                f"Image job {job.get('variant')} unexpectedly freezes a temporal protocol."
            )
    elif job.get("temporal_protocol_snapshot") != expected_temporal_snapshot:
        raise ValueError(
            f"Job {job.get('variant')} temporal protocol differs from its frozen model config."
        )
    else:
        is_pilot = expected_temporal_snapshot["qualification"] == "pilot"
        if bool(job.get("temporal_pilot_authorized")) != is_pilot:
            raise ValueError(
                f"Job {job.get('variant')} temporal pilot authorization is inconsistent."
            )

    if model_name in SEGMENTED_TEMPORAL_MODELS:
        assert expected_temporal_snapshot is not None
        expected_segmented = {
            "checkpoint_set": expected_temporal_snapshot["checkpoint_set"],
            "checkpoint_set_sha256": expected_temporal_snapshot["checkpoint_set_sha256"],
            "segmented_temporal_contract": SEGMENTED_PROTOCOL_CONTRACT,
            "artifact_manifest_sha256": expected_temporal_snapshot["protocol"][
                "artifact_manifest_sha256"
            ],
        }
        observed_segmented = {key: job.get(key) for key in expected_segmented}
        if observed_segmented != expected_segmented:
            raise ValueError(
                f"Job {job.get('variant')} segmented checkpoint/protocol bindings drifted."
            )
        expected_artifact = Path(str(expected_temporal_snapshot["protocol"]["artifact_manifest"]))
        if not expected_artifact.is_absolute():
            expected_artifact = project_root() / expected_artifact
        if Path(str(job.get("artifact_manifest"))).resolve() != expected_artifact.resolve():
            raise ValueError(f"Job {job.get('variant')} artifact manifest path drifted.")
    else:
        forbidden_segmented = {
            "checkpoint_set",
            "checkpoint_set_sha256",
            "segmented_temporal_contract",
            "artifact_manifest",
            "artifact_manifest_sha256",
        }
        present_segmented = sorted(forbidden_segmented & set(job))
        if present_segmented:
            raise ValueError(
                f"Non-segmented job {job.get('variant')} has segmented bindings: "
                f"{present_segmented}."
            )

    root = Path(str(job["base_config"])).resolve().parents[1]
    prompt_suite = load_yaml(Path(str(job["prompt_suite"])))
    rows = {
        str(row["prompt_id"]): row
        for row in prompt_suite.get("prompts") or []
        if isinstance(row, dict) and "prompt_id" in row
    }
    if prompt_id not in rows:
        raise ValueError(f"Frozen prompt suite no longer contains {prompt_id}.")
    row = rows[prompt_id]
    prompt_field = str(prompt_snapshot["selected_prompt_field"])
    tree_field = str(prompt_snapshot["selected_concept_tree_field"])
    if job.get("stage") == FLUX_COMMON_SEED_V3_STAGE or is_flux1_job_v3(job):
        # V3 intentionally separates the compact <=77-token CLIP view from
        # the complete T5 view.  Its versioned module authenticates both views,
        # their tokenizer fingerprints, and the independent prompt-contract
        # file.  Requiring equality with the legacy image_prompt here would
        # silently erase that repair and route the old truncated prompt.
        if prompt_field != "v3_source_prompt_clip_prompt":
            raise ValueError(
                f"Job {job.get('variant')} v3 prompt snapshot does not identify the "
                "compact CLIP source view."
            )
    elif str(row.get(prompt_field)) != str(job["prompt"]):
        raise ValueError(
            f"Job {job.get('variant')} prompt does not match {prompt_field} in its frozen prompt suite."
        )
    modality = "image" if task == "text_to_image" else "video"
    target_reference_field = f"{modality}_target_reference_prompt"
    configured_target_reference = row.get(target_reference_field)
    if configured_target_reference is not None and str(
        prompt_snapshot.get("target_reference_prompt")
    ) != str(configured_target_reference):
        raise ValueError(
            f"Job {job.get('variant')} target reference does not match {target_reference_field} "
            "in its frozen prompt suite."
        )
    selected_tree = Path(str(row.get(tree_field)))
    if not selected_tree.is_absolute():
        selected_tree = root / selected_tree
    if selected_tree.resolve() != Path(str(job["concept_tree"])).resolve():
        raise ValueError(
            f"Job {job.get('variant')} concept tree does not match {tree_field} in its prompt suite."
        )
    if load_yaml(Path(str(job["concept_tree"]))) != tree_snapshot:
        raise ValueError(
            f"Job {job.get('variant')} concept tree differs from its frozen semantic snapshot."
        )
    negative_prompts = _load_negative_prompts(Path(str(job["negative_prompt_config"])))
    if is_flux1_job_v3(job):
        plan = job["generation"].get("flux_dual_view_conditioning")
        negative_mode = job["flux1_dual_view_route_v3"].get("negative_mode")
        if negative_mode == "paired_registered_negative":
            negative_plan = plan.get("negative") if isinstance(plan, Mapping) else None
            if (
                not isinstance(negative_plan, Mapping)
                or str(job["negative_prompt"])
                != str(negative_plan.get("clip_negative_prompt"))
            ):
                raise ValueError(
                    f"Job {job.get('variant')} does not bind the registered compact Flux-v3 "
                    "negative CLIP view."
                )
        elif negative_mode == "explicit_none_control":
            if job.get("negative_prompt") is not None:
                raise ValueError(
                    f"Job {job.get('variant')} explicit-none control has a non-null negative."
                )
        elif negative_prompts[prompt_id][task] != str(job["negative_prompt"]):
            raise ValueError(
                f"Job {job.get('variant')} inactive negative metadata differs from its frozen config."
            )
    elif negative_prompts[prompt_id][task] != str(job["negative_prompt"]):
        raise ValueError(
            f"Job {job.get('variant')} negative prompt differs from its frozen config."
        )
    if model_name == "hunyuan_video":
        expected_plan = _build_hunyuan_conditioning_plan(
            prompt_id=prompt_id,
            base_prompt=str(job["prompt"]),
            negative_prompt=str(job["negative_prompt"]),
            concept_tree_snapshot=tree_snapshot,
            prompt_entry=row,
        )
        if job.get("hunyuan_dual_view_conditioning") != expected_plan:
            raise ValueError(
                f"Job {job.get('variant')} Hunyuan dual-view plan differs from frozen inputs."
            )
    elif "hunyuan_dual_view_conditioning" in job:
        raise ValueError(
            f"Non-Hunyuan job {job.get('variant')} unexpectedly contains a Hunyuan plan."
        )
    if is_flux1_job_v3(job):
        validate_flux1_job_v3(job, project_root=root, mode=FLUX1_V3_MODE_AUDIT)


def _shared_runner_config_from_authenticated_inputs(
    job: Mapping[str, Any],
    root: Path,
    *,
    base: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild the shared runner config from authenticated snapshot payloads.

    This mirrors ``benchmarks.benign_park._runner_config`` while avoiding a
    read of mutable live YAML during historical completed-output audit.
    """

    benchmark_name = str(job.get("benchmark", BENCHMARK_NAME))
    raw_seed = job.get("seed", 0)
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise ValueError("Authenticated runner job seed must be one exact integer.")
    seed = raw_seed
    variant_spec = job["variant_spec"]
    generation = deepcopy(dict(job["generation"]))
    generation.update({"prompt": job["prompt"], "prompt_file": None})

    model = deepcopy(dict(model_config.get("model", {})))
    model["guidance_scale"] = generation.get("guidance_scale")
    dtype = (job.get("runtime") or {}).get("dtype")
    if dtype:
        model["torch_dtype"] = dtype
    runtime = dict(job.get("runtime") or {})
    if "cpu_offload" in runtime:
        cpu_offload = runtime["cpu_offload"]
        if cpu_offload in (False, "false"):
            model.pop("cpu_offload", None)
        elif cpu_offload in (None, "model"):
            pass
        elif cpu_offload == "true":
            model["cpu_offload"] = True
        else:
            model["cpu_offload"] = cpu_offload

    kind = _normalize_variant_kind(str(variant_spec["kind"]))
    native_negative_prompt = (
        {"prompt": job["negative_prompt"]} if kind == "native_negative_prompt" else {}
    )
    if kind == "baseline":
        steering = {"mode": "none", "enabled": False}
    elif kind == "conceptsteer":
        steering = _steering_config(dict(variant_spec))
    else:
        steering = {"mode": "none", "enabled": False}
    override = {
        "project": {"name": benchmark_name, "seed": seed},
        "runtime": {
            "device": runtime.get("device", "cuda"),
            "dtype": model.get("torch_dtype", "bfloat16"),
        },
        "model": model,
        "generation": generation,
        "concepts": {"hierarchy_path": job["concept_tree"]},
        "steering": steering,
        "native_negative_prompt": native_negative_prompt,
        "logging": {
            "output_dir": job["output_dir"],
            "tensorboard": (job.get("logging") or {}).get("tensorboard", True),
            "level": "INFO",
        },
        "output": job["output"],
        "benchmark": {
            "name": benchmark_name,
            "stage": job["stage"],
            "variant": job["variant"],
            "prompt_id": job["prompt_id"],
            "seed": seed,
            "negative_prompt": job["negative_prompt"],
            "active_pair_ids": variant_spec.get("active_pair_ids", []),
            "strength": variant_spec.get("strength"),
            "margin": variant_spec.get("margin"),
            "tau": variant_spec.get("tau"),
            "schedule": variant_spec.get("schedule"),
            "schedule_window": variant_spec.get("schedule_window"),
            "local_mask": variant_spec.get("local_mask"),
            "normalize_directions": variant_spec.get("normalize_directions"),
            "prompt_composition": variant_spec.get("prompt_composition"),
            "step_stride": variant_spec.get("step_stride", 1),
        },
    }
    config = deep_merge(dict(base), override)
    config["generation"] = _sanitize_generation_for_task(
        dict(config.get("generation", {}))
    )
    return config


def _augment_runner_config(
    job: Mapping[str, Any], root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    config = deepcopy(dict(config))
    variant_spec = dict(job["variant_spec"])
    kind = _normalize_variant_kind(str(variant_spec["kind"]))
    benchmark = dict(config.get("benchmark", {}))
    model = dict(config.get("model", {}))
    model["project_root"] = str(root.resolve())
    if is_flux1_job_v3(job):
        model["flux_dual_view_conditioning"] = deepcopy(
            job["generation"]["flux_dual_view_conditioning"]
        )
    config["model"] = model
    benchmark.update(
        {
            "condition_id": str(job["condition_id"]),
            "attempt": int(job["attempt"]),
            "manifest_sha256": job.get("launch_manifest_sha256"),
            "launch_manifest_job_index": job.get("launch_manifest_job_index"),
            "model_revision": str(job["model_revision"]),
        }
    )
    if is_flux1_job_v3(job):
        route = job["flux1_dual_view_route_v3"]
        manifest_index = job.get("launch_manifest_job_index")
        if route.get("mode") == FLUX1_V3_MODE_EXECUTION and (
            isinstance(manifest_index, bool) or not isinstance(manifest_index, int)
        ):
            raise ValueError("Flux-v3 execution config requires an exact manifest job index.")
        benchmark["flux1_dual_view_route_v3"] = {
            "contract_id": route["contract_id"],
            "negative_mode": route["negative_mode"],
            "native_pipeline_execution": route["native_pipeline_execution"],
            "plan_sha256": validate_flux1_runtime_v3(job)["plan_sha256"],
        }
    if str(job["model_name"]) in SEGMENTED_TEMPORAL_MODELS:
        if not isinstance(job.get("launch_manifest_sha256"), str):
            raise ValueError(
                "Schema-2 segmented jobs require an immutable launch-manifest binding."
            )
        benchmark.update(
            {
                "checkpoint_set": deepcopy(job["checkpoint_set"]),
                "checkpoint_set_sha256": str(job["checkpoint_set_sha256"]),
                "segmented_temporal_contract": deepcopy(job["segmented_temporal_contract"]),
                "artifact_manifest": str(job["artifact_manifest"]),
                "artifact_manifest_sha256": str(job["artifact_manifest_sha256"]),
            }
        )
    if str(job["model_name"]) == "hunyuan_video":
        plan = job.get("hunyuan_dual_view_conditioning")
        if not isinstance(plan, dict):
            raise ValueError("Hunyuan jobs require a frozen dual-view conditioning plan.")
        model = dict(config.get("model", {}))
        model[HUNYUAN_DUAL_VIEW_CONFIG_KEY] = deepcopy(plan)
        config["model"] = model
        benchmark["hunyuan_dual_view_plan_sha256"] = str(plan["plan_sha256"])
    if kind == "native_negative_prompt":
        native_options = dict(variant_spec.get("native_negative_prompt_options", {}))
        native_config = dict(config.get("native_negative_prompt", {}))
        native_config.update(native_options)
        config["native_negative_prompt"] = native_config
        benchmark["native_negative_prompt_options"] = native_options
    elif kind == "shapley_concept_steering":
        steering = _steering_config(variant_spec)
        steering["mode"] = "shapley_concept_steering"
        steering["shapley"] = deepcopy(variant_spec["shapley"])
        config["steering"] = steering
        benchmark["shapley"] = deepcopy(variant_spec["shapley"])
        benchmark["shapley_provenance"] = deepcopy(variant_spec["shapley_provenance"])
        benchmark["active_pair_ids"] = list(variant_spec["active_pair_ids"])
    config["benchmark"] = benchmark
    if is_flux1_job_v3(job):
        validate_flux1_runtime_v3(job, runner_config=config)
    return config


def _runner_config(job: dict[str, Any], root: Path) -> dict[str, Any]:
    if job.get("model_name") == "flux1_dev":
        if job.get("stage") == FLUX_COMMON_SEED_V2_STAGE:
            pass
        elif not is_flux1_job_v3(job):
            raise ValueError(
                "Generic Flux.1 execution is forbidden without the dual-view v3 route marker."
            )
        else:
            route_mode = str(job["flux1_dual_view_route_v3"].get("mode"))
            if route_mode not in {FLUX1_V3_MODE_PREVIEW, FLUX1_V3_MODE_EXECUTION}:
                raise ValueError("Flux-v3 runner configuration has an invalid route mode.")
            # Configuration construction is read-only and is also exercised by
            # the preregistered native-equivalence diagnostic.  All actual
            # generation/publication boundaries still require execution_v3.
            validate_flux1_job_v3(job, project_root=root, mode=route_mode)
    return _augment_runner_config(job, root, _shared_runner_config(job, root))


def _lexical_path_below_root(path: str | Path, root: Path, *, label: str) -> Path:
    def inspected(value: str | Path, value_label: str) -> Path:
        try:
            raw = os.fspath(value)
        except TypeError as exc:
            raise ValueError(f"{value_label} is not path-like.") from exc
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise ValueError(f"{value_label} must be one non-empty filesystem path.")
        normalized_separators = raw.replace(os.altsep, os.sep) if os.altsep else raw
        components = normalized_separators.split(os.sep)
        allowed_empty = {0} if normalized_separators.startswith(os.sep) else set()
        if any(
            component in {".", ".."}
            or (component == "" and index not in allowed_empty)
            for index, component in enumerate(components)
        ):
            raise ValueError(f"{value_label} contains an explicit lexical alias.")
        return Path(os.path.expanduser(raw))

    raw_root = inspected(root, f"{label} trusted root")
    raw_path = inspected(path, label)
    if any(part in {".", ".."} for part in (*raw_root.parts, *raw_path.parts)):
        raise ValueError(f"{label} contains an explicit lexical alias.")
    trusted_root = raw_root if raw_root.is_absolute() else (Path.cwd() / raw_root).absolute()
    candidate = raw_path if raw_path.is_absolute() else trusted_root / raw_path
    candidate = candidate.absolute()
    try:
        candidate.relative_to(trusted_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its trusted root: {candidate}.") from exc
    return candidate


@contextmanager
def _pinned_nofollow_directory(
    path: Path,
    *,
    trusted_root: Path,
    label: str,
):
    """Pin every absolute directory component and reauthenticate its name chain."""

    candidate = _lexical_path_below_root(path, trusted_root, label=label)
    parts = candidate.parts
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptors: list[int] = [os.open(candidate.anchor, flags)]
    links: list[tuple[int, str, int, tuple[int, int]]] = []
    try:
        for name in parts[1:]:
            parent = descriptors[-1]
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"{label} contains a symlink or non-directory: {candidate}.")
            child = os.open(name, flags, dir_fd=parent)
            opened = os.fstat(child)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            ):
                os.close(child)
                raise ValueError(f"{label} changed during no-follow traversal: {candidate}.")
            descriptors.append(child)
            links.append((parent, name, child, (opened.st_dev, opened.st_ino)))
        yield candidate, descriptors[-1], links
        for parent, name, child, identity in links:
            opened = os.fstat(child)
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or stat.S_ISLNK(named.st_mode)
                or (opened.st_dev, opened.st_ino) != identity
                or (named.st_dev, named.st_ino) != identity
            ):
                raise ValueError(f"{label} directory chain changed during evidence audit.")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is absent: {candidate}.") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _relative_evidence_parts(path: str | Path, *, label: str) -> tuple[str, ...]:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError(f"{label} relative path is malformed.")
    normalized = raw.replace(os.altsep, os.sep) if os.altsep else raw
    if normalized.startswith(os.sep):
        raise ValueError(f"{label} must be relative to the pinned output directory.")
    parts = tuple(normalized.split(os.sep))
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} contains an explicit lexical alias.")
    return parts


def _read_relative_nofollow_inode_stable(
    output_descriptor: int,
    relative_path: str | Path,
    *,
    label: str,
) -> dict[str, Any]:
    """Read one regular, single-link leaf below an already pinned directory."""

    parts = _relative_evidence_parts(relative_path, label=label)
    parents: list[int] = []
    links: list[tuple[int, str, int, tuple[int, int]]] = []
    parent = output_descriptor
    leaf_descriptor: int | None = None
    try:
        for name in parts[:-1]:
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ValueError(f"{label} contains a symlink or non-directory parent.")
            child = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent,
            )
            opened = os.fstat(child)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            ):
                os.close(child)
                raise ValueError(f"{label} parent changed during no-follow traversal.")
            parents.append(child)
            links.append((parent, name, child, (opened.st_dev, opened.st_ino)))
            parent = child

        leaf_name = parts[-1]
        before = os.stat(leaf_name, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is symlinked or is not a regular file.")
        if before.st_nlink != 1:
            raise ValueError(
                f"{label} has {before.st_nlink} hard links; completed evidence requires nlink=1."
            )
        leaf_descriptor = os.open(
            leaf_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent,
        )
        opened = os.fstat(leaf_descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError(f"{label} changed while its no-follow descriptor was opened.")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(leaf_descriptor, 16 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(leaf_descriptor)
        named = os.stat(leaf_name, dir_fd=parent, follow_symlinks=False)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
            opened.st_nlink,
        )
        if (
            len(raw) != opened.st_size
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_nlink,
            )
            != identity
            or stat.S_ISLNK(named.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or (
                named.st_dev,
                named.st_ino,
                named.st_size,
                named.st_mtime_ns,
                named.st_ctime_ns,
                named.st_nlink,
            )
            != identity
        ):
            raise ValueError(f"{label} changed while being read.")
        for link_parent, name, child, directory_identity in links:
            child_stat = os.fstat(child)
            named_child = os.stat(name, dir_fd=link_parent, follow_symlinks=False)
            if (
                not stat.S_ISDIR(child_stat.st_mode)
                or stat.S_ISLNK(named_child.st_mode)
                or (child_stat.st_dev, child_stat.st_ino) != directory_identity
                or (named_child.st_dev, named_child.st_ino) != directory_identity
            ):
                raise ValueError(f"{label} parent changed while the leaf was read.")
        return {
            "raw": raw,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "identity": identity,
            "parent_identities": [
                {"name": name, "device": identity_[0], "inode": identity_[1]}
                for _, name, _, identity_ in links
            ],
        }
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is absent below the pinned output directory.") from exc
    finally:
        if leaf_descriptor is not None:
            os.close(leaf_descriptor)
        for descriptor in reversed(parents):
            os.close(descriptor)


def _same_authenticated_leaf(
    before: Mapping[str, Any], after: Mapping[str, Any], *, label: str
) -> None:
    if (
        before.get("raw") != after.get("raw")
        or before.get("identity") != after.get("identity")
        or before.get("parent_identities") != after.get("parent_identities")
    ):
        raise ValueError(f"{label} changed across the completed-evidence audit.")


def _read_nofollow_inode_stable(
    path: Path,
    *,
    trusted_root: Path,
    label: str,
) -> bytes:
    """Open every component without following aliases and read one stable inode."""

    candidate = _lexical_path_below_root(path, trusted_root, label=label)
    with _pinned_nofollow_directory(
        candidate.parent,
        trusted_root=trusted_root,
        label=f"{label} parent",
    ) as (_parent_path, parent_descriptor, _links):
        first = _read_relative_nofollow_inode_stable(
            parent_descriptor, candidate.name, label=label
        )
        second = _read_relative_nofollow_inode_stable(
            parent_descriptor, candidate.name, label=label
        )
        _same_authenticated_leaf(first, second, label=label)
        return bytes(first["raw"])


def _json_from_authenticated_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object.")
    return payload


def _canonical_json_bytes(value: Any, *, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not canonical finite JSON data: {exc}") from exc


def _require_canonical_json_equal(actual: Any, expected: Any, *, label: str) -> None:
    if _canonical_json_bytes(actual, label=label) != _canonical_json_bytes(
        expected, label=f"expected {label}"
    ):
        raise ValueError(f"{label} differs from its exact authenticated value.")


def _list_media_below_pinned_output(output_descriptor: int) -> tuple[str, ...]:
    """Enumerate media without following aliases and with per-directory stability checks."""

    media_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".gif"}
    found: list[str] = []

    def walk(descriptor: int, prefix: tuple[str, ...]) -> None:
        names_before = sorted(os.listdir(descriptor))
        identities: dict[str, tuple[int, int, int]] = {}
        for name in names_before:
            before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            identities[name] = (before.st_dev, before.st_ino, before.st_mode)
            relative = (*prefix, name)
            if stat.S_ISLNK(before.st_mode):
                raise ValueError(
                    "Completed FLUX-v3 output contains a symlink: " + "/".join(relative)
                )
            if stat.S_ISDIR(before.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                try:
                    opened = os.fstat(child)
                    if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                        raise ValueError("Output directory changed during media enumeration.")
                    walk(child, relative)
                    after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if (
                        stat.S_ISLNK(after.st_mode)
                        or not stat.S_ISDIR(after.st_mode)
                        or (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino)
                    ):
                        raise ValueError("Output directory changed during media enumeration.")
                finally:
                    os.close(child)
            elif stat.S_ISREG(before.st_mode) and Path(name).suffix.lower() in media_suffixes:
                found.append("/".join(relative))
        names_after = sorted(os.listdir(descriptor))
        if names_after != names_before:
            raise ValueError("Output directory entries changed during media enumeration.")
        for name, identity in identities.items():
            after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if (after.st_dev, after.st_ino, after.st_mode) != identity:
                raise ValueError("Output entry changed during media enumeration.")

    walk(output_descriptor, ())
    return tuple(sorted(found))


def _png_validation_from_authenticated_bytes(
    raw: bytes,
    *,
    canonical_path: Path,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    if not raw:
        raise RuntimeError(f"Expected non-empty PNG was not produced: {canonical_path}")
    with Image.open(BytesIO(raw)) as image:
        image.verify()
    with Image.open(BytesIO(raw)) as image:
        image.load()
        width, height = image.size
        mode = image.mode
        image_format = image.format
    expected_width = job["generation"].get("width")
    expected_height = job["generation"].get("height")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (expected_width, expected_height)
    ):
        raise ValueError("FLUX-v3 image dimensions must be exact integer job fields.")
    if image_format != "PNG" or (width, height) != (expected_width, expected_height):
        raise RuntimeError(
            f"Authenticated PNG contract failed at {canonical_path}: "
            f"format={image_format!r}, size={width}x{height}, "
            f"expected={expected_width}x{expected_height}."
        )
    return {
        "path": str(canonical_path),
        "media_type": "image/png",
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "width": width,
        "height": height,
        "mode": mode,
        "decode_verified": True,
    }


def _snapshot_yaml_payload(
    manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    records = job.get("input_files")
    record = records.get(role) if isinstance(records, Mapping) else None
    if not isinstance(record, Mapping):
        raise ValueError(f"FLUX-v3 audit job lacks snapshot input role {role!r}.")
    digest = record.get("sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"FLUX-v3 audit input role {role!r} has an invalid digest.")
    snapshot_root, objects = _snapshot_object_paths_by_sha256(manifest)
    object_path = objects.get(digest)
    if object_path is None:
        raise ValueError(f"FLUX-v3 audit snapshot lacks input role {role!r}.")
    raw = _read_nofollow_inode_stable(
        object_path,
        trusted_root=snapshot_root,
        label=f"FLUX-v3 snapshot {role}",
    )
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"FLUX-v3 snapshot {role!r} digest changed during audit.")
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot decode FLUX-v3 snapshot {role!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"FLUX-v3 snapshot {role!r} must contain one YAML mapping.")
    return payload


def _audit_expected_runner_config(
    manifest: Mapping[str, Any], job: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    base = _snapshot_yaml_payload(manifest, job, role="base_config")
    if any(key in base for key in ("base_config", "model_config", "concept_config")):
        raise ValueError(
            "Authenticated FLUX-v3 base config uses an uncaptured recursive config reference."
        )
    base["_meta"] = {
        "config_path": str(Path(str(job["base_config"])).resolve()),
        "project_root": str(root.resolve()),
    }
    model_config = _snapshot_yaml_payload(manifest, job, role="model_config")
    shared = _shared_runner_config_from_authenticated_inputs(
        job,
        root,
        base=base,
        model_config=model_config,
    )
    return _augment_runner_config(job, root, shared)


def _validate_flux1_runtime_payloads_v3(
    job: Mapping[str, Any],
    *,
    run_timing: Mapping[str, Any],
    report: Mapping[str, Any],
    runner_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    interpretability = report.get("interpretability")
    trace = interpretability.get("timesteps") if isinstance(interpretability, Mapping) else None
    native_negative = job.get("variant_spec", {}).get("kind") == "native_negative_prompt"
    return validate_flux1_runtime_v3(
        job,
        runner_config=runner_config,
        run_timing=run_timing,
        sample_report=report,
        conditioning_cache=(
            report.get("conditioning_cache")
            if isinstance(report.get("conditioning_cache"), Mapping)
            else None
        ),
        native_trace=trace if native_negative else None,
    )


_FLUX1_EVIDENCE_RELATIVE_PATHS = {
    "resolved_config": "resolved_config.yaml",
    "run_timing": "run_timing.json",
    "sample_report": "sample_0000/report.json",
    "steering_trace": "sample_0000/steering_trace.json",
    "media": "sample_0000/image_000.png",
}
_FLUX1_RUNTIME_RECEIPT_CONTRACT = "finer_detailing_flux1_completed_runtime_receipt_v3"


def _read_flux1_evidence_bundle(
    output_descriptor: int, *, include_result: bool, include_trace: bool
) -> dict[str, dict[str, Any]]:
    paths = dict(_FLUX1_EVIDENCE_RELATIVE_PATHS)
    if not include_trace:
        paths.pop("steering_trace")
    if include_result:
        paths["result"] = "benchmark_job_result.json"
    return {
        key: _read_relative_nofollow_inode_stable(
            output_descriptor,
            relative,
            label=f"completed FLUX-v3 {key}",
        )
        for key, relative in paths.items()
    }


def _json_value_from_authenticated_bytes(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode {label}: {exc}") from exc


def _decode_resolved_config(raw: bytes) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot decode FLUX-v3 resolved config: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Completed FLUX-v3 resolved config must contain one mapping.")
    return payload


def _expected_flux1_output_paths(
    job: Mapping[str, Any], config: Mapping[str, Any], output_dir: Path
) -> tuple[dict[str, str], dict[str, str]]:
    sample = output_dir / "sample_0000"
    native = job["variant_spec"]["kind"] == "native_negative_prompt"
    initial: dict[str, str] = {}
    if not native and config["output"].get("save_latents") is True:
        initial["latents"] = str(sample / "final_latents.pt")
    if native or config["output"].get("save_traces") is True:
        initial["trace"] = str(sample / "steering_trace.json")
    initial["image_0"] = str(sample / "image_000.png")
    completed = {
        **initial,
        "report": str(sample / "report.json"),
        "timing": str(sample / "timing.json"),
    }
    return initial, completed


def _validate_flux1_evidence_payloads(
    job: Mapping[str, Any],
    *,
    output_dir: Path,
    resolved_config: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    run_timing: Mapping[str, Any],
    report: Mapping[str, Any],
    steering_trace: Any | None,
    media_raw: bytes,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Validate exact config/timing/report/path/media semantics from pinned bytes."""

    _require_canonical_json_equal(
        resolved_config, expected_config, label="completed FLUX-v3 resolved config"
    )
    native = job["variant_spec"]["kind"] == "native_negative_prompt"
    timing_keys = (
        {
            "schema_version",
            "status",
            "reason",
            "started_at",
            "ended_at",
            "total_seconds",
            "pipeline_load_seconds",
            "model",
            "generation",
            "benchmark",
            "records",
        }
        if native
        else {
            "schema_version",
            "status",
            "started_at",
            "ended_at",
            "total_seconds",
            "adapter_load_seconds",
            "conditioning_preflight_seconds",
            "conditioning_preflight",
            "model",
            "generation",
            "benchmark",
            "records",
        }
    )
    report_keys = (
        {
            "schema_version",
            "prompt",
            "sample_id",
            "task",
            "benchmark",
            "model",
            "condition",
            "generation",
            "conditioning_provenance",
            "steering",
            "output_paths",
            "interpretability",
        }
        if native
        else {
            "schema_version",
            "prompt",
            "sample_id",
            "task",
            "benchmark",
            "model",
            "condition",
            "generation",
            "conditioning_provenance",
            "steering",
            "concept_hierarchy",
            "output_paths",
            "interpretability",
            "conditioning_cache",
        }
    )
    if set(run_timing) != timing_keys:
        raise ValueError("Completed FLUX-v3 run_timing schema is not exact.")
    if set(report) != report_keys:
        raise ValueError("Completed FLUX-v3 sample report schema is not exact.")
    if (
        type(run_timing.get("schema_version")) is not int
        or run_timing.get("schema_version") != 1
        or run_timing.get("status") != "completed"
        or (native and run_timing.get("reason") is not None)
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("prompt") != job["prompt"]
        or report.get("sample_id") != "sample_0000"
        or report.get("task") != "text_to_image"
    ):
        raise ValueError("Completed FLUX-v3 timing/report scalar identity drifted.")
    for field in (
        "total_seconds",
        "pipeline_load_seconds" if native else "adapter_load_seconds",
        *(("conditioning_preflight_seconds",) if not native else ()),
    ):
        value = run_timing.get(field)
        if type(value) is not float or not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Completed FLUX-v3 run_timing {field!r} must be one finite float."
            )
    if any(
        not isinstance(run_timing.get(field), str) or not run_timing[field]
        for field in ("started_at", "ended_at")
    ):
        raise ValueError("Completed FLUX-v3 run_timing timestamps are malformed.")
    _require_canonical_json_equal(
        run_timing.get("benchmark"),
        expected_config["benchmark"],
        label="completed FLUX-v3 run_timing benchmark",
    )
    _require_canonical_json_equal(
        report.get("benchmark"),
        expected_config["benchmark"],
        label="completed FLUX-v3 report benchmark",
    )
    _require_canonical_json_equal(
        run_timing.get("generation"),
        expected_config["generation"],
        label="completed FLUX-v3 run_timing generation",
    )
    _require_canonical_json_equal(
        report.get("generation"),
        expected_config["generation"],
        label="completed FLUX-v3 report generation",
    )
    _require_canonical_json_equal(
        report.get("steering"),
        expected_config["steering"],
        label="completed FLUX-v3 report steering",
    )
    expected_runtime_model = {
        "adapter": "flux_dual_view",
        "model_id": "black-forest-labs/FLUX.1-dev",
        "revision": EXPECTED_MODEL_REVISIONS["flux1_dev"],
        "pipeline_class": "FluxPipeline",
    }
    _require_canonical_json_equal(
        run_timing.get("model"),
        expected_runtime_model,
        label="completed FLUX-v3 run_timing model",
    )
    _require_canonical_json_equal(
        report.get("model"),
        expected_runtime_model,
        label="completed FLUX-v3 report model",
    )
    initial_paths, completed_paths = _expected_flux1_output_paths(
        job, expected_config, output_dir
    )
    _require_canonical_json_equal(
        report.get("output_paths"),
        initial_paths,
        label="completed FLUX-v3 report output paths",
    )
    records = run_timing.get("records")
    expected_records = [
        {
            "prompt": job["prompt"],
            "sample_id": "sample_0000",
            "output_paths": completed_paths,
        }
    ]
    _require_canonical_json_equal(
        records, expected_records, label="completed FLUX-v3 run records"
    )
    condition = report.get("condition")
    expected_condition_keys = (
        {
            "steering_mode",
            "decode_outputs",
            "is_native_negative_prompt",
            "negative_prompt",
            "negative_prompt_2",
            "flux1_native_negative_calibration",
        }
        if native
        else {
            "steering_mode",
            "decode_outputs",
            "is_native_negative_prompt",
            "negative_prompt",
        }
    )
    if (
        not isinstance(condition, Mapping)
        or set(condition) != expected_condition_keys
        or condition.get("decode_outputs") is not True
        or condition.get("is_native_negative_prompt") is not native
    ):
        raise ValueError("Completed FLUX-v3 report condition schema/identity drifted.")
    if native:
        native_config = expected_config.get("native_negative_prompt")
        if not isinstance(native_config, Mapping):
            raise ValueError("Completed FLUX-v3 native config is malformed.")
        calibration_mode = native_config.get("calibration_negative_prompt_mode")
        expected_calibration = None
        if calibration_mode is not None:
            true_cfg_scale = native_config.get("true_cfg_scale")
            if type(true_cfg_scale) is not float or not math.isfinite(true_cfg_scale):
                raise ValueError("Completed FLUX-v3 calibration scale is not one finite float.")
            expected_calibration = {
                "schema_version": 1,
                "calibration_id": native_config.get("calibration_id"),
                "calibration_config_sha256": native_config.get(
                    "calibration_config_sha256"
                ),
                "calibration_role": native_config.get("calibration_role"),
                "negative_prompt_mode": calibration_mode,
                "negative_prompt_is_none": job["negative_prompt"] is None,
                "true_cfg_scale": true_cfg_scale,
                "only_controlled_call_difference": "paired_negative_prompt_values",
            }
        negative_plan = job["generation"]["flux_dual_view_conditioning"]["negative"]
        expected_negative_prompt_2 = (
            None
            if negative_plan is None
            else negative_plan["t5_negative_prompt_2"]
        )
        if (
            condition.get("steering_mode") != "native_negative_prompt"
            or condition.get("negative_prompt") != job["negative_prompt"]
            or condition.get("negative_prompt_2") != expected_negative_prompt_2
        ):
            raise ValueError("Completed FLUX-v3 native-negative condition drifted.")
        _require_canonical_json_equal(
            condition.get("flux1_native_negative_calibration"),
            expected_calibration,
            label="completed FLUX-v3 native calibration provenance",
        )
    else:
        expected_condition = {
            "steering_mode": expected_config["steering"]["mode"],
            "decode_outputs": True,
            "is_native_negative_prompt": False,
            "negative_prompt": None,
        }
        _require_canonical_json_equal(
            condition,
            expected_condition,
            label="completed ordinary FLUX-v3 report condition",
        )
    report_timesteps = (report.get("interpretability") or {}).get("timesteps")
    trace_required = native or expected_config["output"].get("save_traces") is True
    if trace_required:
        if not isinstance(steering_trace, list) or not steering_trace:
            raise ValueError("Completed FLUX-v3 output lacks its canonical non-empty trace.")
        _require_canonical_json_equal(
            steering_trace,
            report_timesteps,
            label="completed FLUX-v3 trace/report timestep binding",
        )
    elif steering_trace is not None:
        raise ValueError("FLUX-v3 output has an unregistered steering-trace payload.")
    media_validation = _png_validation_from_authenticated_bytes(
        media_raw,
        canonical_path=output_dir / _FLUX1_EVIDENCE_RELATIVE_PATHS["media"],
        job=job,
    )
    runtime_validation = _validate_flux1_runtime_payloads_v3(
        job,
        run_timing=run_timing,
        report=report,
        runner_config=resolved_config,
    )
    return runtime_validation, media_validation, expected_records


def _flux1_runtime_receipt(
    job: Mapping[str, Any],
    *,
    bundle: Mapping[str, Mapping[str, Any]],
    runtime_validation: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_digest = job.get("launch_manifest_sha256")
    manifest_index = job.get("launch_manifest_job_index")
    if (
        not isinstance(manifest_digest, str)
        or _SHA256_RE.fullmatch(manifest_digest) is None
        or isinstance(manifest_index, bool)
        or not isinstance(manifest_index, int)
        or manifest_index < 0
    ):
        raise ValueError("FLUX-v3 runtime receipt lacks an exact manifest row binding.")
    return {
        "schema_version": 3,
        "status": "passed",
        "contract": _FLUX1_RUNTIME_RECEIPT_CONTRACT,
        "condition_id": job["condition_id"],
        "job_sha256": flux1_canonical_sha256(job),
        "manifest_sha256": manifest_digest,
        "manifest_job_index": manifest_index,
        "seed": job["seed"],
        "variant": job["variant"],
        "runtime_validation": deepcopy(dict(runtime_validation)),
        "evidence_sha256": {
            key: str(bundle[key]["sha256"])
            for key in (
                "resolved_config",
                "run_timing",
                "sample_report",
                *(("steering_trace",) if "steering_trace" in bundle else ()),
                "media",
            )
        },
    }


def _validate_flux1_runtime_outputs_v3(
    job: Mapping[str, Any], output_dir: Path, *, root: Path
) -> dict[str, Any]:
    """Authenticate actual dual-view evidence before publishing a success result."""

    expected_config = _runner_config(dict(job), root)
    trusted_root = _lexical_path_below_root(root, root, label="FLUX-v3 project root")
    supplied_output_dir = _lexical_path_below_root(
        output_dir, trusted_root, label="supplied FLUX-v3 output directory"
    )
    expected_output_dir = _lexical_path_below_root(
        os.fspath(job.get("output_dir", "")), trusted_root, label="FLUX-v3 output directory"
    )
    if supplied_output_dir != expected_output_dir:
        raise ValueError("Supplied FLUX-v3 output directory differs from the launch job.")
    output_dir = expected_output_dir
    with _pinned_nofollow_directory(
        output_dir,
        trusted_root=trusted_root,
        label="FLUX-v3 output directory",
    ) as (pinned_output, output_descriptor, _links):
        trace_required = (
            job["variant_spec"]["kind"] == "native_negative_prompt"
            or expected_config["output"].get("save_traces") is True
        )
        first = _read_flux1_evidence_bundle(
            output_descriptor,
            include_result=False,
            include_trace=trace_required,
        )
        if _list_media_below_pinned_output(output_descriptor) != (
            _FLUX1_EVIDENCE_RELATIVE_PATHS["media"],
        ):
            raise ValueError("Completed FLUX-v3 output does not contain exactly one canonical PNG.")
        resolved_config = _decode_resolved_config(first["resolved_config"]["raw"])
        run_timing = _json_from_authenticated_bytes(
            first["run_timing"]["raw"], label="completed FLUX-v3 run_timing"
        )
        report = _json_from_authenticated_bytes(
            first["sample_report"]["raw"], label="completed FLUX-v3 sample report"
        )
        steering_trace = (
            _json_value_from_authenticated_bytes(
                first["steering_trace"]["raw"],
                label="completed FLUX-v3 steering trace",
            )
            if trace_required
            else None
        )
        runtime_validation, _media_validation, _records = _validate_flux1_evidence_payloads(
            job,
            output_dir=pinned_output,
            resolved_config=resolved_config,
            expected_config=expected_config,
            run_timing=run_timing,
            report=report,
            steering_trace=steering_trace,
            media_raw=first["media"]["raw"],
        )
        second = _read_flux1_evidence_bundle(
            output_descriptor,
            include_result=False,
            include_trace=trace_required,
        )
        for key in first:
            _same_authenticated_leaf(first[key], second[key], label=f"FLUX-v3 {key}")
        if _list_media_below_pinned_output(output_descriptor) != (
            _FLUX1_EVIDENCE_RELATIVE_PATHS["media"],
        ):
            raise ValueError("FLUX-v3 media set changed during runtime validation.")
        return _flux1_runtime_receipt(
            job, bundle=first, runtime_validation=runtime_validation
        )


def reopen_completed_flux1_output_v3(
    job: Mapping[str, Any],
    *,
    root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    manifest_job_index: int,
    result_path: Path | None = None,
) -> dict[str, Any]:
    """Reopen one completed FLUX-v3 output as one pinned evidence bundle.

    Downstream qualification, selection, smoke, and plotting code must not
    trust a copied ``status=completed`` or a self-authored runtime receipt.
    This boundary authenticates the historical manifest and content-addressed
    snapshot in audit mode, pins the entire output-directory inode chain,
    rejects symlinks and generic hardlink aliases, and reads every result,
    config, timing, report, and media leaf twice from that held directory.

    Hardlink policy is intentionally strict: each evidence leaf must have
    ``st_nlink == 1``.  A future publication protocol that legitimately uses
    hardlinks must authenticate those aliases in a dedicated canonical
    hardlink manifest; arbitrary aliases are never accepted here.
    """

    raw_root = root.expanduser()
    absolute_root = raw_root if raw_root.is_absolute() else (Path.cwd() / raw_root).absolute()
    trusted_root = _lexical_path_below_root(
        absolute_root, absolute_root, label="FLUX-v3 project root"
    )
    authenticated_manifest_path = _lexical_path_below_root(
        manifest_path, trusted_root, label="FLUX-v3 launch manifest"
    )
    if (
        not isinstance(manifest_sha256, str)
        or _SHA256_RE.fullmatch(manifest_sha256) is None
        or isinstance(manifest_job_index, bool)
        or not isinstance(manifest_job_index, int)
        or manifest_job_index < 0
    ):
        raise ValueError("Strict FLUX-v3 reopening requires an exact manifest digest/index.")
    manifest = read_manifest_for_audit(authenticated_manifest_path, trusted_root)
    if manifest.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Strict FLUX-v3 manifest digest differs from the requested binding.")
    rows = manifest.get("jobs")
    if not isinstance(rows, list) or manifest_job_index >= len(rows):
        raise IndexError("Strict FLUX-v3 manifest job index is outside the manifest row set.")
    manifest_job = rows[manifest_job_index]
    if not isinstance(manifest_job, Mapping) or not is_flux1_job_v3(manifest_job):
        raise ValueError("Strict FLUX-v3 output reopening received a non-v3 job.")
    expected_job = {
        **deepcopy(dict(manifest_job)),
        "launch_manifest_sha256": manifest_sha256,
        "launch_manifest_job_index": manifest_job_index,
    }
    _require_canonical_json_equal(
        job, expected_job, label="completed FLUX-v3 launch-bound job"
    )
    expected_config = _audit_expected_runner_config(
        manifest, expected_job, root=trusted_root
    )
    output_dir = _lexical_path_below_root(
        os.fspath(expected_job.get("output_dir", "")),
        trusted_root,
        label="FLUX-v3 output directory",
    )
    canonical_result = output_dir / "benchmark_job_result.json"
    supplied_result = _lexical_path_below_root(
        canonical_result if result_path is None else result_path,
        trusted_root,
        label="FLUX-v3 result path",
    )
    if supplied_result != canonical_result:
        raise ValueError("Completed FLUX-v3 result path is not canonical for its output row.")

    with _pinned_nofollow_directory(
        output_dir,
        trusted_root=trusted_root,
        label="FLUX-v3 output directory",
    ) as (pinned_output, output_descriptor, _links):
        trace_required = (
            expected_job["variant_spec"]["kind"] == "native_negative_prompt"
            or expected_config["output"].get("save_traces") is True
        )
        first = _read_flux1_evidence_bundle(
            output_descriptor,
            include_result=True,
            include_trace=trace_required,
        )
        if _list_media_below_pinned_output(output_descriptor) != (
            _FLUX1_EVIDENCE_RELATIVE_PATHS["media"],
        ):
            raise ValueError("Completed FLUX-v3 output does not contain exactly one canonical PNG.")
        result = _json_from_authenticated_bytes(
            first["result"]["raw"], label="completed FLUX-v3 result"
        )
        resolved_config = _decode_resolved_config(first["resolved_config"]["raw"])
        run_timing = _json_from_authenticated_bytes(
            first["run_timing"]["raw"], label="completed FLUX-v3 run_timing"
        )
        report = _json_from_authenticated_bytes(
            first["sample_report"]["raw"], label="completed FLUX-v3 sample report"
        )
        steering_trace = (
            _json_value_from_authenticated_bytes(
                first["steering_trace"]["raw"],
                label="completed FLUX-v3 steering trace",
            )
            if trace_required
            else None
        )
        runtime_validation, media_validation, expected_records = (
            _validate_flux1_evidence_payloads(
                expected_job,
                output_dir=pinned_output,
                resolved_config=resolved_config,
                expected_config=expected_config,
                run_timing=run_timing,
                report=report,
                steering_trace=steering_trace,
                media_raw=first["media"]["raw"],
            )
        )
        native = expected_job["variant_spec"]["kind"] == "native_negative_prompt"
        exact_result_keys = (
            {
                "schema_version",
                "status",
                "job",
                "result",
                "reason",
                "media_validation",
                "segmented_temporal_validation",
                "flux1_dual_view_runtime_validation",
                "validated_media_paths",
            }
            if native
            else {
                "schema_version",
                "status",
                "job",
                "runner_output_dir",
                "records",
                "media_validation",
                "segmented_temporal_validation",
                "flux1_dual_view_runtime_validation",
                "validated_media_paths",
            }
        )
        if (
            set(result) != exact_result_keys
            or isinstance(result.get("schema_version"), bool)
            or not isinstance(result.get("schema_version"), int)
            or result.get("schema_version") != 2
            or result.get("status") != "completed"
            or result.get("segmented_temporal_validation") is not None
        ):
            raise ValueError("Completed FLUX-v3 result schema/status is not exact.")
        _require_canonical_json_equal(
            result.get("job"), expected_job, label="completed FLUX-v3 embedded job"
        )
        if native:
            expected_native_result = {
                "output_dir": str(pinned_output),
                "records": expected_records,
            }
            _require_canonical_json_equal(
                result.get("result"),
                expected_native_result,
                label="completed FLUX-v3 native result",
            )
            if result.get("reason") is not None:
                raise ValueError("Completed FLUX-v3 native result unexpectedly has a reason.")
        else:
            if result.get("runner_output_dir") != str(pinned_output):
                raise ValueError("Completed FLUX-v3 runner output directory drifted.")
            _require_canonical_json_equal(
                result.get("records"),
                expected_records,
                label="completed FLUX-v3 result records",
            )
        _require_canonical_json_equal(
            result.get("media_validation"),
            media_validation,
            label="completed FLUX-v3 media validation",
        )
        _require_canonical_json_equal(
            result.get("validated_media_paths"),
            [str(pinned_output / _FLUX1_EVIDENCE_RELATIVE_PATHS["media"])],
            label="completed FLUX-v3 validated media paths",
        )
        recomputed_receipt = _flux1_runtime_receipt(
            expected_job, bundle=first, runtime_validation=runtime_validation
        )
        _require_canonical_json_equal(
            result.get("flux1_dual_view_runtime_validation"),
            recomputed_receipt,
            label="completed FLUX-v3 stored runtime receipt",
        )

        second = _read_flux1_evidence_bundle(
            output_descriptor,
            include_result=True,
            include_trace=trace_required,
        )
        for key in first:
            _same_authenticated_leaf(first[key], second[key], label=f"FLUX-v3 {key}")
        if _list_media_below_pinned_output(output_descriptor) != (
            _FLUX1_EVIDENCE_RELATIVE_PATHS["media"],
        ):
            raise ValueError("FLUX-v3 media set changed during completed-output reopening.")

        artifact_bindings = {
            key: {
                "path": str(pinned_output / (
                    "benchmark_job_result.json"
                    if key == "result"
                    else _FLUX1_EVIDENCE_RELATIVE_PATHS[key]
                )),
                "sha256": str(record["sha256"]),
                "size_bytes": int(record["identity"][2]),
            }
            for key, record in first.items()
        }
        return {
            "schema_version": 3,
            "status": "passed",
            "contract": "finer_detailing_flux1_completed_output_reopen_v3",
            "manifest_path": str(authenticated_manifest_path),
            "manifest_sha256": manifest_sha256,
            "manifest_job_index": manifest_job_index,
            "job_sha256": flux1_canonical_sha256(expected_job),
            "result": result,
            "resolved_config": resolved_config,
            "run_timing": run_timing,
            "sample_report": report,
            "native_trace": steering_trace if native else None,
            "runtime_validation": runtime_validation,
            "media_validation": media_validation,
            "media_bytes": bytes(first["media"]["raw"]),
            "artifact_bindings": artifact_bindings,
            "evidence_bindings": deepcopy(artifact_bindings),
            "result_sha256": str(first["result"]["sha256"]),
            "resolved_config_sha256": str(first["resolved_config"]["sha256"]),
            "run_timing_sha256": str(first["run_timing"]["sha256"]),
            "sample_report_sha256": str(first["sample_report"]["sha256"]),
            "steering_trace_sha256": (
                str(first["steering_trace"]["sha256"])
                if "steering_trace" in first
                else None
            ),
            "media_sha256": str(first["media"]["sha256"]),
        }


def run_job(job: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    if job.get("model_name") == "flux1_dev" and job.get("stage") != FLUX_COMMON_SEED_V2_STAGE:
        validate_flux1_job_v3(job, project_root=root, mode=FLUX1_V3_MODE_EXECUTION)
    reject_reserved_flux_common_seed_outputs(job, root)
    common_seed_stage = (
        str(job["stage"]) if job.get("stage") in FLUX_COMMON_SEED_STAGES else None
    )
    common_seed_module = (
        flux_common_seed_module_for_stage(common_seed_stage)
        if common_seed_stage is not None
        else None
    )
    if common_seed_module is not None:
        common_seed_module.validate_bound_job_for_launch(job, root=root)
    # Fail on provenance drift before creating output files or loading a model.
    _verify_implementation_provenance(job, root)
    _verify_job_input_files(job)
    _verify_job_semantic_snapshots(job)
    if job.get("snapshot_bundle") is not None:
        _verify_snapshot_bundle(job)
    output_dir = Path(str(job["output_dir"]))
    _require_fresh_attempt(output_dir, job, root=root)
    if common_seed_module is not None:
        common_seed_module.validate_attempt_launch_authorization(
            job, output_dir=output_dir, root=root
        )
    ensure_dir(output_dir)
    # Runtime scheduler identity is deliberately separate from the frozen job
    # mapping.  This preserves manifest equality checks while making an attempt
    # traceable even if Slurm terminates Python before a result can be written.
    write_execution_identity(output_dir)
    write_yaml(output_dir / "benchmark_job.yaml", job)
    _write_job_notes(output_dir, job, status="started")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        config = _runner_config(job, root)
        if job["variant_spec"]["kind"] == "native_negative_prompt":
            if job["variant_spec"].get("capability") == "not_supported":
                _assert_no_generated_media(output_dir)
                payload = {
                    "schema_version": 2,
                    "status": "not_supported",
                    "job": job,
                    "result": None,
                    "reason": job["variant_spec"]["reason"],
                    "media_validation": None,
                    "validated_media_paths": [],
                }
                _write_outer_timing(output_dir, started_at, started, "not_supported")
                write_json(output_dir / "benchmark_job_result.json", payload)
                _write_job_notes(output_dir, job, status="not_supported", payload=payload)
                return payload
            native_result = run_native_negative_prompt_baseline(config, str(job["prompt"]))
            status = str(native_result["status"])
            if status == "completed":
                media_validation = validate_exact_media(output_dir, job, decode_video=True)
                segmented_validation = _validate_segmented_temporal_run(output_dir, job)
                flux1_runtime_validation = (
                    _validate_flux1_runtime_outputs_v3(job, output_dir, root=root)
                    if is_flux1_job_v3(job)
                    else None
                )
            elif status == "not_supported":
                if bool(job.get("expected_native_negative_support")):
                    raise RuntimeError(
                        f"{job['model_name']} was expected to support a native negative prompt but reported: "
                        f"{native_result.get('reason')}"
                    )
                _assert_no_generated_media(output_dir)
                media_validation = None
                segmented_validation = None
                flux1_runtime_validation = None
            else:
                raise RuntimeError(f"Unexpected native negative-prompt status: {status!r}")
            payload = {
                "schema_version": 2,
                "status": status,
                "job": job,
                "result": native_result.get("result"),
                "reason": native_result.get("reason"),
                "media_validation": media_validation,
                "segmented_temporal_validation": segmented_validation,
                "flux1_dual_view_runtime_validation": flux1_runtime_validation,
                "validated_media_paths": [media_validation["path"]] if media_validation else [],
            }
        else:
            result = GenerationRunner(config).run(prompt=str(job["prompt"]))
            media_validation = validate_exact_media(output_dir, job, decode_video=True)
            segmented_validation = _validate_segmented_temporal_run(output_dir, job)
            flux1_runtime_validation = (
                _validate_flux1_runtime_outputs_v3(job, output_dir, root=root)
                if is_flux1_job_v3(job)
                else None
            )
            payload = {
                "schema_version": 2,
                "status": "completed",
                "job": job,
                "runner_output_dir": result.output_dir,
                "records": [asdict(record) for record in result.records],
                "media_validation": media_validation,
                "segmented_temporal_validation": segmented_validation,
                "flux1_dual_view_runtime_validation": flux1_runtime_validation,
                "validated_media_paths": [media_validation["path"]],
            }
        _write_outer_timing(output_dir, started_at, started, payload["status"])
        write_json(output_dir / "benchmark_job_result.json", payload)
        _write_job_notes(output_dir, job, status=str(payload["status"]), payload=payload)
        return payload
    except Exception as exc:
        payload = {
            "schema_version": 2,
            "status": "failed",
            "job": job,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_outer_timing(output_dir, started_at, started, "failed")
        write_json(output_dir / "benchmark_job_result.json", payload)
        _write_job_notes(output_dir, job, status="failed", payload=payload)
        raise


def validate_exact_media(
    output_dir: Path, job: dict[str, Any], decode_video: bool
) -> dict[str, Any]:
    task = str(job["generation"]["task"])
    if task == "text_to_image":
        expected = output_dir / "sample_0000" / "image_000.png"
        _assert_only_expected_media(output_dir, expected)
        return _validate_png(expected, job)
    if task == "text_to_video":
        expected = output_dir / "sample_0000" / "video_000.mp4"
        _assert_only_expected_media(output_dir, expected)
        return _validate_mp4(expected, job, decode_video=decode_video)
    raise ValueError(f"Unsupported generation task {task!r}.")


def _validate_segmented_temporal_run(
    output_dir: Path,
    job: dict[str, Any],
) -> dict[str, Any] | None:
    """Reopen schema-2 evidence and its complete segment trace before success."""

    if str(job["model_name"]) not in SEGMENTED_TEMPORAL_MODELS:
        if job.get("segmented_temporal_contract") is not None:
            raise RuntimeError("A non-segmented model unexpectedly declares schema-2 evidence.")
        return None
    contract = job.get("segmented_temporal_contract")
    if contract != SEGMENTED_PROTOCOL_CONTRACT:
        raise RuntimeError("Segmented job contract differs from the frozen benchmark schema.")
    sample_dir = output_dir / "sample_0000"
    evidence_path = sample_dir / "temporal_evidence.json"
    report_path = sample_dir / "report.json"
    evidence = read_temporal_evidence(evidence_path)
    if not report_path.is_file():
        raise FileNotFoundError(f"Segmented report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_manifest = job.get("launch_manifest_sha256")
    expected_identity = {
        "condition_id": job["condition_id"],
        "attempt": job["attempt"],
        "manifest_sha256": expected_manifest,
    }
    observed_identity = {key: evidence.get(key) for key in expected_identity}
    if observed_identity != expected_identity:
        raise RuntimeError("Temporal evidence execution identity differs from the launch job.")
    protocol_snapshot = job.get("temporal_protocol_snapshot")
    if not isinstance(protocol_snapshot, dict) or not isinstance(
        protocol_snapshot.get("protocol"), dict
    ):
        raise RuntimeError("Segmented job lacks its frozen temporal protocol snapshot.")
    if evidence.get("temporal_protocol") != protocol_snapshot["protocol"]:
        raise RuntimeError("Temporal evidence protocol differs from the frozen launch job.")
    expected_metadata = {
        "checkpoint_set": job.get("checkpoint_set"),
        "checkpoint_set_sha256": job.get("checkpoint_set_sha256"),
        "artifact_manifest": job.get("artifact_manifest"),
        "artifact_manifest_sha256": job.get("artifact_manifest_sha256"),
        "segmented_temporal_contract": job.get("segmented_temporal_contract"),
    }
    metadata = evidence.get("metadata")
    if not isinstance(metadata, dict) or any(
        metadata.get(key) != value for key, value in expected_metadata.items()
    ):
        raise RuntimeError("Temporal evidence checkpoint/artifact metadata drifted from the job.")
    benchmark = report.get("benchmark")
    if not isinstance(benchmark, dict) or any(
        benchmark.get(key) != value
        for key, value in {
            **expected_identity,
            **expected_metadata,
        }.items()
    ):
        raise RuntimeError("Segmented report benchmark identity differs from the launch job.")
    interpretation = report.get("interpretability")
    segment_validation = (
        interpretation.get("segment_trace_validation") if isinstance(interpretation, dict) else None
    )
    if (
        not isinstance(segment_validation, dict)
        or segment_validation.get("schema_version") != 1
        or segment_validation.get("status") != "passed"
    ):
        raise RuntimeError("Segmented run lacks passed segment-trace schema 1 evidence.")
    segments = evidence.get("segments")
    traced = segment_validation.get("segments")
    if (
        not isinstance(segments, list)
        or not isinstance(traced, list)
        or segment_validation.get("segment_count") != len(segments)
        or len(traced) != len(segments)
    ):
        raise RuntimeError("Segmented trace coverage differs from temporal evidence.")
    global_start = 0
    for expected_index, (segment, trace_segment) in enumerate(zip(segments, traced, strict=True)):
        if not isinstance(segment, dict) or not isinstance(trace_segment, dict):
            raise RuntimeError("Segmented trace contains a malformed segment record.")
        scheduler = segment.get("scheduler")
        if not isinstance(scheduler, dict):
            raise RuntimeError("Temporal evidence segment scheduler record is missing.")
        local_steps = scheduler.get("num_inference_steps", scheduler.get("denoising_steps"))
        if isinstance(local_steps, bool) or not isinstance(local_steps, int):
            raise RuntimeError("Temporal evidence lacks an integer local step count.")
        exact_identity = {
            "segment_index": expected_index,
            "model_role": segment.get("model_role"),
            "model_id": segment.get("model_id"),
            "model_revision": segment.get("model_revision"),
            "anchor_sha256": segment.get("anchor_sha256"),
            "segment_seed": segment.get("segment_seed"),
            "local_num_steps": local_steps,
            "global_step_start": global_start,
            "global_step_end": global_start + local_steps - 1,
        }
        if any(trace_segment.get(key) != value for key, value in exact_identity.items()):
            raise RuntimeError(
                f"Segment trace identity differs from temporal evidence at segment {expected_index}."
            )
        if trace_segment.get("condition_epoch") != expected_index:
            raise RuntimeError("Segment trace condition epoch did not advance exactly once.")
        global_start += local_steps
    if segment_validation.get("global_num_steps") != global_start:
        raise RuntimeError("Segment trace global step arithmetic is inconsistent.")
    temporal_report_binding = report.get("temporal_evidence")
    if (
        not isinstance(temporal_report_binding, dict)
        or Path(str(temporal_report_binding.get("path", ""))).resolve() != evidence_path.resolve()
    ):
        raise RuntimeError("Report does not bind the published temporal evidence path.")
    if temporal_report_binding.get("temporal_protocol_sha256") != evidence.get(
        "temporal_protocol_sha256"
    ):
        raise RuntimeError("Report/evidence temporal protocol digests differ.")
    return {
        "schema_version": 1,
        "status": "passed",
        "temporal_evidence_path": str(evidence_path),
        "temporal_evidence_document_sha256": evidence["document_sha256"],
        "temporal_protocol_sha256": evidence["temporal_protocol_sha256"],
        "segment_trace_schema_version": 1,
        "segment_count": len(segments),
        "global_num_steps": global_start,
        "final_media_sha256": evidence["binding"]["final_media_sha256"],
        "final_decoded_rgb_sha256": evidence["binding"]["final_decoded_rgb_sha256"],
    }


def _validate_png(path: Path, job: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Expected non-empty PNG was not produced: {path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
        image_format = image.format
    expected_width = int(job["generation"]["width"])
    expected_height = int(job["generation"]["height"])
    if image_format != "PNG":
        raise RuntimeError(f"Expected PNG encoding at {path}, found {image_format!r}.")
    if (width, height) != (expected_width, expected_height):
        raise RuntimeError(
            f"PNG dimensions mismatch at {path}: got {width}x{height}, "
            f"expected {expected_width}x{expected_height}."
        )
    return {
        "path": str(path),
        "media_type": "image/png",
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "width": width,
        "height": height,
        "mode": mode,
        "decode_verified": True,
    }


def _validate_mp4(path: Path, job: dict[str, Any], decode_video: bool) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Expected non-empty MP4 was not produced: {path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-show_entries",
        "format=format_name,duration",
        "-of",
        "json",
        str(path),
    ]
    probe = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
    metadata = json.loads(probe.stdout)
    streams = metadata.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"Expected exactly one video stream in {path}, found {len(streams)}.")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    expected_width = int(job["generation"]["width"])
    expected_height = int(job["generation"]["height"])
    if (width, height) != (expected_width, expected_height):
        raise RuntimeError(
            f"MP4 dimensions mismatch at {path}: got {width}x{height}, "
            f"expected {expected_width}x{expected_height}."
        )
    frame_count_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_count_value in (None, "N/A"):
        raise RuntimeError(f"ffprobe did not report a frame count for {path}.")
    frame_count = int(frame_count_value)
    expected_frames = int(job["generation"]["num_frames"])
    if frame_count != expected_frames:
        raise RuntimeError(
            f"MP4 frame count mismatch at {path}: got {frame_count}, expected {expected_frames}."
        )
    fps = float(Fraction(str(stream["avg_frame_rate"])))
    expected_fps = float(job["generation"]["fps"])
    if abs(fps - expected_fps) > 1.0e-3:
        raise RuntimeError(f"MP4 FPS mismatch at {path}: got {fps}, expected {expected_fps}.")
    duration = float((metadata.get("format") or {}).get("duration") or stream.get("duration"))
    expected_duration = expected_frames / expected_fps
    if abs(duration - expected_duration) > (1.0 / expected_fps + 1.0e-3):
        raise RuntimeError(
            f"MP4 duration mismatch at {path}: got {duration}, expected approximately {expected_duration}."
        )
    if decode_video:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    return {
        "path": str(path),
        "media_type": "video/mp4",
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "codec_name": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "decode_verified": bool(decode_video),
    }


def _snapshot_bundle_path(path: Path) -> Path:
    return Path(f"{path}.snapshot")


def _materialize_snapshot_bundle(
    manifest: dict[str, Any],
    path: Path,
    root: Path,
) -> dict[str, Any]:
    """Create a content-addressed source/config archive beside a manifest."""

    snapshot_root = _snapshot_bundle_path(path)
    if snapshot_root.exists():
        raise FileExistsError(f"Refusing to overwrite immutable snapshot bundle: {snapshot_root}")
    snapshot_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{snapshot_root.name}.tmp-",
            dir=snapshot_root.parent,
        )
    )
    objects_root = temporary_root / "objects" / "sha256"
    objects_root.mkdir(parents=True, exist_ok=False)

    try:
        implementation_files = dict(manifest["implementation_files"])
        object_sources: dict[str, Path] = {}
        for relative, digest in implementation_files.items():
            source = root.resolve() / relative
            if _sha256_file(source) != digest:
                raise ValueError(f"Implementation changed while snapshotting: {source}")
            object_sources.setdefault(digest, source)

        input_records: dict[str, dict[str, Any]] = {}
        for job in manifest["jobs"]:
            for role, record in job["input_files"].items():
                source = Path(str(record["path"])).resolve()
                digest = str(record["sha256"])
                if _sha256_file(source) != digest:
                    raise ValueError(f"Frozen input changed while snapshotting: {source}")
                key = str(source)
                existing = input_records.setdefault(
                    key,
                    {"sha256": digest, "roles": set()},
                )
                if existing["sha256"] != digest:
                    raise ValueError(f"One input path has conflicting frozen hashes: {source}")
                existing["roles"].add(str(role))
                object_sources.setdefault(digest, source)

        objects: dict[str, dict[str, Any]] = {}
        for digest, source in sorted(object_sources.items()):
            object_path = objects_root / digest[:2] / digest
            object_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, object_path)
            copied_digest = _sha256_file(object_path)
            if copied_digest != digest:
                raise RuntimeError(
                    f"Content-addressed snapshot copy failed verification: {object_path}"
                )
            objects[digest] = {
                "path": object_path.relative_to(temporary_root).as_posix(),
                "size_bytes": object_path.stat().st_size,
            }

        index = {
            "schema_version": 1,
            "benchmark": str(manifest.get("benchmark", BENCHMARK_NAME)),
            "implementation_files": implementation_files,
            "implementation_files_sha256": manifest["implementation_files_sha256"],
            "input_files": {
                original: {
                    "sha256": record["sha256"],
                    "roles": sorted(record["roles"]),
                }
                for original, record in sorted(input_records.items())
            },
            "objects": objects,
        }
        temporary_index_path = temporary_root / "index.json"
        write_json(temporary_index_path, index)
        index_sha256 = _sha256_file(temporary_index_path)
        # The final archive name appears only after every copied object and the
        # index have passed their SHA-256 checks.
        temporary_root.rename(snapshot_root)
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise

    index_path = snapshot_root / "index.json"
    if _sha256_file(index_path) != index_sha256:
        # This should be impossible after a same-filesystem rename, but fail
        # closed and do not publish a manifest if storage violated the contract.
        shutil.rmtree(snapshot_root)
        raise RuntimeError(f"Atomically published snapshot index failed verification: {index_path}")
    return {
        "schema_version": 1,
        "root_path": str(snapshot_root.resolve()),
        "index_path": str(index_path.resolve()),
        "index_sha256": index_sha256,
        "num_objects": len(objects),
        "content_addressing": "sha256",
    }


def _verify_snapshot_bundle(container: dict[str, Any]) -> None:
    """Verify an archive without consulting live project source/config files."""

    descriptor = container.get("snapshot_bundle")
    if not isinstance(descriptor, dict):
        raise ValueError("Immutable manifest/job is missing its snapshot_bundle descriptor.")
    index_path = Path(str(descriptor.get("index_path", "")))
    snapshot_root = Path(str(descriptor.get("root_path", "")))
    if not index_path.is_file() or not snapshot_root.is_dir():
        raise FileNotFoundError(
            f"Immutable snapshot bundle disappeared: root={snapshot_root}, index={index_path}"
        )
    expected_index_digest = str(descriptor.get("index_sha256", ""))
    actual_index_digest = _sha256_file(index_path)
    if actual_index_digest != expected_index_digest:
        raise ValueError(
            f"Snapshot index changed: {index_path}; expected={expected_index_digest}, "
            f"actual={actual_index_digest}."
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected_benchmark = str(container.get("benchmark", BENCHMARK_NAME))
    if index.get("benchmark") != expected_benchmark:
        raise ValueError(f"Snapshot benchmark mismatch in {index_path}.")
    if index.get("implementation_files") != container.get("implementation_files"):
        raise ValueError("Snapshot implementation mapping differs from the manifest/job mapping.")
    if index.get("implementation_files_sha256") != container.get("implementation_files_sha256"):
        raise ValueError("Snapshot implementation digest differs from the manifest/job digest.")

    objects = index.get("objects")
    if not isinstance(objects, dict) or len(objects) != int(descriptor.get("num_objects", -1)):
        raise ValueError("Snapshot object count is inconsistent.")
    resolved_root = snapshot_root.resolve()
    for digest, record in sorted(objects.items()):
        if not isinstance(record, dict):
            raise ValueError(f"Malformed snapshot object record for {digest}.")
        object_path = (snapshot_root / str(record.get("path", ""))).resolve()
        if resolved_root not in object_path.parents:
            raise ValueError(f"Snapshot object escapes its archive root: {object_path}")
        if not object_path.is_file() or _sha256_file(object_path) != digest:
            raise ValueError(f"Snapshot object is missing or corrupted: {object_path}")
        if object_path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"Snapshot object size is inconsistent: {object_path}")

    input_index = index.get("input_files")
    if not isinstance(input_index, dict):
        raise ValueError("Snapshot index has no input_files mapping.")
    if "input_files" in container:
        for record in container["input_files"].values():
            original = str(Path(str(record["path"])).resolve())
            frozen = input_index.get(original)
            if not isinstance(frozen, dict) or frozen.get("sha256") != record.get("sha256"):
                raise ValueError(f"Snapshot index does not preserve frozen job input {original}.")
    if "jobs" in container:
        for job in container["jobs"]:
            for record in job["input_files"].values():
                original = str(Path(str(record["path"])).resolve())
                frozen = input_index.get(original)
                if not isinstance(frozen, dict) or frozen.get("sha256") != record.get("sha256"):
                    raise ValueError(
                        f"Snapshot index does not preserve frozen manifest input {original}."
                    )


def _snapshot_object_paths_by_sha256(
    container: Mapping[str, Any],
) -> tuple[Path, dict[str, Path]]:
    """Return lexical archive-object paths after `_verify_snapshot_bundle`."""

    descriptor = container.get("snapshot_bundle")
    if not isinstance(descriptor, Mapping):
        raise ValueError("Snapshot object lookup requires a verified bundle descriptor.")
    snapshot_root = Path(str(descriptor.get("root_path", "")))
    index_path = Path(str(descriptor.get("index_path", "")))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    objects = index.get("objects")
    if not snapshot_root.is_absolute() or not isinstance(objects, Mapping):
        raise ValueError("Snapshot object lookup has malformed root/object metadata.")
    return snapshot_root, {
        str(digest): snapshot_root / str(record["path"])
        for digest, record in objects.items()
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }


def _manifest_provenance_consistency(
    manifest: dict[str, Any], *, require_live_protocol: bool = False
) -> None:
    top = _provenance_from_container(manifest)
    if not isinstance(top["implementation_files"], dict):
        raise ValueError("Manifest is missing launch-blocking implementation provenance.")
    for index, job in enumerate(manifest["jobs"]):
        if _provenance_from_container(job) != top:
            raise ValueError(
                f"Job {index} implementation provenance differs from the manifest snapshot."
            )

    # Historical schema-3 manifests without explicit Shapley protocol fields
    # remain audit-readable.  Once protocol v2 is declared, top-level, job,
    # and runner inputs are exact deep-copy bindings rather than loose hints.
    if "shapley_config" in manifest or (
        isinstance(manifest.get("shapley_provenance"), dict)
        and manifest["shapley_provenance"].get("protocol_version") is not None
    ):
        top_config = manifest.get("shapley_config")
        top_provenance = manifest.get("shapley_provenance")
        if not isinstance(top_config, dict) or not isinstance(top_provenance, dict):
            raise ValueError("Manifest Shapley-v2 top-level bindings must be mappings.")
        if (
            top_provenance.get("protocol_version") != 2
            or top_provenance.get("trace_schema_version") != 2
        ):
            raise ValueError("Manifest Shapley protocol/trace schemas must both be version 2.")
        if require_live_protocol and (
            top_config != SHAPLEY_CONFIG or top_provenance != SHAPLEY_PROVENANCE
        ):
            raise ValueError("Launch manifest Shapley-v2 config differs from the live protocol.")
        for index, job in enumerate(manifest["jobs"]):
            variant = job.get("variant_spec")
            if not isinstance(variant, dict):
                raise ValueError(f"Job {index} has no variant_spec mapping.")
            if variant.get("kind") == "shapley_concept_steering":
                if variant.get("shapley") != top_config:
                    raise ValueError(f"Job {index} Shapley config differs from manifest top level.")
                if variant.get("shapley_provenance") != top_provenance:
                    raise ValueError(
                        f"Job {index} Shapley provenance differs from manifest top level."
                    )
            elif "shapley" in variant or "shapley_provenance" in variant:
                raise ValueError(f"Non-Shapley job {index} unexpectedly claims Shapley protocol.")

    if "segmented_temporal_contract" in manifest:
        top_contract = manifest.get("segmented_temporal_contract")
        if not isinstance(top_contract, dict):
            raise ValueError("Manifest segmented temporal contract must be a mapping.")
        if require_live_protocol and top_contract != SEGMENTED_PROTOCOL_CONTRACT:
            raise ValueError("Launch manifest segmented temporal schemas differ from live code.")
        checkpoint_sets = manifest.get("checkpoint_sets_by_model")
        checkpoint_digests = manifest.get("checkpoint_set_sha256_by_model")
        if not isinstance(checkpoint_sets, dict) or not isinstance(checkpoint_digests, dict):
            raise ValueError("Manifest segmented checkpoint-set snapshots are missing.")
        segmented_models = {
            str(job["model_name"])
            for job in manifest["jobs"]
            if job.get("segmented_temporal_contract") is not None
        }
        if set(checkpoint_sets) != segmented_models or set(checkpoint_digests) != segmented_models:
            raise ValueError("Manifest segmented checkpoint-set model coverage is inconsistent.")
        for index, job in enumerate(manifest["jobs"]):
            model_name = str(job["model_name"])
            if model_name in segmented_models:
                if job.get("segmented_temporal_contract") != top_contract:
                    raise ValueError(
                        f"Job {index} segmented schemas differ from manifest top level."
                    )
                if (
                    job.get("checkpoint_set") != checkpoint_sets[model_name]
                    or job.get("checkpoint_set_sha256") != checkpoint_digests[model_name]
                ):
                    raise ValueError(f"Job {index} checkpoint set differs from manifest top level.")
                canonical = json.dumps(
                    job["checkpoint_set"], sort_keys=True, separators=(",", ":")
                ).encode()
                if hashlib.sha256(canonical).hexdigest() != job["checkpoint_set_sha256"]:
                    raise ValueError(f"Job {index} checkpoint-set digest is invalid.")
            elif any(
                key in job
                for key in ("checkpoint_set", "checkpoint_set_sha256", "artifact_manifest")
            ):
                raise ValueError(f"Non-segmented job {index} claims segmented artifacts.")

    # Preserve read-only audit compatibility for historical schema-v3 manifests,
    # while fail-closing every newly generated manifest that declares the temporal
    # pilot fields.
    if (
        "unvalidated_temporal_pilot_jobs" in manifest
        or "allows_unvalidated_temporal_pilot" in manifest
    ):
        pilot_jobs = [
            job
            for job in manifest["jobs"]
            if (job.get("temporal_protocol_snapshot") or {}).get("qualification") == "pilot"
        ]
        if manifest.get("unvalidated_temporal_pilot_jobs") != len(pilot_jobs):
            raise ValueError("Manifest temporal pilot job count is inconsistent.")
        if bool(manifest.get("allows_unvalidated_temporal_pilot")) != bool(pilot_jobs):
            raise ValueError("Manifest temporal pilot authorization is inconsistent.")
        if any(job.get("temporal_pilot_authorized") is not True for job in pilot_jobs):
            raise ValueError("A temporal pilot job lacks explicit per-job authorization.")


def _write_text_exclusive_atomic(path: Path, content: str) -> None:
    """Publish a fully flushed file without ever overwriting an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails with FileExistsError
        # instead of replacing an immutable artifact in a race.
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


_ATOMIC_CAMPAIGN_LADDER_MANIFEST_ROOT = Path(
    "debugging/manifests/finer_detailing_seed_qualification_v1"
)
_ATOMIC_CAMPAIGN_FINAL_MANIFEST_ROOT = Path(
    "debugging/manifests/finer_detailing_correction_selected_seed_v1"
)
_ATOMIC_CAMPAIGN_LADDER_OUTPUT_ROOT = Path("outputs/finer_detailing_seed_qualification")
_ATOMIC_CAMPAIGN_FINAL_OUTPUT_ROOT = Path("outputs/finer_detailing_correction_selected_seed")
_ATOMIC_QUALIFICATION_ROOTS = {
    "q1": Path("debugging/manifests/finer_detailing_fresh_qualification_q1_v1"),
    "q2": Path("debugging/manifests/finer_detailing_fresh_qualification_q2_v1"),
}
_ATOMIC_QUALIFICATION_OUTPUT_ROOT = Path("outputs/finer_detailing_fresh_qualification_v1")
_ORDINARY_CAMPAIGN_STAGE = "full_and_single_pair_conceptsteer_and_shapley"


def _path_has_symlink_component(path: Path, *, root: Path) -> bool:
    """Return whether an existing component below ``root`` is a symlink."""

    if path != root and root not in path.parents:
        return True
    current = path
    while True:
        if current.is_symlink():
            return True
        if current == root:
            return False
        current = current.parent


def _validate_atomic_campaign_staging_authorization(
    manifest: Mapping[str, Any],
    *,
    physical_path: Path,
    raw_physical_path: Path,
    logical_path: Path,
    raw_logical_path: Path,
    root: Path,
) -> None:
    """Admit only one exact ladder/final member in its private sibling tree.

    This check intentionally duplicates only the small publication namespace
    boundary.  The complete 288/504 scientific topology remains owned by
    :mod:`hierasafe_flow.evaluation.finer_detailing_campaign` and is validated
    both before staging and again from reopened staged bytes.
    """

    if not raw_physical_path.is_absolute() or not raw_logical_path.is_absolute():
        raise ValueError("Atomic campaign staging requires absolute physical and logical paths.")
    # ``resolve`` must not silently turn a caller-supplied alias, ``..`` path,
    # or symlink route into an authorized canonical publication target.
    if str(raw_physical_path) != str(physical_path) or str(raw_logical_path) != str(logical_path):
        raise ValueError("Atomic campaign staging rejects resolved path aliases.")
    if _path_has_symlink_component(raw_physical_path.parent, root=root) or (
        _path_has_symlink_component(raw_logical_path.parent, root=root)
    ):
        raise ValueError("Atomic campaign staging rejects symlink path components.")

    ladder_root = (root / _ATOMIC_CAMPAIGN_LADDER_MANIFEST_ROOT).resolve()
    final_root = (root / _ATOMIC_CAMPAIGN_FINAL_MANIFEST_ROOT).resolve()
    logical_kind: str
    if logical_path.parent == ladder_root:
        logical_kind = "seed_ladder"
        expected_names = {f"seed_{seed:08d}.json" for seed in range(8)}
    elif logical_path.parent == final_root:
        logical_kind = "selected_seed_final"
        expected_names = {
            f"{prompt_id}__{model_name}__{family}.json"
            for prompt_id in PROMPT_IDS
            for model_name in MODEL_NAMES
            for family in ("standard", "shapley")
        }
    else:
        raise ValueError(
            "Atomic campaign staging is restricted to the two canonical campaign roots."
        )
    if logical_path.name not in expected_names:
        raise ValueError("Atomic campaign staging received a noncanonical member filename.")

    canonical_root = ladder_root if logical_kind == "seed_ladder" else final_root
    staging_root = physical_path.parent
    if (
        staging_root.parent != canonical_root.parent
        or not staging_root.name.startswith(f".{canonical_root.name}.staging-")
        or physical_path.name != logical_path.name
    ):
        raise ValueError(
            "Atomic campaign physical storage is not the matching hidden sibling cohort."
        )

    jobs = manifest.get("jobs")
    if (
        manifest.get("benchmark") != BENCHMARK_NAME
        or not isinstance(jobs, list)
        or not jobs
        or any(
            not isinstance(job, Mapping)
            or job.get("benchmark") != BENCHMARK_NAME
            or job.get("stage") != _ORDINARY_CAMPAIGN_STAGE
            for job in jobs
        )
        or manifest.get("stage") in FLUX_COMMON_SEED_STAGES
    ):
        raise ValueError(
            "Atomic campaign staging rejects common-seed, smoke, reserved, or noncampaign data."
        )

    if logical_kind == "seed_ladder":
        seed = int(logical_path.stem.removeprefix("seed_"))
        expected_output_root = (root / _ATOMIC_CAMPAIGN_LADDER_OUTPUT_ROOT).resolve()
        header_ok = (
            manifest.get("seed") == seed
            and manifest.get("attempt") == seed + 1
            and manifest.get("models") == list(MODEL_NAMES)
            and manifest.get("prompt_ids") == list(PROMPT_IDS)
            and manifest.get("variation_groups") == ["01_baseline"]
            and manifest.get("num_jobs") == 36
            and len(jobs) == 36
            and {(str(job.get("prompt_id", "")), str(job.get("model_name", ""))) for job in jobs}
            == {(prompt_id, model_name) for prompt_id in PROMPT_IDS for model_name in MODEL_NAMES}
            and all(
                job.get("seed") == seed
                and job.get("attempt") == seed + 1
                and job.get("seed_scoped_output") is True
                and job.get("variation") == "01_baseline"
                and job.get("variant") == "01_baseline"
                and job.get("variant_spec") == {"kind": "baseline"}
                and job.get("expected_media") is True
                for job in jobs
            )
        )
    else:
        matched = next(
            (
                (prompt_id, model_name, family)
                for prompt_id in PROMPT_IDS
                for model_name in MODEL_NAMES
                for family in ("standard", "shapley")
                if logical_path.name == f"{prompt_id}__{model_name}__{family}.json"
            ),
            None,
        )
        assert matched is not None
        prompt_id, model_name, family = matched
        expected_output_root = (root / _ATOMIC_CAMPAIGN_FINAL_OUTPUT_ROOT).resolve()
        expected_variations = (
            [
                "01_baseline",
                "02_negative_prompt",
                "03_concept_steering",
                "05_concept_steering_single_pair",
            ]
            if family == "standard"
            else [
                "04_shapley_concept_steering",
                "06_shapley_concept_steering_single_pair",
            ]
        )
        expected_rows = 8 if family == "standard" else 6
        header_ok = (
            manifest.get("attempt") == 1
            and manifest.get("models") == [model_name]
            and manifest.get("prompt_ids") == [prompt_id]
            and manifest.get("variation_groups") == expected_variations
            and manifest.get("num_jobs") == expected_rows
            and len(jobs) == expected_rows
            and all(
                job.get("prompt_id") == prompt_id
                and job.get("model_name") == model_name
                and job.get("seed") == manifest.get("seed")
                and job.get("attempt") == 1
                and job.get("seed_scoped_output") is True
                and job.get("variation") in expected_variations
                for job in jobs
            )
            and (
                [job.get("variation") for job in jobs].count("05_concept_steering_single_pair") == 5
                if family == "standard"
                else [job.get("variation") for job in jobs].count(
                    "06_shapley_concept_steering_single_pair"
                )
                == 5
            )
            and all(
                [job.get("variation") for job in jobs].count(variation) == 1
                for variation in expected_variations
                if not variation.endswith("single_pair")
            )
        )
    raw_output_root = Path(str(manifest.get("output_root", ""))).expanduser()
    if (
        not header_ok
        or not raw_output_root.is_absolute()
        or raw_output_root != expected_output_root
        or raw_output_root.resolve(strict=False) != raw_output_root
    ):
        raise ValueError(
            "Atomic campaign staging rejects common-seed, smoke, reserved, or noncampaign data."
        )
    for job in jobs:
        raw_output = Path(str(job.get("output_dir", ""))).expanduser()
        if (
            not raw_output.is_absolute()
            or raw_output.resolve(strict=False) != raw_output
            or raw_output == expected_output_root
            or expected_output_root not in raw_output.parents
        ):
            raise ValueError("Atomic campaign staging rejects a reserved/noncampaign output path.")


def _validate_atomic_qualification_staging_authorization(
    manifest: Mapping[str, Any],
    *,
    physical_path: Path,
    raw_physical_path: Path,
    logical_path: Path,
    raw_logical_path: Path,
    root: Path,
) -> None:
    """Admit only one exact Q1/Q2 role in its matching private sibling."""

    if not raw_physical_path.is_absolute() or not raw_logical_path.is_absolute():
        raise ValueError(
            "Atomic qualification staging requires absolute physical and logical paths."
        )
    if str(raw_physical_path) != str(physical_path) or str(raw_logical_path) != str(logical_path):
        raise ValueError("Atomic qualification staging rejects resolved path aliases.")
    if _path_has_symlink_component(raw_physical_path.parent, root=root) or (
        _path_has_symlink_component(raw_logical_path.parent, root=root)
    ):
        raise ValueError("Atomic qualification staging rejects symlink path components.")

    canonical_roots = {
        phase: (root / relative).resolve()
        for phase, relative in _ATOMIC_QUALIFICATION_ROOTS.items()
    }
    phase = next(
        (
            candidate
            for candidate, cohort in canonical_roots.items()
            if logical_path.parent == cohort
        ),
        None,
    )
    role_contracts = {
        "image_seed000.json": (
            0,
            tuple(
                model
                for model in MODEL_NAMES
                if model
                not in {
                    "cogvideox_5b",
                    "hunyuan_video",
                    "joyai_echo",
                    "ltx_23",
                    "wan22_t2v_a14b",
                }
            ),
            105 if phase == "q1" else 21,
        ),
        "video_seed000.json": (
            0,
            (
                "cogvideox_5b",
                "hunyuan_video",
                "joyai_echo",
                "ltx_23",
                "wan22_t2v_a14b",
            ),
            75 if phase == "q1" else 15,
        ),
        "video_seed001.json": (
            1,
            (
                "cogvideox_5b",
                "hunyuan_video",
                "joyai_echo",
                "ltx_23",
                "wan22_t2v_a14b",
            ),
            75 if phase == "q1" else 15,
        ),
        "video_seed002.json": (
            2,
            (
                "cogvideox_5b",
                "hunyuan_video",
                "joyai_echo",
                "ltx_23",
                "wan22_t2v_a14b",
            ),
            75 if phase == "q1" else 15,
        ),
    }
    if phase is None or logical_path.name not in role_contracts:
        raise ValueError(
            "Atomic qualification staging is restricted to exact canonical Q1/Q2 roles."
        )
    canonical_root = canonical_roots[phase]
    staging_root = physical_path.parent
    if (
        staging_root.parent != canonical_root.parent
        or not staging_root.name.startswith(f".{canonical_root.name}.staging-")
        or physical_path.name != logical_path.name
    ):
        raise ValueError(
            "Atomic qualification physical storage is not the matching hidden sibling cohort."
        )

    seed, models, rows = role_contracts[logical_path.name]
    variations = (
        [
            "01_baseline",
            "02_negative_prompt",
            "03_concept_steering",
            "05_concept_steering_single_pair",
            "06_shapley_concept_steering_single_pair",
        ]
        if phase == "q1"
        else ["04_shapley_concept_steering"]
    )
    jobs = manifest.get("jobs")
    expected_output_root = (root / _ATOMIC_QUALIFICATION_OUTPUT_ROOT).resolve()
    raw_output_root = Path(str(manifest.get("output_root", ""))).expanduser()
    header_ok = (
        manifest.get("benchmark") == BENCHMARK_NAME
        and isinstance(jobs, list)
        and len(jobs) == rows
        and manifest.get("num_jobs") == rows
        and manifest.get("attempt") == 1
        and manifest.get("seed") == seed
        and manifest.get("seed_scoped_output") is True
        and manifest.get("models") == list(models)
        and manifest.get("prompt_ids") == list(PROMPT_IDS)
        and manifest.get("variation_groups") == variations
        and raw_output_root.is_absolute()
        and raw_output_root == expected_output_root
        and raw_output_root.resolve(strict=False) == raw_output_root
        and all(
            isinstance(job, Mapping)
            and job.get("benchmark") == BENCHMARK_NAME
            and job.get("stage") == _ORDINARY_CAMPAIGN_STAGE
            and job.get("attempt") == 1
            and job.get("seed") == seed
            and (raw_output := Path(str(job.get("output_dir", ""))).expanduser()).is_absolute()
            and raw_output.resolve(strict=False) == raw_output
            and raw_output != expected_output_root
            and expected_output_root in raw_output.parents
            for job in (jobs or [])
        )
        and {
            (
                str(job.get("prompt_id", "")),
                str(job.get("model_name", "")),
                str(job.get("variation", "")),
            )
            for job in (jobs or [])
        }
        == {
            (prompt_id, model_name, variation)
            for prompt_id in PROMPT_IDS
            for model_name in models
            for variation in variations
        }
        and all(
            job.get("model_name") in models
            and job.get("prompt_id") in PROMPT_IDS
            and job.get("variation") in variations
            and job.get("seed_scoped_output") is True
            for job in (jobs or [])
        )
    )
    if not header_ok or manifest.get("stage") in FLUX_COMMON_SEED_STAGES:
        raise ValueError(
            "Atomic qualification staging rejects common-seed, smoke, reserved, or "
            "nonqualification data."
        )


def write_manifest_immutable(
    manifest: dict[str, Any],
    path: Path,
    root: Path,
    *,
    logical_publication_path: Path | None = None,
    allow_atomic_common_seed_cohort_staging: bool = False,
    allow_atomic_smoke_bundle_staging: bool = False,
    allow_atomic_campaign_cohort_staging: bool = False,
    allow_atomic_qualification_cohort_staging: bool = False,
) -> None:
    root = root.expanduser().resolve()
    raw_physical_path = path.expanduser()
    if not raw_physical_path.is_absolute():
        raw_physical_path = root / raw_physical_path
    path = raw_physical_path.resolve()
    raw_logical_path = (
        raw_physical_path
        if logical_publication_path is None
        else logical_publication_path.expanduser()
    )
    if not raw_logical_path.is_absolute():
        raw_logical_path = root / raw_logical_path
    logical_path = raw_logical_path.resolve()
    common_seed_claimed = flux_common_seed_stage_from_manifest(manifest) is not None
    staging_modes = sum(
        (
            allow_atomic_common_seed_cohort_staging,
            allow_atomic_smoke_bundle_staging,
            allow_atomic_campaign_cohort_staging,
            allow_atomic_qualification_cohort_staging,
        )
    )
    if staging_modes > 1:
        raise ValueError("Manifest publication cannot select multiple atomic staging contracts.")
    if common_seed_claimed and not allow_atomic_common_seed_cohort_staging:
        raise ValueError(
            "Common-seed manifests may be published only by the atomic eight-manifest "
            "cohort writer (v2) or the atomic versioned cohort writer for their "
            "declared stage."
        )
    if allow_atomic_common_seed_cohort_staging and not common_seed_claimed:
        raise ValueError("Atomic common-seed staging cannot publish an ordinary manifest.")
    if allow_atomic_smoke_bundle_staging and common_seed_claimed:
        raise ValueError("Atomic smoke staging cannot publish a common-seed manifest.")
    if allow_atomic_campaign_cohort_staging:
        _validate_atomic_campaign_staging_authorization(
            manifest,
            physical_path=path,
            raw_physical_path=raw_physical_path,
            logical_path=logical_path,
            raw_logical_path=raw_logical_path,
            root=root,
        )
    if allow_atomic_qualification_cohort_staging:
        _validate_atomic_qualification_staging_authorization(
            manifest,
            physical_path=path,
            raw_physical_path=raw_physical_path,
            logical_path=logical_path,
            raw_logical_path=raw_logical_path,
            root=root,
        )
    if logical_path != path and not (
        allow_atomic_common_seed_cohort_staging
        or allow_atomic_smoke_bundle_staging
        or allow_atomic_campaign_cohort_staging
        or allow_atomic_qualification_cohort_staging
    ):
        raise ValueError("A logical publication path is restricted to atomic cohort staging.")
    digest_path = path.with_suffix(path.suffix + ".sha256")
    snapshot_path = _snapshot_bundle_path(path)
    if path.exists() or digest_path.exists() or snapshot_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable manifest, digest, or snapshot bundle: {path}"
        )
    incoming_digest = manifest_digest(manifest)
    if incoming_digest != manifest.get("manifest_sha256"):
        raise ValueError("Manifest digest is inconsistent before write.")

    frozen = deepcopy(manifest)
    frozen.pop("snapshot_bundle", None)
    for job in frozen["jobs"]:
        job.pop("snapshot_bundle", None)
    if not isinstance(frozen.get("implementation_files"), dict):
        if not frozen["jobs"]:
            raise ValueError("Cannot recover implementation provenance from an empty manifest.")
        frozen.update(_provenance_copy(_provenance_from_container(frozen["jobs"][0])))
    _manifest_provenance_consistency(frozen, require_live_protocol=True)
    _verify_implementation_provenance(frozen, root)
    hash_cache: dict[Path, str] = {}
    for job in frozen["jobs"]:
        _verify_job_input_files(job, hash_cache)
        _verify_job_semantic_snapshots(job)
        if job.get("model_name") == "flux1_dev" and job.get("stage") != FLUX_COMMON_SEED_V2_STAGE:
            validate_flux1_job_v3(job, project_root=root, mode=FLUX1_V3_MODE_EXECUTION)
    _verify_stage_specific_generation_contract(frozen, root)

    bundle_published = False
    digest_published = False
    manifest_published = False
    try:
        descriptor = _materialize_snapshot_bundle(frozen, path, root)
        bundle_published = True
        if logical_path != path:
            logical_snapshot_root = _snapshot_bundle_path(logical_path)
            descriptor["root_path"] = str(logical_snapshot_root)
            descriptor["index_path"] = str(logical_snapshot_root / "index.json")
        frozen["snapshot_bundle"] = deepcopy(descriptor)
        for job in frozen["jobs"]:
            job["snapshot_bundle"] = deepcopy(descriptor)
        frozen["manifest_sha256"] = manifest_digest(frozen)
        # The manifest is the final commit marker: readers can never observe a
        # final manifest that points to an absent archive or digest sidecar.
        _write_text_exclusive_atomic(
            digest_path,
            f"{frozen['manifest_sha256']}  {path.name}\n",
        )
        digest_published = True
        _write_text_exclusive_atomic(
            path,
            json.dumps(frozen, indent=2, sort_keys=True),
        )
        manifest_published = True
    except Exception:
        if not manifest_published:
            if digest_published:
                digest_path.unlink(missing_ok=True)
            if bundle_published and snapshot_path.exists():
                shutil.rmtree(snapshot_path)
        raise
    manifest.clear()
    manifest.update(frozen)


def _read_manifest_structure(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_absolute():
        path = root / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("benchmark") != BENCHMARK_NAME:
        raise ValueError(f"Manifest benchmark mismatch in {path}: {manifest.get('benchmark')!r}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != int(manifest.get("num_jobs", -1)):
        raise ValueError(f"Manifest job count is invalid in {path}.")
    manifest_seed = int(manifest.get("seed", -1))
    _validate_generation_seed(manifest_seed)
    job_seeds = {int(job.get("seed", -1)) for job in jobs}
    if job_seeds != {manifest_seed}:
        raise ValueError(
            f"Manifest jobs must share the frozen top-level seed in {path}: "
            f"manifest={manifest_seed}, jobs={sorted(job_seeds)}."
        )
    expected = str(manifest.get("manifest_sha256", ""))
    actual = manifest_digest(manifest)
    if expected != actual:
        raise ValueError(
            f"Manifest content digest mismatch in {path}: expected {expected}, got {actual}."
        )
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if digest_path.exists():
        sidecar_digest = digest_path.read_text(encoding="utf-8").split()[0]
        if sidecar_digest != actual:
            raise ValueError(f"Manifest sidecar digest mismatch in {digest_path}.")
    output_dirs = [str(job["output_dir"]) for job in jobs]
    if len(output_dirs) != len(set(output_dirs)):
        raise ValueError(f"Manifest contains duplicate output directories: {path}")
    condition_ids = [str(job.get("condition_id")) for job in jobs]
    if len(condition_ids) != len(set(condition_ids)) or "None" in condition_ids:
        raise ValueError(f"Manifest contains missing or duplicate condition IDs: {path}")
    expected_media = sum(bool(job.get("expected_media")) for job in jobs)
    expected_not_supported = len(jobs) - expected_media
    if expected_media != int(manifest.get("expected_media_jobs", -1)):
        raise ValueError(f"Manifest expected-media count is inconsistent in {path}.")
    if expected_not_supported != int(manifest.get("expected_not_supported_jobs", -1)):
        raise ValueError(f"Manifest not-supported count is inconsistent in {path}.")
    return manifest


def read_manifest_for_audit(path: Path, root: Path) -> dict[str, Any]:
    """Read historical results from the archive even after live source repairs."""

    manifest = _read_manifest_structure(path, root)
    _manifest_provenance_consistency(manifest)
    _verify_snapshot_bundle(manifest)
    snapshot_root, snapshot_objects = _snapshot_object_paths_by_sha256(manifest)
    for job in manifest["jobs"]:
        if job.get("model_name") != "flux1_dev":
            continue
        if job.get("stage") == FLUX_COMMON_SEED_V2_STAGE:
            continue
        archived_static_inputs: dict[str, Path] = {}
        for role in FLUX1_V3_SOURCE_INPUT_ROLES:
            record = (job.get("input_files") or {}).get(role)
            if not isinstance(record, Mapping):
                raise ValueError(f"Flux-v3 audit job lacks frozen static role {role!r}.")
            digest = str(record.get("sha256", ""))
            if digest not in snapshot_objects:
                raise ValueError(
                    f"Flux-v3 audit snapshot lacks the object for static role {role!r}."
                )
            archived_static_inputs[role] = snapshot_objects[digest]
        validate_flux1_job_v3(
            job,
            project_root=root,
            mode=FLUX1_V3_MODE_AUDIT,
            audit_snapshot_root=snapshot_root,
            audit_snapshot_input_paths=archived_static_inputs,
        )
    return manifest


def _verify_stage_specific_generation_contract(
    manifest: dict[str, Any],
    root: Path,
    *,
    manifest_path: Path | None = None,
    require_cohort_commit: bool = False,
) -> None:
    """Dispatch special stages to their complete launch-time validators.

    Common-seed manifests deliberately retain the ordinary benchmark name so
    they can reuse the baseline runner.  Input-role checks alone are not their
    scientific contract, so every generic generation reader must also execute
    the exact 3-row/seed/runtime/output/protocol validator.
    """

    jobs = manifest.get("jobs")
    if isinstance(jobs, list):
        for job in jobs:
            if isinstance(job, Mapping):
                reject_reserved_flux_common_seed_outputs(job, root)
    common_seed_stage = flux_common_seed_stage_from_manifest(manifest)
    if common_seed_stage is not None:
        module = flux_common_seed_module_for_stage(common_seed_stage)
        module.validate_seed_manifest(
            manifest,
            root=root.resolve(),
            require_live_inputs=True,
            require_launchable=True,
        )
        if require_cohort_commit:
            if manifest_path is None:
                raise ValueError("Common-seed launch validation requires its manifest path.")
            module.validate_cohort_commit_for_member(
                manifest_path=manifest_path,
                manifest=manifest,
                root=root.resolve(),
            )


def read_manifest(path: Path, root: Path) -> dict[str, Any]:
    """Strict generation reader: archived and live inputs must both match."""

    resolved_path = path if path.is_absolute() else root / path
    resolved_path = resolved_path.resolve()
    manifest = read_manifest_for_audit(resolved_path, root)
    _manifest_provenance_consistency(manifest, require_live_protocol=True)
    _verify_implementation_provenance(manifest, root)
    hash_cache: dict[Path, str] = {}
    for job in manifest["jobs"]:
        _verify_job_input_files(job, hash_cache)
        _verify_job_semantic_snapshots(job)
        if job.get("model_name") == "flux1_dev" and job.get("stage") != FLUX_COMMON_SEED_V2_STAGE:
            validate_flux1_job_v3(job, project_root=root, mode=FLUX1_V3_MODE_EXECUTION)
    _verify_stage_specific_generation_contract(
        manifest,
        root,
        manifest_path=resolved_path,
        require_cohort_commit=True,
    )
    return manifest


def manifest_digest(manifest: dict[str, Any]) -> str:
    canonical = dict(manifest)
    canonical.pop("manifest_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_prompts(path: Path, root: Path) -> dict[str, dict[str, Any]]:
    data = load_yaml(path)
    if data.get("benchmark") != BENCHMARK_NAME or int(data.get("seed", -1)) != 0:
        raise ValueError(f"Prompt configuration contract mismatch: {path}")
    prompts: dict[str, dict[str, Any]] = {}
    for raw in data.get("prompts") or []:
        row = dict(raw)
        prompt_id = str(row["prompt_id"])
        if prompt_id not in PAIR_IDS_BY_PROMPT:
            raise ValueError(f"Unknown prompt ID {prompt_id!r} in {path}.")
        image_prompt, image_prompt_field = _modality_value(
            row,
            primary="image_prompt",
            fallback="prompt",
            prompt_id=prompt_id,
        )
        video_prompt, video_prompt_field = _modality_value(
            row,
            primary="video_prompt",
            fallback="prompt",
            prompt_id=prompt_id,
        )
        for model_name, field in MODEL_SPECIFIC_VIDEO_PROMPT_FIELDS.items():
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Prompt {prompt_id} must define non-empty {field!r} for {model_name}."
                )
        image_tree_value, image_tree_field = _modality_value(
            row,
            primary="image_concept_tree",
            fallback="concept_tree",
            prompt_id=prompt_id,
        )
        video_tree_value, video_tree_field = _modality_value(
            row,
            primary="video_concept_tree",
            fallback="concept_tree",
            prompt_id=prompt_id,
        )
        trees_by_task: dict[str, str] = {}
        for task, tree_value in (
            ("text_to_image", image_tree_value),
            ("text_to_video", video_tree_value),
        ):
            concept_tree = Path(str(tree_value))
            if not concept_tree.is_absolute():
                concept_tree = root / concept_tree
            hierarchy = ConceptHierarchy.from_yaml_file(concept_tree)
            expected_pair_ids = PAIR_IDS_BY_PROMPT[prompt_id]
            actual_pair_ids = tuple(pair.id for pair in hierarchy.pairs)
            if actual_pair_ids != expected_pair_ids:
                raise ValueError(
                    f"Concept pair contract mismatch for {prompt_id}/{task} in {concept_tree}: "
                    f"expected={expected_pair_ids}, actual={actual_pair_ids}."
                )
            trees_by_task[task] = str(concept_tree.resolve())
        row["prompts_by_task"] = {
            "text_to_image": str(image_prompt),
            "text_to_video": str(video_prompt),
        }
        row["prompt_fields_by_task"] = {
            "text_to_image": image_prompt_field,
            "text_to_video": video_prompt_field,
        }
        row["concept_trees_by_task"] = trees_by_task
        row["concept_tree_fields_by_task"] = {
            "text_to_image": image_tree_field,
            "text_to_video": video_tree_field,
        }
        prompts[prompt_id] = row
    if tuple(prompts) != PROMPT_IDS:
        raise ValueError(f"Prompt IDs/order mismatch in {path}: {tuple(prompts)}")
    return prompts


def _load_negative_prompts(path: Path) -> dict[str, dict[str, str]]:
    data = load_yaml(path)
    if data.get("benchmark") != BENCHMARK_NAME:
        raise ValueError(f"Negative-prompt configuration contract mismatch: {path}")
    prompts: dict[str, dict[str, str]] = {}
    for key, value in (data.get("negative_prompts") or {}).items():
        prompt_id = str(key)
        if isinstance(value, str):
            prompts[prompt_id] = {"text_to_image": value, "text_to_video": value}
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"Negative prompt {prompt_id} must be a string or modality mapping in {path}."
            )
        image = _first_present(value, "text_to_image", "image", "image_prompt")
        video = _first_present(value, "text_to_video", "video", "video_prompt")
        if image is None or video is None:
            raise ValueError(
                f"Negative prompt {prompt_id} must provide image and video values in {path}; got {value}."
            )
        prompts[prompt_id] = {"text_to_image": str(image), "text_to_video": str(video)}
    if tuple(prompts) != PROMPT_IDS:
        raise ValueError(f"Negative-prompt IDs/order mismatch in {path}: {tuple(prompts)}")
    return prompts


def _modality_value(
    row: dict[str, Any],
    *,
    primary: str,
    fallback: str,
    prompt_id: str,
) -> tuple[Any, str]:
    if primary in row and row[primary] not in (None, ""):
        return row[primary], primary
    if fallback in row and row[fallback] not in (None, ""):
        return row[fallback], fallback
    raise ValueError(
        f"Prompt {prompt_id} is missing required {primary!r} (and legacy {fallback!r})."
    )


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _selection(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return allowed
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(selected) - set(allowed))
    if not selected or unknown:
        raise ValueError(
            f"Invalid {label} selection {value!r}; unknown={unknown}, allowed={list(allowed)}"
        )
    return selected


def _require_fresh_attempt(
    output_dir: Path,
    job: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> None:
    """Accept only the exact authenticated startup files for this job class."""

    root = (root or project_root()).resolve()
    from hierasafe_flow.benchmarks.finer_detailing_smoke_launch import (
        SMOKE_LAUNCH_AUTHORIZATION_DIGEST_FILENAME,
        SMOKE_LAUNCH_AUTHORIZATION_FILENAME,
        is_canonical_smoke_job,
        require_smoke_launch_authorization_for_job,
    )
    from hierasafe_flow.benchmarks.finer_detailing_campaign_launch import (
        LAUNCH_AUTHORIZATION_DIGEST_FILENAME,
        LAUNCH_AUTHORIZATION_FILENAME,
        is_canonical_production_campaign_job,
        require_campaign_launch_authorization_for_job,
    )
    from hierasafe_flow.benchmarks.finer_detailing_qualification_launch import (
        is_fresh_qualification_job,
        require_qualification_launch_authorization_for_job,
    )

    is_smoke = is_canonical_smoke_job(job, root)
    is_campaign = is_canonical_production_campaign_job(job, root)
    is_qualification = is_fresh_qualification_job(job, root)
    if sum((is_smoke, is_campaign, is_qualification)) > 1:
        raise ValueError("Generation job belongs to multiple reserved launch classes.")
    required = (
        environment_preflight_required()
        or job.get("stage") in FLUX_COMMON_SEED_STAGES
        or is_smoke
        or is_campaign
        or is_qualification
    )
    if not output_dir.exists():
        if required:
            raise FileNotFoundError(
                f"Required environment preflight attempt directory is missing: {output_dir}"
            )
        return
    entries = {path.name for path in output_dir.iterdir()}
    if not entries:
        if required:
            raise FileNotFoundError(
                f"Required environment preflight is missing from empty attempt: {output_dir}"
            )
        return
    preflight_entries = {
        ENVIRONMENT_PREFLIGHT_FILENAME,
        ENVIRONMENT_PREFLIGHT_DIGEST_FILENAME,
    }
    expected_entries = set(preflight_entries)
    if is_smoke:
        expected_entries.update(
            {
                SMOKE_LAUNCH_AUTHORIZATION_FILENAME,
                SMOKE_LAUNCH_AUTHORIZATION_DIGEST_FILENAME,
            }
        )
    if is_campaign:
        expected_entries.update(
            {
                LAUNCH_AUTHORIZATION_FILENAME,
                LAUNCH_AUTHORIZATION_DIGEST_FILENAME,
            }
        )
    if entries != expected_entries:
        raise FileExistsError(
            f"Attempt directory does not contain the exact authenticated startup set: "
            f"{output_dir}. Use a new --attempt and immutable retry manifest."
        )
    raw_job_index = job.get("launch_manifest_job_index")
    if required and (isinstance(raw_job_index, bool) or not isinstance(raw_job_index, int)):
        raise ValueError(
            "Required environment preflight cannot be rebound without an exact launch "
            "manifest job index."
        )
    read_environment_preflight(
        output_dir,
        expected_job=job,
        expected_job_index=raw_job_index if isinstance(raw_job_index, int) else None,
    )
    if is_smoke:
        # This reopens the plan, complete lineage, exact manifest row, submission
        # registry, Slurm identity, preflight, and the full direct-runner job.
        # It therefore cannot be replaced by a handful of matching identifiers.
        require_smoke_launch_authorization_for_job(job, root=root)
    if is_campaign:
        require_campaign_launch_authorization_for_job(job, root=root)
    if is_qualification:
        require_qualification_launch_authorization_for_job(job, root=root)


def _assert_only_expected_media(output_dir: Path, expected: Path) -> None:
    media_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".gif"}
    media = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in media_suffixes
    )
    if media != [expected]:
        raise RuntimeError(
            f"Exact media contract failed under {output_dir}: expected {[expected]}, found {media}."
        )


def _assert_no_generated_media(output_dir: Path) -> None:
    media_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".gif"}
    media = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in media_suffixes
    )
    if media:
        raise RuntimeError(f"Unsupported native-negative job unexpectedly contains media: {media}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_outer_timing(output_dir: Path, started_at: str, started: float, status: str) -> None:
    write_json(
        output_dir / "experiment_timing.json",
        {
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "status": status,
        },
    )


if __name__ == "__main__":
    main()
