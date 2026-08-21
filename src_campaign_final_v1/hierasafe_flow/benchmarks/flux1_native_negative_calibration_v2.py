"""Fail-closed Flux.1 native-negative calibration-v2 manifest builder.

Unlike calibration v1, this cohort may exist only after the separate
source-only common-seed study has selected one seed.  The public builder has
no seed, prompt, scale, role, output-root, or attempt overrides: it authenticates
the sealed protocol and the complete common-seed evidence chain, then builds
one fresh 45-row manifest whose every row uses that selected seed.

The checked-in protocol is intentionally a draft and both digest pins are
``None``.  Consequently :func:`build_calibration_manifest` and immutable
publication fail before constructing a manifest until a later, explicit
preregistration change seals real evidence.  Internal pure builders exist only
to make the frozen topology testable before that sealing change.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import secrets
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hierasafe_flow.benchmarks import finer_detailing_correction as finer
from hierasafe_flow.evaluation import flux1_common_seed_v3 as common_seed
from hierasafe_flow.evaluation.flux1_dual_view_jobs_v3 import (
    CONTRACT_ID as FLUX1_V3_CONTRACT_ID,
    EXECUTION_STATUS as FLUX1_V3_EXECUTION_STATUS,
    MODE_AUDIT as FLUX1_V3_MODE_AUDIT,
    MODE_EXECUTION as FLUX1_V3_MODE_EXECUTION,
    MODE_PREVIEW as FLUX1_V3_MODE_PREVIEW,
    NEGATIVE_MODE_EXPLICIT_NONE_CONTROL,
    NEGATIVE_MODE_NOT_APPLIED,
    NEGATIVE_MODE_PAIRED_REGISTERED,
    PREVIEW_STATUS as FLUX1_V3_PREVIEW_STATUS,
    SOURCE_PROTOCOL_INPUT_ROLES as FLUX1_V3_SOURCE_INPUT_ROLES,
    expected_flux1_protocol_input_roles_v3,
    project_flux1_job_v3,
    validate_flux1_job_v3,
)
from hierasafe_flow.utils.config import load_yaml


CALIBRATION_ID = "flux1_native_negative_true_cfg_scale_v2"
CALIBRATION_STAGE = "flux1_native_negative_scale_calibration_v2"
CALIBRATION_MANIFEST_SCHEMA_VERSION = 2
CALIBRATION_ROW_SCHEMA_VERSION = 3
CONFIG_RELATIVE = Path("configs/experiments/flux1_native_negative_scale_calibration_v2.yaml")
PROTOCOL_RELATIVE = Path(
    "debugging/audits/finer_detailing_20260720/"
    "flux1_native_negative_scale_calibration_v2_PROTOCOL_DRAFT.md"
)

# Intentionally unset while CONFIG_RELATIVE/PROTOCOL_RELATIVE are drafts.
# Sealing must pin raw-file digests after the selected-seed evidence block is
# complete and before any manifest is constructed or written.
SEALED_CONFIG_SHA256: str | None = None
SEALED_PROTOCOL_SHA256: str | None = None

OUTPUT_ROOT_RELATIVE = Path("outputs/finer_detailing_flux1_native_negative_scale_calibration_v2")
V1_OUTPUT_ROOT_RELATIVE = Path(
    "outputs/finer_detailing_flux1_native_negative_scale_calibration_20260720"
)
MODEL_NAME = "flux1_dev"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
PIPELINE_CLASS = "FluxPipeline"
PROMPT_IDS = tuple(finer.PROMPT_IDS)
ATTEMPT = 2
TRUE_CFG_SCALES = (
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
)
ELIGIBLE_SCALES = TRUE_CFG_SCALES[1:]
ROLE_OFFICIAL = "official_study_baseline"
ROLE_CONTROL = "native_no_negative_control"
ROLE_LADDER = "native_negative_scale_ladder"
ROLE_ORDER = (ROLE_OFFICIAL, ROLE_CONTROL, ROLE_LADDER)
EXECUTION_PATH_OFFICIAL = "GenerationRunner_flux_dual_view"
EXECUTION_PATH_NATIVE = "FluxPipeline_flux_dual_view_native_call"
EXPECTED_ROLE_COUNTS = {ROLE_OFFICIAL: 3, ROLE_CONTROL: 3, ROLE_LADDER: 39}
EXPECTED_ROWS = 45
# PyTorch generators accept signed 64-bit non-negative seeds.  The selected
# seed is authenticated by the bound common-seed receipt and by every manifest
# row; it must not be coupled to one particular candidate-ladder cardinality.
MAX_SELECTED_COMMON_SEED = (1 << 63) - 1
IMAGE_LAYOUTS = {
    "01_sad_young_girl": {"width": 832, "height": 1216},
    "02_angry_old_man": {"width": 832, "height": 1216},
    "03_empty_outdoor_mall": {"width": 1216, "height": 832},
}
EVIDENCE_ROLES = (
    "selection_receipt",
    "selection_receipt_sidecar",
    "selection_publication_commit",
    "selection_publication_commit_sidecar",
    "authentication_report",
    "reviewer_manifest",
    "private_unblinding_map",
    "normalized_manual_review_ledger",
)
PROTOCOL_INPUT_ROLES = (
    "calibration_v2_config",
    "calibration_v2_config_sidecar",
    "calibration_v2_protocol",
    "calibration_v2_protocol_sidecar",
)
CALIBRATION_V2_INPUT_ROLES = frozenset((*PROTOCOL_INPUT_ROLES, *EVIDENCE_ROLES))
BASE_INPUT_ROLES = frozenset(
    {"base_config", "model_config", "prompt_suite", "concept_tree", "negative_prompt_config"}
)
GENERATION_INPUT_RELATIVES = {
    "base_config": Path("configs/default.yaml"),
    "model_config": Path("configs/models/t2i_flux1_dev_dual_view_v3.yaml"),
    "prompt_suite": Path("configs/experiments/finer_detailing_correction_prompts.yaml"),
    "negative_prompt_config": Path(
        "configs/experiments/finer_detailing_correction_negative_prompts.yaml"
    ),
    "concept_tree_01_sad_young_girl": Path(
        "configs/concepts/finer_detailing_01_sad_young_girl.yaml"
    ),
    "concept_tree_02_angry_old_man": Path("configs/concepts/finer_detailing_02_angry_old_man.yaml"),
    "concept_tree_03_empty_outdoor_mall": Path(
        "configs/concepts/finer_detailing_03_empty_outdoor_mall_t2i.yaml"
    ),
}
DOCUMENT_EVIDENCE_ROLES = tuple(
    role
    for role in EVIDENCE_ROLES
    if role not in {"selection_receipt_sidecar", "selection_publication_commit_sidecar"}
)
SLURM_JOB_NAME = "flux1-neg-v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def project_root() -> Path:
    return finer.project_root()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scale_label(scale: float) -> str:
    if isinstance(scale, bool) or float(scale) not in TRUE_CFG_SCALES:
        raise ValueError(f"Scale {scale!r} is not preregistered for calibration v2.")
    return f"{float(scale):.2f}".replace(".", "p")


def _selected_seed(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_SELECTED_COMMON_SEED
    ):
        raise ValueError(
            "The selected common seed must be one exact non-negative signed-64-bit integer."
        )
    return value


def _aware_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be one explicit timezone-aware timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not valid ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset.")
    return parsed


def _blind_id(
    blinding_nonce: str,
    prompt_id: str,
    role: str,
    scale: float | None,
    seed: int,
) -> str:
    if not _SHA256_RE.fullmatch(blinding_nonce):
        raise ValueError("Calibration-v2 review blinding nonce must be 32 random bytes.")
    payload = f"{CALIBRATION_ID}|{seed}|{prompt_id}|{role}|{scale!r}".encode()
    digest = hmac.new(bytes.fromhex(blinding_nonce), payload, hashlib.sha256).hexdigest()
    return f"blind_{digest[:24]}"


def _record(path: Path, *, document_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Required immutable evidence is absent or empty: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if document_sha256 is not None:
        result["document_sha256"] = document_sha256
    return result


def _authenticate_generation_inputs(
    config: Mapping[str, Any], root: Path
) -> dict[str, dict[str, str]]:
    raw = config.get("generation_inputs")
    if not isinstance(raw, Mapping) or set(raw) != set(GENERATION_INPUT_RELATIVES):
        raise ValueError("Calibration-v2 generation-input pins are incomplete.")
    records: dict[str, dict[str, str]] = {}
    for role, relative in GENERATION_INPUT_RELATIVES.items():
        configured = raw[role]
        expected = str(configured.get("sha256", "")) if isinstance(configured, Mapping) else ""
        path = (root / relative).resolve()
        if (
            not _SHA256_RE.fullmatch(expected)
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise ValueError(f"Sealed calibration-v2 generation input drifted: {role!r}.")
        records[role] = {"path": str(path), "sha256": expected}
    return records


def _validate_runtime_guard_pin(expected_config_sha256: str) -> None:
    # finer_detailing_correction imports this runtime module before importing
    # calibration v2, so this local import cannot create a cycle.
    from hierasafe_flow.cli import run_redteam_tri_condition as runtime_cli

    runtime_pin = runtime_cli._FLUX1_NEGATIVE_CALIBRATION_V2_CONFIG_SHA256
    if runtime_pin != expected_config_sha256:
        raise ValueError(
            "Calibration-v2 builder/runtime config pins differ; native execution remains blocked."
        )


def _document_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return common_seed.canonical_sha256(canonical)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return payload


def _resolve_config_record(root: Path, raw: Any, role: str) -> tuple[Path, str, str | None]:
    expected_keys = {"path", "sha256"} | (
        {"document_sha256"} if role in DOCUMENT_EVIDENCE_ROLES else set()
    )
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise ValueError(f"Selected-seed evidence {role!r} must contain {sorted(expected_keys)}.")
    relative = Path(str(raw.get("path") or ""))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Selected-seed evidence path must be project-relative: {role!r}.")
    resolved = (root / relative).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError(f"Selected-seed evidence path escapes the project root: {role!r}.")
    digest = str(raw.get("sha256") or "")
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"Selected-seed evidence raw digest is malformed: {role!r}.")
    document = raw.get("document_sha256")
    if document is not None and not _SHA256_RE.fullmatch(str(document)):
        raise ValueError(f"Selected-seed evidence document digest is malformed: {role!r}.")
    return resolved, digest, None if document is None else str(document)


def _authenticate_selected_seed_evidence(
    config: Mapping[str, Any], *, root: Path, reauthenticate_sources: bool
) -> tuple[int, dict[str, dict[str, Any]]]:
    """Authenticate the selected receipt and every document it derives from."""

    block = config.get("selected_common_seed_evidence")
    if not isinstance(block, Mapping) or set(block) != {
        "selected_common_seed",
        *EVIDENCE_ROLES,
    }:
        raise ValueError("Calibration v2 selected-common-seed evidence block is incomplete.")
    selected = _selected_seed(block["selected_common_seed"])
    records: dict[str, dict[str, Any]] = {}
    documents: dict[str, dict[str, Any]] = {}
    for role in EVIDENCE_ROLES:
        path, expected_raw, expected_document = _resolve_config_record(root, block[role], role)
        actual_raw = sha256_file(path) if path.is_file() else ""
        if actual_raw != expected_raw:
            raise ValueError(
                f"Selected-seed evidence {role!r} changed: expected={expected_raw}, actual={actual_raw}."
            )
        if expected_document is None:
            records[role] = _record(path)
            continue
        payload = _load_json(path, role)
        if payload.get("document_sha256") != expected_document:
            raise ValueError(f"Selected-seed evidence {role!r} claims another document digest.")
        if _document_digest(payload) != expected_document:
            raise ValueError(f"Selected-seed evidence {role!r} canonical digest is invalid.")
        records[role] = _record(path, document_sha256=expected_document)
        documents[role] = payload

    receipt_path = Path(records["selection_receipt"]["path"])
    sidecar_path = Path(records["selection_receipt_sidecar"]["path"])
    if sidecar_path != receipt_path.with_suffix(receipt_path.suffix + ".sha256"):
        raise ValueError("Common-seed selection receipt sidecar path is not canonical.")
    if sidecar_path.read_text(encoding="utf-8").split() != [
        records["selection_receipt"]["sha256"],
        receipt_path.name,
    ]:
        raise ValueError("Common-seed selection receipt raw-file sidecar is inconsistent.")
    publication_commit_path = Path(records["selection_publication_commit"]["path"])
    publication_commit_sidecar = Path(records["selection_publication_commit_sidecar"]["path"])
    if publication_commit_path != common_seed.selection_publication_commit_path(receipt_path):
        raise ValueError("Common-seed selection publication commit path is not canonical.")
    if publication_commit_sidecar != publication_commit_path.with_suffix(
        publication_commit_path.suffix + ".sha256"
    ):
        raise ValueError("Common-seed selection publication-commit sidecar path is not canonical.")
    if publication_commit_sidecar.read_text(encoding="utf-8").split() != [
        records["selection_publication_commit"]["sha256"],
        publication_commit_path.name,
    ]:
        raise ValueError("Common-seed selection publication-commit sidecar is inconsistent.")

    authentication = documents["authentication_report"]
    reviewer = documents["reviewer_manifest"]
    unblinding = documents["private_unblinding_map"]
    manual = documents["normalized_manual_review_ledger"]
    receipt = documents["selection_receipt"]
    common_seed.validate_authentication_report(authentication)
    # Public validators are deliberately used when available; these are part
    # of the common-seed stage's final evidence API, not duplicated heuristics.
    validate_reviewer = getattr(common_seed, "validate_reviewer_manifest", None)
    if callable(validate_reviewer):
        validate_reviewer(reviewer)
    else:
        common_seed._validate_reviewer_manifest(reviewer)
    validate_unblinding = getattr(common_seed, "validate_unblinding_map", None)
    if callable(validate_unblinding):
        validate_unblinding(unblinding, reviewer, authentication)
    else:
        common_seed._validate_unblinding_map(unblinding, reviewer, authentication)
    validate_ledger = getattr(common_seed, "validate_normalized_manual_ledger", None)
    if callable(validate_ledger):
        validate_ledger(manual, reviewer)
    else:
        normalized = common_seed.normalize_manual_ledger(manual, reviewer)
        if normalized != manual:
            raise ValueError("Bound common-seed manual ledger is not the normalized ledger.")
    if reauthenticate_sources:
        common_seed.reauthenticate_authentication_report(authentication, root=root)

    # The public reader is the canonical receipt API.  It reopens the raw-file
    # sidecar, validates the selected/minimum-seed arithmetic, the immutable
    # normalized-ledger binding, and all 24 exact PNG copies actually reviewed.
    reopened = common_seed.read_selection_report(
        receipt_path,
        reviewer_manifest=reviewer,
        verify_files=True,
    )
    if reopened != receipt:
        raise ValueError("Common-seed selection receipt changed between reads.")
    # A receipt can be internally self-consistent yet lie about which seed the
    # bound manual ledger actually selected.  Recompute the substantive
    # decision from all four authenticated source documents; schema/digest
    # validation of the receipt alone is intentionally insufficient here.
    common_seed.validate_selection_report_against_evidence(
        receipt,
        authentication=authentication,
        reviewer_manifest=reviewer,
        unblinding_map=unblinding,
        normalized_manual_ledger=manual,
        verify_files=True,
    )

    if (
        receipt.get("selection") != common_seed.SELECTION_NAME
        or receipt.get("status") != "selected"
        or receipt.get("selected_common_seed") != selected
        or receipt.get("document_sha256") != records["selection_receipt"]["document_sha256"]
    ):
        raise ValueError("Common-seed selection receipt is not one exact selected-seed decision.")
    source_digests = receipt.get("source_document_sha256") or {}
    expected_sources = {
        "authentication": records["authentication_report"]["document_sha256"],
        "reviewer_manifest": records["reviewer_manifest"]["document_sha256"],
        "private_unblinding_map": records["private_unblinding_map"]["document_sha256"],
        "manual_review_ledger": records["normalized_manual_review_ledger"]["document_sha256"],
        "reviewed_media": receipt["reviewed_media"]["document_sha256"],
    }
    if source_digests != expected_sources:
        raise ValueError("Selection receipt does not bind the exact source evidence documents.")
    if receipt.get("normalized_manual_review_ledger") != records["normalized_manual_review_ledger"]:
        raise ValueError("Selection receipt binds another normalized manual-review ledger.")
    later = receipt.get("later_calibration_v2_binding") or {}
    if later != {
        "calibration_id": CALIBRATION_ID,
        "required_selected_common_seed": selected,
        "required_rows": EXPECTED_ROWS,
        "required_prompt_ids": list(PROMPT_IDS),
        "every_row_must_use_selected_common_seed": True,
        "v1_scale_outputs_may_be_reused": False,
        "scale_selection_rule_may_be_relaxed": False,
        "launch_authorized": True,
    }:
        raise ValueError("Selection receipt does not authorize the exact calibration-v2 cohort.")
    return selected, records


def _flux1_v3_execution_inputs_from_authentication(
    authentication: Mapping[str, Any], *, root: Path
) -> dict[str, dict[str, str]]:
    """Extract the exact shared execution route from authenticated v3 evidence.

    Calibration v2 must consume the model/source/equivalence bindings already
    used by the selected common-seed-v3 cohort.  It may not assemble a second
    route from live files or silently fall back to the legacy Flux adapter.
    """

    source_bindings = authentication.get("source_bindings")
    available = (
        source_bindings.get("protocol_inputs")
        if isinstance(source_bindings, Mapping)
        else None
    )
    required = expected_flux1_protocol_input_roles_v3(FLUX1_V3_MODE_EXECUTION)
    if not isinstance(available, Mapping) or not required <= set(available):
        missing = sorted(required - set(available or {}))
        raise ValueError(
            "Selected common-seed-v3 evidence omits shared Flux execution inputs: "
            f"{missing}."
        )
    extracted: dict[str, dict[str, str]] = {}
    for role in sorted(required):
        record = available[role]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or not Path(str(record.get("path", ""))).is_absolute()
            or not _SHA256_RE.fullmatch(str(record.get("sha256", "")))
        ):
            raise ValueError(f"Common-seed-v3 Flux execution binding is malformed: {role!r}.")
        extracted[role] = {
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
        }
    expected_model = (root.resolve() / GENERATION_INPUT_RELATIVES["model_config"]).resolve()
    if Path(extracted["model_config"]["path"]) != expected_model:
        raise ValueError("Calibration v2 common-seed evidence binds a legacy model config.")
    adapter_binding = available.get("flux_dual_view_adapter")
    expected_adapter = (root.resolve() / common_seed.V3_ADAPTER_SOURCE_RELATIVE).resolve()
    if (
        not isinstance(adapter_binding, Mapping)
        or set(adapter_binding) != {"path", "sha256"}
        or Path(str(adapter_binding.get("path", ""))) != expected_adapter
        or adapter_binding.get("sha256") != sha256_file(expected_adapter)
    ):
        raise ValueError(
            "Selected common-seed-v3 evidence does not bind the exact dual-view adapter."
        )
    return extracted


def _validate_static_config(config: Mapping[str, Any]) -> None:
    expected_top_level = {
        "schema_version",
        "calibration_id",
        "benchmark",
        "status",
        "preregistered_at",
        "selected_common_seed_evidence",
        "generation_inputs",
        "scope",
        "output_root",
        "review_blinding",
        "roles",
        "expected_rows",
        "evaluation_contract",
        "selection_rule",
        "reuse_prohibitions",
    }
    if set(config) != expected_top_level:
        raise ValueError("Calibration-v2 config top-level schema drifted.")
    expected_scope = {
        "model_name": MODEL_NAME,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pipeline_class": PIPELINE_CLASS,
        "prompt_ids": list(PROMPT_IDS),
        "seed_rule": "exact_selected_common_seed_from_authenticated_receipt",
        "attempt": ATTEMPT,
        "num_inference_steps": 28,
        "embedded_guidance_scale": 3.5,
        "num_outputs_per_prompt": 1,
        "image_layout_by_prompt": IMAGE_LAYOUTS,
    }
    if (
        config.get("schema_version") != 1
        or config.get("calibration_id") != CALIBRATION_ID
        or config.get("benchmark") != finer.BENCHMARK_NAME
        or config.get("scope") != expected_scope
        or config.get("status")
        not in {"implementation_draft_do_not_generate", "preregistered_before_generation"}
    ):
        raise ValueError("Calibration-v2 config identity or fixed scope drifted.")
    if Path(str(config.get("output_root", ""))) != OUTPUT_ROOT_RELATIVE:
        raise ValueError("Calibration-v2 output root drifted.")
    generation_inputs = config.get("generation_inputs")
    if not isinstance(generation_inputs, Mapping) or set(generation_inputs) != set(
        GENERATION_INPUT_RELATIVES
    ):
        raise ValueError("Calibration-v2 generation-input role coverage drifted.")
    draft = config.get("status") == "implementation_draft_do_not_generate"
    for role, relative in GENERATION_INPUT_RELATIVES.items():
        record = generation_inputs[role]
        digest = record.get("sha256") if isinstance(record, Mapping) else None
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or Path(str(record.get("path", ""))) != relative
            or (digest is not None and not _SHA256_RE.fullmatch(str(digest)))
            or (not draft and not _SHA256_RE.fullmatch(str(digest or "")))
        ):
            raise ValueError(f"Calibration-v2 generation input {role!r} is not frozen.")
    if config.get("review_blinding") != {
        "nonce_generation": "secrets_token_hex_32_once_at_manifest_freeze",
        "blind_id_derivation": "hmac_sha256_nonce_over_seed_prompt_role_and_scale",
        "nonce_excluded_from_reviewer_package": True,
        "ladder_role_groups_excluded_from_reviewer_package": True,
        "source_and_parity_reviewers_access_disjoint": True,
        "objective_and_unblinding_evidence_kept_in_separate_private_root": True,
    }:
        raise ValueError("Calibration-v2 reviewer-blinding policy drifted.")
    roles = config.get("roles")
    expected_roles = {
        ROLE_OFFICIAL: {
            "count_per_prompt": 1,
            "execution_path": EXECUTION_PATH_OFFICIAL,
            "negative_prompt_applied": False,
            "eligible_for_scale_selection": False,
        },
        ROLE_CONTROL: {
            "count_per_prompt": 1,
            "execution_path": EXECUTION_PATH_NATIVE,
            "negative_prompt": None,
            "true_cfg_scale": 1.0,
            "eligible_for_scale_selection": False,
        },
        ROLE_LADDER: {
            "count_per_prompt": 13,
            "execution_path": EXECUTION_PATH_NATIVE,
            "negative_prompt": "registered_paired_clip_t5_views",
            "true_cfg_scales": list(TRUE_CFG_SCALES),
            "ineligible_scales": [1.0],
        },
    }
    if not isinstance(roles, Mapping) or tuple(roles) != ROLE_ORDER or roles != expected_roles:
        raise ValueError("Calibration-v2 role order or semantics drifted.")
    if config.get("expected_rows") != {**EXPECTED_ROLE_COUNTS, "total": EXPECTED_ROWS}:
        raise ValueError("Calibration-v2 exact row counts drifted.")
    if config.get("evaluation_contract") != {
        "identical_to_v1_except_selected_common_seed_and_v2_provenance": True,
        "native_call_scale_integrity": (
            "exact_executed_guidance_scale_and_true_cfg_scale"
        ),
        "inert_negative_integrity": "exact_decoded_rgb_equality",
        "path_parity": (
            "both_controls_pass_full_source_fidelity_and_no_material_manual_difference"
        ),
        "objective_noncollapse_thresholds_may_change": False,
        "positive_target_concepts_must_not_be_scored": True,
        "minimum_unambiguous_suppressed_source_concepts_per_prompt": 2,
    }:
        raise ValueError("Calibration-v2 frozen v1 evaluation policy drifted.")
    if config.get("selection_rule") != {
        "eligible_scales": list(ELIGIBLE_SCALES),
        "cross_prompt_requirement": ("one_global_scale_must_pass_every_gate_on_all_three_prompts"),
        "choice": "smallest_eligible_scale",
        "per_prompt_tuning_forbidden": True,
        "averaging_across_prompts_forbidden": True,
        "threshold_relaxation_after_viewing_forbidden": True,
        "no_passing_scale_outcome": ("mark_flux1_native_negative_not_qualified_and_do_not_select"),
    }:
        raise ValueError("Calibration-v2 global scale-selection rule drifted.")
    if config.get("reuse_prohibitions") != {
        "v1_manifest_or_media_may_be_reused": False,
        "common_seed_candidate_media_may_be_reused": False,
        "every_v2_row_must_have_a_fresh_output_path": True,
        "every_v2_attempt_requires_authenticated_environment_preflight": True,
        "every_result_must_bind_exact_manifest_index": True,
    }:
        raise ValueError("Calibration-v2 reuse/preflight prohibitions drifted.")


def _load_launch_contract(
    root: Path, *, reauthenticate_sources: bool = True
) -> tuple[
    dict[str, Any],
    int,
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    """Load only a fully preregistered protocol; drafts never reach evidence parsing."""

    root = root.resolve()
    config_path = (root / CONFIG_RELATIVE).resolve()
    protocol_path = (root / PROTOCOL_RELATIVE).resolve()
    for path in (config_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(f"Calibration-v2 protocol input is absent: {path}")
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("Calibration-v2 YAML must contain one mapping.")
    _validate_static_config(config)
    if config.get("status") != "preregistered_before_generation" or not config.get(
        "preregistered_at"
    ):
        raise ValueError("Calibration-v2 manifest publication is blocked: config is a draft.")
    preregistered_at = _aware_timestamp(
        config["preregistered_at"], "Calibration-v2 preregistered_at"
    )
    if preregistered_at.astimezone(timezone.utc) > datetime.now(timezone.utc):
        raise ValueError("Calibration-v2 preregistration timestamp is in the future.")
    protocol_text = protocol_path.read_text(encoding="utf-8")
    if "Status: `preregistered_before_generation`." not in protocol_text or re.search(
        r"\bdraft\b|do not generate|not a preregistration", protocol_text, re.I
    ):
        raise ValueError("Calibration-v2 protocol text is not a sealed preregistration.")
    pins = (SEALED_CONFIG_SHA256, SEALED_PROTOCOL_SHA256)
    if any(pin is None or not _SHA256_RE.fullmatch(str(pin)) for pin in pins):
        raise ValueError(
            "Calibration-v2 publication is blocked until both sealed digests are pinned."
        )
    _validate_runtime_guard_pin(str(SEALED_CONFIG_SHA256))
    inputs: dict[str, dict[str, Any]] = {}
    for role, path, expected in (
        ("calibration_v2_config", config_path, str(SEALED_CONFIG_SHA256)),
        ("calibration_v2_protocol", protocol_path, str(SEALED_PROTOCOL_SHA256)),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Sealed calibration-v2 {role} digest drifted.")
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split() != [
            expected,
            path.name,
        ]:
            raise ValueError(f"Sealed calibration-v2 sidecar is absent/inconsistent: {sidecar}")
        inputs[role] = _record(path)
        inputs[f"{role}_sidecar"] = _record(sidecar)
    selected, evidence = _authenticate_selected_seed_evidence(
        config, root=root, reauthenticate_sources=reauthenticate_sources
    )
    generation_inputs = _authenticate_generation_inputs(config, root)
    authentication = _load_json(
        Path(evidence["authentication_report"]["path"]),
        "selected common-seed-v3 authentication report",
    )
    flux1_v3_inputs = _flux1_v3_execution_inputs_from_authentication(
        authentication,
        root=root,
    )
    return config, selected, inputs, evidence, generation_inputs, flux1_v3_inputs


def _baseline_jobs(
    root: Path,
    output_root: Path,
    selected_seed: int,
    *,
    flux1_route_mode: str,
    flux1_protocol_inputs: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, Any]]:
    args = finer.build_parser().parse_args(
        [
            "--models",
            MODEL_NAME,
            "--prompt-ids",
            ",".join(PROMPT_IDS),
            "--variations",
            "baseline",
            "--output-root",
            str(output_root),
            "--attempt",
            str(ATTEMPT),
            "--seed",
            str(selected_seed),
            "--no-tensorboard",
            "--no-save-latents",
            "--no-save-traces",
        ]
    )
    source = finer.build_manifest(
        args,
        root,
        flux1_route_context=flux1_route_mode,
        flux1_protocol_inputs=flux1_protocol_inputs,
    )
    jobs = {str(job["prompt_id"]): job for job in source["jobs"]}
    if tuple(jobs) != PROMPT_IDS or len(jobs) != len(PROMPT_IDS):
        raise RuntimeError("Baseline builder did not return the exact three prompt rows.")
    return jobs


def _generation_inputs_from_jobs(
    source_jobs: Mapping[str, Mapping[str, Any]], root: Path
) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    shared_roles = ("base_config", "model_config", "prompt_suite", "negative_prompt_config")
    for role in shared_roles:
        values = [dict(source_jobs[prompt_id]["input_files"][role]) for prompt_id in PROMPT_IDS]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Baseline builder returned inconsistent {role!r} bindings.")
        records[role] = values[0]
    for prompt_id in PROMPT_IDS:
        role = f"concept_tree_{prompt_id}"
        records[role] = dict(source_jobs[prompt_id]["input_files"]["concept_tree"])
    if set(records) != set(GENERATION_INPUT_RELATIVES):
        raise ValueError("Baseline builder generation-input role coverage drifted.")
    for role, relative in GENERATION_INPUT_RELATIVES.items():
        record = records[role]
        if (
            set(record) != {"path", "sha256"}
            or Path(str(record["path"])).resolve() != (root / relative).resolve()
            or not _SHA256_RE.fullmatch(str(record["sha256"]))
        ):
            raise ValueError(f"Baseline builder rebound canonical generation input {role!r}.")
    return records


def _native_options(role: str, scale: float, config_sha256: str) -> dict[str, Any]:
    if role not in {ROLE_CONTROL, ROLE_LADDER}:
        raise ValueError(f"Native options are forbidden for calibration role {role!r}.")
    return {
        "true_cfg_scale": float(scale),
        "calibration_negative_prompt_mode": (
            NEGATIVE_MODE_EXPLICIT_NONE_CONTROL
            if role == ROLE_CONTROL
            else NEGATIVE_MODE_PAIRED_REGISTERED
        ),
        "calibration_id": CALIBRATION_ID,
        "calibration_config_sha256": config_sha256,
        "calibration_role": role,
    }


def _row_path(output_root: Path, prompt_id: str, role: str, scale: float | None, seed: int) -> Path:
    base = output_root / prompt_id / MODEL_NAME
    if role == ROLE_OFFICIAL:
        branch = base / "01_official_study_baseline"
    elif role == ROLE_CONTROL:
        branch = base / "02_native_no_negative_control" / "true_cfg_scale_1p00"
    elif role == ROLE_LADDER and scale is not None:
        branch = base / "03_native_negative_scale_ladder" / f"true_cfg_scale_{_scale_label(scale)}"
    else:
        raise ValueError(f"Invalid calibration-v2 path role={role!r}, scale={scale!r}.")
    return branch / f"selected_common_seed_{seed:08d}" / "attempts" / f"attempt_{ATTEMPT:03d}"


def _row_id(prompt_id: str, role: str, scale: float | None, seed: int) -> str:
    suffix = role if scale is None else f"{role}__true_cfg_scale_{_scale_label(scale)}"
    return f"{prompt_id}__{MODEL_NAME}__calibration_v2__{suffix}__seed_{seed:08d}"


def _make_row(
    source_job: Mapping[str, Any],
    *,
    root: Path,
    output_root: Path,
    selected_seed: int,
    blinding_nonce: str,
    protocol_inputs: Mapping[str, Mapping[str, Any]],
    selection_evidence: Mapping[str, Mapping[str, Any]],
    flux1_protocol_inputs: Mapping[str, Mapping[str, str]],
    flux1_route_mode: str,
    role: str,
    scale: float | None,
) -> dict[str, Any]:
    job = deepcopy(dict(source_job))
    prompt_id = str(job["prompt_id"])
    config_sha = str(protocol_inputs["calibration_v2_config"]["sha256"])
    output_dir = _row_path(output_root, prompt_id, role, scale, selected_seed)
    bindings = {
        "selected_common_seed": selected_seed,
        "protocol_inputs": deepcopy(dict(protocol_inputs)),
        "selection_evidence": deepcopy(dict(selection_evidence)),
    }
    job.update(
        {
            "schema_version": 4,
            "benchmark": CALIBRATION_ID,
            "stage": CALIBRATION_STAGE,
            "attempt": ATTEMPT,
            "variation": role,
            "variant": role if scale is None else f"{role}__{_scale_label(scale)}",
            "condition_id": _row_id(prompt_id, role, scale, selected_seed),
            "seed": selected_seed,
            "seed_scoped_output": False,
            "variation_dir": str(output_dir.parents[2]),
            "output_dir": str(output_dir),
            "expected_media": True,
            "calibration_v2_bindings": bindings,
            "calibration_row": {
                "schema_version": CALIBRATION_ROW_SCHEMA_VERSION,
                "calibration_id": CALIBRATION_ID,
                "role": role,
                "execution_path": (
                    EXECUTION_PATH_OFFICIAL if role == ROLE_OFFICIAL else EXECUTION_PATH_NATIVE
                ),
                "prompt_id": prompt_id,
                "selected_common_seed": selected_seed,
                "true_cfg_scale": scale,
                "negative_prompt_mode": (
                    NEGATIVE_MODE_NOT_APPLIED
                    if role == ROLE_OFFICIAL
                    else (
                        NEGATIVE_MODE_EXPLICIT_NONE_CONTROL
                        if role == ROLE_CONTROL
                        else NEGATIVE_MODE_PAIRED_REGISTERED
                    )
                ),
                "eligible_for_scale_selection": role == ROLE_LADDER and scale in ELIGIBLE_SCALES,
                "blind_id": _blind_id(blinding_nonce, prompt_id, role, scale, selected_seed),
                "fresh_v2_media_required": True,
            },
        }
    )
    # These exact ten extra roles are intentionally present in input_files so
    # the shared content-addressed manifest snapshot archives the complete
    # preregistration and selected-seed evidence chain.  Central validation
    # admits them only for CALIBRATION_STAGE/CALIBRATION_ID.
    extra_inputs = {
        role: {"path": str(record["path"]), "sha256": str(record["sha256"])}
        for role, record in {**protocol_inputs, **selection_evidence}.items()
    }
    if set(extra_inputs) != CALIBRATION_V2_INPUT_ROLES:
        raise ValueError("Calibration-v2 job input evidence role coverage is not exact.")
    job["input_files"].update(extra_inputs)
    if role == ROLE_OFFICIAL:
        job["variant_spec"] = {"kind": "baseline"}
        job["native_negative_prompt_options"] = {}
        negative_mode = NEGATIVE_MODE_NOT_APPLIED
        native_pipeline_execution = False
    else:
        assert scale is not None
        options = _native_options(role, scale, config_sha)
        job["variant_spec"] = {
            "kind": "native_negative_prompt",
            "capability": "supported",
            "native_negative_prompt_options": deepcopy(options),
        }
        job["native_negative_prompt_options"] = deepcopy(options)
        negative_mode = (
            NEGATIVE_MODE_EXPLICIT_NONE_CONTROL
            if role == ROLE_CONTROL
            else NEGATIVE_MODE_PAIRED_REGISTERED
        )
        native_pipeline_execution = True
    return project_flux1_job_v3(
        job,
        project_root=root,
        protocol_inputs=flux1_protocol_inputs,
        mode=flux1_route_mode,
        negative_mode=negative_mode,
        native_pipeline_execution=native_pipeline_execution,
    )


def _build_manifest_from_contract(
    *,
    root: Path,
    selected_seed: int,
    protocol_inputs: Mapping[str, Mapping[str, Any]],
    selection_evidence: Mapping[str, Mapping[str, Any]],
    generation_inputs: Mapping[str, Mapping[str, str]] | None = None,
    flux1_v3_protocol_inputs: Mapping[str, Mapping[str, str]] | None = None,
    flux1_route_mode: str = FLUX1_V3_MODE_PREVIEW,
    blinding_nonce: str | None = None,
) -> dict[str, Any]:
    """Pure topology builder used after, never instead of, launch authentication."""

    root = root.resolve()
    selected_seed = _selected_seed(selected_seed)
    blinding_nonce = blinding_nonce or secrets.token_hex(32)
    if not _SHA256_RE.fullmatch(blinding_nonce):
        raise ValueError("Calibration-v2 blinding nonce must encode exactly 32 random bytes.")
    output_root = (root / OUTPUT_ROOT_RELATIVE).resolve()
    if output_root == (root / V1_OUTPUT_ROOT_RELATIVE).resolve():
        raise ValueError("Calibration v2 must never share the v1 output root.")
    if flux1_route_mode not in {FLUX1_V3_MODE_PREVIEW, FLUX1_V3_MODE_EXECUTION}:
        raise ValueError(f"Unknown calibration-v2 Flux route mode {flux1_route_mode!r}.")
    if flux1_route_mode == FLUX1_V3_MODE_EXECUTION and flux1_v3_protocol_inputs is None:
        raise ValueError("Calibration-v2 execution projection requires authenticated v3 inputs.")
    source_jobs = _baseline_jobs(
        root,
        output_root,
        selected_seed,
        flux1_route_mode=flux1_route_mode,
        flux1_protocol_inputs=flux1_v3_protocol_inputs,
    )
    first_route = source_jobs[PROMPT_IDS[0]]["flux1_dual_view_route_v3"]
    bound_flux1_inputs = deepcopy(dict(first_route["protocol_inputs"]))
    if any(
        job["flux1_dual_view_route_v3"]["protocol_inputs"] != bound_flux1_inputs
        for job in source_jobs.values()
    ):
        raise ValueError("Calibration-v2 source rows disagree on shared Flux-v3 inputs.")
    if (
        flux1_v3_protocol_inputs is not None
        and dict(flux1_v3_protocol_inputs) != bound_flux1_inputs
    ):
        raise ValueError("Calibration-v2 projector rebound authenticated Flux-v3 inputs.")
    derived_generation_inputs = _generation_inputs_from_jobs(source_jobs, root)
    if generation_inputs is not None and dict(generation_inputs) != derived_generation_inputs:
        raise ValueError("Sealed generation-input pins differ from the canonical baseline jobs.")
    generation_inputs = derived_generation_inputs
    jobs: list[dict[str, Any]] = []
    for prompt_id in PROMPT_IDS:
        source = source_jobs[prompt_id]
        jobs.append(
            _make_row(
                source,
                root=root,
                output_root=output_root,
                selected_seed=selected_seed,
                blinding_nonce=blinding_nonce,
                protocol_inputs=protocol_inputs,
                selection_evidence=selection_evidence,
                flux1_protocol_inputs=bound_flux1_inputs,
                flux1_route_mode=flux1_route_mode,
                role=ROLE_OFFICIAL,
                scale=None,
            )
        )
        jobs.append(
            _make_row(
                source,
                root=root,
                output_root=output_root,
                selected_seed=selected_seed,
                blinding_nonce=blinding_nonce,
                protocol_inputs=protocol_inputs,
                selection_evidence=selection_evidence,
                flux1_protocol_inputs=bound_flux1_inputs,
                flux1_route_mode=flux1_route_mode,
                role=ROLE_CONTROL,
                scale=1.0,
            )
        )
        jobs.extend(
            _make_row(
                source,
                root=root,
                output_root=output_root,
                selected_seed=selected_seed,
                blinding_nonce=blinding_nonce,
                protocol_inputs=protocol_inputs,
                selection_evidence=selection_evidence,
                flux1_protocol_inputs=bound_flux1_inputs,
                flux1_route_mode=flux1_route_mode,
                role=ROLE_LADDER,
                scale=scale,
            )
            for scale in TRUE_CFG_SCALES
        )
    implementation = finer._provenance_from_container(jobs[0])
    route_status = (
        FLUX1_V3_PREVIEW_STATUS
        if flux1_route_mode == FLUX1_V3_MODE_PREVIEW
        else FLUX1_V3_EXECUTION_STATUS
    )
    manifest: dict[str, Any] = {
        "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
        "benchmark": CALIBRATION_ID,
        "calibration_id": CALIBRATION_ID,
        "stage": CALIBRATION_STAGE,
        "status": (
            "implementation_preview_not_launchable"
            if flux1_route_mode == FLUX1_V3_MODE_PREVIEW
            else "frozen_before_generation"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_common_seed": selected_seed,
        "review_blinding_nonce": blinding_nonce,
        "attempt": ATTEMPT,
        "model_name": MODEL_NAME,
        "prompt_ids": list(PROMPT_IDS),
        "true_cfg_scales": list(TRUE_CFG_SCALES),
        "eligible_scales": list(ELIGIBLE_SCALES),
        "output_root": str(output_root),
        "num_jobs": EXPECTED_ROWS,
        "expected_media_jobs": EXPECTED_ROWS,
        "expected_not_supported_jobs": 0,
        "role_counts": deepcopy(EXPECTED_ROLE_COUNTS),
        "protocol_inputs": deepcopy(dict(protocol_inputs)),
        "selection_evidence": deepcopy(dict(selection_evidence)),
        "generation_inputs": deepcopy(dict(generation_inputs)),
        "flux1_dual_view_route_v3": {
            "schema_version": 1,
            "contract_id": FLUX1_V3_CONTRACT_ID,
            "mode": flux1_route_mode,
            "status": route_status,
            "protocol_inputs": deepcopy(bound_flux1_inputs),
        },
        "requires_environment_preflight": True,
        "requires_exact_manifest_job_index": True,
        "v1_or_common_seed_media_reuse_allowed": False,
        **finer._provenance_copy(implementation),
        "jobs": jobs,
    }
    manifest["manifest_sha256"] = finer.manifest_digest(manifest)
    validate_calibration_manifest(manifest, root=root, require_live_inputs=False)
    return manifest


def build_calibration_manifest(root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    (
        _config,
        selected,
        inputs,
        evidence,
        generation_inputs,
        flux1_v3_inputs,
    ) = _load_launch_contract(root)
    manifest = _build_manifest_from_contract(
        root=root,
        selected_seed=selected,
        protocol_inputs=inputs,
        selection_evidence=evidence,
        generation_inputs=generation_inputs,
        flux1_v3_protocol_inputs=flux1_v3_inputs,
        flux1_route_mode=FLUX1_V3_MODE_EXECUTION,
    )
    occupied = [job["output_dir"] for job in manifest["jobs"] if Path(job["output_dir"]).exists()]
    if occupied:
        raise FileExistsError(
            "Calibration-v2 manifest construction found pre-existing attempt paths; "
            f"a new preregistered attempt is required: {occupied[:3]}"
        )
    validate_calibration_manifest(manifest, root=root, require_live_inputs=True)
    return manifest


def _expected_rows(output_root: Path, seed: int) -> list[tuple[str, str, float | None, Path]]:
    return [
        (prompt_id, role, scale, _row_path(output_root, prompt_id, role, scale, seed))
        for prompt_id in PROMPT_IDS
        for role, scales in (
            (ROLE_OFFICIAL, (None,)),
            (ROLE_CONTROL, (1.0,)),
            (ROLE_LADDER, TRUE_CFG_SCALES),
        )
        for scale in scales
    ]


def _validate_bound_files(
    records: Mapping[str, Any],
    expected_roles: tuple[str, ...],
    *,
    snapshot_bundle: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(records, Mapping) or set(records) != set(expected_roles):
        raise ValueError(f"Immutable evidence role coverage drifted: {tuple(records)}.")
    for role, record in records.items():
        required = {"path", "sha256", "size_bytes"} | (
            {"document_sha256"} if role in DOCUMENT_EVIDENCE_ROLES else set()
        )
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValueError(f"Immutable evidence record is malformed: {role!r}.")
        path = Path(str(record["path"])).resolve()
        size = record["size_bytes"]
        if (
            not Path(str(record["path"])).is_absolute()
            or not _SHA256_RE.fullmatch(str(record["sha256"]))
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or (
                role in DOCUMENT_EVIDENCE_ROLES
                and not _SHA256_RE.fullmatch(str(record["document_sha256"]))
            )
        ):
            raise ValueError(f"Immutable evidence record values are malformed: {role!r}.")
        live_matches = (
            path.is_file() and path.stat().st_size == size and sha256_file(path) == record["sha256"]
        )
        authenticated_path = path
        if isinstance(snapshot_bundle, Mapping):
            snapshot_root = Path(str(snapshot_bundle.get("root_path", ""))).resolve()
            index_path = Path(str(snapshot_bundle.get("index_path", ""))).resolve()
            index = _load_json(index_path, "calibration-v2 snapshot index")
            frozen = (index.get("input_files") or {}).get(str(path))
            digest = str(record["sha256"])
            object_record = (index.get("objects") or {}).get(digest)
            if not isinstance(frozen, Mapping) or frozen.get("sha256") != digest:
                raise ValueError(f"Snapshot does not archive calibration-v2 evidence {role!r}.")
            if role not in set(frozen.get("roles") or ()):
                raise ValueError(f"Snapshot evidence role is missing for {role!r}.")
            if not isinstance(object_record, Mapping):
                raise ValueError(
                    f"Snapshot object is missing for calibration-v2 evidence {role!r}."
                )
            archived = (snapshot_root / str(object_record.get("path", ""))).resolve()
            if (
                snapshot_root not in archived.parents
                or not archived.is_file()
                or archived.stat().st_size != size
                or sha256_file(archived) != digest
            ):
                raise ValueError(f"Snapshot object drifted for calibration-v2 evidence {role!r}.")
            authenticated_path = archived
        elif not live_matches:
            raise ValueError(f"Immutable evidence disappeared or drifted: {role!r}.")
        if role in DOCUMENT_EVIDENCE_ROLES:
            payload = _load_json(authenticated_path, f"archived calibration-v2 {role}")
            expected_document = str(record["document_sha256"])
            if (
                payload.get("document_sha256") != expected_document
                or _document_digest(payload) != expected_document
            ):
                raise ValueError(
                    f"Canonical document digest drifted for calibration-v2 evidence {role!r}."
                )


def validate_calibration_manifest(
    manifest: Mapping[str, Any], *, root: Path | None = None, require_live_inputs: bool
) -> None:
    root = (root or project_root()).resolve()
    seed = _selected_seed(manifest.get("selected_common_seed"))
    blinding_nonce = str(manifest.get("review_blinding_nonce", ""))
    if not _SHA256_RE.fullmatch(blinding_nonce):
        raise ValueError("Calibration-v2 manifest review-blinding nonce is malformed.")
    protocol_inputs = manifest.get("protocol_inputs")
    selection_evidence = manifest.get("selection_evidence")
    generation_inputs = manifest.get("generation_inputs")
    flux1_route = manifest.get("flux1_dual_view_route_v3")
    if not isinstance(flux1_route, Mapping) or set(flux1_route) != {
        "schema_version",
        "contract_id",
        "mode",
        "status",
        "protocol_inputs",
    }:
        raise ValueError("Calibration-v2 top-level Flux-v3 route binding is malformed.")
    flux1_route_mode = flux1_route.get("mode")
    if flux1_route_mode not in {FLUX1_V3_MODE_PREVIEW, FLUX1_V3_MODE_EXECUTION}:
        raise ValueError("Calibration-v2 top-level Flux-v3 route mode is invalid.")
    expected_flux1_status = (
        FLUX1_V3_PREVIEW_STATUS
        if flux1_route_mode == FLUX1_V3_MODE_PREVIEW
        else FLUX1_V3_EXECUTION_STATUS
    )
    flux1_inputs = flux1_route.get("protocol_inputs")
    expected_flux1_roles = expected_flux1_protocol_input_roles_v3(str(flux1_route_mode))
    if (
        flux1_route.get("schema_version") != 1
        or flux1_route.get("contract_id") != FLUX1_V3_CONTRACT_ID
        or flux1_route.get("status") != expected_flux1_status
        or not isinstance(flux1_inputs, Mapping)
        or set(flux1_inputs) != set(expected_flux1_roles)
    ):
        raise ValueError("Calibration-v2 top-level Flux-v3 route semantics drifted.")
    if require_live_inputs:
        (
            _config,
            live_seed,
            live_inputs,
            live_evidence,
            live_generation_inputs,
            live_flux1_inputs,
        ) = _load_launch_contract(root)
        if flux1_route_mode != FLUX1_V3_MODE_EXECUTION:
            raise ValueError("Calibration-v2 preview manifests are never launchable.")
        if (seed, protocol_inputs, selection_evidence, generation_inputs, flux1_inputs) != (
            live_seed,
            live_inputs,
            live_evidence,
            live_generation_inputs,
            live_flux1_inputs,
        ):
            raise ValueError("Manifest differs from the live sealed calibration-v2 contract.")
    else:
        if not isinstance(protocol_inputs, Mapping) or not protocol_inputs:
            raise ValueError("Calibration-v2 manifest lacks protocol bindings.")
        if not isinstance(selection_evidence, Mapping):
            raise ValueError("Calibration-v2 manifest lacks common-seed evidence bindings.")
    if not isinstance(generation_inputs, Mapping) or set(generation_inputs) != set(
        GENERATION_INPUT_RELATIVES
    ):
        raise ValueError("Calibration-v2 manifest generation-input bindings are incomplete.")
    for role, relative in GENERATION_INPUT_RELATIVES.items():
        record = generation_inputs[role]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or Path(str(record["path"])).resolve() != (root / relative).resolve()
            or not _SHA256_RE.fullmatch(str(record["sha256"]))
        ):
            raise ValueError(f"Calibration-v2 generation-input binding drifted: {role!r}.")
    snapshot_bundle = manifest.get("snapshot_bundle")
    audit_snapshot_root: Path | None = None
    audit_snapshot_objects: dict[str, Path] = {}
    if isinstance(snapshot_bundle, Mapping):
        finer._verify_snapshot_bundle(dict(manifest))
        audit_snapshot_root, audit_snapshot_objects = finer._snapshot_object_paths_by_sha256(
            dict(manifest)
        )
    _validate_bound_files(
        protocol_inputs,
        PROTOCOL_INPUT_ROLES,
        snapshot_bundle=snapshot_bundle if not require_live_inputs else None,
    )
    _validate_bound_files(
        selection_evidence,
        EVIDENCE_ROLES,
        snapshot_bundle=snapshot_bundle if not require_live_inputs else None,
    )
    output_root = (root / OUTPUT_ROOT_RELATIVE).resolve()
    expected_top = {
        "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
        "benchmark": CALIBRATION_ID,
        "calibration_id": CALIBRATION_ID,
        "stage": CALIBRATION_STAGE,
        "status": (
            "implementation_preview_not_launchable"
            if flux1_route_mode == FLUX1_V3_MODE_PREVIEW
            else "frozen_before_generation"
        ),
        "selected_common_seed": seed,
        "review_blinding_nonce": blinding_nonce,
        "attempt": ATTEMPT,
        "model_name": MODEL_NAME,
        "prompt_ids": list(PROMPT_IDS),
        "true_cfg_scales": list(TRUE_CFG_SCALES),
        "eligible_scales": list(ELIGIBLE_SCALES),
        "output_root": str(output_root),
        "num_jobs": EXPECTED_ROWS,
        "expected_media_jobs": EXPECTED_ROWS,
        "expected_not_supported_jobs": 0,
        "role_counts": EXPECTED_ROLE_COUNTS,
        "requires_environment_preflight": True,
        "requires_exact_manifest_job_index": True,
        "v1_or_common_seed_media_reuse_allowed": False,
        "generation_inputs": generation_inputs,
        "flux1_dual_view_route_v3": flux1_route,
    }
    drift = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected_top.items()
        if manifest.get(key) != value
    }
    if drift:
        raise ValueError(f"Calibration-v2 top-level contract drifted: {drift}")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != EXPECTED_ROWS:
        raise ValueError("Calibration-v2 manifest must contain exactly 45 rows.")
    paths: set[str] = set()
    conditions: set[str] = set()
    blinds: set[str] = set()
    v1_root = (root / V1_OUTPUT_ROOT_RELATIVE).resolve()
    for index, (job, expected) in enumerate(
        zip(jobs, _expected_rows(output_root, seed), strict=True)
    ):
        prompt_id, role, scale, output_dir = expected
        row = job.get("calibration_row") if isinstance(job, Mapping) else None
        negative_mode = (
            NEGATIVE_MODE_NOT_APPLIED
            if role == ROLE_OFFICIAL
            else (
                NEGATIVE_MODE_EXPLICIT_NONE_CONTROL
                if role == ROLE_CONTROL
                else NEGATIVE_MODE_PAIRED_REGISTERED
            )
        )
        expected_row = {
            "schema_version": CALIBRATION_ROW_SCHEMA_VERSION,
            "calibration_id": CALIBRATION_ID,
            "role": role,
            "execution_path": (
                EXECUTION_PATH_OFFICIAL if role == ROLE_OFFICIAL else EXECUTION_PATH_NATIVE
            ),
            "prompt_id": prompt_id,
            "selected_common_seed": seed,
            "true_cfg_scale": scale,
            "negative_prompt_mode": negative_mode,
            "eligible_for_scale_selection": role == ROLE_LADDER and scale in ELIGIBLE_SCALES,
            "blind_id": _blind_id(blinding_nonce, prompt_id, role, scale, seed),
            "fresh_v2_media_required": True,
        }
        if row != expected_row:
            raise ValueError(f"Calibration-v2 row {index} identity drifted.")
        expected_condition = _row_id(prompt_id, role, scale, seed)
        expected_variant = role if scale is None else f"{role}__{_scale_label(scale)}"
        fixed = {
            "schema_version": 4,
            "benchmark": CALIBRATION_ID,
            "stage": CALIBRATION_STAGE,
            "attempt": ATTEMPT,
            "seed": seed,
            "seed_scoped_output": False,
            "prompt_id": prompt_id,
            "variation": role,
            "variant": expected_variant,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "condition_id": expected_condition,
            "variation_dir": str(output_dir.parents[2]),
            "output_dir": str(output_dir),
            "expected_media": True,
            "expected_native_negative_support": True,
            "runtime": {"device": "cuda", "dtype": None},
            "logging": {"tensorboard": False},
            "output": {
                "decode": True,
                "save_latents": False,
                "save_traces": False,
                "image_format": "png",
                "video_format": "mp4",
            },
            "calibration_v2_bindings": {
                "selected_common_seed": seed,
                "protocol_inputs": protocol_inputs,
                "selection_evidence": selection_evidence,
            },
        }
        if any(job.get(key) != value for key, value in fixed.items()):
            raise ValueError(f"Calibration-v2 row {index} frozen field drifted.")
        layout = IMAGE_LAYOUTS[prompt_id]
        generation = job.get("generation")
        expected_generation = {
            "task": "text_to_image",
            "num_inference_steps": 28,
            "height": layout["height"],
            "width": layout["width"],
            "guidance_scale": 3.5,
            "num_outputs_per_prompt": 1,
        }
        if (
            not isinstance(generation, Mapping)
            or set(generation) != {*expected_generation, "flux_dual_view_conditioning"}
            or any(generation.get(key) != value for key, value in expected_generation.items())
        ):
            raise ValueError(f"Calibration-v2 row {index} generation contract drifted.")
        shared_kwargs: dict[str, Any] = {}
        if not require_live_inputs and audit_snapshot_root is not None:
            archived_static_inputs: dict[str, Path] = {}
            for source_role in FLUX1_V3_SOURCE_INPUT_ROLES:
                source_record = (job.get("input_files") or {}).get(source_role)
                digest = (
                    str(source_record.get("sha256", ""))
                    if isinstance(source_record, Mapping)
                    else ""
                )
                if digest not in audit_snapshot_objects:
                    raise ValueError(
                        "Calibration-v2 audit snapshot lacks the shared Flux-v3 "
                        f"static object for {source_role!r}."
                    )
                archived_static_inputs[source_role] = audit_snapshot_objects[digest]
            shared_kwargs = {
                "audit_snapshot_root": audit_snapshot_root,
                "audit_snapshot_input_paths": archived_static_inputs,
            }
        shared_validation = validate_flux1_job_v3(
            job,
            project_root=root,
            mode=(
                FLUX1_V3_MODE_EXECUTION
                if require_live_inputs
                else FLUX1_V3_MODE_AUDIT
            ),
            **shared_kwargs,
        )
        job_route = job["flux1_dual_view_route_v3"]
        expected_native_pipeline = role != ROLE_OFFICIAL
        if (
            job_route.get("mode") != flux1_route_mode
            or job_route.get("status") != expected_flux1_status
            or job_route.get("protocol_inputs") != flux1_inputs
            or job_route.get("negative_mode") != negative_mode
            or job_route.get("native_pipeline_execution") is not expected_native_pipeline
            or shared_validation.get("mode") != flux1_route_mode
        ):
            raise ValueError(f"Calibration-v2 row {index} shared Flux-v3 role drifted.")
        if role == ROLE_OFFICIAL:
            if (
                job.get("variant_spec") != {"kind": "baseline"}
                or job.get("native_negative_prompt_options") != {}
            ):
                raise ValueError(f"Calibration-v2 official row {index} is not exact baseline.")
        else:
            assert scale is not None
            options = _native_options(
                role, scale, str(protocol_inputs["calibration_v2_config"]["sha256"])
            )
            if job.get("native_negative_prompt_options") != options or job.get("variant_spec") != {
                "kind": "native_negative_prompt",
                "capability": "supported",
                "native_negative_prompt_options": options,
            }:
                raise ValueError(f"Calibration-v2 native row {index} options drifted.")
        expected_extra_inputs = {
            evidence_role: {
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
            }
            for evidence_role, record in {
                **protocol_inputs,
                **selection_evidence,
            }.items()
        }
        input_files = job.get("input_files")
        expected_base_inputs = {
            "base_config": generation_inputs["base_config"],
            "model_config": generation_inputs["model_config"],
            "prompt_suite": generation_inputs["prompt_suite"],
            "negative_prompt_config": generation_inputs["negative_prompt_config"],
            "concept_tree": generation_inputs[f"concept_tree_{prompt_id}"],
        }
        if (
            not isinstance(input_files, Mapping)
            or set(input_files)
            != BASE_INPUT_ROLES | CALIBRATION_V2_INPUT_ROLES | expected_flux1_roles
            or any(input_files.get(key) != value for key, value in expected_extra_inputs.items())
            or any(input_files.get(key) != value for key, value in expected_base_inputs.items())
            or any(
                Path(str(job.get(key, ""))).resolve() != Path(str(value["path"])).resolve()
                for key, value in expected_base_inputs.items()
            )
        ):
            raise ValueError(f"Calibration-v2 row {index} snapshot input roles drifted.")
        resolved_output = Path(str(job["output_dir"])).resolve()
        if resolved_output == v1_root or v1_root in resolved_output.parents:
            raise ValueError("Calibration-v2 row attempts to reuse a v1 output path.")
        paths.add(str(resolved_output))
        conditions.add(str(job["condition_id"]))
        blinds.add(str(row["blind_id"]))
        if require_live_inputs or not isinstance(snapshot_bundle, Mapping):
            finer._verify_job_input_files(dict(job))
            finer._verify_job_semantic_snapshots(dict(job))
        # Historical audit has already authenticated the immutable manifest,
        # its complete content-addressed input bundle, and the archived shared
        # source views above.  Re-entering the generic live semantic reader
        # here would incorrectly make an audit depend on mutable live paths.
    if any(len(values) != EXPECTED_ROWS for values in (paths, conditions, blinds)):
        raise ValueError("Calibration-v2 paths, conditions, and blind IDs must be unique.")
    if manifest.get("manifest_sha256") != finer.manifest_digest(dict(manifest)):
        raise ValueError("Calibration-v2 manifest digest is inconsistent.")
    finer._manifest_provenance_consistency(
        dict(manifest), require_live_protocol=require_live_inputs
    )
    if require_live_inputs:
        finer._verify_implementation_provenance(dict(manifest), root)


def write_calibration_manifest_immutable(
    manifest: dict[str, Any], path: Path, root: Path | None = None
) -> None:
    root = (root or project_root()).resolve()
    validate_calibration_manifest(manifest, root=root, require_live_inputs=True)
    finer.write_manifest_immutable(manifest, path, root)
    validate_calibration_manifest(manifest, root=root, require_live_inputs=True)


def read_calibration_manifest(
    path: Path, root: Path | None = None, *, require_live_inputs: bool = True
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    resolved = path if path.is_absolute() else root / path
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    validate_calibration_manifest(manifest, root=root, require_live_inputs=require_live_inputs)
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    fields = sidecar.read_text(encoding="utf-8").split() if sidecar.is_file() else []
    if fields != [manifest["manifest_sha256"], resolved.name]:
        raise ValueError("Calibration-v2 manifest sidecar is absent or inconsistent.")
    if not isinstance(manifest.get("snapshot_bundle"), Mapping):
        raise ValueError("Calibration-v2 manifest lacks its immutable snapshot bundle.")
    finer._verify_snapshot_bundle(manifest)
    return manifest


def read_calibration_manifest_for_audit(path: Path, root: Path | None = None) -> dict[str, Any]:
    return read_calibration_manifest(path, root, require_live_inputs=False)


def execute_calibration_job(
    manifest: dict[str, Any], index: int, root: Path | None = None
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    validate_calibration_manifest(manifest, root=root, require_live_inputs=True)
    if isinstance(index, bool) or index < 0 or index >= EXPECTED_ROWS:
        raise IndexError(f"Calibration-v2 index {index!r} is outside 0..44.")
    bound = finer._job_with_launch_manifest_binding(
        manifest["jobs"][index], manifest, job_index=index
    )
    return finer.run_job(bound, root)


def protocol_status(root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    config = load_yaml(root / CONFIG_RELATIVE)
    _validate_static_config(config)
    superficially_sealed = (
        config.get("status") == "preregistered_before_generation"
        and SEALED_CONFIG_SHA256 is not None
        and SEALED_PROTOCOL_SHA256 is not None
    )
    launchable = False
    blocking_reason = "draft_or_unpinned_protocol"
    if superficially_sealed:
        try:
            _load_launch_contract(root)
        except (OSError, ValueError) as exc:
            blocking_reason = str(exc)
        else:
            launchable = True
            blocking_reason = None
    return {
        "status": "launchable" if launchable else "implementation_draft_not_launchable",
        "calibration_id": CALIBRATION_ID,
        "expected_rows": EXPECTED_ROWS,
        "attempt": ATTEMPT,
        "output_root": str((root / OUTPUT_ROOT_RELATIVE).resolve()),
        "manifest_publication_allowed": launchable,
        "blocking_reason": blocking_reason,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--write-manifest")
    mode.add_argument("--manifest")
    parser.add_argument("--index", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = project_root()
    if args.status:
        if args.index is not None or args.validate_only:
            raise ValueError("--status does not accept execution/validation flags.")
        print(json.dumps(protocol_status(root), indent=2, sort_keys=True))
        return
    if args.write_manifest:
        if args.index is not None or args.validate_only:
            raise ValueError("Manifest publication cannot select a row or validate-only mode.")
        manifest = build_calibration_manifest(root)
        write_calibration_manifest_immutable(manifest, Path(args.write_manifest), root)
        print(
            json.dumps(
                {
                    "status": "immutable_manifest_written",
                    "path": str(Path(args.write_manifest)),
                    "num_jobs": EXPECTED_ROWS,
                    "manifest_sha256": manifest["manifest_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    manifest = read_calibration_manifest(Path(args.manifest), root)
    if args.validate_only:
        if args.index is not None:
            raise ValueError("--validate-only cannot select an execution row.")
        print(json.dumps({"status": "strictly_validated", "num_jobs": EXPECTED_ROWS}, indent=2))
        return
    if args.index is None:
        raise ValueError("Calibration-v2 execution requires one exact --index.")
    print(json.dumps(execute_calibration_job(manifest, args.index, root), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
