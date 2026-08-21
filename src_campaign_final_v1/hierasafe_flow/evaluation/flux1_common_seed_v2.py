"""Target-blind common-seed qualification for repaired Flux.1 baselines.

The first Flux.1 native-negative ladder used a source-poor seed.  This module
implements a separate, launch-disabled-by-default stage that qualifies one
common seed using baseline images only.  It deliberately does not import or
read the v1 calibration outputs.

The stage has three fail-closed boundaries:

* exactly eight immutable seed-homogeneous manifests (seeds 0..7), each with
  the same three production-shaped baseline jobs;
* complete artifact, environment-preflight, registry, and Slurm-identity
  authentication before a reviewer can see a randomized source-only package;
* a fixed selection rule: the smallest seed whose three images pass every
  registered hard gate, otherwise no selection.

The checked-in config and protocol remain drafts.  Manifest publication is
therefore intentionally impossible until both documents are preregistered,
sealed, and their SHA-256 digests are pinned below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import secrets
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from hierasafe_flow.benchmarks import finer_detailing_correction as finer
from hierasafe_flow.benchmarks.slurm_tracking import (
    read_environment_preflight,
    read_execution_identity,
    read_submission_registry,
)
from hierasafe_flow.utils.config import load_yaml


PROTOCOL_ID = "flux1_common_seed_source_ladder_v2"
STAGE = "flux1_common_seed_source_ladder_v2"
CONFIG_RELATIVE = Path("configs/experiments/flux1_common_seed_source_ladder_v2.yaml")
PROTOCOL_RELATIVE = Path(
    "debugging/audits/finer_detailing_20260720/"
    "flux1_common_seed_source_ladder_v2_PROTOCOL_PREREGISTERED.md"
)

# These two values must remain None while the documents above are drafts.  A
# future launch-preparation change must pin their final raw-file SHA-256 values.
SEALED_CONFIG_SHA256: str | None = (
    "13adefb51ec592ebfe27402ef7fa438f9099057478882fe7418ccc44ce3cc477"
)
SEALED_PROTOCOL_SHA256: str | None = (
    "f772036b82f76e8060d63502cbcfb9fe269017de71fa5f3047db58db6d5b94cd"
)

MODEL_NAME = "flux1_dev"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
PIPELINE_CLASS = "FluxPipeline"
PROMPT_IDS = tuple(finer.PROMPT_IDS)
SEEDS = tuple(range(8))
EXPECTED_MANIFESTS = 8
ROWS_PER_MANIFEST = 3
EXPECTED_ROWS = 24
OUTPUT_ROOT_RELATIVE = Path("outputs/finer_detailing_flux1_common_seed_source_ladder_v2")
MANIFEST_ROOT_RELATIVE = Path(
    "debugging/manifests/finer_detailing_20260720/flux1_common_seed_source_ladder_v2"
)
COHORT_COMMIT_FILENAME = "cohort_commit.json"
COHORT_COMMIT_NAME = "flux1_common_seed_manifest_cohort_commit_v2"
REVIEW_PACKAGE_COMMIT_FILENAME = "review_package_commit.json"
REVIEW_PACKAGE_COMMIT_NAME = "flux1_common_seed_blinded_review_package_commit_v2"
SELECTION_PUBLICATION_COMMIT_NAME = "flux1_common_seed_selection_publication_commit_v2"
PHASE_ID = "flux1_common_seed_source_ladder_v2_attempt001"
PHASE_PLAN_RELATIVE = Path(
    "debugging/plans/finer_detailing_20260720/flux1_common_seed_source_ladder_v2.json"
)
PHASE_STATE_RELATIVE = Path(
    "debugging/phase_state/finer_detailing_20260720/flux1_common_seed_source_ladder_v2.json"
)
PHASE_LAUNCH_COMMIT_RELATIVE = Path(
    "debugging/phase_state/finer_detailing_20260720/"
    "flux1_common_seed_source_ladder_v2_launch_commit.json"
)
PHASE_RELEASE_RECEIPT_RELATIVE = Path(
    "debugging/phase_state/finer_detailing_20260720/"
    "flux1_common_seed_source_ladder_v2_release_receipt.json"
)
PHASE_REGISTRY_ROOT_RELATIVE = Path(
    "debugging/submissions/finer_detailing_20260720/flux1_common_seed_source_ladder_v2"
)
PHASE_LAUNCHER_RELATIVE = Path("slurm/flux1_common_seed_v2_h100.sbatch")
SLURM_JOB_NAME = "flux1-seed-v2"
IMAGE_LAYOUTS = {
    "01_sad_young_girl": {"width": 832, "height": 1216},
    "02_angry_old_man": {"width": 832, "height": 1216},
    "03_empty_outdoor_mall": {"width": 1216, "height": 832},
}
REVIEW_NAME = "flux1_common_baseline_source_review_v2"
AUTHENTICATION_NAME = "flux1_common_seed_candidate_authentication_v2"
UNBLINDING_NAME = "flux1_common_seed_private_unblinding_v2"
MANUAL_REVIEW_NAME = "flux1_common_baseline_manual_review_v2"
SELECTION_NAME = "flux1_common_seed_selection_v2"
CALIBRATION_V2_ID = "flux1_native_negative_true_cfg_scale_v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These prompt-specific semantic gates are required to appear explicitly in
# the preregistered YAML.  Keeping them here as an independent validator stops
# a later config edit from weakening exact sign text or escalator direction.
MANDATORY_PROMPT_GATES: dict[str, tuple[str, ...]] = {
    "03_empty_outdoor_mall": (
        "repeated_signboards_each_exactly_and_legibly_read_50_percent_sale",
        "two_parallel_working_escalators_with_correctly_opposing_up_and_down_directions",
    )
}


def project_root() -> Path:
    return finer.project_root()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def document_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_aware_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO-8601 with timezone.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include an explicit timezone.")
    return parsed


def _binding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Cannot bind missing or empty artifact: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {resolved}")
    return payload


def _file_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _validate_frozen_input_records(
    records: Mapping[str, Any], *, root: Path
) -> dict[str, dict[str, str]]:
    if not isinstance(records, Mapping) or not records:
        raise ValueError("Common-seed protocol requires non-empty frozen_inputs.")
    normalized: dict[str, dict[str, str]] = {}
    for role, raw in records.items():
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise ValueError(f"Frozen input {role!r} must contain exactly path and sha256.")
        relative = Path(str(raw["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Frozen input {role!r} must be project relative.")
        path = (root / relative).resolve()
        expected = str(raw["sha256"])
        if not _SHA256_RE.fullmatch(expected) or not path.is_file():
            raise ValueError(f"Frozen input {role!r} is missing or has a malformed digest.")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen input {role!r} drifted: expected {expected}, got {actual}.")
        normalized[str(role)] = {"path": str(path), "sha256": actual}
    return normalized


def _load_protocol(
    root: Path, *, require_sealed: bool
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    root = root.resolve()
    config_path = (root / CONFIG_RELATIVE).resolve()
    protocol_path = (root / PROTOCOL_RELATIVE).resolve()
    for path in (config_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(f"Common-seed protocol input is absent: {path}")
    config = load_yaml(config_path)
    if not isinstance(config, dict):
        raise ValueError("Common-seed YAML must contain one mapping.")

    expected_scope = {
        "model_name": MODEL_NAME,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pipeline_class": PIPELINE_CLASS,
        "execution_path": "GenerationRunner_FluxAdapter_repaired_native_equivalent",
        "prompt_ids": list(PROMPT_IDS),
        "candidate_seeds": list(SEEDS),
        "attempt_rule": "seed_plus_one",
        "manifests": EXPECTED_MANIFESTS,
        "rows_per_manifest": ROWS_PER_MANIFEST,
        "total_rows": EXPECTED_ROWS,
        "variation": "01_baseline",
        "native_negative_prompt_applied": False,
        "concept_steering_applied": False,
        "shapley_concept_steering_applied": False,
        "num_inference_steps": 28,
        "embedded_guidance_scale": 3.5,
        "num_outputs_per_prompt": 1,
        "seed_scoped_output": True,
        "image_layout_by_prompt": IMAGE_LAYOUTS,
    }
    if config.get("schema_version") != 1 or config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Common-seed protocol schema or identity drifted.")
    if config.get("benchmark") != finer.BENCHMARK_NAME:
        raise ValueError("Common-seed source benchmark drifted.")
    if config.get("scope") != expected_scope:
        raise ValueError("Common-seed 8x3 production scope drifted.")
    if Path(str(config.get("output_root", ""))) != OUTPUT_ROOT_RELATIVE:
        raise ValueError("Common-seed output root drifted.")
    if Path(str(config.get("manifest_root", ""))) != MANIFEST_ROOT_RELATIVE:
        raise ValueError("Common-seed manifest root drifted.")

    gates = config.get("source_only_hard_gates")
    if not isinstance(gates, dict) or tuple(gates) != PROMPT_IDS:
        raise ValueError("Source-only hard-gate prompt order/coverage drifted.")
    for prompt_id in PROMPT_IDS:
        values = gates[prompt_id]
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError(f"Hard-gate list is invalid for {prompt_id}.")

    expected_review = {
        "source_baseline_media_only": True,
        "exact_original_resolution_review_required": True,
        "randomized_blind_ids": True,
        "reviewer_manifest_must_not_contain_seed_values": True,
        "reviewer_manifest_must_not_contain_negative_prompts": True,
        "reviewer_manifest_must_not_contain_target_concepts": True,
        "reviewer_manifest_must_not_contain_steered_outputs": True,
        "binary_noncompensatory_hard_gates": True,
        "evidence_notes_required_for_every_gate": True,
        "quality_scoring_or_ranking_forbidden": True,
    }
    if config.get("review_protocol") != expected_review:
        raise ValueError("Target-blind reviewer contract drifted.")
    expected_selection = {
        "eligible_common_seed": "all_three_prompts_pass_every_registered_hard_gate",
        "choice": "numerically_lowest_eligible_common_seed",
        "prompt_specific_seeds_forbidden": True,
        "averaging_across_prompts_forbidden": True,
        "post_review_threshold_or_gate_relaxation_forbidden": True,
        "no_eligible_seed_outcome": ("publish_no_selection_and_repair_then_rerun_all_eight"),
        "v1_native_negative_ladder_outputs_are_ineligible_and_must_not_be_inspected": True,
    }
    if config.get("selection_rule") != expected_selection:
        raise ValueError("Common-seed selection rule drifted.")
    expected_later = {
        "calibration_id": CALIBRATION_V2_ID,
        "required_rows": 45,
        "required_prompt_ids": list(PROMPT_IDS),
        "every_row_must_use_selected_common_seed": True,
        "scale_ladder_and_selection_rule_must_remain_frozen": True,
        "v1_scale_outputs_must_not_be_reused": True,
    }
    if config.get("later_calibration_v2_binding") != expected_later:
        raise ValueError("Later 45-row calibration-v2 binding drifted.")

    status = config.get("status")
    if require_sealed:
        if status != "preregistered_before_generation":
            raise ValueError(
                "Manifest publication is blocked: common-seed config is not preregistered."
            )
        preregistered_at = _parse_aware_timestamp(
            config.get("preregistered_at"), "common-seed preregistered_at"
        )
        if preregistered_at > datetime.now(timezone.utc):
            raise ValueError("Common-seed preregistration timestamp is in the future.")
        draft_locations = []
        for label, path in (("config", config_path), ("protocol", protocol_path)):
            text = path.read_text(encoding="utf-8")
            if re.search(r"\bdraft\b", text, flags=re.IGNORECASE):
                draft_locations.append(label)
        if draft_locations:
            raise ValueError(
                f"Sealed common-seed inputs still contain DRAFT language: {draft_locations}."
            )
        pins = (SEALED_CONFIG_SHA256, SEALED_PROTOCOL_SHA256)
        if any(pin is None or not _SHA256_RE.fullmatch(str(pin)) for pin in pins):
            raise ValueError(
                "Manifest publication is blocked until config/protocol digests are pinned."
            )
        if sha256_file(config_path) != SEALED_CONFIG_SHA256:
            raise ValueError("Sealed common-seed config digest drifted.")
        if sha256_file(protocol_path) != SEALED_PROTOCOL_SHA256:
            raise ValueError("Sealed common-seed protocol digest drifted.")
        for path, digest in (
            (config_path, SEALED_CONFIG_SHA256),
            (protocol_path, SEALED_PROTOCOL_SHA256),
        ):
            sidecar = path.with_suffix(path.suffix + ".sha256")
            if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split() != [
                digest,
                path.name,
            ]:
                raise ValueError(f"Sealed sidecar is absent or inconsistent: {sidecar}")
    elif status not in {
        "implementation_draft_do_not_generate",
        "preregistered_before_generation",
    }:
        raise ValueError("Unknown common-seed protocol status.")

    inputs = {
        "common_seed_config": _file_record(config_path),
        "common_seed_protocol": _file_record(protocol_path),
        **_validate_frozen_input_records(config.get("frozen_inputs") or {}, root=root),
    }
    return config, inputs


def _hard_gate_ids_from_config(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    gates: dict[str, tuple[str, ...]] = {}
    for prompt_id in PROMPT_IDS:
        configured = tuple(config["source_only_hard_gates"][prompt_id])
        missing = tuple(
            gate for gate in MANDATORY_PROMPT_GATES.get(prompt_id, ()) if gate not in configured
        )
        if missing:
            raise ValueError(
                f"Registered hard gates for {prompt_id} omit mandatory semantics: {missing}."
            )
        gates[prompt_id] = configured
    return gates


def hard_gate_ids(
    root: Path | None = None, *, require_sealed: bool = False
) -> dict[str, tuple[str, ...]]:
    config, _ = _load_protocol((root or project_root()).resolve(), require_sealed=require_sealed)
    return _hard_gate_ids_from_config(config)


def default_manifest_path(root: Path, seed: int) -> Path:
    _validate_seed(seed)
    return root.resolve() / MANIFEST_ROOT_RELATIVE / f"seed_{seed:08d}" / "manifest.json"


def is_reserved_common_seed_output(job: Mapping[str, Any], root: Path | None = None) -> bool:
    """Return whether a job resolves inside the protected common-seed output tree."""

    root = (root or project_root()).resolve()
    raw = str(job.get("output_dir", "")).strip()
    if not raw:
        return False
    output = Path(raw)
    if not output.is_absolute():
        output = root / output
    output = output.resolve(strict=False)
    reserved = (root / OUTPUT_ROOT_RELATIVE).resolve(strict=False)
    return output == reserved or reserved in output.parents


def reject_reserved_common_seed_output_without_stage(
    job: Mapping[str, Any], root: Path | None = None
) -> None:
    if is_reserved_common_seed_output(job, root) and job.get("stage") != STAGE:
        raise RuntimeError(
            "The common-seed output namespace is reserved for committed cohort rows with "
            "complete held-phase authorization."
        )


def cohort_commit_path(root: Path) -> Path:
    return root.resolve() / MANIFEST_ROOT_RELATIVE / COHORT_COMMIT_FILENAME


def default_phase_plan_path(root: Path) -> Path:
    return root.resolve() / PHASE_PLAN_RELATIVE


def default_phase_registry_path(root: Path, seed: int) -> Path:
    _validate_seed(seed)
    return (
        root.resolve() / PHASE_REGISTRY_ROOT_RELATIVE / f"seed_{seed:08d}_submission_registry.json"
    )


def default_phase_launch_commit_path(root: Path) -> Path:
    return (root.resolve() / PHASE_LAUNCH_COMMIT_RELATIVE).resolve()


def default_phase_release_receipt_path(root: Path) -> Path:
    return (root.resolve() / PHASE_RELEASE_RECEIPT_RELATIVE).resolve()


def validate_cohort_commit_for_member(
    *, manifest_path: Path, manifest: Mapping[str, Any], root: Path
) -> None:
    """Require one atomically visible commit covering all eight seed manifests."""

    root = root.resolve()
    seed = _validate_seed(manifest.get("seed"))
    expected_member_path = default_manifest_path(root, seed)
    if manifest_path.expanduser().resolve() != expected_member_path:
        raise ValueError("Common-seed manifest is outside its canonical cohort path.")
    cohort_root = (root / MANIFEST_ROOT_RELATIVE).resolve()
    if (
        cohort_root.is_symlink()
        or not cohort_root.is_dir()
        or cohort_root.stat().st_mode & 0o222
        or any(item.is_symlink() for item in cohort_root.rglob("*"))
        or any(item.stat().st_mode & 0o222 for item in cohort_root.rglob("*") if item.is_dir())
    ):
        raise ValueError("Common-seed cohort is an uncommitted or aliased directory claim.")
    commit_path = cohort_commit_path(root)
    commit = _load_json(commit_path, "common-seed cohort commit")
    expected_top_fields = {
        "schema_version",
        "commit",
        "status",
        "created_at_utc",
        "protocol_id",
        "stage",
        "seeds",
        "rows_per_manifest",
        "expected_rows",
        "protocol_inputs",
        "members",
        "document_sha256",
    }
    if (
        set(commit) != expected_top_fields
        or commit.get("schema_version") != 1
        or commit.get("commit") != COHORT_COMMIT_NAME
        or commit.get("status") != "complete_before_any_launch"
        or commit.get("protocol_id") != PROTOCOL_ID
        or commit.get("stage") != STAGE
        or commit.get("seeds") != list(SEEDS)
        or commit.get("rows_per_manifest") != ROWS_PER_MANIFEST
        or commit.get("expected_rows") != EXPECTED_ROWS
        or commit.get("protocol_inputs") != manifest.get("common_seed_protocol_inputs")
        or commit.get("document_sha256") != document_sha256(commit)
    ):
        raise ValueError("Common-seed cohort commit identity/digest is invalid.")
    _parse_aware_timestamp(commit.get("created_at_utc"), "cohort commit created_at_utc")
    commit_sidecar = commit_path.with_suffix(commit_path.suffix + ".sha256")
    commit_raw_sha256 = sha256_file(commit_path) if commit_path.is_file() else ""
    if (
        not commit_sidecar.is_file()
        or commit_sidecar.read_text(encoding="utf-8").split()
        != [
            commit_raw_sha256,
            commit_path.name,
        ]
        or any(path.stat().st_mode & 0o222 for path in (commit_path, commit_sidecar))
    ):
        raise ValueError("Common-seed cohort commit sidecar is inconsistent.")
    members = commit.get("members")
    if not isinstance(members, list) or len(members) != EXPECTED_MANIFESTS:
        raise ValueError("Common-seed cohort commit must cover exactly eight manifests.")
    expected_member_fields = {
        "seed",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_sha256",
        "sidecar_path",
        "sidecar_sha256",
        "snapshot_index_path",
        "snapshot_index_sha256",
    }
    observed_seeds: list[int] = []
    selected_member: Mapping[str, Any] | None = None
    for member in members:
        if not isinstance(member, Mapping) or set(member) != expected_member_fields:
            raise ValueError("Common-seed cohort member schema drifted.")
        member_seed = _validate_seed(member.get("seed"))
        observed_seeds.append(member_seed)
        canonical_manifest = default_manifest_path(root, member_seed)
        canonical_sidecar = canonical_manifest.with_suffix(canonical_manifest.suffix + ".sha256")
        canonical_index = Path(f"{canonical_manifest}.snapshot") / "index.json"
        if (
            Path(str(member.get("manifest_path", ""))).resolve() != canonical_manifest
            or Path(str(member.get("sidecar_path", ""))).resolve() != canonical_sidecar
            or Path(str(member.get("snapshot_index_path", ""))).resolve()
            != canonical_index.resolve()
        ):
            raise ValueError("Common-seed cohort member path is noncanonical.")
        if any(
            not path.is_file() for path in (canonical_manifest, canonical_sidecar, canonical_index)
        ):
            raise FileNotFoundError("A committed common-seed manifest artifact is absent.")
        if any(
            path.is_symlink() or path.stat().st_mode & 0o222
            for path in (canonical_manifest, canonical_sidecar, canonical_index)
        ):
            raise ValueError("A committed common-seed manifest artifact is writable or aliased.")
        loaded = _load_json(canonical_manifest, "committed common-seed manifest")
        if (
            sha256_file(canonical_manifest) != member["manifest_file_sha256"]
            or loaded.get("manifest_sha256") != member["manifest_sha256"]
            or finer.manifest_digest(loaded) != member["manifest_sha256"]
            or loaded.get("seed") != member_seed
            or loaded.get("common_seed_protocol_inputs") != commit["protocol_inputs"]
            or sha256_file(canonical_sidecar) != member["sidecar_sha256"]
            or canonical_sidecar.read_text(encoding="utf-8").split()
            != [member["manifest_sha256"], canonical_manifest.name]
            or sha256_file(canonical_index) != member["snapshot_index_sha256"]
        ):
            raise ValueError("A committed common-seed member changed or is inconsistent.")
        descriptor = loaded.get("snapshot_bundle") or {}
        if (
            Path(str(descriptor.get("root_path", ""))).resolve() != canonical_index.parent.resolve()
            or Path(str(descriptor.get("index_path", ""))).resolve() != canonical_index.resolve()
            or descriptor.get("index_sha256") != member["snapshot_index_sha256"]
        ):
            raise ValueError("Committed common-seed snapshot descriptor drifted.")
        if member_seed == seed:
            selected_member = member
            if loaded != manifest:
                raise ValueError("Launch manifest bytes differ from their cohort member.")
    if observed_seeds != list(SEEDS) or selected_member is None:
        raise ValueError("Common-seed cohort commit seed order/coverage drifted.")


def validate_common_seed_phase_contract(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    state_path: Path,
    submissions: Sequence[Any],
    root: Path,
) -> None:
    """Require one indivisible 8-manifest/24-row launch plan."""

    root = root.resolve()
    canonical_plan = default_phase_plan_path(root)
    canonical_launcher = (root / PHASE_LAUNCHER_RELATIVE).resolve()
    if (
        plan_path.resolve() != canonical_plan
        or plan.get("phase_id") != PHASE_ID
        or state_path.resolve() != (root / PHASE_STATE_RELATIVE).resolve()
        or len(submissions) != EXPECTED_MANIFESTS
    ):
        raise ValueError("Common-seed phase identity/path/count contract drifted.")
    observed_seeds: list[int] = []
    for position, submission in enumerate(submissions):
        manifest = submission.manifest
        seed = _validate_seed(manifest.get("seed"))
        observed_seeds.append(seed)
        expected_manifest = default_manifest_path(root, seed)
        expected_registry = default_phase_registry_path(root, seed)
        expected_environment = {
            "FINER_DETAILING_MANIFEST": str(expected_manifest),
            "FLUX1_COMMON_SEED_PHASE_PLAN": str(canonical_plan),
        }
        if (
            position != seed
            or submission.submission_id != f"seed_{seed:08d}"
            or submission.manifest_path != expected_manifest
            or submission.indices != tuple(range(ROWS_PER_MANIFEST))
            or submission.array_spec != "0-2"
            or submission.registry_path != expected_registry
            or submission.slurm_job_name != SLURM_JOB_NAME
            or submission.launcher_path != canonical_launcher
            or submission.exported_environment != expected_environment
            or submission.pilot_environment != {}
            or manifest.get("stage") != STAGE
        ):
            raise ValueError(
                f"Common-seed phase submission {position} does not cover one exact seed array."
            )
    if observed_seeds != list(SEEDS):
        raise ValueError("Common-seed phase does not cover seeds 0..7 exactly once in order.")


def build_common_seed_phase_plan(root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    launcher = (root / PHASE_LAUNCHER_RELATIVE).resolve()
    if not launcher.is_file():
        raise FileNotFoundError(f"Common-seed H100 launcher is absent: {launcher}")
    plan_path = default_phase_plan_path(root)
    submissions: list[dict[str, Any]] = []
    for seed in SEEDS:
        manifest_path = default_manifest_path(root, seed)
        manifest = finer.read_manifest(manifest_path, root)
        occupied = [
            Path(str(job["output_dir"])).resolve()
            for job in manifest["jobs"]
            if Path(str(job["output_dir"])).resolve().exists()
        ]
        if occupied:
            raise FileExistsError(
                f"Common-seed phase requires all 24 canonical attempts to be fresh; "
                f"seed {seed} already has {occupied}."
            )
        submissions.append(
            {
                "submission_id": f"seed_{seed:08d}",
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "launcher_path": str(launcher),
                "launcher_sha256": sha256_file(launcher),
                "array_spec": "0-2",
                "registry_path": str(default_phase_registry_path(root, seed)),
                "slurm_job_name": SLURM_JOB_NAME,
                "exported_environment": {
                    "FINER_DETAILING_MANIFEST": str(manifest_path),
                    "FLUX1_COMMON_SEED_PHASE_PLAN": str(plan_path),
                },
                "pilot_environment": {},
            }
        )
    plan: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": finer.BENCHMARK_NAME,
        "phase_id": PHASE_ID,
        "created_at_utc": _utc_now(),
        "state_path": str((root / PHASE_STATE_RELATIVE).resolve()),
        "submissions": submissions,
    }
    from hierasafe_flow.benchmarks.finer_detailing_phase import phase_plan_digest

    plan["phase_plan_sha256"] = phase_plan_digest(plan)
    return plan


def write_common_seed_phase_plan_immutable(root: Path | None = None) -> Path:
    root = (root or project_root()).resolve()
    plan = build_common_seed_phase_plan(root)
    path = default_phase_plan_path(root)
    from hierasafe_flow.benchmarks.finer_detailing_phase import (
        validate_phase_plan,
        write_phase_plan_immutable,
    )

    write_phase_plan_immutable(plan, path)
    validate_phase_plan(path, root=root)
    return path


def _build_common_seed_phase_launch_commit(
    *,
    validated: Any,
    state: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Derive the immutable prerelease proof for eight already-held arrays."""

    from hierasafe_flow.benchmarks.finer_detailing_phase import build_sbatch_command

    root = root.resolve()
    if (
        validated.plan_path != default_phase_plan_path(root)
        or validated.state_path != (root / PHASE_STATE_RELATIVE).resolve()
        or validated.plan.get("phase_id") != PHASE_ID
        or state.get("status") != "complete"
        or state.get("phase_id") != PHASE_ID
        or state.get("phase_plan_sha256") != validated.phase_plan_sha256
        or state.get("union_proof") != validated.union_proof
        or len(validated.submissions) != EXPECTED_MANIFESTS
    ):
        raise ValueError("Common-seed held-phase state is not complete and canonical.")
    state_entries = state.get("submissions")
    if not isinstance(state_entries, list) or len(state_entries) != EXPECTED_MANIFESTS:
        raise ValueError("Common-seed held-phase state lacks eight submission records.")

    submission_bindings: list[dict[str, Any]] = []
    job_ids: list[str] = []
    for seed, (submission, saved) in enumerate(
        zip(validated.submissions, state_entries, strict=True)
    ):
        expected_command = build_sbatch_command(submission)
        if "--hold" not in expected_command:
            raise ValueError("Common-seed sbatch command is not held before phase commit.")
        registry = read_submission_registry(submission.registry_path)
        expected_indices = list(range(ROWS_PER_MANIFEST))
        job_id = str(registry.get("slurm_array_job_id", ""))
        if (
            submission.manifest.get("seed") != seed
            or saved.get("status") != "registered"
            or saved.get("manifest_sha256") != submission.manifest_sha256
            or saved.get("registry_path") != str(submission.registry_path)
            or saved.get("slurm_array_job_id") != job_id
            or saved.get("registry_sha256") != registry.get("registry_sha256")
            or saved.get("sbatch_command") != expected_command
            or registry.get("manifest_path") != str(submission.manifest_path)
            or registry.get("manifest_sha256") != submission.manifest_sha256
            or registry.get("array_spec") != "0-2"
            or registry.get("slurm_job_name") != SLURM_JOB_NAME
            or registry.get("num_registered_tasks") != ROWS_PER_MANIFEST
            or [row.get("job_index") for row in registry.get("submissions", ())] != expected_indices
            or not job_id.isdigit()
        ):
            raise ValueError(f"Held common-seed submission {seed} is not exactly registered.")
        job_ids.append(job_id)
        submission_bindings.append(
            {
                "seed": seed,
                "submission_id": submission.submission_id,
                "manifest_path": str(submission.manifest_path),
                "manifest_sha256": submission.manifest_sha256,
                "array_spec": submission.array_spec,
                "indices": expected_indices,
                "launcher_path": str(submission.launcher_path),
                "launcher_sha256": submission.launcher_sha256,
                "sbatch_command": expected_command,
                "registry": {
                    **_binding(submission.registry_path),
                    "registry_sha256": registry["registry_sha256"],
                    "slurm_array_job_id": job_id,
                    "slurm_job_name": registry["slurm_job_name"],
                },
            }
        )
    if len(set(job_ids)) != EXPECTED_MANIFESTS:
        raise ValueError("Common-seed held arrays do not have eight unique Slurm job IDs.")

    cohort_path = cohort_commit_path(root)
    cohort = _load_json(cohort_path, "common-seed cohort commit")
    release_command = ["scontrol", "release", ",".join(job_ids)]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "flux1_common_seed_complete_held_phase_launch_commit_v2",
        "status": "complete_held_phase_authorized_for_release",
        "created_at_utc": state["completed_at_utc"],
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "phase_plan": {
            **_binding(validated.plan_path),
            "phase_plan_sha256": validated.phase_plan_sha256,
        },
        "phase_state": {
            **_binding(validated.state_path),
            "phase_state_sha256": state["phase_state_sha256"],
            "status": state["status"],
        },
        "cohort_commit": {
            **_binding(cohort_path),
            "document_sha256": cohort["document_sha256"],
        },
        "union_proof": deepcopy(validated.union_proof),
        "submissions": submission_bindings,
        "release_command": release_command,
    }
    payload["document_sha256"] = document_sha256(payload)
    return payload


def read_common_seed_phase_launch_commit(root: Path | None = None) -> dict[str, Any]:
    """Rebuild and authenticate the complete held-phase prerelease proof."""

    root = (root or project_root()).resolve()
    from hierasafe_flow.benchmarks.finer_detailing_phase import (
        read_phase_state,
        validate_phase_plan,
    )

    validated = validate_phase_plan(default_phase_plan_path(root), root=root)
    state = read_phase_state((root / PHASE_STATE_RELATIVE).resolve())
    expected = _build_common_seed_phase_launch_commit(
        validated=validated,
        state=state,
        root=root,
    )
    path = default_phase_launch_commit_path(root)
    observed = _load_json(path, "common-seed phase launch commit")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        observed != expected
        or observed.get("document_sha256") != document_sha256(observed)
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split() != [sha256_file(path), path.name]
    ):
        raise ValueError("Common-seed phase launch commit changed or is inconsistent.")
    return observed


def authorize_and_release_complete_common_seed_phase(
    *,
    validated: Any,
    state: Mapping[str, Any],
    root: Path,
    run: Callable[..., Any],
) -> dict[str, Any]:
    """Publish the all-held commit, then release all eight arrays in one command."""

    root = root.resolve()
    commit_path = default_phase_launch_commit_path(root)
    receipt_path = default_phase_release_receipt_path(root)
    if commit_path.exists() or commit_path.with_suffix(commit_path.suffix + ".sha256").exists():
        raise RuntimeError(
            "A common-seed launch commit already exists; release outcome must be reconciled "
            "without replaying scontrol."
        )
    if receipt_path.exists() or receipt_path.with_suffix(receipt_path.suffix + ".sha256").exists():
        raise RuntimeError("A common-seed release receipt already exists; refusing replay.")
    commit = _build_common_seed_phase_launch_commit(
        validated=validated,
        state=state,
        root=root,
    )
    _write_documents_transactional({commit_path: commit})
    if read_common_seed_phase_launch_commit(root) != commit:
        raise RuntimeError("Published common-seed launch commit failed immediate reauthentication.")
    try:
        result = run(
            commit["release_command"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        raise RuntimeError(
            "Common-seed release outcome is ambiguous after immutable launch commit; "
            "do not replay automatically."
        ) from exc
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "contract": "flux1_common_seed_complete_phase_release_receipt_v2",
        "status": "release_command_completed",
        "released_at_utc": _utc_now(),
        "launch_commit": {
            **_binding(commit_path),
            "document_sha256": commit["document_sha256"],
        },
        "release_command": deepcopy(commit["release_command"]),
        "scheduler_stdout": str(getattr(result, "stdout", "")).strip(),
        "scheduler_stderr": str(getattr(result, "stderr", "")).strip(),
    }
    receipt["document_sha256"] = document_sha256(receipt)
    _write_documents_transactional({receipt_path: receipt})
    reopened = _load_json(receipt_path, "common-seed phase release receipt")
    if reopened != receipt:
        raise RuntimeError("Common-seed release receipt failed immediate reauthentication.")
    return receipt


def read_common_seed_phase_release_receipt(root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    commit = read_common_seed_phase_launch_commit(root)
    commit_path = default_phase_launch_commit_path(root)
    path = default_phase_release_receipt_path(root)
    payload = _load_json(path, "common-seed phase release receipt")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected_keys = {
        "schema_version",
        "contract",
        "status",
        "released_at_utc",
        "launch_commit",
        "release_command",
        "scheduler_stdout",
        "scheduler_stderr",
        "document_sha256",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("contract") != "flux1_common_seed_complete_phase_release_receipt_v2"
        or payload.get("status") != "release_command_completed"
        or payload.get("document_sha256") != document_sha256(payload)
        or payload.get("launch_commit")
        != {**_binding(commit_path), "document_sha256": commit["document_sha256"]}
        or payload.get("release_command") != commit["release_command"]
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split() != [sha256_file(path), path.name]
    ):
        raise ValueError("Common-seed release receipt changed or is inconsistent.")
    _parse_aware_timestamp(payload.get("released_at_utc"), "phase released_at_utc")
    return payload


def build_common_seed_launch_authorization(
    *,
    phase_plan_path: Path,
    manifest_path: Path,
    job_index: int,
    root: Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Authorize one row only after all eight arrays are held and registered."""

    root = root.resolve()
    phase_plan_path = phase_plan_path.expanduser()
    manifest_path = manifest_path.expanduser()
    phase_plan_path = (
        phase_plan_path if phase_plan_path.is_absolute() else root / phase_plan_path
    ).resolve()
    manifest_path = (
        manifest_path if manifest_path.is_absolute() else root / manifest_path
    ).resolve()
    from hierasafe_flow.benchmarks.finer_detailing_phase import validate_phase_plan

    validated = validate_phase_plan(phase_plan_path, root=root)
    if isinstance(job_index, bool) or job_index not in range(ROWS_PER_MANIFEST):
        raise ValueError("Common-seed launch index must be exactly one of 0, 1, 2.")
    matches = [
        submission
        for submission in validated.submissions
        if submission.manifest_path == manifest_path
    ]
    if len(matches) != 1 or job_index not in matches[0].indices:
        raise ValueError("Common-seed phase does not authorize this manifest/index.")
    submission = matches[0]
    job = submission.manifest["jobs"][job_index]
    launch_commit = read_common_seed_phase_launch_commit(root)
    commit_submission = [
        row for row in launch_commit["submissions"] if row["manifest_path"] == str(manifest_path)
    ]
    if len(commit_submission) != 1:
        raise ValueError("Held-phase launch commit does not bind this manifest.")
    registry_binding = commit_submission[0]["registry"]
    registry_path = Path(registry_binding["path"]).resolve()
    registry = read_submission_registry(registry_path)
    registry_rows = [row for row in registry["submissions"] if row.get("job_index") == job_index]
    if len(registry_rows) != 1:
        raise ValueError("Canonical common-seed registry does not bind this exact task.")
    registry_row = registry_rows[0]
    values = os.environ if environ is None else environ
    live = {
        "SLURM_ARRAY_JOB_ID": str(values.get("SLURM_ARRAY_JOB_ID", "")),
        "SLURM_ARRAY_TASK_ID": str(values.get("SLURM_ARRAY_TASK_ID", "")),
        "SLURM_JOB_ID": str(values.get("SLURM_JOB_ID", "")),
        "SLURM_JOB_NAME": str(values.get("SLURM_JOB_NAME", "")),
    }
    expected_live = {
        "SLURM_ARRAY_JOB_ID": str(registry["slurm_array_job_id"]),
        "SLURM_ARRAY_TASK_ID": str(job_index),
        "SLURM_JOB_NAME": SLURM_JOB_NAME,
    }
    if (
        any(live[key] != value for key, value in expected_live.items())
        or not live["SLURM_JOB_ID"].isdigit()
    ):
        raise RuntimeError(
            f"Live Slurm identity differs from the complete held phase: "
            f"expected={expected_live}, observed={live}."
        )
    if registry_row.get("slurm_task_id") != f"{registry['slurm_array_job_id']}_{job_index}":
        raise ValueError("Common-seed registry composite task identity drifted.")
    commit_path = cohort_commit_path(root)
    commit = _load_json(commit_path, "common-seed cohort commit")
    launch_commit_path = default_phase_launch_commit_path(root)
    authorization = {
        "schema_version": 1,
        "authorization": "flux1_common_seed_complete_phase_launch_v2",
        "status": "authorized_before_model_load",
        "protocol_id": PROTOCOL_ID,
        "phase_plan_path": str(phase_plan_path),
        "phase_plan_sha256": validated.phase_plan_sha256,
        "phase_plan_file_sha256": sha256_file(phase_plan_path),
        "phase_plan_sidecar_sha256": sha256_file(
            phase_plan_path.with_suffix(phase_plan_path.suffix + ".sha256")
        ),
        "phase_id": PHASE_ID,
        "union_proof": deepcopy(validated.union_proof),
        "phase_launch_commit_path": str(launch_commit_path),
        "phase_launch_commit_document_sha256": launch_commit["document_sha256"],
        "phase_launch_commit_file_sha256": sha256_file(launch_commit_path),
        "phase_state_sha256": launch_commit["phase_state"]["phase_state_sha256"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": submission.manifest_sha256,
        "manifest_job_index": job_index,
        "condition_id": job["condition_id"],
        "cohort_commit_path": str(commit_path),
        "cohort_commit_document_sha256": commit["document_sha256"],
        "cohort_commit_file_sha256": sha256_file(commit_path),
        "submission_registry_path": str(registry_path),
        "submission_registry_file_sha256": sha256_file(registry_path),
        "submission_registry_sha256": registry["registry_sha256"],
        "slurm_array_job_id": str(registry["slurm_array_job_id"]),
        "slurm_array_task_id": job_index,
        "slurm_task_id": registry_row["slurm_task_id"],
        "slurm_job_id": live["SLURM_JOB_ID"],
        "slurm_job_name": live["SLURM_JOB_NAME"],
    }
    return authorization


def _validate_seed(seed: Any) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise ValueError("Common-seed candidate must be one exact integer in 0..7.")
    return seed


def _build_one_seed_manifest(
    root: Path,
    *,
    seed: int,
    config: Mapping[str, Any],
    protocol_inputs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    attempt = seed + 1
    args = finer.build_parser().parse_args(
        [
            "--models",
            MODEL_NAME,
            "--prompt-ids",
            ",".join(PROMPT_IDS),
            "--variations",
            "baseline",
            "--output-root",
            str(OUTPUT_ROOT_RELATIVE),
            "--attempt",
            str(attempt),
            "--seed",
            str(seed),
            "--seed-scoped-output",
            "--no-tensorboard",
            "--no-save-latents",
            "--no-save-traces",
        ]
    )
    manifest = finer.build_manifest(
        args,
        root,
        flux1_route_context=finer.FLUX1_ROUTE_CONTEXT_LEGACY_V2,
    )
    if len(manifest["jobs"]) != ROWS_PER_MANIFEST:
        raise RuntimeError("Baseline builder did not return exactly three Flux.1 rows.")
    protocol_status = str(config["status"])
    top_candidate = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_status": protocol_status,
        "seed": seed,
        "attempt": attempt,
        "prompt_ids": list(PROMPT_IDS),
        "rows": ROWS_PER_MANIFEST,
        "baseline_only": True,
        "eligible_for_common_seed_review": protocol_status == "preregistered_before_generation",
    }
    manifest.update(
        {
            "stage": STAGE,
            "common_seed_protocol": deepcopy(top_candidate),
            "common_seed_protocol_inputs": deepcopy(protocol_inputs),
            "status": (
                "frozen_before_generation"
                if protocol_status == "preregistered_before_generation"
                else "implementation_preview_not_launchable"
            ),
        }
    )
    for job in manifest["jobs"]:
        job["stage"] = STAGE
        job["common_seed_candidate"] = {
            **deepcopy(top_candidate),
            "prompt_id": job["prompt_id"],
        }
        job["common_seed_protocol_inputs"] = deepcopy(protocol_inputs)
        job["input_files"].update(deepcopy(protocol_inputs))
    manifest["manifest_sha256"] = finer.manifest_digest(manifest)
    return manifest


def build_seed_manifests(
    root: Path | None = None, *, require_sealed: bool = False
) -> dict[int, dict[str, Any]]:
    """Build the exact 8x3 cohort in memory without writing any manifest."""

    root = (root or project_root()).resolve()
    config, protocol_inputs = _load_protocol(root, require_sealed=require_sealed)
    manifests = {
        seed: _build_one_seed_manifest(
            root,
            seed=seed,
            config=config,
            protocol_inputs=protocol_inputs,
        )
        for seed in SEEDS
    }
    for seed, manifest in manifests.items():
        validate_seed_manifest(
            manifest,
            root=root,
            expected_seed=seed,
            require_live_inputs=True,
            require_launchable=require_sealed,
        )
    if sum(len(manifest["jobs"]) for manifest in manifests.values()) != EXPECTED_ROWS:
        raise RuntimeError("Common-seed manifest cohort is not exactly 24 rows.")
    return manifests


def validate_seed_manifest(
    manifest: Mapping[str, Any],
    *,
    root: Path,
    expected_seed: int | None = None,
    require_live_inputs: bool,
    require_launchable: bool,
) -> None:
    root = root.resolve()
    seed = _validate_seed(manifest.get("seed"))
    if expected_seed is not None and seed != _validate_seed(expected_seed):
        raise ValueError("Common-seed manifest has the wrong seed identity.")
    if require_live_inputs:
        config, inputs = _load_protocol(root, require_sealed=require_launchable)
        protocol_status = str(config["status"])
        canonical_jobs = _build_one_seed_manifest(
            root,
            seed=seed,
            config=config,
            protocol_inputs=inputs,
        )["jobs"]
    else:
        inputs = manifest.get("common_seed_protocol_inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("Historical common-seed manifest lacks protocol input bindings.")
        inputs = deepcopy(dict(inputs))
        protocol_status = str(
            (manifest.get("common_seed_protocol") or {}).get("protocol_status", "")
        )
        canonical_jobs = None
    attempt = seed + 1
    expected_status = (
        "frozen_before_generation"
        if protocol_status == "preregistered_before_generation"
        else "implementation_preview_not_launchable"
    )
    expected_top = {
        "benchmark": finer.BENCHMARK_NAME,
        "stage": STAGE,
        "seed": seed,
        "seed_scoped_output": True,
        "attempt": attempt,
        "models": [MODEL_NAME],
        "prompt_ids": list(PROMPT_IDS),
        "variation_groups": ["01_baseline"],
        "num_jobs": ROWS_PER_MANIFEST,
        "expected_media_jobs": ROWS_PER_MANIFEST,
        "expected_not_supported_jobs": 0,
        "output_root": str((root / OUTPUT_ROOT_RELATIVE).resolve()),
        "status": expected_status,
        "common_seed_protocol_inputs": inputs,
    }
    drift = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_top.items()
        if manifest.get(key) != expected
    }
    if drift:
        raise ValueError(f"Common-seed manifest contract drifted: {drift}")
    expected_candidate = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_status": protocol_status,
        "seed": seed,
        "attempt": attempt,
        "prompt_ids": list(PROMPT_IDS),
        "rows": ROWS_PER_MANIFEST,
        "baseline_only": True,
        "eligible_for_common_seed_review": protocol_status == "preregistered_before_generation",
    }
    if manifest.get("common_seed_protocol") != expected_candidate:
        raise ValueError("Top-level common-seed candidate provenance drifted.")
    if require_launchable and expected_status != "frozen_before_generation":
        raise ValueError("Draft common-seed manifests cannot be launched.")
    if manifest.get("manifest_sha256") != finer.manifest_digest(dict(manifest)):
        raise ValueError("Common-seed manifest digest is inconsistent.")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != ROWS_PER_MANIFEST:
        raise ValueError("Every common-seed manifest must contain exactly three jobs.")
    seen_output: set[str] = set()
    for job_index, (job, prompt_id) in enumerate(zip(jobs, PROMPT_IDS, strict=True)):
        layout = IMAGE_LAYOUTS[prompt_id]
        candidate = {**expected_candidate, "prompt_id": prompt_id}
        expected_condition = f"{prompt_id}__{MODEL_NAME}__01_baseline__seed_{seed:08d}"
        expected_output = (
            root
            / OUTPUT_ROOT_RELATIVE
            / prompt_id
            / MODEL_NAME
            / "01_baseline"
            / f"seed_{seed:08d}"
            / "attempts"
            / f"attempt_{attempt:03d}"
        ).resolve()
        fixed = {
            "benchmark": finer.BENCHMARK_NAME,
            "stage": STAGE,
            "attempt": attempt,
            "variation": "01_baseline",
            "variant": "01_baseline",
            "condition_id": expected_condition,
            "variant_spec": {"kind": "baseline"},
            "prompt_id": prompt_id,
            "seed": seed,
            "seed_scoped_output": True,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "expected_media": True,
            "native_negative_prompt_options": {},
            "logging": {"tensorboard": False},
            "common_seed_candidate": candidate,
            "common_seed_protocol_inputs": inputs,
        }
        job_drift = {
            key: {"expected": expected, "actual": job.get(key)}
            for key, expected in fixed.items()
            if job.get(key) != expected
        }
        if job_drift:
            raise ValueError(f"Common-seed job {job_index} contract drifted: {job_drift}")
        generation = job.get("generation") or {}
        expected_generation = {
            "task": "text_to_image",
            "num_inference_steps": 28,
            "height": layout["height"],
            "width": layout["width"],
            "guidance_scale": 3.5,
            "num_outputs_per_prompt": 1,
        }
        if generation != expected_generation:
            raise ValueError(f"Common-seed job {job_index} generation contract drifted.")
        if canonical_jobs is not None:
            observed_exact = deepcopy(dict(job))
            # Immutable publication adds only this content-addressed archive
            # descriptor to each job.  Every generation-affecting field must
            # otherwise equal a freshly reconstructed canonical baseline row.
            observed_exact.pop("snapshot_bundle", None)
            if observed_exact != canonical_jobs[job_index]:
                raise ValueError(
                    f"Common-seed job {job_index} differs from the exact canonical "
                    "baseline projection (runtime/output/logging or another field drifted)."
                )
        if Path(str(job.get("output_dir", ""))).resolve() != expected_output:
            raise ValueError(f"Common-seed job {job_index} output path drifted.")
        if str(expected_output) in seen_output:
            raise ValueError("Common-seed manifest contains duplicate output paths.")
        seen_output.add(str(expected_output))
        if job.get("input_files") is None or any(
            job["input_files"].get(role) != record for role, record in inputs.items()
        ):
            raise ValueError(f"Common-seed job {job_index} protocol inputs drifted.")
        serialized_variant = json.dumps(job["variant_spec"], sort_keys=True)
        if any(
            token in serialized_variant for token in ("native_negative", "shapley", "conceptsteer")
        ):
            raise ValueError("A common-seed candidate is not baseline-only.")


def write_seed_manifests_immutable(
    root: Path | None = None,
) -> dict[int, Path]:
    """Publish the complete cohort only after both draft inputs are sealed.

    In the current repository state this function intentionally raises before
    creating any path because the protocol documents remain drafts.
    """

    root = (root or project_root()).resolve()
    manifests = build_seed_manifests(root, require_sealed=True)
    paths = {seed: default_manifest_path(root, seed) for seed in SEEDS}
    cohort_root = (root / MANIFEST_ROOT_RELATIVE).resolve()
    cohort_root.parent.mkdir(parents=True, exist_ok=True)
    if cohort_root.exists():
        raise FileExistsError(
            "Refusing common-seed publication because its canonical cohort root "
            f"already exists: {cohort_root}"
        )
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{cohort_root.name}.staging-",
            dir=cohort_root.parent,
        )
    )
    try:
        for seed in SEEDS:
            physical_path = (staging_root / f"seed_{seed:08d}" / "manifest.json").resolve()
            finer.write_manifest_immutable(
                manifests[seed],
                physical_path,
                root,
                logical_publication_path=paths[seed],
                allow_atomic_common_seed_cohort_staging=True,
            )
            validate_seed_manifest(
                manifests[seed],
                root=root,
                expected_seed=seed,
                require_live_inputs=True,
                require_launchable=True,
            )
        members: list[dict[str, Any]] = []
        for seed in SEEDS:
            physical_manifest = staging_root / f"seed_{seed:08d}" / "manifest.json"
            physical_sidecar = physical_manifest.with_suffix(".json.sha256")
            physical_index = Path(f"{physical_manifest}.snapshot") / "index.json"
            members.append(
                {
                    "seed": seed,
                    "manifest_path": str(paths[seed]),
                    "manifest_file_sha256": sha256_file(physical_manifest),
                    "manifest_sha256": manifests[seed]["manifest_sha256"],
                    "sidecar_path": str(paths[seed].with_suffix(".json.sha256")),
                    "sidecar_sha256": sha256_file(physical_sidecar),
                    "snapshot_index_path": str(Path(f"{paths[seed]}.snapshot") / "index.json"),
                    "snapshot_index_sha256": sha256_file(physical_index),
                }
            )
        commit: dict[str, Any] = {
            "schema_version": 1,
            "commit": COHORT_COMMIT_NAME,
            "status": "complete_before_any_launch",
            "created_at_utc": _utc_now(),
            "protocol_id": PROTOCOL_ID,
            "stage": STAGE,
            "seeds": list(SEEDS),
            "rows_per_manifest": ROWS_PER_MANIFEST,
            "expected_rows": EXPECTED_ROWS,
            "protocol_inputs": deepcopy(manifests[0]["common_seed_protocol_inputs"]),
            "members": members,
        }
        commit["document_sha256"] = document_sha256(commit)
        _write_documents_transactional({staging_root / COHORT_COMMIT_FILENAME: commit})
        staged_commit_path = staging_root / COHORT_COMMIT_FILENAME
        staged_commit = _load_json(staged_commit_path, "staged cohort commit")
        staged_commit_sidecar = staged_commit_path.with_suffix(
            staged_commit_path.suffix + ".sha256"
        )
        if (
            staged_commit != commit
            or staged_commit.get("document_sha256") != document_sha256(staged_commit)
            or staged_commit_sidecar.read_text(encoding="utf-8").split()
            != [sha256_file(staged_commit_path), staged_commit_path.name]
        ):
            raise RuntimeError("Staged common-seed cohort commit failed byte authentication.")
        for seed in SEEDS:
            physical_manifest = staging_root / f"seed_{seed:08d}" / "manifest.json"
            physical_sidecar = physical_manifest.with_suffix(".json.sha256")
            physical_index = Path(f"{physical_manifest}.snapshot") / "index.json"
            staged_manifest = _load_json(physical_manifest, "staged common-seed manifest")
            descriptor = staged_manifest.get("snapshot_bundle") or {}
            if (
                staged_manifest != manifests[seed]
                or finer.manifest_digest(staged_manifest) != staged_manifest.get("manifest_sha256")
                or physical_sidecar.read_text(encoding="utf-8").split()
                != [staged_manifest["manifest_sha256"], physical_manifest.name]
                or descriptor.get("root_path") != str(Path(f"{paths[seed]}.snapshot"))
                or descriptor.get("index_path")
                != str(Path(f"{paths[seed]}.snapshot") / "index.json")
                or descriptor.get("index_sha256") != sha256_file(physical_index)
            ):
                raise RuntimeError(
                    f"Staged common-seed manifest {seed} failed byte authentication."
                )
        for file_path in staging_root.rglob("*"):
            if file_path.is_file():
                file_path.chmod(0o444)
        for directory in sorted(
            (path for path in staging_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        staging_root.chmod(0o555)
        _publish_directory_new(
            staging_root,
            cohort_root,
            commit_filename=COHORT_COMMIT_FILENAME,
        )
        for seed in SEEDS:
            published = _load_json(paths[seed], "published common-seed manifest")
            validate_seed_manifest(
                published,
                root=root,
                expected_seed=seed,
                require_live_inputs=True,
                require_launchable=True,
            )
            validate_cohort_commit_for_member(
                manifest_path=paths[seed], manifest=published, root=root
            )
            manifests[seed].clear()
            manifests[seed].update(published)
    except BaseException:
        if staging_root.exists():
            staging_root.chmod(0o755)
            for directory in staging_root.rglob("*"):
                if directory.is_dir():
                    directory.chmod(0o755)
            shutil.rmtree(staging_root)
        raise
    return paths


def read_seed_manifest_for_audit(path: str | Path, root: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    manifest = finer.read_manifest_for_audit(resolved, root.resolve())
    validate_seed_manifest(
        manifest,
        root=root,
        require_live_inputs=False,
        require_launchable=True,
    )
    seed = _validate_seed(manifest.get("seed"))
    if resolved != default_manifest_path(root, seed):
        raise ValueError("Common-seed evidence manifest path is not its canonical cohort member.")
    validate_cohort_commit_for_member(
        manifest_path=resolved,
        manifest=manifest,
        root=root.resolve(),
    )
    return manifest


def _bound_job(job: Mapping[str, Any], manifest_sha256: str, job_index: int) -> dict[str, Any]:
    if isinstance(job_index, bool) or not 0 <= job_index < ROWS_PER_MANIFEST:
        raise IndexError(
            f"Common-seed manifest job index {job_index!r} is outside 0..{ROWS_PER_MANIFEST - 1}."
        )
    bound = deepcopy(dict(job))
    bound["launch_manifest_sha256"] = manifest_sha256
    bound["launch_manifest_job_index"] = job_index
    return bound


def validate_bound_job_for_launch(job: Mapping[str, Any], *, root: Path) -> None:
    """Reconstruct a direct runner call from its canonical committed manifest."""

    root = root.resolve()
    seed = _validate_seed(job.get("seed"))
    raw_index = job.get("launch_manifest_job_index")
    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
        raise ValueError("Common-seed bound job lacks an exact manifest row index.")
    manifest_path = default_manifest_path(root, seed)
    manifest = finer.read_manifest(manifest_path, root)
    expected = _bound_job(manifest["jobs"][raw_index], manifest["manifest_sha256"], raw_index)
    if dict(job) != expected:
        raise ValueError(
            "Direct common-seed runner job differs from its committed canonical manifest row."
        )


def validate_attempt_launch_authorization(
    job: Mapping[str, Any], *, output_dir: Path, root: Path
) -> None:
    """Require the dispatcher's immutable complete-phase authorization."""

    raw_index = job.get("launch_manifest_job_index")
    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
        raise ValueError("Common-seed launch authorization requires a manifest row index.")
    preflight = read_environment_preflight(
        output_dir,
        expected_job=job,
        expected_job_index=raw_index,
    )
    expected = build_common_seed_launch_authorization(
        phase_plan_path=default_phase_plan_path(root),
        manifest_path=default_manifest_path(root, _validate_seed(job.get("seed"))),
        job_index=raw_index,
        root=root,
    )
    if preflight.get("common_seed_launch_authorization") != expected:
        raise ValueError(
            "Attempt preflight lacks the exact complete common-seed phase authorization."
        )


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and nonnegative.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and nonnegative.") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return numeric


def _expected_timing_bindings(
    job: Mapping[str, Any], manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    benchmark = {
        "name": finer.BENCHMARK_NAME,
        "stage": STAGE,
        "condition_id": job["condition_id"],
        "prompt_id": job["prompt_id"],
        "variant": "01_baseline",
        "seed": job["seed"],
        "attempt": job["attempt"],
        "manifest_sha256": manifest_sha256,
        "model_revision": MODEL_REVISION,
    }
    generation = {
        **deepcopy(dict(job["generation"])),
        "prompt": job["prompt"],
        "prompt_file": None,
    }
    model = {
        "adapter": "flux",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "pipeline_class": PIPELINE_CLASS,
    }
    return benchmark, generation, model


def _require_mapping_subset(payload: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if drift:
        raise ValueError(f"{label} binding drifted: {drift}")


def _validate_timing(
    path: Path,
    label: str,
    *,
    job: Mapping[str, Any],
    manifest_sha256: str,
    media_path: Path,
) -> dict[str, Any]:
    payload = _load_json(path, label)
    if payload.get("status") != "completed":
        raise ValueError(f"{label} is not completed: {path}")
    if label == "experiment timing":
        if set(payload) != {
            "started_at_utc",
            "finished_at_utc",
            "wall_seconds",
            "status",
        }:
            raise ValueError("Experiment timing schema differs from the exact writer schema.")
        started = _parse_aware_timestamp(
            payload.get("started_at_utc"), "experiment timing started_at_utc"
        )
        finished = _parse_aware_timestamp(
            payload.get("finished_at_utc"), "experiment timing finished_at_utc"
        )
        if finished < started:
            raise ValueError("Experiment timing finishes before it starts.")
        _finite_nonnegative(payload.get("wall_seconds"), f"{label} wall_seconds")
        return payload

    benchmark, generation, model = _expected_timing_bindings(job, manifest_sha256)
    if payload.get("schema_version") != 1:
        raise ValueError(f"{label} schema_version must be exactly 1.")
    started = _parse_aware_timestamp(payload.get("started_at"), f"{label} started_at")
    ended = _parse_aware_timestamp(payload.get("ended_at"), f"{label} ended_at")
    if ended < started:
        raise ValueError(f"{label} finishes before it starts.")
    _finite_nonnegative(payload.get("total_seconds"), f"{label} total_seconds")
    _require_mapping_subset(payload.get("benchmark"), benchmark, f"{label} benchmark")
    if payload.get("generation") != generation:
        raise ValueError(f"{label} generation binding drifted.")
    if payload.get("model") != model:
        raise ValueError(f"{label} model binding drifted.")
    if label == "run timing":
        _finite_nonnegative(payload.get("adapter_load_seconds"), "run timing adapter_load_seconds")
        records = payload.get("records")
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError("Run timing must bind exactly one generated record.")
        record = records[0]
        _require_mapping_subset(
            record,
            {"prompt": job["prompt"], "sample_id": "sample_0000"},
            "run timing record",
        )
        output_paths = record.get("output_paths")
        if not isinstance(output_paths, Mapping) or output_paths.get("image_0") != str(media_path):
            raise ValueError("Run timing record does not bind the exact PNG.")
    elif label == "sample timing":
        if (
            payload.get("sample_id") != "sample_0000"
            or payload.get("prompt") != job["prompt"]
            or payload.get("task") != "text_to_image"
        ):
            raise ValueError("Sample timing identity/prompt/task drifted.")
        media = payload.get("media")
        if not isinstance(media, Mapping) or media.get("present") is not True:
            raise ValueError("Sample timing does not attest generated image media.")
        output_paths = payload.get("output_paths")
        if not isinstance(output_paths, Mapping) or output_paths.get("image_0") != str(media_path):
            raise ValueError("Sample timing does not bind the exact PNG.")
    else:
        raise ValueError(f"Unknown timing role: {label!r}.")
    return payload


def _validate_one_result(
    *,
    job: Mapping[str, Any],
    job_index: int,
    manifest_sha256: str,
    manifest_path: Path,
    root: Path,
    registry_entry: Mapping[str, Any],
    media_validator: Callable[[Path, dict[str, Any], bool], dict[str, Any]],
    preflight_reader: Callable[..., dict[str, Any]],
    launch_authorization_builder: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    bound = _bound_job(job, manifest_sha256, job_index)
    output_dir = Path(str(job["output_dir"])).resolve()
    result_path = output_dir / "benchmark_job_result.json"
    result = _load_json(result_path, "benchmark result")
    if (
        result.get("schema_version") != 2
        or result.get("status") != "completed"
        or result.get("job") != bound
    ):
        raise ValueError(
            f"Completed result/launch-job binding is invalid at manifest index {job_index}."
        )
    validation = media_validator(output_dir, bound, True)
    media_path = Path(str(validation["path"])).resolve()
    if result.get("media_validation") != validation or result.get("validated_media_paths") != [
        str(media_path)
    ]:
        raise ValueError(f"Result media binding drifted at manifest index {job_index}.")
    if validation.get("sha256") != sha256_file(media_path):
        raise ValueError(f"Media digest drifted at manifest index {job_index}.")
    expected_media_contract = {
        "path": str((output_dir / "sample_0000" / "image_000.png").resolve()),
        "media_type": "image/png",
        "width": int(job["generation"]["width"]),
        "height": int(job["generation"]["height"]),
        "mode": "RGB",
        "decode_verified": True,
        "sha256": sha256_file(media_path),
        "size_bytes": media_path.stat().st_size,
    }
    if validation != expected_media_contract:
        raise ValueError(f"Exact RGB PNG contract drifted at manifest index {job_index}.")

    preflight = preflight_reader(
        output_dir,
        expected_job=bound,
        expected_job_index=job_index,
    )
    preflight_path = output_dir / "environment_preflight.json"
    preflight_sidecar = output_dir / "environment_preflight.json.sha256"
    identity_path = output_dir / "execution_identity.json"
    identity = read_execution_identity(identity_path)
    expected_identity = {
        "SLURM_ARRAY_JOB_ID": str(registry_entry["slurm_array_job_id"]),
        "SLURM_ARRAY_TASK_ID": str(job_index),
        "SLURM_JOB_NAME": SLURM_JOB_NAME,
        "slurm_task_id": str(registry_entry["slurm_task_id"]),
    }
    observed_identity = {key: identity.get(key) for key in expected_identity}
    if observed_identity != expected_identity:
        raise ValueError(f"Runtime Slurm identity differs from registry at index {job_index}.")
    if registry_entry.get("slurm_array_task_id") != job_index:
        raise ValueError("Registry task index does not equal manifest job index.")
    expected_authorization = launch_authorization_builder(
        phase_plan_path=default_phase_plan_path(root),
        manifest_path=manifest_path,
        job_index=job_index,
        root=root,
        environ={
            "SLURM_ARRAY_JOB_ID": str(identity.get("SLURM_ARRAY_JOB_ID", "")),
            "SLURM_ARRAY_TASK_ID": str(identity.get("SLURM_ARRAY_TASK_ID", "")),
            "SLURM_JOB_ID": str(identity.get("SLURM_JOB_ID", "")),
            "SLURM_JOB_NAME": str(identity.get("SLURM_JOB_NAME", "")),
        },
    )
    if preflight.get("common_seed_launch_authorization") != expected_authorization:
        raise ValueError(
            f"Attempt {job_index} did not persist the exact complete held-phase authorization."
        )

    experiment_path = output_dir / "experiment_timing.json"
    run_path = output_dir / "run_timing.json"
    sample_path = output_dir / "sample_0000" / "timing.json"
    timing_kwargs = {
        "job": job,
        "manifest_sha256": manifest_sha256,
        "media_path": media_path,
    }
    experiment = _validate_timing(experiment_path, "experiment timing", **timing_kwargs)
    run = _validate_timing(run_path, "run timing", **timing_kwargs)
    sample = _validate_timing(sample_path, "sample timing", **timing_kwargs)
    return {
        "seed": int(job["seed"]),
        "prompt_id": str(job["prompt_id"]),
        "prompt": str(job["prompt"]),
        "condition_id": str(job["condition_id"]),
        "attempt": int(job["attempt"]),
        "manifest_job_index": job_index,
        "width": int(job["generation"]["width"]),
        "height": int(job["generation"]["height"]),
        "media_path": str(media_path),
        "media_sha256": validation["sha256"],
        "media_size_bytes": media_path.stat().st_size,
        "runtime_seconds": {
            "experiment_wall": float(experiment["wall_seconds"]),
            "run_total": float(run["total_seconds"]),
            "sample_total": float(sample["total_seconds"]),
        },
        "environment_status": preflight["status"],
        "common_seed_launch_authorization": deepcopy(expected_authorization),
        "artifact_bindings": {
            "benchmark_job_result": _binding(result_path),
            "media": _binding(media_path),
            "environment_preflight": _binding(preflight_path),
            "environment_preflight_sidecar": _binding(preflight_sidecar),
            "execution_identity": _binding(identity_path),
            "experiment_timing": _binding(experiment_path),
            "run_timing": _binding(run_path),
            "sample_timing": _binding(sample_path),
        },
    }


def collect_candidate_artifacts(
    *,
    manifest_paths: Sequence[str | Path],
    registry_paths: Sequence[str | Path],
    root: Path | None = None,
    media_validator: Callable[
        [Path, dict[str, Any], bool], dict[str, Any]
    ] = finer.validate_exact_media,
    preflight_reader: Callable[..., dict[str, Any]] = read_environment_preflight,
) -> dict[str, Any]:
    """Authenticate the complete 24-image baseline cohort before blinding."""

    root = (root or project_root()).resolve()
    sealed_config, sealed_inputs = _load_protocol(root, require_sealed=True)
    sealed_gate_ids = _hard_gate_ids_from_config(sealed_config)
    if len(manifest_paths) != EXPECTED_MANIFESTS or len(registry_paths) != EXPECTED_MANIFESTS:
        raise ValueError("Authentication requires exactly eight manifests and eight registries.")
    expected_manifest_paths = {default_manifest_path(root, seed) for seed in SEEDS}
    observed_manifest_paths = {Path(path).expanduser().resolve() for path in manifest_paths}
    expected_registry_paths = {default_phase_registry_path(root, seed) for seed in SEEDS}
    observed_registry_paths = {Path(path).expanduser().resolve() for path in registry_paths}
    if observed_manifest_paths != expected_manifest_paths:
        raise ValueError("Authentication requires the eight canonical committed manifest paths.")
    if observed_registry_paths != expected_registry_paths:
        raise ValueError("Authentication requires the eight canonical phase registry paths.")
    launch_commit = read_common_seed_phase_launch_commit(root)
    release_receipt = read_common_seed_phase_release_receipt(root)
    manifests: dict[int, tuple[Path, dict[str, Any]]] = {}
    for raw_path in manifest_paths:
        path = Path(raw_path).expanduser().resolve()
        manifest = read_seed_manifest_for_audit(path, root)
        if manifest.get("common_seed_protocol_inputs") != sealed_inputs:
            raise ValueError(
                "Common-seed manifest protocol inputs differ from the currently "
                "sealed review/selection contract."
            )
        seed = _validate_seed(manifest.get("seed"))
        if seed in manifests:
            raise ValueError(f"Duplicate manifest ownership for seed {seed}.")
        manifests[seed] = (path, manifest)
    if tuple(sorted(manifests)) != SEEDS:
        raise ValueError("Manifest cohort does not cover exact seeds 0..7.")

    registries: dict[str, tuple[Path, dict[str, Any]]] = {}
    for raw_path in registry_paths:
        path = Path(raw_path).expanduser().resolve()
        registry = read_submission_registry(path)
        digest = str(registry.get("manifest_sha256", ""))
        if digest in registries:
            raise ValueError("More than one registry claims the same manifest.")
        registries[digest] = (path, registry)
    _assert_unique_slurm_array_ids([registry for _path, registry in registries.values()])

    rows: list[dict[str, Any]] = []
    source_manifests: list[dict[str, Any]] = []
    for seed in SEEDS:
        manifest_path, manifest = manifests[seed]
        manifest_sha = str(manifest["manifest_sha256"])
        if manifest_sha not in registries:
            raise ValueError(f"Seed {seed} has no matching submission registry.")
        registry_path, registry = registries[manifest_sha]
        expected_indices = list(range(ROWS_PER_MANIFEST))
        if (
            registry.get("benchmark") != finer.BENCHMARK_NAME
            or Path(str(registry.get("manifest_path", ""))).resolve() != manifest_path
            or registry.get("manifest_sha256") != manifest_sha
            or registry.get("slurm_job_name") != SLURM_JOB_NAME
            or registry.get("num_registered_tasks") != ROWS_PER_MANIFEST
            or [entry.get("job_index") for entry in registry.get("submissions", ())]
            != expected_indices
        ):
            raise ValueError(f"Seed {seed} registry does not cover exact indices 0..2.")
        by_index = {int(row["job_index"]): row for row in registry["submissions"]}
        source_manifests.append(
            {
                "seed": seed,
                "manifest": {
                    **_binding(manifest_path),
                    "manifest_sha256": manifest_sha,
                },
                "manifest_sidecar": _binding(
                    manifest_path.with_suffix(manifest_path.suffix + ".sha256")
                ),
                "snapshot_index": _binding(Path(f"{manifest_path}.snapshot") / "index.json"),
                "submission_registry": {
                    **_binding(registry_path),
                    "registry_sha256": registry["registry_sha256"],
                    "slurm_array_job_id": registry["slurm_array_job_id"],
                },
            }
        )
        for index, job in enumerate(manifest["jobs"]):
            rows.append(
                _validate_one_result(
                    job=job,
                    job_index=index,
                    manifest_sha256=manifest_sha,
                    manifest_path=manifest_path,
                    root=root,
                    registry_entry=by_index[index],
                    media_validator=media_validator,
                    preflight_reader=preflight_reader,
                    launch_authorization_builder=build_common_seed_launch_authorization,
                )
            )
    identities = {(row["seed"], row["prompt_id"]) for row in rows}
    expected = {(seed, prompt_id) for seed in SEEDS for prompt_id in PROMPT_IDS}
    if identities != expected or len(rows) != EXPECTED_ROWS:
        raise ValueError("Authenticated cohort is not the exact seed-by-prompt grid.")
    if len({row["media_path"] for row in rows}) != EXPECTED_ROWS:
        raise ValueError("Authenticated cohort contains duplicate media paths.")
    return {
        "rows": rows,
        "source_manifests": source_manifests,
        "config_binding": _binding(root / CONFIG_RELATIVE),
        "protocol_binding": _binding(root / PROTOCOL_RELATIVE),
        "protocol_inputs": deepcopy(sealed_inputs),
        "phase_evidence": {
            "cohort_commit": {
                **_binding(cohort_commit_path(root)),
                "document_sha256": _load_json(
                    cohort_commit_path(root), "common-seed cohort commit"
                )["document_sha256"],
            },
            "phase_plan": {
                **_binding(default_phase_plan_path(root)),
                "phase_plan_sha256": launch_commit["phase_plan"]["phase_plan_sha256"],
            },
            "phase_state": deepcopy(launch_commit["phase_state"]),
            "phase_launch_commit": {
                **_binding(default_phase_launch_commit_path(root)),
                "document_sha256": launch_commit["document_sha256"],
            },
            "phase_release_receipt": {
                **_binding(default_phase_release_receipt_path(root)),
                "document_sha256": release_receipt["document_sha256"],
            },
        },
        "source_only_hard_gate_ids": {
            prompt_id: list(sealed_gate_ids[prompt_id]) for prompt_id in PROMPT_IDS
        },
    }


def _assert_unique_slurm_array_ids(registries: Sequence[Mapping[str, Any]]) -> None:
    if len(registries) != EXPECTED_MANIFESTS:
        raise ValueError("Exactly eight registries are required for array-ID validation.")
    array_job_ids = [str(registry.get("slurm_array_job_id", "")) for registry in registries]
    if (
        any(not re.fullmatch(r"[0-9]+", job_id) for job_id in array_job_ids)
        or len(set(array_job_ids)) != EXPECTED_MANIFESTS
    ):
        raise ValueError(
            "Each seed manifest requires its own Slurm array job ID; reused IDs are forbidden."
        )


def build_authentication_report(collected: Mapping[str, Any]) -> dict[str, Any]:
    rows = deepcopy(list(collected.get("rows") or ()))
    manifests = deepcopy(list(collected.get("source_manifests") or ()))
    if len(rows) != EXPECTED_ROWS or len(manifests) != EXPECTED_MANIFESTS:
        raise ValueError("Cannot authenticate an incomplete common-seed cohort.")
    report: dict[str, Any] = {
        "schema_version": 1,
        "authentication": AUTHENTICATION_NAME,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "protocol_id": PROTOCOL_ID,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "coverage": {
            "seeds": list(SEEDS),
            "prompt_ids": list(PROMPT_IDS),
            "manifests": EXPECTED_MANIFESTS,
            "rows": EXPECTED_ROWS,
            "completed_media": EXPECTED_ROWS,
            "environment_preflights": EXPECTED_ROWS,
            "slurm_execution_identities": EXPECTED_ROWS,
        },
        "source_bindings": {
            "config": deepcopy(collected["config_binding"]),
            "protocol": deepcopy(collected["protocol_binding"]),
            "protocol_inputs": deepcopy(collected["protocol_inputs"]),
            "phase_evidence": deepcopy(collected["phase_evidence"]),
            "manifests_and_registries": manifests,
        },
        "source_only_hard_gate_ids": deepcopy(collected["source_only_hard_gate_ids"]),
        "rows": rows,
        "review_admission": {
            "baseline_media_only": True,
            "all_24_rows_authenticated": True,
            "v1_calibration_outputs_consulted": False,
            "negative_or_target_or_steered_outputs_consulted": False,
        },
    }
    report["document_sha256"] = document_sha256(report)
    validate_authentication_report(report)
    return report


def validate_authentication_report(payload: Mapping[str, Any]) -> None:
    expected_top_fields = {
        "schema_version",
        "authentication",
        "status",
        "created_at_utc",
        "protocol_id",
        "model_name",
        "model_revision",
        "coverage",
        "source_bindings",
        "source_only_hard_gate_ids",
        "rows",
        "review_admission",
        "document_sha256",
    }
    if (
        set(payload) != expected_top_fields
        or payload.get("schema_version") != 1
        or payload.get("authentication") != AUTHENTICATION_NAME
        or payload.get("status") != "complete"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("model_name") != MODEL_NAME
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("document_sha256") != document_sha256(payload)
    ):
        raise ValueError("Common-seed authentication report identity/digest is invalid.")
    _parse_aware_timestamp(payload.get("created_at_utc"), "authentication created_at_utc")
    coverage = payload.get("coverage") or {}
    expected_coverage = {
        "seeds": list(SEEDS),
        "prompt_ids": list(PROMPT_IDS),
        "manifests": EXPECTED_MANIFESTS,
        "rows": EXPECTED_ROWS,
        "completed_media": EXPECTED_ROWS,
        "environment_preflights": EXPECTED_ROWS,
        "slurm_execution_identities": EXPECTED_ROWS,
    }
    rows = payload.get("rows")
    source_bindings = payload.get("source_bindings") or {}
    if not isinstance(source_bindings, Mapping) or set(source_bindings) != {
        "config",
        "protocol",
        "protocol_inputs",
        "phase_evidence",
        "manifests_and_registries",
    }:
        raise ValueError("Authentication report source-binding schema drifted.")
    manifests = source_bindings.get("manifests_and_registries")
    if (
        coverage != expected_coverage
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_ROWS
        or not isinstance(manifests, list)
        or len(manifests) != EXPECTED_MANIFESTS
    ):
        raise ValueError("Common-seed authentication coverage is incomplete.")
    protocol_inputs = source_bindings.get("protocol_inputs")
    if (
        not isinstance(protocol_inputs, Mapping)
        or set(protocol_inputs) != set(finer.FLUX_COMMON_SEED_PROTOCOL_INPUT_ROLES)
        or {key: (source_bindings.get("config") or {}).get(key) for key in ("path", "sha256")}
        != protocol_inputs.get("common_seed_config")
        or {key: (source_bindings.get("protocol") or {}).get(key) for key in ("path", "sha256")}
        != protocol_inputs.get("common_seed_protocol")
    ):
        raise ValueError("Authentication report protocol-input binding is incomplete.")
    phase_evidence = source_bindings.get("phase_evidence")
    if not isinstance(phase_evidence, Mapping) or set(phase_evidence) != {
        "cohort_commit",
        "phase_plan",
        "phase_state",
        "phase_launch_commit",
        "phase_release_receipt",
    }:
        raise ValueError("Authentication report complete-phase evidence is incomplete.")
    for role in (
        "cohort_commit",
        "phase_plan",
        "phase_state",
        "phase_launch_commit",
        "phase_release_receipt",
    ):
        record = phase_evidence[role]
        if (
            not isinstance(record, Mapping)
            or _SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
        ):
            raise ValueError(f"Authentication phase evidence {role} is malformed.")
    frozen_gates = payload.get("source_only_hard_gate_ids")
    if (
        not isinstance(frozen_gates, Mapping)
        or set(frozen_gates) != set(PROMPT_IDS)
        or any(
            not isinstance(frozen_gates[prompt_id], list)
            or not frozen_gates[prompt_id]
            or len(frozen_gates[prompt_id]) != len(set(frozen_gates[prompt_id]))
            or any(
                not isinstance(gate, str) or not gate.strip() for gate in frozen_gates[prompt_id]
            )
            or any(
                gate not in frozen_gates[prompt_id]
                for gate in MANDATORY_PROMPT_GATES.get(prompt_id, ())
            )
            for prompt_id in PROMPT_IDS
        )
    ):
        raise ValueError("Authentication report hard-gate contract is invalid.")
    identities = {(row.get("seed"), row.get("prompt_id")) for row in rows}
    if identities != {(seed, prompt) for seed in SEEDS for prompt in PROMPT_IDS}:
        raise ValueError("Common-seed authentication row grid is incomplete or duplicated.")
    admission = payload.get("review_admission")
    if admission != {
        "baseline_media_only": True,
        "all_24_rows_authenticated": True,
        "v1_calibration_outputs_consulted": False,
        "negative_or_target_or_steered_outputs_consulted": False,
    }:
        raise ValueError("Authentication report does not attest source-only admission.")


_FORBIDDEN_TARGET_PHRASES = (
    "pink daytime sky",
    "fixed marble staircase",
    "ceramic tile",
    "new arrival",
    "passenger cars",
    "happy smiling",
    "happy relaxed",
    "full-body walking",
    "red and blue fabric",
    "away from her mouth",
    "away from his mouth",
    "dynamic candid",
)


def _contains_seed_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            "seed" in str(key).lower() or _contains_seed_key(item) for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_seed_key(item) for item in value)
    return False


def _validate_reviewer_manifest(payload: Mapping[str, Any]) -> None:
    expected_top_fields = {
        "schema_version",
        "review",
        "status",
        "created_at_utc",
        "protocol_id",
        "model_name",
        "authentication_document_sha256",
        "coverage",
        "review_instructions",
        "source_only_hard_gate_ids",
        "rows",
        "document_sha256",
    }
    if (
        set(payload) != expected_top_fields
        or payload.get("schema_version") != 1
        or payload.get("review") != REVIEW_NAME
        or payload.get("status") != "ready_for_source_only_review"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("model_name") != MODEL_NAME
        or payload.get("document_sha256") != document_sha256(payload)
    ):
        raise ValueError("Blinded reviewer manifest identity/digest is invalid.")
    _parse_aware_timestamp(payload.get("created_at_utc"), "reviewer created_at_utc")
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("authentication_document_sha256", ""))):
        raise ValueError("Blinded reviewer authentication digest is malformed.")
    if payload.get("coverage") != {
        "candidate_images": EXPECTED_ROWS,
        "prompt_ids": list(PROMPT_IDS),
        "candidates_per_prompt": len(SEEDS),
    }:
        raise ValueError("Blinded reviewer coverage contract drifted.")
    if payload.get("review_instructions") != {
        "view_every_image_at_original_resolution": True,
        "score_every_hard_gate_as_boolean_with_nonempty_notes": True,
        "source_fidelity_preservation_and_framing_only": True,
        "quality_scores_or_candidate_ranking_forbidden": True,
        "do_not_open_parent_directory_or_private_unblinding_document": True,
        "do_not_inspect_negative_target_or_steered_material": True,
    }:
        raise ValueError("Blinded reviewer instructions drifted.")
    frozen_gates = payload.get("source_only_hard_gate_ids")
    if (
        not isinstance(frozen_gates, Mapping)
        or set(frozen_gates) != set(PROMPT_IDS)
        or any(
            not isinstance(frozen_gates[prompt_id], list)
            or not frozen_gates[prompt_id]
            or len(frozen_gates[prompt_id]) != len(set(frozen_gates[prompt_id]))
            or any(
                not isinstance(gate, str) or not gate.strip() for gate in frozen_gates[prompt_id]
            )
            or any(
                gate not in frozen_gates[prompt_id]
                for gate in MANDATORY_PROMPT_GATES.get(prompt_id, ())
            )
            for prompt_id in PROMPT_IDS
        )
    ):
        raise ValueError("Blinded reviewer hard-gate contract is invalid.")
    if _contains_seed_key(payload):
        raise ValueError("Blinded reviewer manifest exposes a seed-named field.")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise ValueError("Blinded reviewer manifest must contain exactly 24 rows.")
    if len({row.get("blind_id") for row in rows}) != EXPECTED_ROWS:
        raise ValueError("Blinded reviewer IDs are missing or duplicated.")
    if {row.get("prompt_id") for row in rows} != set(PROMPT_IDS):
        raise ValueError("Blinded reviewer prompt coverage is incomplete.")
    if {
        prompt_id: sum(row.get("prompt_id") == prompt_id for row in rows)
        for prompt_id in PROMPT_IDS
    } != {prompt_id: len(SEEDS) for prompt_id in PROMPT_IDS}:
        raise ValueError("Blinded reviewer prompt multiplicities are not exactly 8 each.")
    serialized = json.dumps(payload, sort_keys=True).lower()
    if re.search(r"seed_[0-9]", serialized):
        raise ValueError("Blinded reviewer manifest leaks a seed-scoped source path.")
    if any(phrase in serialized for phrase in _FORBIDDEN_TARGET_PHRASES):
        raise ValueError("Blinded reviewer manifest contains a steering target phrase.")
    expected_row_fields = {
        "review_position",
        "blind_id",
        "prompt_id",
        "source_prompt",
        "media_file",
        "media_sha256",
        "media_size_bytes",
        "width",
        "height",
        "hard_gate_ids",
        "original_resolution_review_required",
    }
    for expected_position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise ValueError("A blinded reviewer row differs from the exact frozen schema.")
        prompt_id = str(row.get("prompt_id"))
        if row.get("review_position") != expected_position:
            raise ValueError("Blinded reviewer positions must be exactly 1..24 in row order.")
        if not isinstance(row.get("source_prompt"), str) or not str(row["source_prompt"]).strip():
            raise ValueError("Blinded reviewer source prompt is absent or malformed.")
        layout = IMAGE_LAYOUTS[prompt_id]
        if (
            row.get("width") != layout["width"]
            or row.get("height") != layout["height"]
            or row.get("original_resolution_review_required") is not True
            or isinstance(row.get("media_size_bytes"), bool)
            or not isinstance(row.get("media_size_bytes"), int)
            or row["media_size_bytes"] <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("media_sha256", "")))
        ):
            raise ValueError("Blinded reviewer media contract is malformed.")
        expected_gates = frozen_gates[prompt_id]
        if row.get("hard_gate_ids") != expected_gates:
            raise ValueError(f"Blinded hard-gate rubric drifted for {prompt_id}.")
        if not re.fullmatch(r"blind_[0-9a-f]{24}", str(row.get("blind_id", ""))):
            raise ValueError("Blinded candidate ID is malformed.")
        if row.get("media_file") != f"media/{row['blind_id']}.png":
            raise ValueError("Blinded media filename is not opaque and canonical.")


def _validate_unblinding_map(
    payload: Mapping[str, Any],
    reviewer_manifest: Mapping[str, Any],
    authentication: Mapping[str, Any],
) -> None:
    _validate_reviewer_manifest(reviewer_manifest)
    validate_authentication_report(authentication)
    expected_top_fields = {
        "schema_version",
        "unblinding",
        "status",
        "created_at_utc",
        "protocol_id",
        "authentication_document_sha256",
        "reviewer_manifest_document_sha256",
        "randomization_nonce_sha256",
        "rows",
        "document_sha256",
    }
    if (
        set(payload) != expected_top_fields
        or payload.get("schema_version") != 1
        or payload.get("unblinding") != UNBLINDING_NAME
        or payload.get("status") != "private_until_manual_review_complete"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("reviewer_manifest_document_sha256")
        != reviewer_manifest.get("document_sha256")
        or payload.get("authentication_document_sha256") != authentication.get("document_sha256")
        or payload.get("document_sha256") != document_sha256(payload)
    ):
        raise ValueError("Private common-seed unblinding map is invalid.")
    _parse_aware_timestamp(payload.get("created_at_utc"), "unblinding created_at_utc")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(payload.get("randomization_nonce_sha256", ""))
    ) or reviewer_manifest.get("authentication_document_sha256") != authentication.get(
        "document_sha256"
    ):
        raise ValueError("Private unblinding provenance binding is invalid.")
    if reviewer_manifest.get("source_only_hard_gate_ids") != authentication.get(
        "source_only_hard_gate_ids"
    ):
        raise ValueError("Reviewer rubric differs from the authenticated frozen hard gates.")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise ValueError("Private unblinding map must contain exactly 24 rows.")
    reviewer_by_id = {row["blind_id"]: row for row in reviewer_manifest["rows"]}
    reviewer_ids = set(reviewer_by_id)
    if {row.get("blind_id") for row in rows} != reviewer_ids:
        raise ValueError("Unblinding and reviewer candidate IDs differ.")
    identities = {(row.get("seed"), row.get("prompt_id")) for row in rows}
    if identities != {(seed, prompt) for seed in SEEDS for prompt in PROMPT_IDS}:
        raise ValueError("Private unblinding map does not cover the exact 8x3 grid.")
    auth_by_condition = {row["condition_id"]: row for row in authentication.get("rows") or ()}
    if len(auth_by_condition) != EXPECTED_ROWS:
        raise ValueError("Authentication report has duplicate/missing condition identities.")
    expected_fields = {
        "blind_id",
        "seed",
        "prompt_id",
        "condition_id",
        "attempt",
        "manifest_job_index",
        "source_media_path",
        "source_media_sha256",
        "source_media_size_bytes",
        "width",
        "height",
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError("Private unblinding row fields drifted.")
        condition_id = str(row["condition_id"])
        source = auth_by_condition.get(condition_id)
        if source is None:
            raise ValueError("Unblinding row does not identify an authenticated candidate.")
        expected_source = {
            "seed": source["seed"],
            "prompt_id": source["prompt_id"],
            "condition_id": source["condition_id"],
            "attempt": source["attempt"],
            "manifest_job_index": source["manifest_job_index"],
            "source_media_path": source["media_path"],
            "source_media_sha256": source["media_sha256"],
            "source_media_size_bytes": source["media_size_bytes"],
            "width": source["width"],
            "height": source["height"],
        }
        observed_source = {key: row.get(key) for key in expected_source}
        if observed_source != expected_source:
            raise ValueError(
                "Unblinding row was rotated or differs from its authenticated source row."
            )
        review_row = reviewer_by_id[row["blind_id"]]
        expected_review = {
            "prompt_id": source["prompt_id"],
            "source_prompt": source["prompt"],
            "media_sha256": source["media_sha256"],
            "media_size_bytes": source["media_size_bytes"],
            "width": source["width"],
            "height": source["height"],
        }
        if any(review_row.get(key) != value for key, value in expected_review.items()):
            raise ValueError(
                "Blinded reviewer row differs from the authenticated unblinded candidate."
            )


def build_review_documents(
    collected: Mapping[str, Any], *, randomization_nonce: str | None = None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build authenticated public/private review documents without writing files."""

    authentication = build_authentication_report(collected)
    nonce = randomization_nonce or secrets.token_hex(32)
    if not isinstance(nonce, str) or len(nonce) < 32:
        raise ValueError("Review randomization nonce must contain at least 32 characters.")
    rows = list(authentication["rows"])
    blinded: list[tuple[str, dict[str, Any]]] = []
    for ordinal, row in enumerate(rows):
        token = "|".join(
            (
                nonce,
                str(row["condition_id"]),
                str(row["media_sha256"]),
                str(ordinal),
            )
        )
        blind_id = f"blind_{hashlib.sha256(token.encode()).hexdigest()[:24]}"
        blinded.append((blind_id, row))
    if len({blind_id for blind_id, _ in blinded}) != EXPECTED_ROWS:
        raise RuntimeError("Randomized blind-ID collision; choose a new nonce.")
    rng = random.Random(int(hashlib.sha256(nonce.encode()).hexdigest(), 16))
    rng.shuffle(blinded)
    gates = {
        prompt_id: tuple(authentication["source_only_hard_gate_ids"][prompt_id])
        for prompt_id in PROMPT_IDS
    }
    reviewer_rows = [
        {
            "review_position": position,
            "blind_id": blind_id,
            "prompt_id": row["prompt_id"],
            "source_prompt": row["prompt"],
            "media_file": f"media/{blind_id}.png",
            "media_sha256": row["media_sha256"],
            "media_size_bytes": row["media_size_bytes"],
            "width": row["width"],
            "height": row["height"],
            "hard_gate_ids": list(gates[row["prompt_id"]]),
            "original_resolution_review_required": True,
        }
        for position, (blind_id, row) in enumerate(blinded, start=1)
    ]
    reviewer: dict[str, Any] = {
        "schema_version": 1,
        "review": REVIEW_NAME,
        "status": "ready_for_source_only_review",
        "created_at_utc": _utc_now(),
        "protocol_id": PROTOCOL_ID,
        "model_name": MODEL_NAME,
        "authentication_document_sha256": authentication["document_sha256"],
        "coverage": {
            "candidate_images": EXPECTED_ROWS,
            "prompt_ids": list(PROMPT_IDS),
            "candidates_per_prompt": len(SEEDS),
        },
        "review_instructions": {
            "view_every_image_at_original_resolution": True,
            "score_every_hard_gate_as_boolean_with_nonempty_notes": True,
            "source_fidelity_preservation_and_framing_only": True,
            "quality_scores_or_candidate_ranking_forbidden": True,
            "do_not_open_parent_directory_or_private_unblinding_document": True,
            "do_not_inspect_negative_target_or_steered_material": True,
        },
        "source_only_hard_gate_ids": {
            prompt_id: list(gates[prompt_id]) for prompt_id in PROMPT_IDS
        },
        "rows": reviewer_rows,
    }
    reviewer["document_sha256"] = document_sha256(reviewer)
    unblinding_rows = [
        {
            "blind_id": blind_id,
            "seed": row["seed"],
            "prompt_id": row["prompt_id"],
            "condition_id": row["condition_id"],
            "attempt": row["attempt"],
            "manifest_job_index": row["manifest_job_index"],
            "source_media_path": row["media_path"],
            "source_media_sha256": row["media_sha256"],
            "source_media_size_bytes": row["media_size_bytes"],
            "width": row["width"],
            "height": row["height"],
        }
        for blind_id, row in blinded
    ]
    unblinding: dict[str, Any] = {
        "schema_version": 1,
        "unblinding": UNBLINDING_NAME,
        "status": "private_until_manual_review_complete",
        "created_at_utc": _utc_now(),
        "protocol_id": PROTOCOL_ID,
        "authentication_document_sha256": authentication["document_sha256"],
        "reviewer_manifest_document_sha256": reviewer["document_sha256"],
        "randomization_nonce_sha256": hashlib.sha256(nonce.encode()).hexdigest(),
        "rows": unblinding_rows,
    }
    unblinding["document_sha256"] = document_sha256(unblinding)
    template: dict[str, Any] = {
        "schema_version": 1,
        "review": MANUAL_REVIEW_NAME,
        "status": "draft_incomplete",
        "reviewer_identity": "",
        "reviewed_at_utc": "",
        "reviewer_manifest_document_sha256": reviewer["document_sha256"],
        "decisions": [
            {
                "blind_id": row["blind_id"],
                "prompt_id": row["prompt_id"],
                "original_resolution_viewed": False,
                "hard_gates": {
                    gate_id: {"passed": None, "notes": ""} for gate_id in row["hard_gate_ids"]
                },
            }
            for row in reviewer_rows
        ],
        "attestation": {
            "source_baseline_media_only": False,
            "no_seed_values_or_unblinding_map_seen": False,
            "no_negative_target_or_steered_outputs_seen": False,
            "no_quality_ranking_or_gate_relaxation_used": False,
        },
    }
    _validate_reviewer_manifest(reviewer)
    _validate_unblinding_map(unblinding, reviewer, authentication)
    return authentication, reviewer, unblinding, template


def normalize_manual_ledger(
    draft: Mapping[str, Any], reviewer_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    _validate_reviewer_manifest(reviewer_manifest)
    expected_keys = {
        "schema_version",
        "review",
        "status",
        "reviewer_identity",
        "reviewed_at_utc",
        "reviewer_manifest_document_sha256",
        "decisions",
        "attestation",
    }
    allowed = expected_keys | {"document_sha256"}
    if set(draft) != expected_keys and set(draft) != allowed:
        raise ValueError("Manual ledger fields differ from the exact frozen schema.")
    if (
        draft.get("schema_version") != 1
        or draft.get("review") != MANUAL_REVIEW_NAME
        or draft.get("status") != "completed"
        or draft.get("reviewer_manifest_document_sha256") != reviewer_manifest["document_sha256"]
    ):
        raise ValueError("Manual common-seed review identity/status is invalid.")
    reviewer_identity = str(draft.get("reviewer_identity", "")).strip()
    reviewed_at = str(draft.get("reviewed_at_utc", "")).strip()
    if not reviewer_identity:
        raise ValueError("Completed manual review requires reviewer_identity.")
    try:
        parsed_time = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("reviewed_at_utc must be ISO-8601 with timezone.") from exc
    if parsed_time.tzinfo is None or parsed_time.utcoffset() is None:
        raise ValueError("reviewed_at_utc must include a timezone.")
    expected_attestation = {
        "source_baseline_media_only": True,
        "no_seed_values_or_unblinding_map_seen": True,
        "no_negative_target_or_steered_outputs_seen": True,
        "no_quality_ranking_or_gate_relaxation_used": True,
    }
    if draft.get("attestation") != expected_attestation:
        raise ValueError("Manual review source-only/blinding attestation is incomplete.")
    expected_rows = {row["blind_id"]: row for row in reviewer_manifest["rows"]}
    decisions = draft.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != EXPECTED_ROWS:
        raise ValueError("Manual ledger must contain exactly 24 decisions.")
    normalized: list[dict[str, Any]] = []
    for decision in decisions:
        if not isinstance(decision, Mapping) or set(decision) != {
            "blind_id",
            "prompt_id",
            "original_resolution_viewed",
            "hard_gates",
        }:
            raise ValueError("A manual candidate decision has invalid fields.")
        blind_id = str(decision["blind_id"])
        if blind_id not in expected_rows:
            raise ValueError(f"Unknown blinded candidate {blind_id!r}.")
        expected_row = expected_rows[blind_id]
        if (
            decision["prompt_id"] != expected_row["prompt_id"]
            or decision["original_resolution_viewed"] is not True
        ):
            raise ValueError("Manual decision prompt/viewing attestation drifted.")
        hard_gates = decision["hard_gates"]
        expected_gate_ids = expected_row["hard_gate_ids"]
        if not isinstance(hard_gates, Mapping) or set(hard_gates) != set(expected_gate_ids):
            raise ValueError("Manual decision hard-gate coverage drifted.")
        normalized_gates: dict[str, dict[str, Any]] = {}
        for gate_id in expected_gate_ids:
            value = hard_gates[gate_id]
            if not isinstance(value, Mapping) or set(value) != {"passed", "notes"}:
                raise ValueError(f"Hard gate {gate_id!r} has invalid fields.")
            passed = value["passed"]
            notes = str(value["notes"]).strip()
            if not isinstance(passed, bool) or not notes:
                raise ValueError(f"Hard gate {gate_id!r} requires a Boolean and evidence notes.")
            normalized_gates[gate_id] = {"passed": passed, "notes": notes}
        normalized.append(
            {
                "blind_id": blind_id,
                "prompt_id": expected_row["prompt_id"],
                "original_resolution_viewed": True,
                "hard_gates": normalized_gates,
            }
        )
    if {row["blind_id"] for row in normalized} != set(expected_rows):
        raise ValueError("Manual ledger duplicates or omits blinded candidates.")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "review": MANUAL_REVIEW_NAME,
        "status": "completed",
        "reviewer_identity": reviewer_identity,
        "reviewed_at_utc": reviewed_at,
        "reviewer_manifest_document_sha256": reviewer_manifest["document_sha256"],
        "decisions": normalized,
        "attestation": expected_attestation,
    }
    payload["document_sha256"] = document_sha256(payload)
    supplied = draft.get("document_sha256")
    if supplied is not None and supplied != payload["document_sha256"]:
        raise ValueError("Supplied manual ledger document digest is inconsistent.")
    return payload


def validate_reviewed_media_evidence(
    payload: Mapping[str, Any],
    reviewer_manifest: Mapping[str, Any],
    *,
    verify_files: bool,
) -> None:
    _validate_reviewer_manifest(reviewer_manifest)
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "authenticated_after_manual_review"
        or payload.get("reviewer_manifest_document_sha256") != reviewer_manifest["document_sha256"]
        or payload.get("document_sha256") != document_sha256(payload)
    ):
        raise ValueError("Reviewed-media evidence identity/digest is invalid.")
    records = payload.get("records")
    if (
        not isinstance(records, list)
        or len(records) != EXPECTED_ROWS
        or payload.get("candidate_count") != EXPECTED_ROWS
        or payload.get("records_sha256") != canonical_sha256(records)
    ):
        raise ValueError("Reviewed-media evidence coverage/digest is incomplete.")
    reviewer_by_id = {row["blind_id"]: row for row in reviewer_manifest["rows"]}
    if {row.get("blind_id") for row in records} != set(reviewer_by_id):
        raise ValueError("Reviewed-media records differ from blinded reviewer coverage.")
    media_root = Path(str(payload.get("reviewed_media_root", ""))).resolve()
    review_root = Path(str(payload.get("review_root", ""))).resolve()
    if media_root != (review_root / "media").resolve():
        raise ValueError("Reviewed-media root is not the canonical review/media directory.")
    expected_fields = {
        "blind_id",
        "path",
        "sha256",
        "size_bytes",
        "width",
        "height",
        "mode",
        "format",
        "decode_verified",
    }
    expected_paths: set[Path] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise ValueError("Reviewed-media record fields drifted.")
        blind_id = str(record["blind_id"])
        review = reviewer_by_id[blind_id]
        expected_path = (media_root / f"{blind_id}.png").resolve()
        expected_paths.add(expected_path)
        expected = {
            "path": str(expected_path),
            "sha256": review["media_sha256"],
            "size_bytes": review["media_size_bytes"],
            "width": review["width"],
            "height": review["height"],
            "mode": "RGB",
            "format": "PNG",
            "decode_verified": True,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Reviewed-media record differs from its blinded review row.")
        if verify_files:
            path = expected_path
            if not path.is_file() or path.stat().st_size != record["size_bytes"]:
                raise ValueError(f"Reviewed media is missing or changed: {path}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"Reviewed media digest changed: {path}")
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                observed = (image.format, image.mode, image.width, image.height)
            if observed != (
                "PNG",
                "RGB",
                int(record["width"]),
                int(record["height"]),
            ):
                raise ValueError(f"Reviewed media decode contract changed: {path}")
    if verify_files:
        discovered = {path.resolve() for path in media_root.rglob("*") if path.is_file()}
        if discovered != expected_paths:
            raise ValueError("Reviewed media directory contains missing or additional files.")


def authenticate_reviewed_media(
    reviewer_manifest_path: str | Path,
    reviewer_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen the exact PNG copies that the blind reviewer actually inspected."""

    _validate_reviewer_manifest(reviewer_manifest)
    reviewer_path = Path(reviewer_manifest_path).expanduser().resolve()
    review_root = reviewer_path.parent
    media_root = (review_root / "media").resolve()
    records: list[dict[str, Any]] = []
    for review in reviewer_manifest["rows"]:
        blind_id = str(review["blind_id"])
        relative = Path(str(review["media_file"]))
        path = (review_root / relative).resolve()
        if path.parent != media_root or path.name != f"{blind_id}.png":
            raise ValueError("Reviewer media path escapes the exact blinded media root.")
        if not path.is_file():
            raise FileNotFoundError(f"Reviewed PNG is absent: {path}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            image_format = image.format
            mode = image.mode
            width, height = image.size
        records.append(
            {
                "blind_id": blind_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format,
                "decode_verified": True,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "authenticated_after_manual_review",
        "authenticated_at_utc": _utc_now(),
        "reviewer_manifest_document_sha256": reviewer_manifest["document_sha256"],
        "review_root": str(review_root),
        "reviewed_media_root": str(media_root),
        "candidate_count": EXPECTED_ROWS,
        "records_sha256": canonical_sha256(records),
        "records": records,
    }
    payload["document_sha256"] = document_sha256(payload)
    validate_reviewed_media_evidence(payload, reviewer_manifest, verify_files=True)
    return payload


def _compute_selection_outcomes(
    unblinding_map: Mapping[str, Any], ledger: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[int], int | None]:
    by_blind = {row["blind_id"]: row for row in ledger["decisions"]}
    outcomes: list[dict[str, Any]] = []
    eligible: list[int] = []
    for seed in SEEDS:
        mapped = sorted(
            (row for row in unblinding_map["rows"] if row["seed"] == seed),
            key=lambda row: PROMPT_IDS.index(row["prompt_id"]),
        )
        if [row["prompt_id"] for row in mapped] != list(PROMPT_IDS):
            raise ValueError(f"Unblinding order/coverage drifted for seed {seed}.")
        prompt_outcomes: dict[str, bool] = {}
        for row in mapped:
            decision = by_blind[row["blind_id"]]
            prompt_outcomes[row["prompt_id"]] = all(
                gate["passed"] for gate in decision["hard_gates"].values()
            )
        passes = all(prompt_outcomes.values())
        outcomes.append(
            {
                "seed": seed,
                "all_registered_gates_pass_by_prompt": prompt_outcomes,
                "eligible_common_seed": passes,
            }
        )
        if passes:
            eligible.append(seed)
    return outcomes, eligible, min(eligible) if eligible else None


def build_selection_report(
    authentication: Mapping[str, Any],
    reviewer_manifest: Mapping[str, Any],
    unblinding_map: Mapping[str, Any],
    manual_ledger: Mapping[str, Any],
    reviewed_media: Mapping[str, Any],
    manual_ledger_binding: Mapping[str, Any],
) -> dict[str, Any]:
    validate_authentication_report(authentication)
    _validate_reviewer_manifest(reviewer_manifest)
    _validate_unblinding_map(unblinding_map, reviewer_manifest, authentication)
    ledger = normalize_manual_ledger(manual_ledger, reviewer_manifest)
    validate_reviewed_media_evidence(reviewed_media, reviewer_manifest, verify_files=False)
    expected_ledger_binding = {
        "path",
        "sha256",
        "size_bytes",
        "document_sha256",
    }
    ledger_size = (
        manual_ledger_binding.get("size_bytes")
        if isinstance(manual_ledger_binding, Mapping)
        else None
    )
    if (
        not isinstance(manual_ledger_binding, Mapping)
        or set(manual_ledger_binding) != expected_ledger_binding
        or manual_ledger_binding.get("document_sha256") != ledger["document_sha256"]
        or not _SHA256_RE.fullmatch(str(manual_ledger_binding.get("sha256", "")))
        or isinstance(ledger_size, bool)
        or not isinstance(ledger_size, int)
        or ledger_size <= 0
    ):
        raise ValueError("Normalized manual-ledger publication binding is invalid.")
    if reviewer_manifest.get("authentication_document_sha256") != authentication.get(
        "document_sha256"
    ) or unblinding_map.get("authentication_document_sha256") != authentication.get(
        "document_sha256"
    ):
        raise ValueError("Review/unblinding documents bind another authentication report.")
    outcomes, eligible, selected = _compute_selection_outcomes(unblinding_map, ledger)
    report: dict[str, Any] = {
        "schema_version": 1,
        "selection": SELECTION_NAME,
        "status": "selected" if selected is not None else "no_selection",
        "created_at_utc": _utc_now(),
        "protocol_id": PROTOCOL_ID,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selection_rule": {
            "eligibility": "all_three_prompts_pass_every_registered_hard_gate",
            "choice": "numerically_lowest_eligible_common_seed",
            "quality_scores_or_averaging_used": False,
            "threshold_or_gate_relaxation_used": False,
        },
        "eligible_common_seeds": eligible,
        "selected_common_seed": selected,
        "seed_outcomes": outcomes,
        "source_document_sha256": {
            "authentication": authentication["document_sha256"],
            "reviewer_manifest": reviewer_manifest["document_sha256"],
            "private_unblinding_map": unblinding_map["document_sha256"],
            "manual_review_ledger": ledger["document_sha256"],
            "reviewed_media": reviewed_media["document_sha256"],
        },
        "normalized_manual_review_ledger": deepcopy(dict(manual_ledger_binding)),
        "reviewed_media": deepcopy(dict(reviewed_media)),
        "later_calibration_v2_binding": {
            "calibration_id": CALIBRATION_V2_ID,
            "required_selected_common_seed": selected,
            "required_rows": 45,
            "required_prompt_ids": list(PROMPT_IDS),
            "every_row_must_use_selected_common_seed": True,
            "v1_scale_outputs_may_be_reused": False,
            "scale_selection_rule_may_be_relaxed": False,
            "launch_authorized": selected is not None,
        },
        "no_selection_action": (
            None
            if selected is not None
            else "repair_protocol_then_rerun_all_eight_seeds_without_reusing_this_cohort"
        ),
    }
    report["document_sha256"] = document_sha256(report)
    validate_selection_report(report)
    return report


def validate_selection_report(
    payload: Mapping[str, Any],
    *,
    reviewer_manifest: Mapping[str, Any] | None = None,
    verify_files: bool = False,
) -> None:
    """Validate the public common-seed receipt consumed by calibration v2."""

    if (
        payload.get("schema_version") != 1
        or payload.get("selection") != SELECTION_NAME
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("model_name") != MODEL_NAME
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("status") not in {"selected", "no_selection"}
        or payload.get("document_sha256") != document_sha256(payload)
    ):
        raise ValueError("Common-seed selection receipt identity/digest is invalid.")
    eligible = payload.get("eligible_common_seeds")
    selected = payload.get("selected_common_seed")
    if (
        not isinstance(eligible, list)
        or any(isinstance(seed, bool) or seed not in SEEDS for seed in eligible)
        or eligible != sorted(set(eligible))
        or (selected is not None and (isinstance(selected, bool) or selected not in SEEDS))
        or selected != (min(eligible) if eligible else None)
        or (payload["status"] == "selected") != (selected is not None)
    ):
        raise ValueError("Common-seed receipt violates the frozen minimum-seed rule.")
    if payload.get("selection_rule") != {
        "eligibility": "all_three_prompts_pass_every_registered_hard_gate",
        "choice": "numerically_lowest_eligible_common_seed",
        "quality_scores_or_averaging_used": False,
        "threshold_or_gate_relaxation_used": False,
    }:
        raise ValueError("Common-seed receipt selection rule drifted.")
    outcomes = payload.get("seed_outcomes")
    if (
        not isinstance(outcomes, list)
        or [row.get("seed") for row in outcomes] != list(SEEDS)
        or [row.get("seed") for row in outcomes if row.get("eligible_common_seed")] != eligible
    ):
        raise ValueError("Common-seed receipt outcome coverage is inconsistent.")
    for row in outcomes:
        if not isinstance(row, Mapping) or set(row) != {
            "seed",
            "all_registered_gates_pass_by_prompt",
            "eligible_common_seed",
        }:
            raise ValueError("Common-seed receipt outcome fields drifted.")
        prompt_outcomes = row["all_registered_gates_pass_by_prompt"]
        if (
            not isinstance(prompt_outcomes, Mapping)
            or set(prompt_outcomes) != set(PROMPT_IDS)
            or any(not isinstance(value, bool) for value in prompt_outcomes.values())
            or row["eligible_common_seed"] is not all(prompt_outcomes.values())
        ):
            raise ValueError("Common-seed receipt prompt outcome arithmetic is invalid.")
    later = payload.get("later_calibration_v2_binding") or {}
    if later != {
        "calibration_id": CALIBRATION_V2_ID,
        "required_selected_common_seed": selected,
        "required_rows": 45,
        "required_prompt_ids": list(PROMPT_IDS),
        "every_row_must_use_selected_common_seed": True,
        "v1_scale_outputs_may_be_reused": False,
        "scale_selection_rule_may_be_relaxed": False,
        "launch_authorized": selected is not None,
    }:
        raise ValueError("Calibration-v2 seed binding is inconsistent.")
    source_documents = payload.get("source_document_sha256")
    if (
        not isinstance(source_documents, Mapping)
        or set(source_documents)
        != {
            "authentication",
            "reviewer_manifest",
            "private_unblinding_map",
            "manual_review_ledger",
            "reviewed_media",
        }
        or any(not _SHA256_RE.fullmatch(str(value)) for value in source_documents.values())
    ):
        raise ValueError("Selection receipt source-document bindings are incomplete.")
    ledger = payload.get("normalized_manual_review_ledger")
    if not isinstance(ledger, Mapping) or set(ledger) != {
        "path",
        "sha256",
        "size_bytes",
        "document_sha256",
    }:
        raise ValueError("Selection receipt lacks normalized manual-ledger binding.")
    if (
        ledger.get("document_sha256") != source_documents["manual_review_ledger"]
        or not _SHA256_RE.fullmatch(str(ledger.get("sha256", "")))
        or isinstance(ledger.get("size_bytes"), bool)
        or not isinstance(ledger.get("size_bytes"), int)
        or ledger["size_bytes"] <= 0
        or not str(ledger.get("path", "")).strip()
    ):
        raise ValueError("Normalized manual-ledger receipt binding is malformed.")
    if verify_files:
        ledger_path = Path(str(ledger["path"])).resolve()
        ledger_sidecar = ledger_path.with_suffix(ledger_path.suffix + ".sha256")
        if (
            not ledger_path.is_file()
            or ledger_path.stat().st_size != ledger["size_bytes"]
            or sha256_file(ledger_path) != ledger["sha256"]
        ):
            raise ValueError("Published normalized manual ledger changed or disappeared.")
        if not ledger_sidecar.is_file() or ledger_sidecar.read_text(encoding="utf-8").split() != [
            ledger["sha256"],
            ledger_path.name,
        ]:
            raise ValueError("Normalized manual-ledger SHA-256 sidecar is inconsistent.")
        normalized = _load_json(ledger_path, "normalized manual ledger")
        if normalized.get("document_sha256") != ledger["document_sha256"]:
            raise ValueError("Normalized manual-ledger document digest drifted.")
    reviewed = payload.get("reviewed_media")
    if (
        not isinstance(reviewed, Mapping)
        or reviewed.get("document_sha256") != source_documents["reviewed_media"]
    ):
        raise ValueError("Selection receipt reviewed-media binding is inconsistent.")
    if reviewer_manifest is not None:
        validate_reviewed_media_evidence(reviewed, reviewer_manifest, verify_files=verify_files)
    else:
        if reviewed.get("document_sha256") != document_sha256(reviewed):
            raise ValueError("Embedded reviewed-media evidence digest is invalid.")
        records = reviewed.get("records")
        if (
            not isinstance(records, list)
            or len(records) != EXPECTED_ROWS
            or reviewed.get("records_sha256") != canonical_sha256(records)
            or len({record.get("blind_id") for record in records}) != EXPECTED_ROWS
            or any(
                record.get("mode") != "RGB"
                or record.get("format") != "PNG"
                or record.get("decode_verified") is not True
                or (record.get("width"), record.get("height"))
                not in {(layout["width"], layout["height"]) for layout in IMAGE_LAYOUTS.values()}
                for record in records
            )
        ):
            raise ValueError("Embedded reviewed-media record coverage is invalid.")
    expected_no_selection = (
        None
        if selected is not None
        else "repair_protocol_then_rerun_all_eight_seeds_without_reusing_this_cohort"
    )
    if payload.get("no_selection_action") != expected_no_selection:
        raise ValueError("Common-seed receipt no-selection action drifted.")


def validate_selection_report_against_evidence(
    payload: Mapping[str, Any],
    *,
    authentication: Mapping[str, Any],
    reviewer_manifest: Mapping[str, Any],
    unblinding_map: Mapping[str, Any],
    normalized_manual_ledger: Mapping[str, Any],
    verify_files: bool,
) -> None:
    """Recompute the substantive seed decision from all bound source evidence."""

    validate_authentication_report(authentication)
    _validate_reviewer_manifest(reviewer_manifest)
    _validate_unblinding_map(unblinding_map, reviewer_manifest, authentication)
    normalized = normalize_manual_ledger(normalized_manual_ledger, reviewer_manifest)
    if normalized != normalized_manual_ledger:
        raise ValueError("Selection evidence is not the exact normalized manual ledger.")
    validate_selection_report(
        payload,
        reviewer_manifest=reviewer_manifest,
        verify_files=verify_files,
    )
    expected_sources = {
        "authentication": authentication["document_sha256"],
        "reviewer_manifest": reviewer_manifest["document_sha256"],
        "private_unblinding_map": unblinding_map["document_sha256"],
        "manual_review_ledger": normalized["document_sha256"],
        "reviewed_media": payload["reviewed_media"]["document_sha256"],
    }
    if payload.get("source_document_sha256") != expected_sources:
        raise ValueError("Selection receipt does not bind the supplied substantive evidence.")
    ledger_record = payload.get("normalized_manual_review_ledger")
    if not isinstance(ledger_record, Mapping) or ledger_record != _planned_document_binding(
        Path(str(ledger_record.get("path", ""))), normalized
    ):
        raise ValueError("Selection receipt does not bind the exact normalized ledger bytes.")
    outcomes, eligible, selected = _compute_selection_outcomes(unblinding_map, normalized)
    expected_substantive = {
        "status": "selected" if selected is not None else "no_selection",
        "eligible_common_seeds": eligible,
        "selected_common_seed": selected,
        "seed_outcomes": outcomes,
        "later_calibration_v2_binding": {
            "calibration_id": CALIBRATION_V2_ID,
            "required_selected_common_seed": selected,
            "required_rows": 45,
            "required_prompt_ids": list(PROMPT_IDS),
            "every_row_must_use_selected_common_seed": True,
            "v1_scale_outputs_may_be_reused": False,
            "scale_selection_rule_may_be_relaxed": False,
            "launch_authorized": selected is not None,
        },
        "no_selection_action": (
            None
            if selected is not None
            else "repair_protocol_then_rerun_all_eight_seeds_without_reusing_this_cohort"
        ),
    }
    drift = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in expected_substantive.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"Selection receipt differs from the recomputed manual evidence: {drift}")


def read_selection_report(
    path: str | Path,
    *,
    reviewer_manifest: Mapping[str, Any] | None = None,
    verify_files: bool = False,
    require_publication_commit: bool = True,
) -> dict[str, Any]:
    """Authenticate a public receipt, its raw sidecar, and optional live evidence."""

    resolved = Path(path).expanduser().resolve()
    payload = _load_json(resolved, "common-seed selection receipt")
    raw_sha256 = sha256_file(resolved)
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split() != [
        raw_sha256,
        resolved.name,
    ]:
        raise ValueError("Common-seed selection receipt sidecar is inconsistent.")
    validate_selection_report(
        payload,
        reviewer_manifest=reviewer_manifest,
        verify_files=verify_files,
    )
    if require_publication_commit:
        _validate_selection_publication_commit(resolved, payload)
    return payload


def reauthenticate_authentication_report(
    payload: Mapping[str, Any], *, root: Path | None = None
) -> None:
    """Reopen all generation evidence immediately before final selection."""

    validate_authentication_report(payload)
    bindings = (payload.get("source_bindings") or {}).get("manifests_and_registries")
    if not isinstance(bindings, list) or len(bindings) != EXPECTED_MANIFESTS:
        raise ValueError("Authentication report lacks eight manifest/registry bindings.")
    manifest_paths = [row["manifest"]["path"] for row in bindings]
    registry_paths = [row["submission_registry"]["path"] for row in bindings]
    recollected = collect_candidate_artifacts(
        manifest_paths=manifest_paths,
        registry_paths=registry_paths,
        root=root or project_root(),
    )
    rebuilt = build_authentication_report(recollected)
    for key in (
        "coverage",
        "source_bindings",
        "source_only_hard_gate_ids",
        "rows",
        "review_admission",
    ):
        if rebuilt[key] != payload[key]:
            raise ValueError(
                f"Live generation evidence differs from authenticated review input: {key}."
            )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _directory_tree(path: Path) -> tuple[set[Path], set[Path]]:
    descendants = list(path.rglob("*"))
    if any(item.is_symlink() for item in descendants):
        raise ValueError(f"Atomic publication tree contains a symlink: {path}.")
    return (
        {item.relative_to(path) for item in descendants if item.is_file()},
        {item.relative_to(path) for item in descendants if item.is_dir()},
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_owned_tree(path: Path, identity: tuple[int, int]) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if path.is_symlink() or (observed.st_dev, observed.st_ino) != identity:
        return
    for directory in sorted(
        (item for item in path.rglob("*") if item.is_dir() and not item.is_symlink()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o700)
    path.chmod(0o700)
    shutil.rmtree(path)


def _remove_owned_claim(
    path: Path,
    identity: tuple[int, int],
    *,
    staging: Path,
) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if path.is_symlink() or (observed.st_dev, observed.st_ino) != identity:
        return
    try:
        staged_files, staged_directories = _directory_tree(staging)
        claimed_files, claimed_directories = _directory_tree(path)
    except (FileNotFoundError, ValueError):
        return
    if not claimed_files.issubset(staged_files) or not claimed_directories.issubset(
        staged_directories
    ):
        return
    for relative in claimed_files:
        source = staging / relative
        target = path / relative
        if (
            not source.is_file()
            or not target.is_file()
            or source.stat().st_dev != target.stat().st_dev
            or source.stat().st_ino != target.stat().st_ino
        ):
            return
    _remove_owned_tree(path, identity)


def _freeze_directory_tree(path: Path) -> None:
    files, directories = _directory_tree(path)
    for relative in files:
        target = path / relative
        descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for relative in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        target = path / relative
        target.chmod(0o555)
        _fsync_directory(target)
    path.chmod(0o555)
    _fsync_directory(path)


def _publish_directory_new(
    temporary: Path,
    destination: Path,
    *,
    commit_filename: str,
) -> None:
    """Publish a tree on Ceph using an O_EXCL claim and commit-last admission.

    CephFS on the experiment mount returns ``EINVAL`` for
    ``renameat2(RENAME_NOREPLACE)``.  ``mkdir`` and hard-link creation do
    enforce no-replace semantics, so the canonical tree is claimed once,
    populated only with staged inodes, and admitted by linking its commit last
    followed by an atomic read-only mode transition.  Readers reject writable
    or commit-less claims.
    """

    temporary = temporary.resolve()
    destination = destination.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable evidence: {destination}")
    _freeze_directory_tree(temporary)
    stage_stat = temporary.lstat()
    stage_identity = (stage_stat.st_dev, stage_stat.st_ino)
    claimed = False
    claim_identity: tuple[int, int] | None = None
    admitted = False
    try:
        try:
            os.mkdir(destination, 0o700)
        except FileExistsError as exc:
            raise FileExistsError(
                f"A competing publisher claimed immutable evidence: {destination}"
            ) from exc
        claimed = True
        claim_stat = destination.lstat()
        claim_identity = (claim_stat.st_dev, claim_stat.st_ino)
        _fsync_directory(destination.parent)
        staged_files, staged_directories = _directory_tree(temporary)
        commit_relative = Path(commit_filename)
        if commit_relative not in staged_files:
            raise RuntimeError(f"Atomic stage lacks release commit {commit_filename!r}.")
        for relative in sorted(staged_directories, key=lambda value: len(value.parts)):
            os.mkdir(destination / relative, 0o700)
        for relative in sorted(staged_files - {commit_relative}):
            os.link(temporary / relative, destination / relative, follow_symlinks=False)
        claimed_files, claimed_directories = _directory_tree(destination)
        if claimed_directories != staged_directories or claimed_files != staged_files - {
            commit_relative
        }:
            raise RuntimeError("Claimed evidence tree differs before release commit.")
        for relative in claimed_files:
            source = temporary / relative
            target = destination / relative
            if (
                source.stat().st_dev != target.stat().st_dev
                or source.stat().st_ino != target.stat().st_ino
                or sha256_file(source) != sha256_file(target)
            ):
                raise RuntimeError(f"Claimed evidence member changed: {relative}.")
        os.link(
            temporary / commit_relative,
            destination / commit_relative,
            follow_symlinks=False,
        )
        committed_files, committed_directories = _directory_tree(destination)
        if committed_files != staged_files or committed_directories != staged_directories:
            raise RuntimeError("Claimed evidence tree changed after release commit.")
        for relative in sorted(
            staged_directories,
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            target = destination / relative
            target.chmod(0o555)
            _fsync_directory(target)
        _fsync_directory(destination)
        destination.chmod(0o555)
        _fsync_directory(destination)
        admitted = True
    finally:
        if claimed and not admitted and claim_identity is not None:
            _remove_owned_claim(destination, claim_identity, staging=temporary)
        _remove_owned_tree(temporary, stage_identity)


def _build_review_package_commit(staging: Path, destination: Path) -> dict[str, Any]:
    files, directories = _directory_tree(staging)
    if Path(REVIEW_PACKAGE_COMMIT_FILENAME) in files:
        raise FileExistsError("Review-package release commit already exists in staging.")
    expected_directories = {Path("review"), Path("review/media")}
    if directories != expected_directories:
        raise ValueError("Blinded review package has missing or extra directories.")
    media_files = sorted(path for path in files if path.parent == Path("review/media"))
    if len(media_files) != EXPECTED_ROWS or any(path.suffix != ".png" for path in media_files):
        raise ValueError("Blinded review package must contain exactly 24 PNG media copies.")
    expected_nonmedia = {
        Path("authentication_report.json"),
        Path("private_unblinding_map.json"),
        Path("review/reviewer_manifest.json"),
        Path("review/manual_review_TEMPLATE.json"),
        Path("review/README.md"),
    }
    if files - set(media_files) != expected_nonmedia:
        raise ValueError("Blinded review package contains an unexpected non-media file.")
    members = [
        {
            "index": index,
            "relative_path": relative.as_posix(),
            "canonical_path": str(destination / relative),
            "sha256": sha256_file(staging / relative),
            "size_bytes": (staging / relative).stat().st_size,
        }
        for index, relative in enumerate(sorted(files))
    ]
    commit: dict[str, Any] = {
        "schema_version": 1,
        "commit": REVIEW_PACKAGE_COMMIT_NAME,
        "status": "complete_blinded_package_before_manual_review",
        "created_at_utc": _utc_now(),
        "protocol_id": PROTOCOL_ID,
        "package_root": str(destination),
        "candidate_media": EXPECTED_ROWS,
        "member_count": len(members),
        "members_sha256": canonical_sha256(members),
        "members": members,
    }
    commit["document_sha256"] = document_sha256(commit)
    return commit


def validate_review_package(root: str | Path) -> dict[str, Any]:
    """Authenticate the exact read-only blinded package and its release commit."""

    package_root = Path(root).expanduser().resolve()
    if (
        package_root.is_symlink()
        or not package_root.is_dir()
        or package_root.stat().st_mode & 0o222
        or any(item.is_symlink() for item in package_root.rglob("*"))
        or any(item.stat().st_mode & 0o222 for item in package_root.rglob("*") if item.is_dir())
    ):
        raise ValueError("Blinded review package is an uncommitted or aliased directory claim.")
    commit_path = package_root / REVIEW_PACKAGE_COMMIT_FILENAME
    commit = _load_json(commit_path, "blinded review package commit")
    expected_commit_fields = {
        "schema_version",
        "commit",
        "status",
        "created_at_utc",
        "protocol_id",
        "package_root",
        "candidate_media",
        "member_count",
        "members_sha256",
        "members",
        "document_sha256",
    }
    members = commit.get("members")
    if (
        set(commit) != expected_commit_fields
        or commit.get("schema_version") != 1
        or commit.get("commit") != REVIEW_PACKAGE_COMMIT_NAME
        or commit.get("status") != "complete_blinded_package_before_manual_review"
        or commit.get("protocol_id") != PROTOCOL_ID
        or commit.get("package_root") != str(package_root)
        or commit.get("candidate_media") != EXPECTED_ROWS
        or not isinstance(members, list)
        or commit.get("member_count") != len(members)
        or commit.get("members_sha256") != canonical_sha256(members)
        or commit.get("document_sha256") != document_sha256(commit)
    ):
        raise ValueError("Blinded review package commit identity/count/digest is invalid.")
    _parse_aware_timestamp(commit.get("created_at_utc"), "review package created_at_utc")
    expected_member_fields = {
        "index",
        "relative_path",
        "canonical_path",
        "sha256",
        "size_bytes",
    }
    expected_files = {Path(REVIEW_PACKAGE_COMMIT_FILENAME)}
    media_count = 0
    for index, member in enumerate(members):
        if (
            not isinstance(member, Mapping)
            or set(member) != expected_member_fields
            or member.get("index") != index
        ):
            raise ValueError("Blinded review package member schema/order is invalid.")
        relative = Path(str(member.get("relative_path", "")))
        physical = (package_root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or physical != package_root / relative
            or member.get("canonical_path") != str(physical)
            or physical.is_symlink()
            or not physical.is_file()
            or physical.stat().st_mode & 0o222
            or physical.stat().st_size != member.get("size_bytes")
            or sha256_file(physical) != member.get("sha256")
        ):
            raise ValueError(f"Blinded review package member changed: {relative}.")
        expected_files.add(relative)
        if relative.parent == Path("review/media") and relative.suffix == ".png":
            media_count += 1
    actual_files, actual_directories = _directory_tree(package_root)
    if (
        actual_files != expected_files
        or actual_directories != {Path("review"), Path("review/media")}
        or media_count != EXPECTED_ROWS
        or commit_path.stat().st_mode & 0o222
    ):
        raise ValueError("Blinded review package tree/coverage differs from its commit.")
    authentication = _load_json(
        package_root / "authentication_report.json", "review-package authentication"
    )
    reviewer = _load_json(
        package_root / "review/reviewer_manifest.json", "review-package reviewer manifest"
    )
    unblinding = _load_json(
        package_root / "private_unblinding_map.json", "review-package unblinding map"
    )
    validate_authentication_report(authentication)
    _validate_reviewer_manifest(reviewer)
    _validate_unblinding_map(unblinding, reviewer, authentication)
    expected_media = {Path("review") / str(row["media_file"]): row for row in reviewer["rows"]}
    actual_media = {
        relative
        for relative in actual_files
        if relative.parent == Path("review/media") and relative.suffix == ".png"
    }
    if set(expected_media) != actual_media:
        raise ValueError("Blinded review media paths differ from the reviewer manifest.")
    for relative, row in expected_media.items():
        media = package_root / relative
        if (
            sha256_file(media) != row["media_sha256"]
            or media.stat().st_size != row["media_size_bytes"]
        ):
            raise ValueError(f"Blinded review media changed: {relative}.")
        with Image.open(media) as image:
            image.verify()
        with Image.open(media) as image:
            image.load()
            observed = (image.format, image.mode, image.width, image.height)
        if observed != ("PNG", "RGB", int(row["width"]), int(row["height"])):
            raise ValueError(f"Blinded review media decode contract changed: {relative}.")
    return commit


def prepare_review_package(
    *,
    manifest_paths: Sequence[str | Path],
    registry_paths: Sequence[str | Path],
    output_root: str | Path,
    root: Path | None = None,
    randomization_nonce: str | None = None,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    collected = collect_candidate_artifacts(
        manifest_paths=manifest_paths,
        registry_paths=registry_paths,
        root=root,
    )
    authentication, reviewer, unblinding, template = build_review_documents(
        collected, randomization_nonce=randomization_nonce
    )
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite review package: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        review_root = temporary / "review"
        media_root = review_root / "media"
        media_root.mkdir(parents=True)
        unblind_by_id = {row["blind_id"]: row for row in unblinding["rows"]}
        for review_row in reviewer["rows"]:
            blind_id = review_row["blind_id"]
            source = Path(unblind_by_id[blind_id]["source_media_path"])
            target = media_root / f"{blind_id}.png"
            shutil.copyfile(source, target)
            if sha256_file(target) != review_row["media_sha256"]:
                raise ValueError("Blinded media copy digest differs from authenticated source.")
            target.chmod(0o444)
        _write_json(temporary / "authentication_report.json", authentication)
        _write_json(temporary / "private_unblinding_map.json", unblinding)
        _write_json(review_root / "reviewer_manifest.json", reviewer)
        _write_json(review_root / "manual_review_TEMPLATE.json", template)
        (review_root / "README.md").write_text(
            "# Source-only blinded Flux.1 baseline review\n\n"
            "Open only this `review` directory. View every PNG at original resolution, "
            "copy `manual_review_TEMPLATE.json` outside this immutable package, complete every "
            "Boolean hard gate with evidence notes, and set all four attestations to true. "
            "Do not modify this package. Do not open the parent directory or inspect any "
            "negative, target, steered, or earlier calibration output.\n",
            encoding="utf-8",
        )
        review_commit = _build_review_package_commit(temporary, destination)
        _write_json(temporary / REVIEW_PACKAGE_COMMIT_FILENAME, review_commit)
        _publish_directory_new(
            temporary,
            destination,
            commit_filename=REVIEW_PACKAGE_COMMIT_FILENAME,
        )
        reopened_commit = validate_review_package(destination)
        if reopened_commit != review_commit:
            raise RuntimeError("Published blinded review package commit changed.")
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "ready_for_blinded_source_review",
        "output_root": str(destination),
        "authenticated_candidates": EXPECTED_ROWS,
        "authentication_document_sha256": authentication["document_sha256"],
        "reviewer_manifest_document_sha256": reviewer["document_sha256"],
        "private_unblinding_document_sha256": unblinding["document_sha256"],
        "review_package_commit_path": str(destination / REVIEW_PACKAGE_COMMIT_FILENAME),
        "review_package_commit_document_sha256": review_commit["document_sha256"],
    }


def _encoded_document(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _planned_document_binding(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    encoded = _encoded_document(payload)
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        "document_sha256": payload["document_sha256"],
    }


def selection_publication_commit_path(selection_path: str | Path) -> Path:
    selection = Path(selection_path).expanduser().resolve()
    return selection.with_name(f"{selection.stem}__publication_commit.json")


def _build_selection_publication_commit(
    *,
    selection_path: Path,
    selection_report: Mapping[str, Any],
    ledger_path: Path,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    commit: dict[str, Any] = {
        "schema_version": 1,
        "commit": SELECTION_PUBLICATION_COMMIT_NAME,
        "status": "complete_selection_and_manual_ledger",
        "created_at_utc": _utc_now(),
        "protocol_id": PROTOCOL_ID,
        "selection": _planned_document_binding(selection_path, selection_report),
        "normalized_manual_review": _planned_document_binding(ledger_path, ledger),
        "source_document_sha256": deepcopy(selection_report["source_document_sha256"]),
        "selected_common_seed": selection_report["selected_common_seed"],
        "selection_status": selection_report["status"],
    }
    commit["document_sha256"] = document_sha256(commit)
    return commit


def _validate_selection_publication_commit(
    selection_path: Path,
    selection_report: Mapping[str, Any],
) -> dict[str, Any]:
    commit_path = selection_publication_commit_path(selection_path)
    commit = _load_json(commit_path, "common-seed selection publication commit")
    expected_fields = {
        "schema_version",
        "commit",
        "status",
        "created_at_utc",
        "protocol_id",
        "selection",
        "normalized_manual_review",
        "source_document_sha256",
        "selected_common_seed",
        "selection_status",
        "document_sha256",
    }
    ledger_binding = selection_report.get("normalized_manual_review_ledger")
    if not isinstance(ledger_binding, Mapping):
        raise ValueError("Selection report lacks its normalized manual-ledger binding.")
    ledger_path = Path(str(ledger_binding.get("path", ""))).resolve()
    if (
        set(commit) != expected_fields
        or commit.get("schema_version") != 1
        or commit.get("commit") != SELECTION_PUBLICATION_COMMIT_NAME
        or commit.get("status") != "complete_selection_and_manual_ledger"
        or commit.get("protocol_id") != PROTOCOL_ID
        or commit.get("selection") != _planned_document_binding(selection_path, selection_report)
        or commit.get("normalized_manual_review") != dict(ledger_binding)
        or commit.get("source_document_sha256") != selection_report.get("source_document_sha256")
        or commit.get("selected_common_seed") != selection_report.get("selected_common_seed")
        or commit.get("selection_status") != selection_report.get("status")
        or commit.get("document_sha256") != document_sha256(commit)
    ):
        raise ValueError("Common-seed selection publication commit identity/digest is invalid.")
    _parse_aware_timestamp(commit.get("created_at_utc"), "selection commit created_at_utc")
    sidecar = commit_path.with_suffix(commit_path.suffix + ".sha256")
    if (
        commit_path.is_symlink()
        or sidecar.is_symlink()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split()
        != [sha256_file(commit_path), commit_path.name]
        or any(path.stat().st_mode & 0o222 for path in (commit_path, sidecar))
    ):
        raise ValueError("Common-seed selection publication commit sidecar is invalid.")
    for label, binding, path in (
        ("selection", commit["selection"], selection_path),
        ("normalized manual review", commit["normalized_manual_review"], ledger_path),
    ):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_mode & 0o222
            or path.stat().st_size != binding.get("size_bytes")
            or sha256_file(path) != binding.get("sha256")
        ):
            raise ValueError(f"Committed common-seed {label} changed or disappeared.")
        payload = _load_json(path, f"committed common-seed {label}")
        if payload.get("document_sha256") != binding.get("document_sha256"):
            raise ValueError(f"Committed common-seed {label} document digest changed.")
    return commit


def _write_documents_transactional(
    documents: Mapping[Path, Mapping[str, Any]],
) -> dict[Path, dict[str, Any]]:
    """Publish all documents and sidecars, rolling back every partial write."""

    resolved_documents = {
        path.expanduser().resolve(): payload for path, payload in documents.items()
    }
    if len(resolved_documents) != len(documents):
        raise ValueError("Transactional document destinations must be distinct.")
    for path in resolved_documents:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if path.exists() or sidecar.exists():
            raise FileExistsError(f"Refusing to overwrite immutable document: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    publications: list[tuple[Path, Path]] = []
    published: list[tuple[Path, tuple[int, int]]] = []
    result: dict[Path, dict[str, Any]] = {}
    try:
        for path, payload in resolved_documents.items():
            encoded = _encoded_document(payload)
            raw_digest = hashlib.sha256(encoded).hexdigest()
            sidecar = path.with_suffix(path.suffix + ".sha256")
            document_fd, document_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            sidecar_fd, sidecar_name = tempfile.mkstemp(prefix=f".{sidecar.name}.", dir=path.parent)
            os.close(document_fd)
            os.close(sidecar_fd)
            document_temp = Path(document_name)
            sidecar_temp = Path(sidecar_name)
            temporary.extend((document_temp, sidecar_temp))
            document_temp.write_bytes(encoded)
            sidecar_temp.write_text(f"{raw_digest}  {path.name}\n", encoding="utf-8")
            publications.extend(((document_temp, path), (sidecar_temp, sidecar)))
        for source, destination in publications:
            source_stat = source.stat(follow_symlinks=False)
            expected_identity = (source_stat.st_dev, source_stat.st_ino)
            os.link(source, destination)
            published.append((destination, expected_identity))
        identity_by_path = dict(published)
        for destination, expected_identity in published:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(destination, flags)
            try:
                stat_result = os.fstat(descriptor)
                if (stat_result.st_dev, stat_result.st_ino) != expected_identity:
                    raise RuntimeError(
                        f"Immutable publication was replaced before chmod: {destination}"
                    )
                os.fchmod(descriptor, 0o444)
            finally:
                os.close(descriptor)
        for path in resolved_documents:
            sidecar = path.with_suffix(path.suffix + ".sha256")
            result[path] = {
                "path": path,
                "sidecar": sidecar,
                "path_identity": identity_by_path[path],
                "sidecar_identity": identity_by_path[sidecar],
            }
        return result
    except Exception:
        for destination, expected_identity in reversed(published):
            try:
                observed = destination.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (observed.st_dev, observed.st_ino) == expected_identity:
                destination.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def _unlink_owned_publication(record: Mapping[str, Any]) -> None:
    """Remove only links whose inode identities still belong to this publisher."""

    for path_key, identity_key in (
        ("path", "path_identity"),
        ("sidecar", "sidecar_identity"),
    ):
        path = Path(record[path_key])
        expected = tuple(record[identity_key])
        try:
            observed = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if (observed.st_dev, observed.st_ino) == expected:
            path.unlink(missing_ok=True)


def _require_owned_publication(record: Mapping[str, Any]) -> None:
    for path_key, identity_key in (
        ("path", "path_identity"),
        ("sidecar", "sidecar_identity"),
    ):
        path = Path(record[path_key])
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            observed = os.fstat(descriptor)
            if (observed.st_dev, observed.st_ino) != tuple(record[identity_key]):
                raise RuntimeError(
                    f"Immutable publication was replaced before final authentication: {path}"
                )
        finally:
            os.close(descriptor)


def finalize_selection(
    *,
    authentication_path: str | Path,
    reviewer_manifest_path: str | Path,
    unblinding_map_path: str | Path,
    manual_review_path: str | Path,
    output_path: str | Path,
    normalized_manual_output_path: str | Path | None = None,
    root: Path | None = None,
    reauthenticate_sources: bool = True,
) -> dict[str, Any]:
    if reauthenticate_sources:
        reviewer_path = Path(reviewer_manifest_path).expanduser().resolve()
        package_root = reviewer_path.parent.parent
        validate_review_package(package_root)
        expected_package_inputs = {
            "authentication": package_root / "authentication_report.json",
            "reviewer": package_root / "review/reviewer_manifest.json",
            "unblinding": package_root / "private_unblinding_map.json",
        }
        observed_package_inputs = {
            "authentication": Path(authentication_path).expanduser().resolve(),
            "reviewer": reviewer_path,
            "unblinding": Path(unblinding_map_path).expanduser().resolve(),
        }
        if observed_package_inputs != expected_package_inputs:
            raise ValueError(
                "Common-seed finalization inputs must come from one exact committed review package."
            )
    authentication = _load_json(authentication_path, "authentication report")
    reviewer = _load_json(reviewer_manifest_path, "reviewer manifest")
    unblinding = _load_json(unblinding_map_path, "private unblinding map")
    manual = _load_json(manual_review_path, "manual review ledger")
    if reauthenticate_sources:
        reauthenticate_authentication_report(authentication, root=root)
    reviewed_media = authenticate_reviewed_media(reviewer_manifest_path, reviewer)
    normalized_manual = normalize_manual_ledger(manual, reviewer)
    selection_path = Path(output_path).expanduser().resolve()
    ledger_path = (
        Path(normalized_manual_output_path).expanduser().resolve()
        if normalized_manual_output_path is not None
        else selection_path.with_name(f"{selection_path.stem}__manual_review_completed.json")
    )
    ledger_binding = _planned_document_binding(ledger_path, normalized_manual)
    report = build_selection_report(
        authentication,
        reviewer,
        unblinding,
        normalized_manual,
        reviewed_media,
        ledger_binding,
    )
    publication_commit_path = selection_publication_commit_path(selection_path)
    publication_commit = _build_selection_publication_commit(
        selection_path=selection_path,
        selection_report=report,
        ledger_path=ledger_path,
        ledger=normalized_manual,
    )
    written_documents = _write_documents_transactional(
        {
            ledger_path: normalized_manual,
            selection_path: report,
            publication_commit_path: publication_commit,
        }
    )
    selection_publication = written_documents[selection_path]
    ledger_publication = written_documents[ledger_path]
    commit_publication = written_documents[publication_commit_path]
    written = Path(selection_publication["path"])
    sidecar = Path(selection_publication["sidecar"])
    written_ledger = Path(ledger_publication["path"])
    ledger_sidecar = Path(ledger_publication["sidecar"])
    try:
        _require_owned_publication(selection_publication)
        _require_owned_publication(ledger_publication)
        _require_owned_publication(commit_publication)
        reopened = read_selection_report(
            written,
            reviewer_manifest=reviewer,
            verify_files=True,
        )
        if reopened != report:
            raise ValueError("Published selection bytes differ from the in-memory report.")
        validate_selection_report_against_evidence(
            reopened,
            authentication=authentication,
            reviewer_manifest=reviewer,
            unblinding_map=unblinding,
            normalized_manual_ledger=normalized_manual,
            verify_files=True,
        )
        reopened_ledger = _load_json(written_ledger, "published normalized manual ledger")
        if reopened_ledger != normalized_manual:
            raise ValueError("Published normalized manual ledger bytes changed.")
        _require_owned_publication(selection_publication)
        _require_owned_publication(ledger_publication)
        _require_owned_publication(commit_publication)
    except Exception:
        _unlink_owned_publication(selection_publication)
        _unlink_owned_publication(ledger_publication)
        _unlink_owned_publication(commit_publication)
        raise
    return {
        "status": report["status"],
        "selected_common_seed": report["selected_common_seed"],
        "eligible_common_seeds": report["eligible_common_seeds"],
        "selection_path": str(written),
        "selection_sidecar": str(sidecar),
        "normalized_manual_review_path": str(written_ledger),
        "normalized_manual_review_sidecar": str(ledger_sidecar),
        "normalized_manual_review_document_sha256": normalized_manual["document_sha256"],
        "publication_commit_path": str(publication_commit_path),
        "publication_commit_sidecar": str(commit_publication["sidecar"]),
        "publication_commit_document_sha256": publication_commit["document_sha256"],
        "document_sha256": report["document_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(project_root()))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preview", help="Validate and summarize the launch-disabled 8x3 plan.")
    commands.add_parser(
        "write-manifests",
        help="Publish manifests only after final preregistration and digest pinning.",
    )
    commands.add_parser(
        "write-phase-plan",
        help="Publish the exact eight-array phase plan after the cohort commit exists.",
    )
    prepare = commands.add_parser("prepare-review")
    prepare.add_argument("--manifest", action="append", required=True)
    prepare.add_argument("--registry", action="append", required=True)
    prepare.add_argument("--output-root", required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--authentication", required=True)
    finalize.add_argument("--reviewer-manifest", required=True)
    finalize.add_argument("--unblinding-map", required=True)
    finalize.add_argument("--manual-review", required=True)
    finalize.add_argument("--output", required=True)
    finalize.add_argument("--normalized-manual-output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    if args.command == "preview":
        manifests = build_seed_manifests(root, require_sealed=False)
        result = {
            "status": "implementation_preview_not_launchable",
            "protocol_id": PROTOCOL_ID,
            "seeds": list(manifests),
            "manifest_count": len(manifests),
            "rows_per_manifest": [len(manifests[seed]["jobs"]) for seed in SEEDS],
            "total_rows": sum(len(value["jobs"]) for value in manifests.values()),
            "manifest_sha256_by_seed": {
                str(seed): manifests[seed]["manifest_sha256"] for seed in SEEDS
            },
        }
    elif args.command == "write-manifests":
        paths = write_seed_manifests_immutable(root)
        result = {"status": "written", "manifest_paths": paths}
    elif args.command == "write-phase-plan":
        path = write_common_seed_phase_plan_immutable(root)
        result = {"status": "written", "phase_plan_path": str(path)}
    elif args.command == "prepare-review":
        result = prepare_review_package(
            manifest_paths=args.manifest,
            registry_paths=args.registry,
            output_root=args.output_root,
            root=root,
        )
    else:
        result = finalize_selection(
            authentication_path=args.authentication,
            reviewer_manifest_path=args.reviewer_manifest,
            unblinding_map_path=args.unblinding_map,
            manual_review_path=args.manual_review,
            output_path=args.output,
            normalized_manual_output_path=args.normalized_manual_output,
            root=root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
