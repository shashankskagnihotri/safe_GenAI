"""Fail-closed production-shaped smoke planning and immutable stage lineage.

Every generation job comes unchanged from the ordinary finer-detailing
``build_manifest`` implementation.  This module only selects sealed slices,
independently reconstructs their identities/paths/contracts, and publishes an
immutable plan.  It never submits or generates.  Later stages can be published
only after gate-specific evaluator reports are reopened and re-evaluated.

The launch order is intentionally explicit and disjoint:

* ``no_generation_preflight``: zero generation rows; source/environment/API gate.
* ``wan_transition``: one complete Wan native-negative transition row.
* ``native_baselines``: Cog, Hunyuan, and Joy native baseline rows.
* ``path_matrix``: 25 baseline/native/exact-one rows (23 media + 2 unsupported).
* ``post_exact_full_modes``: all six ordinary/Shapley full rows, only after the
  exact-one and registered non-regression gates pass.
* ``final_admission``: zero generation rows; independently reopens and validates
  the cumulative 35-row cohort, including both unsupported declarations.

The cumulative union remains exactly 35 logical rows, 33 media, and two
truthful Joy/LTX native-negative unsupported declarations.  Smoke output is
separate from Q1/Q2, the seed ladder, and the final matrix.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hierasafe_flow.benchmarks.finer_detailing_correction import (
    BENCHMARK_NAME,
    EXPECTED_CHECKPOINT_SETS,
    EXPECTED_MODEL_REVISIONS,
    EXPECTED_NATIVE_NEGATIVE_SUPPORT,
    MODEL_NAMES,
    PAIR_IDS_BY_PROMPT,
    _implementation_provenance,
    build_manifest,
    build_parser as build_benchmark_parser,
    manifest_digest,
    project_root,
    read_manifest,
    write_manifest_immutable,
)
from hierasafe_flow.adapters.flux_dual_view_adapter import (
    FLUX_DUAL_VIEW_CONFIG_KEY,
    validate_flux_dual_view_conditioning,
)
from hierasafe_flow.evaluation import flux1_dual_view_jobs_v3 as flux_v3
from hierasafe_flow.utils.immutable_publication import (
    cleanup_owned_staging as _cleanup_owned_staging,
    freeze_tree,
    publish_hardlink_tree_commit_last,
    require_nonwritable_directories,
)


SMOKE_PLAN_SCHEMA_VERSION = 2
SMOKE_PLAN_CONTRACT = "finer_detailing_production_shaped_smoke_plan_v2"

NO_GENERATION_PREFLIGHT = "no_generation_preflight"
WAN_TRANSITION = "wan_transition"
NATIVE_BASELINES = "native_baselines"
PATH_MATRIX = "path_matrix"
POST_EXACT_FULL_MODES = "post_exact_full_modes"
FINAL_ADMISSION = "final_admission"
STAGES = (
    NO_GENERATION_PREFLIGHT,
    WAN_TRANSITION,
    NATIVE_BASELINES,
    PATH_MATRIX,
    POST_EXACT_FULL_MODES,
    FINAL_ADMISSION,
)

NO_GENERATION_GATE = "finer_detailing_no_generation_preflight_gate_v1"
WAN_TRANSITION_GATE = "finer_detailing_wan_native_negative_transition_gate_v2"
NATIVE_BASELINE_GATE = "finer_detailing_native_baseline_gate_v1"
IDEOGRAM_CONDITIONING_GATE = "finer_detailing_ideogram_p3_conditioning_gate_v2"
EXACT_ONE_PATH_GATE = "finer_detailing_smoke_exact_one_path_gate_v2"
FULL_PAIR_NON_REGRESSION_GATE = "finer_detailing_smoke_full_pair_non_regression_gate_v2"
FINAL_CUMULATIVE_ADMISSION_GATE = "finer_detailing_smoke_final_cumulative_35_row_admission_gate_v1"
GATE_CONTRACTS = (
    NO_GENERATION_GATE,
    WAN_TRANSITION_GATE,
    NATIVE_BASELINE_GATE,
    IDEOGRAM_CONDITIONING_GATE,
    EXACT_ONE_PATH_GATE,
    FULL_PAIR_NON_REGRESSION_GATE,
    FINAL_CUMULATIVE_ADMISSION_GATE,
)
UPSTREAM_STAGE = {
    NO_GENERATION_PREFLIGHT: None,
    WAN_TRANSITION: NO_GENERATION_PREFLIGHT,
    NATIVE_BASELINES: WAN_TRANSITION,
    PATH_MATRIX: NATIVE_BASELINES,
    POST_EXACT_FULL_MODES: PATH_MATRIX,
    FINAL_ADMISSION: POST_EXACT_FULL_MODES,
}
REQUIRED_GATES_BY_STAGE = {
    NO_GENERATION_PREFLIGHT: (),
    WAN_TRANSITION: (NO_GENERATION_GATE,),
    NATIVE_BASELINES: (WAN_TRANSITION_GATE,),
    PATH_MATRIX: (NATIVE_BASELINE_GATE, IDEOGRAM_CONDITIONING_GATE),
    POST_EXACT_FULL_MODES: (EXACT_ONE_PATH_GATE, FULL_PAIR_NON_REGRESSION_GATE),
    FINAL_ADMISSION: (FINAL_CUMULATIVE_ADMISSION_GATE,),
}

PROMPT_01 = "01_sad_young_girl"
PROMPT_03 = "03_empty_outdoor_mall"
AFFECT_PAIR = "facial_affect_negative_to_happy"
CIRCULATION_PAIR = "vertical_circulation_escalators_to_marble_stairs"
SEED = 0

VIDEO_MODELS = (
    "cogvideox_5b",
    "hunyuan_video",
    "joyai_echo",
    "ltx_23",
    "wan22_t2v_a14b",
)
IMAGE_MODELS = tuple(model for model in MODEL_NAMES if model not in VIDEO_MODELS)
NON_IDEOGRAM_IMAGE_MODELS = tuple(model for model in IMAGE_MODELS if model != "ideogram4_nf4")

BASELINE = "01_baseline"
NEGATIVE = "02_negative_prompt"
ORDINARY_FULL = "03_concept_steering"
SHAPLEY_FULL = "04_shapley_concept_steering"
ORDINARY_EXACT = "05_concept_steering_single_pair"
SHAPLEY_EXACT = "06_shapley_concept_steering_single_pair"

AUTHORITY_DOCUMENTS = {
    "debugging/audits/finer_detailing_20260719/"
    "composite_segmented_temporal_repair_patch_checklist.md": (
        "df3c00e4cea3845fdc7a4aef2938fd05aaebb86276879744ae1d47d627151e28"
    ),
    "debugging/audits/finer_detailing_20260719/"
    "ltx_wan_ideogram_atomic_patch_checklist_20260719_133502_CEST.md": (
        "cad64826e15466601e14d5423048aea194e772664964c6648780d4b2a343d8b9"
    ),
    "debugging/audits/finer_detailing_20260720/"
    "fresh_qualification_topology_addendum_20260720_151438_CEST.md": (
        "2839615d489243939957b969019b23853a15401c9653524b528066e8130b6f3f"
    ),
}
PLANNER_SOURCE_FILES = (
    "src/hierasafe_flow/benchmarks/finer_detailing_production_smoke.py",
    "src/hierasafe_flow/evaluation/production_smoke.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_smoke_launch.py",
    "scripts/plan_finer_detailing_production_smoke.py",
    "scripts/evaluate_finer_detailing_production_smoke_gate.py",
    "scripts/finer_detailing_smoke_dispatch.py",
    "scripts/orchestrate_finer_detailing_smoke_launch.py",
    "scripts/run_finer_detailing_smoke_dispatched.sh",
    "slurm/finer_detailing_production_smoke_h100.sbatch",
)


@dataclass(frozen=True)
class ManifestSlice:
    role: str
    task: str
    prompt_ids: tuple[str, ...]
    models: tuple[str, ...]
    variations: tuple[str, ...]
    pair_ids: tuple[str, ...] = ()
    permit_temporal_pilot: bool = False


STAGE_SLICES: dict[str, tuple[ManifestSlice, ...]] = {
    NO_GENERATION_PREFLIGHT: (),
    WAN_TRANSITION: (
        ManifestSlice(
            "wan_p1_native_negative_transition",
            "text_to_video",
            (PROMPT_01,),
            ("wan22_t2v_a14b",),
            (NEGATIVE,),
            permit_temporal_pilot=True,
        ),
    ),
    NATIVE_BASELINES: (
        ManifestSlice(
            "cog_p1_native_baseline",
            "text_to_video",
            (PROMPT_01,),
            ("cogvideox_5b",),
            (BASELINE,),
            permit_temporal_pilot=True,
        ),
        ManifestSlice(
            "hunyuan_p1_native_baseline",
            "text_to_video",
            (PROMPT_01,),
            ("hunyuan_video",),
            (BASELINE,),
            permit_temporal_pilot=True,
        ),
        ManifestSlice(
            "joy_p1_native_baseline",
            "text_to_video",
            (PROMPT_01,),
            ("joyai_echo",),
            (BASELINE,),
            permit_temporal_pilot=True,
        ),
    ),
    PATH_MATRIX: (
        ManifestSlice(
            "cog_p1_remaining_paths",
            "text_to_video",
            (PROMPT_01,),
            ("cogvideox_5b",),
            (NEGATIVE, ORDINARY_EXACT, SHAPLEY_EXACT),
            (AFFECT_PAIR,),
            True,
        ),
        ManifestSlice(
            "hunyuan_p1_remaining_paths",
            "text_to_video",
            (PROMPT_01,),
            ("hunyuan_video",),
            (NEGATIVE, ORDINARY_EXACT, SHAPLEY_EXACT),
            (AFFECT_PAIR,),
            True,
        ),
        ManifestSlice(
            "joy_p1_remaining_paths",
            "text_to_video",
            (PROMPT_01,),
            ("joyai_echo",),
            (NEGATIVE, ORDINARY_EXACT, SHAPLEY_EXACT),
            (AFFECT_PAIR,),
            True,
        ),
        ManifestSlice(
            "ltx_p1_paths",
            "text_to_video",
            (PROMPT_01,),
            ("ltx_23",),
            (BASELINE, NEGATIVE, ORDINARY_EXACT, SHAPLEY_EXACT),
            (AFFECT_PAIR,),
            True,
        ),
        ManifestSlice(
            "wan_p1_post_transition_paths",
            "text_to_video",
            (PROMPT_01,),
            ("wan22_t2v_a14b",),
            (BASELINE, ORDINARY_EXACT, SHAPLEY_EXACT),
            (AFFECT_PAIR,),
            True,
        ),
        ManifestSlice(
            "six_non_ideogram_images_p1_exact_shapley",
            "text_to_image",
            (PROMPT_01,),
            NON_IDEOGRAM_IMAGE_MODELS,
            (SHAPLEY_EXACT,),
            (AFFECT_PAIR,),
        ),
        ManifestSlice(
            "ideogram_p3_conditioning_and_circulation",
            "text_to_image",
            (PROMPT_03,),
            ("ideogram4_nf4",),
            (BASELINE, ORDINARY_EXACT, SHAPLEY_EXACT),
            (CIRCULATION_PAIR,),
        ),
    ),
    POST_EXACT_FULL_MODES: (
        ManifestSlice(
            "cog_p1_full_shapley_after_gate",
            "text_to_video",
            (PROMPT_01,),
            ("cogvideox_5b",),
            (ORDINARY_FULL, SHAPLEY_FULL),
            permit_temporal_pilot=True,
        ),
        ManifestSlice(
            "hunyuan_p1_full_modes_after_gate",
            "text_to_video",
            (PROMPT_01,),
            ("hunyuan_video",),
            (ORDINARY_FULL, SHAPLEY_FULL),
            permit_temporal_pilot=True,
        ),
        ManifestSlice(
            "wan_p1_full_ordinary_after_gate",
            "text_to_video",
            (PROMPT_01,),
            ("wan22_t2v_a14b",),
            (ORDINARY_FULL,),
            permit_temporal_pilot=True,
        ),
        ManifestSlice(
            "ideogram_p3_full_ordinary_after_gate",
            "text_to_image",
            (PROMPT_03,),
            ("ideogram4_nf4",),
            (ORDINARY_FULL,),
        ),
    ),
    FINAL_ADMISSION: (),
}

EXPECTED_STAGE_COUNTS = {
    NO_GENERATION_PREFLIGHT: {"logical": 0, "media": 0, "unsupported": 0},
    WAN_TRANSITION: {"logical": 1, "media": 1, "unsupported": 0},
    NATIVE_BASELINES: {"logical": 3, "media": 3, "unsupported": 0},
    PATH_MATRIX: {"logical": 25, "media": 23, "unsupported": 2},
    POST_EXACT_FULL_MODES: {"logical": 6, "media": 6, "unsupported": 0},
    FINAL_ADMISSION: {"logical": 0, "media": 0, "unsupported": 0},
}
EXPECTED_CUMULATIVE_COUNTS = {
    NO_GENERATION_PREFLIGHT: {"logical": 0, "media": 0, "unsupported": 0},
    WAN_TRANSITION: {"logical": 1, "media": 1, "unsupported": 0},
    NATIVE_BASELINES: {"logical": 4, "media": 4, "unsupported": 0},
    PATH_MATRIX: {"logical": 29, "media": 27, "unsupported": 2},
    POST_EXACT_FULL_MODES: {"logical": 35, "media": 33, "unsupported": 2},
    FINAL_ADMISSION: {"logical": 35, "media": 33, "unsupported": 2},
}

# Exact production config dictionaries.  Extra fields are rejected.
GENERATION_BASE = {
    "cosmos3_super_text2image": ("text_to_image", 28, 4.0),
    "flux1_dev": ("text_to_image", 28, 3.5),
    "flux2_dev": ("text_to_image", 50, 4.0),
    "ideogram4_nf4": ("text_to_image", 48, 7.0),
    "qwen_image": ("text_to_image", 50, 4.0),
    "qwen_image_2512": ("text_to_image", 50, 4.0),
    "sd35_large": ("text_to_image", 28, 7.0),
    "cogvideox_5b": ("text_to_video", 50, 6.0, 480, 720),
    "hunyuan_video": ("text_to_video", 50, 6.0, 544, 960),
    "joyai_echo": ("text_to_video", 8, 1.0, 736, 1280),
    "ltx_23": ("text_to_video", 8, 1.0, 768, 1344),
    "wan22_t2v_a14b": ("text_to_video", 50, 5.0, 720, 1280),
}
EXPECTED_RUNTIME = {
    model: (
        {"device": "cuda", "dtype": None, "cpu_offload": "sequential"}
        if model == "flux2_dev"
        else {"device": "cuda", "dtype": None}
    )
    for model in MODEL_NAMES
}
EXPECTED_OUTPUT = {
    "decode": True,
    "save_latents": False,
    "save_traces": True,
    "image_format": "png",
    "video_format": "mp4",
}
EXPECTED_LOGGING = {"tensorboard": True}
TEMPORAL_SNAPSHOT_SHA256 = {
    "cogvideox_5b": "7842c01de604c215eb7c76d0583c725c100c912c12128aec90dccc2bdd00c026",
    "hunyuan_video": "71dca505957c22c9e14c918053d38b880788106e10918179ea5ed963d55bc0a0",
    "joyai_echo": "52d0f389b0bd8d963ff2620e43aaf1429ff0ac3c5ef7d740239c8168f9b6290a",
    "ltx_23": "f2e89b41ad8857b0d2f3613e9c19d96f243a7924fb174dc2769d63070bb454d6",
    "wan22_t2v_a14b": "e1f8c2b3d29f9ab75ec4cb84f4b3acd98eda2b224427357136a9b43897354bdb",
}
EXPECTED_SEGMENTED_ARTIFACTS = {
    "cogvideox_5b": (
        "b295c84046e91e67fdbdcd5b6240c401552cf2218999ca6069084f0a9cd3458d",
        "bb65f9ea7607f2b75d63aeb342b7f3fb71d771f8e438d059441893feab6729bc",
    ),
    "hunyuan_video": (
        "c2b7f4afe4b2c30546c99739d736325e11c9187b637c7fc2465fa2f019d60527",
        "65d49d0de533c374eaeb238dc989269c0d61c957d2fff92f6840cb1337837523",
    ),
    "joyai_echo": (
        "5832273a3ba671a7347e83b254257a836350a262e3194d542c6785aed501cb03",
        "86982ab74eadf086fd8ad7b242f16b9b04500653e0d3e2520b02e1a1e031504b",
    ),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_RE = re.compile(r"^attempt_(\d{3,})$")


@dataclass(frozen=True)
class ValidatedSmokePlan:
    path: Path
    plan: dict[str, Any]
    manifests: dict[str, dict[str, Any]]
    topology_proof: dict[str, Any]
    upstream: "ValidatedSmokePlan | None"
    gate_evaluations: tuple[dict[str, Any], ...]

    @property
    def stage(self) -> str:
        return str(self.plan["stage"])

    @property
    def digest(self) -> str:
        return str(self.plan["smoke_plan_sha256"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _builder_contract_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash the complete deterministic builder contract, excluding publication metadata."""

    contract = deepcopy(dict(manifest))
    contract.pop("created_at_utc", None)
    contract.pop("manifest_sha256", None)
    contract.pop("snapshot_bundle", None)
    jobs = contract.get("jobs")
    if isinstance(jobs, list):
        for job in jobs:
            if isinstance(job, dict):
                job.pop("snapshot_bundle", None)
    return canonical_sha256(contract)


def smoke_plan_digest(plan: Mapping[str, Any]) -> str:
    canonical = dict(plan)
    canonical.pop("smoke_plan_sha256", None)
    return canonical_sha256(canonical)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a timezone-aware ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not valid ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone offset.")
    return parsed


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)


def _resolve_reader_path(
    path: str | Path, root: Path, label: str, *, require_absolute: bool = False
) -> Path:
    raw_text = os.fspath(path)
    if any(segment in {".", ".."} for segment in raw_text.split(os.sep)):
        raise ValueError(f"{label} contains an explicit lexical alias: {raw_text}")
    candidate = Path(path).expanduser()
    if require_absolute and not candidate.is_absolute():
        raise ValueError(f"{label} binding must be an absolute canonical path.")
    if not candidate.is_absolute():
        candidate = root / candidate
    lexical = candidate.absolute()
    if ".." in candidate.parts or str(candidate) != str(lexical):
        raise ValueError(f"{label} contains a noncanonical lexical alias: {candidate}")
    _reject_symlink_components(lexical, label)
    resolved = lexical.resolve(strict=False)
    if resolved != lexical:
        raise ValueError(f"{label} resolves through a noncanonical alias: {candidate}")
    return resolved


def _reject_symlink_components(path: Path, label: str) -> None:
    """Reject every existing symlink component without following it."""

    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")


def require_exact_canonical_path(
    raw: str | Path,
    *,
    expected: Path,
    descendant_root: Path,
    label: str,
) -> Path:
    """Require lexical identity, resolved identity, containment, and no symlinks."""

    raw_text = os.fspath(raw)
    if any(segment in {".", ".."} for segment in raw_text.split(os.sep)):
        raise ValueError(f"{label} contains an explicit lexical alias: {raw_text}")
    candidate = Path(raw).expanduser()
    expected = expected.absolute()
    descendant_root = descendant_root.absolute()
    if not candidate.is_absolute() or str(candidate) != str(expected):
        raise ValueError(
            f"{label} is not the exact canonical lexical path: "
            f"expected={expected}, actual={candidate}."
        )
    if candidate != descendant_root:
        try:
            candidate.relative_to(descendant_root)
        except ValueError as exc:
            raise ValueError(f"{label} is not a descendant of {descendant_root}.") from exc
    _reject_symlink_components(descendant_root, f"{label} root")
    _reject_symlink_components(candidate, label)
    resolved_root = descendant_root.resolve(strict=False)
    resolved = candidate.resolve(strict=False)
    if resolved != expected.resolve(strict=False) or (
        resolved != resolved_root and resolved_root not in resolved.parents
    ):
        raise ValueError(f"{label} resolves outside its exact canonical root.")
    return resolved


def require_canonical_smoke_output_root(raw: str | Path, root: Path) -> Path:
    expected = canonical_smoke_output_root(root).absolute()
    return require_exact_canonical_path(
        raw,
        expected=expected,
        descendant_root=expected,
        label="smoke generation root",
    )


def paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    return first == second or first in second.parents or second in first.parents


def require_external_artifact_path(path: Path, generation_root: Path, label: str) -> None:
    if paths_overlap(path, generation_root):
        raise ValueError(
            f"{label} must be outside and non-ancestral to generation root {generation_root}: {path}"
        )


def _atomic_write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_authenticated_document(
    payload: Mapping[str, Any], path: Path, *, digest_field: str, digest_function: Any
) -> dict[str, Any]:
    frozen = deepcopy(dict(payload))
    frozen[digest_field] = digest_function(frozen)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite authenticated document: {path}")
    sidecar_written = False
    try:
        _atomic_write_new(sidecar, f"{frozen[digest_field]}  {path.name}\n")
        sidecar_written = True
        _atomic_write_new(path, json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    except Exception:
        if sidecar_written and not path.exists():
            sidecar.unlink(missing_ok=True)
        raise
    return frozen


def read_authenticated_document(
    path: Path, *, digest_field: str, digest_function: Any, label: str
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object.")
    actual = digest_function(payload)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (
        payload.get(digest_field) != actual
        or not sidecar.is_file()
        or sidecar.read_text(encoding="utf-8").split() != [actual, path.name]
    ):
        raise ValueError(f"{label} or its sidecar failed authentication: {path}")
    return payload


def _authority_bindings(root: Path) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    for relative, expected in AUTHORITY_DOCUMENTS.items():
        path = root / relative
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Sealed smoke authority is missing or changed: {path}")
        if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split() != [
            expected,
            path.name,
        ]:
            raise ValueError(f"Sealed smoke authority sidecar is invalid: {sidecar}")
        bindings.append({"path": relative, "sha256": expected})
    return bindings


def _planner_source_bindings(root: Path) -> tuple[dict[str, str], str]:
    files: dict[str, str] = {}
    for relative in PLANNER_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required smoke implementation source is missing: {path}")
        files[relative] = _sha256_file(path)
    return files, canonical_sha256(files)


def _builder_arguments(spec: ManifestSlice, *, output_root: Path, attempt: int) -> Any:
    arguments = [
        "--output-root",
        str(output_root),
        "--attempt",
        str(attempt),
        "--models",
        ",".join(spec.models),
        "--prompt-ids",
        ",".join(spec.prompt_ids),
        "--variations",
        ",".join(spec.variations),
        "--seed",
        str(SEED),
        "--seed-scoped-output",
    ]
    if spec.pair_ids:
        arguments.extend(["--pair-ids", ",".join(spec.pair_ids)])
    if spec.permit_temporal_pilot:
        arguments.append("--allow-unvalidated-temporal-pilot")
    return build_benchmark_parser().parse_args(arguments)


def build_smoke_manifests(
    stage: str,
    *,
    output_root: str | Path,
    attempt: int,
    root: Path | None = None,
    flux1_execution_protocol_inputs: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build one exact stage in memory; never publish or submit."""

    root = (root or project_root()).resolve()
    if stage not in STAGES:
        raise ValueError(f"Unknown smoke stage {stage!r}; expected {STAGES}.")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise ValueError("Smoke attempt must be a positive integer.")
    base = _resolve(output_root, root)
    stage_output = base / stage
    manifests: dict[str, dict[str, Any]] = {}
    for spec in STAGE_SLICES[stage]:
        flux_kwargs: dict[str, Any] = {}
        if "flux1_dev" in spec.models and flux1_execution_protocol_inputs is not None:
            flux_kwargs = {
                "flux1_route_context": flux_v3.MODE_EXECUTION,
                "flux1_protocol_inputs": flux1_execution_protocol_inputs,
            }
        manifests[spec.role] = build_manifest(
            _builder_arguments(spec, output_root=stage_output, attempt=attempt),
            root,
            **flux_kwargs,
        )
    validate_smoke_manifests(
        stage,
        manifests,
        output_root=base,
        attempt=attempt,
        root=root,
        require_flux1_execution=flux1_execution_protocol_inputs is not None,
    )
    return manifests


def _job_pair_id(job: Mapping[str, Any]) -> str | None:
    variant = job.get("variant_spec")
    if not isinstance(variant, Mapping) or variant.get("pair_selection") != "single":
        return None
    active = variant.get("active_pair_ids")
    if (
        not isinstance(active, Sequence)
        or isinstance(active, (str, bytes, bytearray))
        or len(active) != 1
    ):
        raise ValueError("Exact-one smoke job must activate exactly one concept pair.")
    return str(active[0])


def _expected_axes(stage: str) -> set[tuple[str, str, str, str, str | None]]:
    rows: set[tuple[str, str, str, str, str | None]] = set()
    for spec in STAGE_SLICES[stage]:
        for prompt in spec.prompt_ids:
            for model in spec.models:
                for variation in spec.variations:
                    pair = (
                        spec.pair_ids[0] if variation in {ORDINARY_EXACT, SHAPLEY_EXACT} else None
                    )
                    rows.add((spec.role, prompt, model, variation, pair))
    return rows


def canonical_variant(variation: str, pair_id: str | None) -> str:
    if variation == ORDINARY_FULL:
        return "conceptsteer_full"
    if variation == SHAPLEY_FULL:
        return "shapley_concept_steering_full"
    if variation == ORDINARY_EXACT:
        return f"conceptsteer_single__{pair_id}"
    if variation == SHAPLEY_EXACT:
        return f"shapley_concept_steering_single__{pair_id}"
    if variation in {BASELINE, NEGATIVE}:
        return variation
    raise ValueError(f"Unknown smoke variation {variation!r}.")


def canonical_variation_parts(variation: str, pair_id: str | None) -> tuple[str, ...]:
    if variation in {ORDINARY_FULL, SHAPLEY_FULL}:
        if pair_id is not None:
            raise ValueError("Full path cannot carry a single pair.")
        return (variation, "full")
    if variation in {ORDINARY_EXACT, SHAPLEY_EXACT}:
        if pair_id is None:
            raise ValueError("Exact-one path requires its pair ID.")
        return (variation, pair_id)
    if pair_id is not None:
        raise ValueError("Non-steering path cannot carry a pair ID.")
    return (variation,)


def canonical_output_paths(
    job: Mapping[str, Any], *, stage_output: Path, attempt: int
) -> tuple[Path, Path]:
    pair_id = _job_pair_id(job)
    variation_dir = stage_output / str(job["prompt_id"]) / str(job["model_name"])
    for part in canonical_variation_parts(str(job["variation"]), pair_id):
        variation_dir /= part
    output = variation_dir / f"seed_{int(job['seed']):08d}" / "attempts" / f"attempt_{attempt:03d}"
    return variation_dir.resolve(strict=False), output.resolve(strict=False)


def _expected_generation(job: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    model = str(job["model_name"])
    prompt = str(job["prompt_id"])
    registered = GENERATION_BASE[model]
    task, steps, guidance = registered[:3]
    if task == "text_to_image":
        dimensions = (
            {"height": 1216, "width": 832}
            if prompt in {"01_sad_young_girl", "02_angry_old_man"}
            else {"height": 832, "width": 1216}
        )
        expected = {
            "task": task,
            **dimensions,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "num_outputs_per_prompt": 1,
        }
        if model == "flux1_dev":
            route = job.get("flux1_dual_view_route_v3")
            source = job.get("flux1_dual_view_source_contract")
            if not isinstance(route, Mapping) or not isinstance(source, Mapping):
                raise ValueError("FLUX smoke row lacks its shared-v3 route/source contract.")
            route_mode = route.get("mode")
            validation_mode = (
                flux_v3.MODE_EXECUTION
                if route_mode == flux_v3.MODE_EXECUTION
                else flux_v3.MODE_PREVIEW
            )
            flux_v3.validate_flux1_job_v3(
                job,
                project_root=root,
                mode=validation_mode,
            )
            negative = (
                deepcopy(source.get("registered_negative"))
                if route.get("negative_mode")
                == flux_v3.NEGATIVE_MODE_PAIRED_REGISTERED
                else None
            )
            expected[FLUX_DUAL_VIEW_CONFIG_KEY] = validate_flux_dual_view_conditioning(
                {
                    "schema_version": 1,
                    "positive": deepcopy(source.get("positive")),
                    "negative": negative,
                }
            )
        return expected
    return {
        "task": task,
        "height": registered[3],
        "width": registered[4],
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "duration_seconds": 15.0,
        "num_frames": 240,
        "fps": 16,
        "num_outputs_per_prompt": 1,
    }


def _validate_temporal_contract(job: Mapping[str, Any]) -> None:
    model = str(job["model_name"])
    snapshot = job.get("temporal_protocol_snapshot")
    if model not in VIDEO_MODELS:
        if snapshot is not None or job.get("temporal_pilot_authorized") is not None:
            raise ValueError("Image smoke row unexpectedly carries temporal authorization.")
        if any(
            field in job
            for field in ("checkpoint_set", "checkpoint_set_sha256", "artifact_manifest_sha256")
        ):
            raise ValueError("Image smoke row unexpectedly carries segmented checkpoint fields.")
        return
    if (
        not isinstance(snapshot, Mapping)
        or canonical_sha256(snapshot) != TEMPORAL_SNAPSHOT_SHA256[model]
    ):
        raise ValueError(f"Frozen temporal protocol digest drifted for {model}.")
    if snapshot.get("qualification") == "pilot":
        if job.get("temporal_pilot_authorized") is not True:
            raise ValueError(f"Pilot authorization is absent for current {model} smoke row.")
    elif job.get("temporal_pilot_authorized") is True:
        raise ValueError(f"Production {model} path falsely claims pilot authorization.")
    if model in EXPECTED_SEGMENTED_ARTIFACTS:
        checkpoint_digest, artifact_digest = EXPECTED_SEGMENTED_ARTIFACTS[model]
        if (
            job.get("checkpoint_set") != EXPECTED_CHECKPOINT_SETS[model]
            or job.get("checkpoint_set_sha256") != checkpoint_digest
            or job.get("artifact_manifest_sha256") != artifact_digest
        ):
            raise ValueError(f"Checkpoint/artifact contract drifted for {model}.")
    elif any(
        field in job
        for field in ("checkpoint_set", "checkpoint_set_sha256", "artifact_manifest_sha256")
    ):
        raise ValueError(f"Non-segmented {model} row has unexpected checkpoint/artifact fields.")


def _validate_variant(job: Mapping[str, Any]) -> None:
    variation = str(job["variation"])
    variant = job.get("variant_spec")
    if not isinstance(variant, Mapping):
        raise ValueError("Smoke row has no variant specification.")
    expected = {
        BASELINE: ("baseline", None),
        NEGATIVE: ("native_negative_prompt", None),
        ORDINARY_FULL: ("conceptsteer", "full"),
        SHAPLEY_FULL: ("shapley_concept_steering", "full"),
        ORDINARY_EXACT: ("conceptsteer", "single"),
        SHAPLEY_EXACT: ("shapley_concept_steering", "single"),
    }
    kind, selection = expected[variation]
    if variant.get("kind") != kind or variant.get("pair_selection") != selection:
        raise ValueError("Smoke variation semantics drifted.")
    pair = _job_pair_id(job)
    if selection == "single" and tuple(variant.get("active_pair_ids") or ()) != (pair,):
        raise ValueError("Exact-one active-pair contract drifted.")
    if (
        selection == "full"
        and tuple(variant.get("active_pair_ids") or ()) != PAIR_IDS_BY_PROMPT[str(job["prompt_id"])]
    ):
        raise ValueError("Full path does not activate all five registered pairs.")
    if variation == NEGATIVE:
        supported = EXPECTED_NATIVE_NEGATIVE_SUPPORT[str(job["model_name"])]
        if variant.get("capability") != ("supported" if supported else "not_supported"):
            raise ValueError("Native-negative capability drifted.")
        if bool(job["expected_media"]) is not supported:
            raise ValueError("Native-negative media declaration drifted.")
        if job["model_name"] == "flux1_dev":
            raise ValueError("Flux native negative remains blocked pending v2 calibration.")
    elif job.get("expected_media") is not True:
        raise ValueError("Only truthful unsupported native-negative rows may omit media.")


def validate_smoke_manifests(
    stage: str,
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    output_root: str | Path,
    attempt: int,
    root: Path | None = None,
    require_flux1_execution: bool = False,
) -> dict[str, Any]:
    """Independently prove all axes and exact production execution contracts."""

    root = (root or project_root()).resolve()
    if stage not in STAGES:
        raise ValueError(f"Unknown smoke stage {stage!r}.")
    specs = STAGE_SLICES[stage]
    if tuple(manifests) != tuple(spec.role for spec in specs):
        raise ValueError("Smoke manifest roles are missing, reordered, or unexpected.")
    base = Path(output_root).expanduser().resolve()
    stage_output = (base / stage).resolve()
    expected_axes = _expected_axes(stage)
    actual_axes: set[tuple[str, str, str, str, str | None]] = set()
    condition_ids: set[str] = set()
    output_dirs: set[str] = set()
    implementations: set[str] = set()
    git_states: set[str] = set()
    rows: list[dict[str, Any]] = []
    media = unsupported = 0
    role_counts: dict[str, dict[str, int]] = {}
    model_counts: Counter[str] = Counter()
    variation_counts: Counter[str] = Counter()
    flux_route_modes: set[str] = set()
    flux_protocol_inputs: dict[str, dict[str, str]] | None = None

    if not specs:
        provenance = _implementation_provenance(root)
        digest = str(provenance["implementation_files_sha256"])
        return {
            "stage": stage,
            **EXPECTED_STAGE_COUNTS[stage],
            "manifest_count": 0,
            "unique_axes": 0,
            "unique_condition_ids": 0,
            "unique_output_dirs": 0,
            "implementation_files_sha256": digest,
            "git_provenance_sha256": canonical_sha256(provenance["git_provenance"]),
            "counts_by_role": {},
            "counts_by_model": {},
            "counts_by_variation": {},
            "axis_union_sha256": canonical_sha256([]),
            "flux1_execution_binding": None,
        }

    for spec in specs:
        manifest = manifests[spec.role]
        if manifest.get("benchmark") != BENCHMARK_NAME or manifest_digest(
            dict(manifest)
        ) != manifest.get("manifest_sha256"):
            raise ValueError(f"Manifest authentication failed for smoke role {spec.role}.")
        jobs = manifest.get("jobs")
        if not isinstance(jobs, list) or len(jobs) != manifest.get("num_jobs"):
            raise ValueError(f"Malformed job list for smoke role {spec.role}.")
        role_flux_jobs = [
            job
            for job in jobs
            if isinstance(job, Mapping) and job.get("model_name") == "flux1_dev"
        ]
        rebuild_kwargs: dict[str, Any] = {}
        if role_flux_jobs:
            first_route = role_flux_jobs[0].get("flux1_dual_view_route_v3")
            if not isinstance(first_route, Mapping):
                raise ValueError(f"Smoke role {spec.role} has an unprojected FLUX row.")
            first_mode = str(first_route.get("mode", ""))
            first_inputs = first_route.get("protocol_inputs")
            if first_mode not in {flux_v3.MODE_PREVIEW, flux_v3.MODE_EXECUTION}:
                raise ValueError(f"Smoke role {spec.role} has an invalid FLUX route mode.")
            if not isinstance(first_inputs, Mapping):
                raise ValueError(f"Smoke role {spec.role} lacks FLUX protocol inputs.")
            rebuild_kwargs = {
                "flux1_route_context": first_mode,
                "flux1_protocol_inputs": first_inputs,
            }
        rebuilt = build_manifest(
            _builder_arguments(spec, output_root=stage_output, attempt=attempt),
            root,
            **rebuild_kwargs,
        )
        if _builder_contract_sha256(manifest) != _builder_contract_sha256(rebuilt):
            raise ValueError(
                f"Smoke role {spec.role} differs from a fresh ordinary-builder reconstruction."
            )
        expected_top = {
            "seed": SEED,
            "seed_scoped_output": True,
            "attempt": attempt,
            "models": list(spec.models),
            "prompt_ids": list(spec.prompt_ids),
            "variation_groups": list(spec.variations),
            "output_root": str(stage_output),
        }
        if any(manifest.get(field) != value for field, value in expected_top.items()):
            raise ValueError(f"Top-level builder axes drifted for smoke role {spec.role}.")
        implementations.add(str(manifest.get("implementation_files_sha256", "")))
        git_states.add(canonical_sha256(manifest.get("git_provenance")))
        role_media = role_unsupported = 0
        for index, job in enumerate(jobs):
            if not isinstance(job, Mapping):
                raise ValueError(f"Smoke job {spec.role}/{index} is not an object.")
            pair = _job_pair_id(job)
            axis = (
                spec.role,
                str(job.get("prompt_id")),
                str(job.get("model_name")),
                str(job.get("variation")),
                pair,
            )
            if axis in actual_axes:
                raise ValueError(f"Duplicate smoke axis {axis}.")
            actual_axes.add(axis)
            if (
                job.get("seed") != SEED
                or job.get("seed_scoped_output") is not True
                or job.get("attempt") != attempt
            ):
                raise ValueError("Smoke attempt/seed scoping drifted.")
            expected_variant = canonical_variant(str(job["variation"]), pair)
            expected_condition = (
                f"{job['prompt_id']}__{job['model_name']}__{expected_variant}__seed_{SEED:08d}"
            )
            if (
                job.get("variant") != expected_variant
                or job.get("condition_id") != expected_condition
            ):
                raise ValueError("Canonical smoke variant/condition identity drifted.")
            expected_variation_dir, expected_output = canonical_output_paths(
                job, stage_output=stage_output, attempt=attempt
            )
            require_exact_canonical_path(
                str(job.get("variation_dir")),
                expected=expected_variation_dir,
                descendant_root=stage_output,
                label="smoke variation directory",
            )
            require_exact_canonical_path(
                str(job.get("output_dir")),
                expected=expected_output,
                descendant_root=stage_output,
                label="smoke attempt output directory",
            )
            if expected_condition in condition_ids or str(expected_output) in output_dirs:
                raise ValueError("Smoke condition or output path overlaps.")
            condition_ids.add(expected_condition)
            output_dirs.add(str(expected_output))
            if job.get("model_name") == "flux1_dev":
                route = job.get("flux1_dual_view_route_v3")
                if not isinstance(route, Mapping):
                    raise ValueError("Smoke FLUX row lacks its shared-v3 route.")
                mode = str(route.get("mode", ""))
                if require_flux1_execution and mode != flux_v3.MODE_EXECUTION:
                    raise ValueError("Production smoke rejects a preview FLUX row.")
                inputs = route.get("protocol_inputs")
                expected_roles = flux_v3.expected_flux1_protocol_input_roles_v3(mode)
                if not isinstance(inputs, Mapping) or set(inputs) != set(expected_roles):
                    raise ValueError("Smoke FLUX row protocol-input role set drifted.")
                normalized_inputs = {
                    str(role): {
                        "path": str(record.get("path", "")),
                        "sha256": str(record.get("sha256", "")),
                    }
                    for role, record in inputs.items()
                    if isinstance(record, Mapping)
                }
                if len(normalized_inputs) != len(inputs):
                    raise ValueError("Smoke FLUX row has a malformed protocol binding.")
                if flux_protocol_inputs is None:
                    flux_protocol_inputs = normalized_inputs
                elif flux_protocol_inputs != normalized_inputs:
                    raise ValueError("Smoke FLUX rows do not share one exact protocol binding.")
                flux_route_modes.add(mode)
            if dict(job.get("generation") or {}) != _expected_generation(job, root=root):
                raise ValueError(f"Exact generation contract drifted for {expected_condition}.")
            if dict(job.get("runtime") or {}) != EXPECTED_RUNTIME[str(job["model_name"])]:
                raise ValueError(
                    f"Exact runtime/device/dtype/offload drifted for {expected_condition}."
                )
            if (
                dict(job.get("output") or {}) != EXPECTED_OUTPUT
                or dict(job.get("logging") or {}) != EXPECTED_LOGGING
            ):
                raise ValueError(f"Output/logging contract drifted for {expected_condition}.")
            if job.get("model_revision") != EXPECTED_MODEL_REVISIONS[str(job["model_name"])]:
                raise ValueError(f"Model revision drifted for {expected_condition}.")
            _validate_temporal_contract(job)
            _validate_variant(job)
            expected_media = bool(job["expected_media"])
            media += int(expected_media)
            unsupported += int(not expected_media)
            role_media += int(expected_media)
            role_unsupported += int(not expected_media)
            model_counts[str(job["model_name"])] += 1
            variation_counts[str(job["variation"])] += 1
            rows.append(
                {
                    "stage": stage,
                    "role": spec.role,
                    "manifest_index": index,
                    "condition_id": expected_condition,
                    "output_dir": str(expected_output),
                    "prompt_id": str(job["prompt_id"]),
                    "model_name": str(job["model_name"]),
                    "variation": str(job["variation"]),
                    "pair_id": pair,
                    "expected_media": expected_media,
                }
            )
        if (
            manifest.get("expected_media_jobs") != role_media
            or manifest.get("expected_not_supported_jobs") != role_unsupported
        ):
            raise ValueError(f"Manifest summary drifted for smoke role {spec.role}.")
        pilot_jobs = sum(
            (job.get("temporal_protocol_snapshot") or {}).get("qualification") == "pilot"
            for job in jobs
        )
        if manifest.get("unvalidated_temporal_pilot_jobs") != pilot_jobs or bool(
            manifest.get("allows_unvalidated_temporal_pilot")
        ) is not bool(pilot_jobs):
            raise ValueError(f"Pilot authorization summary drifted for {spec.role}.")
        role_counts[spec.role] = {
            "logical": len(jobs),
            "media": role_media,
            "unsupported": role_unsupported,
        }

    if actual_axes != expected_axes:
        raise ValueError(
            f"Smoke axes drifted: missing={sorted(expected_axes - actual_axes)}, "
            f"extra={sorted(actual_axes - expected_axes)}."
        )
    if len(implementations) != 1 or _SHA256_RE.fullmatch(next(iter(implementations))) is None:
        raise ValueError("Smoke manifests do not share one authenticated implementation.")
    if len(git_states) != 1:
        raise ValueError("Smoke manifests do not share one Git state.")
    observed = {"logical": len(rows), "media": media, "unsupported": unsupported}
    if observed != EXPECTED_STAGE_COUNTS[stage]:
        raise ValueError(
            f"Smoke count drift: expected={EXPECTED_STAGE_COUNTS[stage]}, actual={observed}."
        )
    if stage == PATH_MATRIX:
        ordinary = sum(row["variation"] == ORDINARY_EXACT for row in rows)
        shapley = sum(row["variation"] == SHAPLEY_EXACT for row in rows)
        image_shapley = {
            row["model_name"]
            for row in rows
            if row["variation"] == SHAPLEY_EXACT and row["model_name"] in IMAGE_MODELS
        }
        if (ordinary, shapley) != (6, 12) or image_shapley != set(IMAGE_MODELS):
            raise ValueError("Exact-one smoke coverage is not 6 ordinary + 12 Shapley.")
    if stage != POST_EXACT_FULL_MODES and any(
        row["variation"] in {ORDINARY_FULL, SHAPLEY_FULL} for row in rows
    ):
        raise ValueError("An ordinary/Shapley full mode appeared before the post-exact stage.")
    if stage == POST_EXACT_FULL_MODES:
        ordinary_full = sum(row["variation"] == ORDINARY_FULL for row in rows)
        shapley_full = sum(row["variation"] == SHAPLEY_FULL for row in rows)
        if (ordinary_full, shapley_full) != (4, 2) or any(
            row["variation"] not in {ORDINARY_FULL, SHAPLEY_FULL} for row in rows
        ):
            raise ValueError(
                "Post-exact smoke coverage is not exactly four ordinary + two Shapley full rows."
            )
    rows.sort(key=lambda row: (row["role"], row["condition_id"]))
    if require_flux1_execution and flux_protocol_inputs is None:
        if any("flux1_dev" in spec.models for spec in specs):
            raise ValueError("Production smoke stage lost all expected FLUX rows.")
    flux_binding = (
        {
            "route_mode": next(iter(flux_route_modes)),
            "protocol_input_count": len(flux_protocol_inputs),
            "protocol_inputs_sha256": canonical_sha256(flux_protocol_inputs),
            "acceptance_receipt": deepcopy(
                flux_protocol_inputs.get("native_equivalence_receipt")
            ),
        }
        if flux_protocol_inputs is not None and len(flux_route_modes) == 1
        else None
    )
    if flux_protocol_inputs is not None and flux_binding is None:
        raise ValueError("Smoke FLUX rows mix route modes.")
    return {
        "stage": stage,
        **observed,
        "manifest_count": len(manifests),
        "unique_axes": len(actual_axes),
        "unique_condition_ids": len(condition_ids),
        "unique_output_dirs": len(output_dirs),
        "implementation_files_sha256": next(iter(implementations)),
        "git_provenance_sha256": next(iter(git_states)),
        "counts_by_role": role_counts,
        "counts_by_model": dict(sorted(model_counts.items())),
        "counts_by_variation": dict(sorted(variation_counts.items())),
        "axis_union_sha256": canonical_sha256(rows),
        "flux1_execution_binding": flux_binding,
    }


def smoke_plan_binding(validated: ValidatedSmokePlan) -> dict[str, Any]:
    return {
        "stage": validated.stage,
        "path": str(validated.path),
        "file_sha256": _sha256_file(validated.path),
        "smoke_plan_sha256": validated.digest,
    }


def _manifest_binding(role: str, path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "logical_rows": int(manifest["num_jobs"]),
        "media_rows": int(manifest["expected_media_jobs"]),
        "unsupported_rows": int(manifest["expected_not_supported_jobs"]),
    }


def _evaluation_binding(path: Path, report: Mapping[str, Any]) -> dict[str, str]:
    return {
        "contract": str(report["contract"]),
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "evaluation_sha256": str(report["evaluation_sha256"]),
    }


def iter_plan_jobs(validated: ValidatedSmokePlan | None):
    if validated is None:
        return
    if validated.upstream is not None:
        yield from iter_plan_jobs(validated.upstream)
    for role, manifest in validated.manifests.items():
        for index, job in enumerate(manifest["jobs"]):
            yield validated.stage, role, index, job


def _cumulative_proof(
    upstream: ValidatedSmokePlan | None,
    current_stage: str,
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    condition_ids: set[str] = set()
    output_dirs: set[str] = set()
    media = unsupported = 0
    if upstream is not None:
        for stage, role, index, job in iter_plan_jobs(upstream):
            rows.append(
                {
                    "stage": stage,
                    "role": role,
                    "index": index,
                    "condition_id": str(job["condition_id"]),
                    "output_dir": str(Path(job["output_dir"]).resolve()),
                    "expected_media": bool(job["expected_media"]),
                }
            )
    for role, manifest in manifests.items():
        for index, job in enumerate(manifest["jobs"]):
            rows.append(
                {
                    "stage": current_stage,
                    "role": role,
                    "index": index,
                    "condition_id": str(job["condition_id"]),
                    "output_dir": str(Path(job["output_dir"]).resolve()),
                    "expected_media": bool(job["expected_media"]),
                }
            )
    for row in rows:
        if row["condition_id"] in condition_ids or row["output_dir"] in output_dirs:
            raise ValueError("Smoke stages overlap in condition or output identity.")
        condition_ids.add(row["condition_id"])
        output_dirs.add(row["output_dir"])
        media += int(row["expected_media"])
        unsupported += int(not row["expected_media"])
    observed = {"logical": len(rows), "media": media, "unsupported": unsupported}
    if observed != EXPECTED_CUMULATIVE_COUNTS[current_stage]:
        raise ValueError(
            f"Cumulative smoke count drift: expected={EXPECTED_CUMULATIVE_COUNTS[current_stage]}, "
            f"actual={observed}."
        )
    rows.sort(key=lambda row: (row["stage"], row["role"], row["condition_id"]))
    return {
        **observed,
        "unique_condition_ids": len(condition_ids),
        "unique_output_dirs": len(output_dirs),
        "union_sha256": canonical_sha256(rows),
    }


def _preflight_fresh_outputs(manifests: Mapping[str, Mapping[str, Any]], attempt: int) -> None:
    for manifest in manifests.values():
        for job in manifest["jobs"]:
            output = Path(job["output_dir"]).resolve()
            if output.exists():
                raise FileExistsError(f"Smoke generation attempt already exists: {output}")
            existing: list[int] = []
            if output.parent.is_dir():
                for child in output.parent.iterdir():
                    match = _ATTEMPT_RE.fullmatch(child.name)
                    if match:
                        existing.append(int(match.group(1)))
            if existing and attempt <= max(existing):
                raise ValueError(
                    f"Attempt {attempt} is not greater than {max(existing)} for "
                    f"{job['condition_id']}."
                )


def canonical_smoke_output_root(root: Path | None = None) -> Path:
    # Preserve the lexical project-root descendant so readers can detect and
    # reject any symlink component instead of silently canonicalizing through it.
    return (
        (root or project_root()).resolve() / "outputs/finer_detailing_engineering_smoke"
    ).absolute()


def _publish_directory_commit_last(source: Path, destination: Path) -> None:
    """Publish a smoke plan with its authenticated plan JSON as the final edge."""

    try:
        publish_hardlink_tree_commit_last(
            source,
            destination,
            commit_relative_path=Path("production_smoke_plan.json"),
        )
    except FileExistsError as exc:
        raise FileExistsError(f"Immutable smoke bundle already exists: {destination}") from exc


def claim_bundle_directory(bundle: Path) -> tuple[Path, tuple[int, int]]:
    """Create a private sibling staging directory; the public bundle stays absent."""

    bundle = bundle.absolute()
    bundle.parent.mkdir(parents=True, exist_ok=True)
    if bundle.exists() or bundle.is_symlink():
        raise FileExistsError(f"Smoke bundle must be previously absent: {bundle}")
    staging = Path(tempfile.mkdtemp(prefix=f".{bundle.name}.staging-", dir=bundle.parent))
    observed = staging.stat()
    return staging, (observed.st_dev, observed.st_ino)


def cleanup_claimed_bundle(bundle: Path, identity: tuple[int, int]) -> None:
    """Remove only the descriptor-authenticated private staging inode."""

    _cleanup_owned_staging(bundle, identity)


def _freeze_bundle_tree(bundle: Path) -> None:
    freeze_tree(bundle, label="Production smoke bundle staging")


def _verify_staged_smoke_bundle(
    *,
    staging: Path,
    canonical_bundle: Path,
    plan: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    plan_path = staging / "production_smoke_plan.json"
    loaded_plan = read_authenticated_document(
        plan_path,
        digest_field="smoke_plan_sha256",
        digest_function=smoke_plan_digest,
        label="staged production smoke plan",
    )
    if loaded_plan != plan:
        raise RuntimeError("Staged smoke plan bytes differ from the planned commit.")
    for role, manifest in manifests.items():
        physical = staging / "manifests" / f"{role}.json"
        loaded = json.loads(physical.read_text(encoding="utf-8"))
        sidecar = physical.with_suffix(physical.suffix + ".sha256")
        snapshot_index = Path(f"{physical}.snapshot") / "index.json"
        canonical_manifest = canonical_bundle / "manifests" / f"{role}.json"
        descriptor = loaded.get("snapshot_bundle") or {}
        if (
            loaded != manifest
            or loaded.get("manifest_sha256") != manifest_digest(loaded)
            or sidecar.read_text(encoding="utf-8").split()
            != [loaded["manifest_sha256"], physical.name]
            or not snapshot_index.is_file()
            or descriptor.get("root_path") != str(Path(f"{canonical_manifest}.snapshot"))
            or descriptor.get("index_path")
            != str(Path(f"{canonical_manifest}.snapshot") / "index.json")
            or descriptor.get("index_sha256") != _sha256_file(snapshot_index)
        ):
            raise RuntimeError(f"Staged smoke manifest {role!r} failed byte authentication.")


def _plan_payload(
    *,
    stage: str,
    output_root: Path,
    attempt: int,
    manifest_paths: Mapping[str, Path],
    manifest_storage_paths: Mapping[str, Path] | None,
    manifests: Mapping[str, Mapping[str, Any]],
    topology: Mapping[str, Any],
    upstream: ValidatedSmokePlan | None,
    evaluation_paths: Sequence[Path],
    evaluations: Sequence[Mapping[str, Any]],
    flux1_execution_protocol_inputs: Mapping[str, Mapping[str, str]],
    root: Path,
) -> dict[str, Any]:
    planner_files, planner_digest = _planner_source_bindings(root)
    plan: dict[str, Any] = {
        "schema_version": SMOKE_PLAN_SCHEMA_VERSION,
        "contract": SMOKE_PLAN_CONTRACT,
        "benchmark": BENCHMARK_NAME,
        "stage": stage,
        "created_at_utc": _utc_now(),
        "attempt": attempt,
        "seed": SEED,
        "output_root": str(output_root),
        "stage_output_root": str(output_root / stage),
        "counts_toward_official_campaign": False,
        "authority_documents": _authority_bindings(root),
        "planner_files": planner_files,
        "planner_files_sha256": planner_digest,
        "implementation_files_sha256": topology["implementation_files_sha256"],
        "flux1_execution_admission": {
            "acceptance_receipt": deepcopy(
                flux1_execution_protocol_inputs["native_equivalence_receipt"]
            ),
            "protocol_input_roles": sorted(flux1_execution_protocol_inputs),
            "protocol_inputs_sha256": canonical_sha256(flux1_execution_protocol_inputs),
        },
        "manifest_bindings": [],
        "topology_proof": deepcopy(dict(topology)),
        "upstream_plan": smoke_plan_binding(upstream) if upstream else None,
        "gate_evaluations": [
            _evaluation_binding(path, report)
            for path, report in zip(evaluation_paths, evaluations, strict=True)
        ],
        "cumulative_proof": _cumulative_proof(upstream, stage, manifests),
        "launch_requirements": {
            "dedicated_smoke_dispatcher_required": True,
            "environment_preflight_required_per_attempt": True,
            "smoke_launch_authorization_required_per_attempt": True,
            "submission_registry_required": True,
            "full_resolution_manual_review_required": True,
            "video_normal_and_slow_motion_review_required": True,
            "video_lossless_segment_and_seam_evidence_required": True,
            "no_shortened_frames_steps_spatial_or_precision_contract": True,
        },
    }
    for spec in STAGE_SLICES[stage]:
        physical = (
            manifest_paths[spec.role]
            if manifest_storage_paths is None
            else manifest_storage_paths[spec.role]
        )
        binding = _manifest_binding(spec.role, physical, manifests[spec.role])
        binding["path"] = str(manifest_paths[spec.role])
        plan["manifest_bindings"].append(binding)
    plan["smoke_plan_sha256"] = smoke_plan_digest(plan)
    return plan


def _validate_plan_shape(plan: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "contract",
        "benchmark",
        "stage",
        "created_at_utc",
        "attempt",
        "seed",
        "output_root",
        "stage_output_root",
        "counts_toward_official_campaign",
        "authority_documents",
        "planner_files",
        "planner_files_sha256",
        "implementation_files_sha256",
        "flux1_execution_admission",
        "manifest_bindings",
        "topology_proof",
        "upstream_plan",
        "gate_evaluations",
        "cumulative_proof",
        "launch_requirements",
        "smoke_plan_sha256",
    }
    if set(plan) != required:
        raise ValueError("Smoke plan has an invalid shape.")
    if (
        plan.get("schema_version") != SMOKE_PLAN_SCHEMA_VERSION
        or plan.get("contract") != SMOKE_PLAN_CONTRACT
        or plan.get("benchmark") != BENCHMARK_NAME
        or plan.get("stage") not in STAGES
        or plan.get("seed") != SEED
        or plan.get("counts_toward_official_campaign") is not False
        or smoke_plan_digest(plan) != plan.get("smoke_plan_sha256")
    ):
        raise ValueError("Smoke plan identity/digest/campaign-isolation contract failed.")
    _timestamp(plan.get("created_at_utc"), "smoke plan created_at_utc")
    if (
        isinstance(plan.get("attempt"), bool)
        or not isinstance(plan.get("attempt"), int)
        or int(plan["attempt"]) <= 0
    ):
        raise ValueError("Smoke plan attempt must be positive.")
    for field in ("planner_files_sha256", "implementation_files_sha256", "smoke_plan_sha256"):
        if _SHA256_RE.fullmatch(str(plan.get(field, ""))) is None:
            raise ValueError(f"Smoke plan lacks {field}.")
    admission = plan.get("flux1_execution_admission")
    expected_roles = sorted(flux_v3.execution_protocol_input_roles_v3())
    if (
        not isinstance(admission, Mapping)
        or set(admission)
        != {"acceptance_receipt", "protocol_input_roles", "protocol_inputs_sha256"}
        or admission.get("protocol_input_roles") != expected_roles
        or _SHA256_RE.fullmatch(str(admission.get("protocol_inputs_sha256", ""))) is None
        or not isinstance(admission.get("acceptance_receipt"), Mapping)
    ):
        raise ValueError("Smoke plan FLUX execution admission binding is malformed.")


def _require_sealed_smoke_bundle(plan_path: Path, plan: Mapping[str, Any] | None = None) -> None:
    """Require the commit-last plan tree to be sealed and structurally exact."""

    if plan_path.name != "production_smoke_plan.json":
        raise ValueError("Production smoke plan must use its canonical commit filename.")
    bundle = plan_path.parent
    if bundle.is_symlink() or not bundle.is_dir():
        raise FileNotFoundError(f"Production smoke bundle is absent: {bundle}")
    require_nonwritable_directories(bundle, label="Production smoke bundle")
    for member in bundle.rglob("*"):
        observed = member.lstat()
        if member.is_symlink() or (not member.is_dir() and not member.is_file()):
            raise ValueError(f"Production smoke bundle contains an aliased member: {member}")
        if member.is_file() and observed.st_mode & 0o222:
            raise ValueError(f"Production smoke bundle contains a writable file: {member}")
    if plan is None:
        return
    roles = tuple(spec.role for spec in STAGE_SLICES[str(plan["stage"])])
    expected_root = {
        "production_smoke_plan.json",
        "production_smoke_plan.json.sha256",
    }
    if roles:
        expected_root.add("manifests")
    if {item.name for item in bundle.iterdir()} != expected_root:
        raise ValueError("Production smoke bundle has unexpected or missing root members.")
    if roles:
        manifest_root = bundle / "manifests"
        expected_manifests = {
            name
            for role in roles
            for name in (
                f"{role}.json",
                f"{role}.json.sha256",
                f"{role}.json.snapshot",
            )
        }
        if {item.name for item in manifest_root.iterdir()} != expected_manifests:
            raise ValueError("Production smoke bundle manifest membership drifted.")


def read_smoke_plan(
    path: str | Path,
    *,
    root: Path | None = None,
    _allow_noncanonical_output_root_for_tests: bool = False,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    resolved = _resolve_reader_path(path, root, "production smoke plan")
    _require_sealed_smoke_bundle(resolved)
    plan = read_authenticated_document(
        resolved,
        digest_field="smoke_plan_sha256",
        digest_function=smoke_plan_digest,
        label="production smoke plan",
    )
    _validate_plan_shape(plan)
    _require_sealed_smoke_bundle(resolved, plan)
    if _allow_noncanonical_output_root_for_tests:
        test_root = Path(str(plan["output_root"])).absolute()
        require_exact_canonical_path(
            plan["output_root"],
            expected=test_root,
            descendant_root=test_root,
            label="test-only smoke generation root",
        )
    else:
        require_canonical_smoke_output_root(plan["output_root"], root)
    return plan


def validate_smoke_plan(
    path: str | Path,
    *,
    root: Path | None = None,
    expected_stage: str | None = None,
    _allow_noncanonical_output_root_for_tests: bool = False,
) -> ValidatedSmokePlan:
    """Reopen manifests, upstream lineage, structured evaluations, and live sources."""

    root = (root or project_root()).resolve()
    resolved = _resolve_reader_path(path, root, "production smoke plan")
    plan = read_smoke_plan(
        resolved,
        root=root,
        _allow_noncanonical_output_root_for_tests=_allow_noncanonical_output_root_for_tests,
    )
    stage = str(plan["stage"])
    if expected_stage is not None and stage != expected_stage:
        raise ValueError(f"Expected smoke stage {expected_stage}, found {stage}.")
    generation_root = Path(plan["output_root"]).resolve()
    require_external_artifact_path(resolved, generation_root, "smoke plan")
    if plan["authority_documents"] != _authority_bindings(root):
        raise ValueError("Smoke authority binding drifted.")
    planner_files, planner_digest = _planner_source_bindings(root)
    if plan["planner_files"] != planner_files or plan["planner_files_sha256"] != planner_digest:
        raise ValueError("Smoke implementation source changed after plan publication.")
    admission_binding = plan["flux1_execution_admission"]
    receipt_binding = admission_binding["acceptance_receipt"]
    if set(receipt_binding) != {"path", "sha256"}:
        raise ValueError("Smoke plan acceptance-receipt binding shape drifted.")
    flux1_execution_protocol_inputs = flux_v3.load_flux1_execution_protocol_inputs_v3(
        str(receipt_binding["path"]),
        project_root=root,
    )
    if (
        flux1_execution_protocol_inputs.get("native_equivalence_receipt")
        != receipt_binding
        or sorted(flux1_execution_protocol_inputs)
        != admission_binding["protocol_input_roles"]
        or canonical_sha256(flux1_execution_protocol_inputs)
        != admission_binding["protocol_inputs_sha256"]
    ):
        raise ValueError("Smoke plan FLUX execution admission changed after publication.")
    specs = STAGE_SLICES[stage]
    bindings = plan.get("manifest_bindings")
    if not isinstance(bindings, list) or [item.get("role") for item in bindings] != [
        spec.role for spec in specs
    ]:
        raise ValueError("Smoke manifest bindings are missing or reordered.")
    manifests: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        manifest_path = _resolve_reader_path(
            binding["path"], root, "smoke manifest", require_absolute=True
        )
        require_external_artifact_path(manifest_path, generation_root, "smoke manifest")
        if _sha256_file(manifest_path) != binding.get("file_sha256"):
            raise ValueError(f"Smoke manifest file binding drifted: {manifest_path}")
        manifest = read_manifest(manifest_path, root)
        counts = {
            "logical_rows": manifest["num_jobs"],
            "media_rows": manifest["expected_media_jobs"],
            "unsupported_rows": manifest["expected_not_supported_jobs"],
        }
        if manifest["manifest_sha256"] != binding.get("manifest_sha256") or any(
            binding.get(field) != value for field, value in counts.items()
        ):
            raise ValueError(f"Smoke manifest identity/count binding drifted: {manifest_path}")
        manifests[str(binding["role"])] = manifest
    topology = validate_smoke_manifests(
        stage,
        manifests,
        output_root=generation_root,
        attempt=int(plan["attempt"]),
        root=root,
        require_flux1_execution=True,
    )
    if (
        topology != plan["topology_proof"]
        or topology["implementation_files_sha256"] != plan["implementation_files_sha256"]
    ):
        raise ValueError("Smoke topology proof does not recompute exactly.")

    upstream: ValidatedSmokePlan | None = None
    reports: list[dict[str, Any]] = []
    expected_upstream = UPSTREAM_STAGE[stage]
    if expected_upstream is None:
        if plan["upstream_plan"] is not None or plan["gate_evaluations"] != []:
            raise ValueError("No-generation stage cannot claim upstream gates.")
    else:
        binding = plan.get("upstream_plan")
        if not isinstance(binding, Mapping):
            raise ValueError("Later smoke stage lacks upstream plan binding.")
        upstream_path = _resolve_reader_path(
            binding["path"], root, "upstream smoke plan", require_absolute=True
        )
        require_external_artifact_path(upstream_path, generation_root, "upstream smoke plan")
        if _sha256_file(upstream_path) != binding.get("file_sha256"):
            raise ValueError("Upstream smoke plan file binding drifted.")
        upstream = validate_smoke_plan(
            upstream_path,
            root=root,
            expected_stage=expected_upstream,
            _allow_noncanonical_output_root_for_tests=_allow_noncanonical_output_root_for_tests,
        )
        if binding != smoke_plan_binding(upstream):
            raise ValueError("Upstream smoke plan canonical binding drifted.")
        if (
            upstream.plan["attempt"] != plan["attempt"]
            or upstream.plan["output_root"] != plan["output_root"]
            or upstream.plan["implementation_files_sha256"] != plan["implementation_files_sha256"]
        ):
            raise ValueError("Smoke cohort attempt/output/source differs from upstream.")
        gate_bindings = plan.get("gate_evaluations")
        contracts = REQUIRED_GATES_BY_STAGE[stage]
        if not isinstance(gate_bindings, list) or [
            item.get("contract") for item in gate_bindings
        ] != list(contracts):
            raise ValueError("Gate evaluator reports are missing or reordered.")
        from hierasafe_flow.evaluation.production_smoke import read_smoke_gate_evaluation

        for gate_binding, contract in zip(gate_bindings, contracts, strict=True):
            report_path = _resolve_reader_path(
                gate_binding["path"], root, "gate evaluation", require_absolute=True
            )
            require_external_artifact_path(report_path, generation_root, "gate evaluation")
            if _sha256_file(report_path) != gate_binding.get("file_sha256"):
                raise ValueError("Gate evaluator report file binding drifted.")
            report = read_smoke_gate_evaluation(
                report_path, subject=upstream, expected_contract=contract, root=root
            )
            if report["evaluation_sha256"] != gate_binding.get("evaluation_sha256"):
                raise ValueError("Gate evaluator report canonical binding drifted.")
            reports.append(report)
        created = _timestamp(plan["created_at_utc"], "smoke plan created_at_utc")
        if any(
            _timestamp(report["created_at_utc"], "gate report time") > created for report in reports
        ):
            raise ValueError("Smoke plan predates a required evaluator report.")
    cumulative = _cumulative_proof(upstream, stage, manifests)
    if cumulative != plan["cumulative_proof"]:
        raise ValueError("Cumulative smoke proof does not recompute exactly.")
    if plan["stage_output_root"] != str(generation_root / stage):
        raise ValueError("Stage output root drifted.")
    exact_launch = {
        "dedicated_smoke_dispatcher_required": True,
        "environment_preflight_required_per_attempt": True,
        "smoke_launch_authorization_required_per_attempt": True,
        "submission_registry_required": True,
        "full_resolution_manual_review_required": True,
        "video_normal_and_slow_motion_review_required": True,
        "video_lossless_segment_and_seam_evidence_required": True,
        "no_shortened_frames_steps_spatial_or_precision_contract": True,
    }
    if plan["launch_requirements"] != exact_launch:
        raise ValueError("Smoke launch requirements were weakened.")
    return ValidatedSmokePlan(
        path=resolved,
        plan=plan,
        manifests=manifests,
        topology_proof=topology,
        upstream=upstream,
        gate_evaluations=tuple(reports),
    )


def publish_smoke_stage(
    stage: str,
    *,
    bundle_dir: str | Path,
    output_root: str | Path,
    attempt: int,
    flux1_native_equivalence_acceptance_receipt_path: str | Path,
    upstream_plan_path: str | Path | None = None,
    gate_evaluation_paths: Mapping[str, str | Path] | None = None,
    root: Path | None = None,
    _allow_noncanonical_output_root_for_tests: bool = False,
) -> ValidatedSmokePlan:
    """Atomically claim and publish one stage bundle; never submit it."""

    root = (root or project_root()).resolve()
    flux1_execution_protocol_inputs = flux_v3.load_flux1_execution_protocol_inputs_v3(
        flux1_native_equivalence_acceptance_receipt_path,
        project_root=root,
    )
    if stage not in STAGES:
        raise ValueError(f"Unknown smoke stage {stage!r}.")
    output = (
        _resolve(output_root, root)
        if _allow_noncanonical_output_root_for_tests
        else require_canonical_smoke_output_root(output_root, root)
    )
    bundle = _resolve_reader_path(bundle_dir, root, "smoke bundle")
    require_external_artifact_path(bundle, output, "smoke bundle")
    expected_upstream = UPSTREAM_STAGE[stage]
    contracts = REQUIRED_GATES_BY_STAGE[stage]
    supplied = dict(gate_evaluation_paths or {})
    upstream: ValidatedSmokePlan | None = None
    evaluation_paths: list[Path] = []
    evaluations: list[dict[str, Any]] = []
    if expected_upstream is None:
        if upstream_plan_path is not None or supplied:
            raise ValueError("No-generation stage cannot claim upstream evidence.")
    else:
        if upstream_plan_path is None:
            raise ValueError(f"Stage {stage} requires upstream stage {expected_upstream}.")
        upstream = validate_smoke_plan(
            upstream_plan_path,
            root=root,
            expected_stage=expected_upstream,
            _allow_noncanonical_output_root_for_tests=_allow_noncanonical_output_root_for_tests,
        )
        require_external_artifact_path(upstream.path, output, "upstream smoke plan")
        if set(supplied) != set(contracts):
            raise ValueError(f"Stage {stage} requires exactly evaluator reports {contracts}.")
        from hierasafe_flow.evaluation.production_smoke import read_smoke_gate_evaluation

        for contract in contracts:
            report_path = _resolve_reader_path(supplied[contract], root, "gate evaluation")
            require_external_artifact_path(report_path, output, "gate evaluation")
            evaluations.append(
                read_smoke_gate_evaluation(
                    report_path, subject=upstream, expected_contract=contract, root=root
                )
            )
            evaluation_paths.append(report_path)

    manifests = build_smoke_manifests(
        stage,
        output_root=output,
        attempt=attempt,
        root=root,
        flux1_execution_protocol_inputs=flux1_execution_protocol_inputs,
    )
    topology = validate_smoke_manifests(
        stage,
        manifests,
        output_root=output,
        attempt=attempt,
        root=root,
        require_flux1_execution=True,
    )
    if upstream is not None:
        if (
            upstream.plan["attempt"] != attempt
            or upstream.plan["output_root"] != str(output)
            or upstream.plan["implementation_files_sha256"]
            != topology["implementation_files_sha256"]
            or upstream.topology_proof["git_provenance_sha256"] != topology["git_provenance_sha256"]
        ):
            raise ValueError("Later smoke stage differs from upstream cohort/source state.")
    _preflight_fresh_outputs(manifests, attempt)

    claimed, inode = claim_bundle_directory(bundle)
    staged_plan_path = claimed / "production_smoke_plan.json"
    canonical_plan_path = bundle / "production_smoke_plan.json"
    manifest_paths = {
        spec.role: bundle / "manifests" / f"{spec.role}.json" for spec in STAGE_SLICES[stage]
    }
    staged_manifest_paths = {
        spec.role: claimed / "manifests" / f"{spec.role}.json" for spec in STAGE_SLICES[stage]
    }
    try:
        for spec in STAGE_SLICES[stage]:
            write_manifest_immutable(
                manifests[spec.role],
                staged_manifest_paths[spec.role],
                root,
                logical_publication_path=manifest_paths[spec.role],
                allow_atomic_smoke_bundle_staging=True,
            )
        plan = _plan_payload(
            stage=stage,
            output_root=output,
            attempt=attempt,
            manifest_paths=manifest_paths,
            manifest_storage_paths=staged_manifest_paths,
            manifests=manifests,
            topology=topology,
            upstream=upstream,
            evaluation_paths=evaluation_paths,
            evaluations=evaluations,
            flux1_execution_protocol_inputs=flux1_execution_protocol_inputs,
            root=root,
        )
        write_authenticated_document(
            plan,
            staged_plan_path,
            digest_field="smoke_plan_sha256",
            digest_function=smoke_plan_digest,
        )
        _verify_staged_smoke_bundle(
            staging=claimed,
            canonical_bundle=bundle,
            plan=plan,
            manifests=manifests,
        )
        _freeze_bundle_tree(claimed)
        _publish_directory_commit_last(claimed, bundle)
        validated = validate_smoke_plan(
            canonical_plan_path,
            root=root,
            expected_stage=stage,
            _allow_noncanonical_output_root_for_tests=_allow_noncanonical_output_root_for_tests,
        )
        cleanup_claimed_bundle(claimed, inode)
        return validated
    except BaseException:
        cleanup_claimed_bundle(claimed, inode)
        raise
