"""Sealed 45-row Flux.1 native-negative true-CFG calibration.

This module is intentionally separate from the ordinary finer-detailing
manifest builder.  It admits exactly three official ``GenerationRunner``
baselines, three same-``FluxPipeline`` explicit-none controls, and thirteen
same-pipeline registered-negative scale rows per prompt.  No prompt, seed,
scale, role, path, or generation parameter is selectable from the command
line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hierasafe_flow.benchmarks import finer_detailing_correction as finer
from hierasafe_flow.utils.config import load_yaml


CALIBRATION_ID = "flux1_native_negative_true_cfg_scale_v1"
CALIBRATION_STAGE = "flux1_native_negative_scale_calibration_v1"
CALIBRATION_MANIFEST_SCHEMA_VERSION = 1
CALIBRATION_CONFIG_RELATIVE = Path(
    "configs/experiments/flux1_native_negative_scale_calibration_v1.yaml"
)
CALIBRATION_CONFIG_SHA256 = (
    "6f972bc66fe02732f2e175e82b92b011fc4cf0244e1adb99799bc54c222e41cd"
)
PREREGISTRATION_RELATIVE = Path(
    "debugging/audits/finer_detailing_20260720/"
    "flux1_native_negative_scale_calibration_preregistration_20260720_153813_CEST.md"
)
PREREGISTRATION_SHA256 = (
    "5428aa75087e21fe675340df80c7ba83689192405da36fd7f33649dea7bcbd8e"
)
PREREGISTRATION_SIDECAR_SHA256 = (
    "eb2c9bdcd21bf78d1430ccc019fe4e4387ba79fea617866a9a8a8c48df9299fc"
)
OUTPUT_ROOT_RELATIVE = Path(
    "outputs/finer_detailing_flux1_native_negative_scale_calibration_20260720"
)
PROMPT_IDS = finer.PROMPT_IDS
MODEL_NAME = "flux1_dev"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
PIPELINE_CLASS = "FluxPipeline"
SEED = 0
ATTEMPT = 1
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
EXPECTED_ROLE_COUNTS = {
    ROLE_OFFICIAL: 3,
    ROLE_CONTROL: 3,
    ROLE_LADDER: 39,
}
EXPECTED_ROWS = 45
IMAGE_LAYOUTS = {
    "01_sad_young_girl": {"width": 832, "height": 1216},
    "02_angry_old_man": {"width": 832, "height": 1216},
    "03_empty_outdoor_mall": {"width": 1216, "height": 832},
}


def project_root() -> Path:
    return finer.project_root()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scale_label(scale: float) -> str:
    if float(scale) not in TRUE_CFG_SCALES:
        raise ValueError(f"Scale {scale!r} is not preregistered.")
    return f"{float(scale):.2f}".replace(".", "p")


def _blind_id(prompt_id: str, role: str, scale: float | None) -> str:
    token = f"{CALIBRATION_ID}|{prompt_id}|{role}|{scale!r}".encode()
    return f"blind_{hashlib.sha256(token).hexdigest()[:16]}"


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256_file(path.resolve())}


def _expected_calibration_inputs(root: Path) -> dict[str, dict[str, str]]:
    preregistration = (root / PREREGISTRATION_RELATIVE).resolve()
    return {
        "calibration_config": {
            "path": str((root / CALIBRATION_CONFIG_RELATIVE).resolve()),
            "sha256": CALIBRATION_CONFIG_SHA256,
        },
        "sealed_preregistration": {
            "path": str(preregistration),
            "sha256": PREREGISTRATION_SHA256,
        },
        "sealed_preregistration_sidecar": {
            "path": str(preregistration.with_suffix(preregistration.suffix + ".sha256")),
            "sha256": PREREGISTRATION_SIDECAR_SHA256,
        },
    }


def _load_protocol(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    root = root.resolve()
    config_path = (root / CALIBRATION_CONFIG_RELATIVE).resolve()
    preregistration_path = (root / PREREGISTRATION_RELATIVE).resolve()
    preregistration_sidecar = preregistration_path.with_suffix(
        preregistration_path.suffix + ".sha256"
    )
    for path in (config_path, preregistration_path, preregistration_sidecar):
        if not path.is_file():
            raise FileNotFoundError(f"Sealed calibration input is absent: {path}")
    if _sha256_file(config_path) != CALIBRATION_CONFIG_SHA256:
        raise ValueError("Authoritative Flux.1 calibration config digest drifted.")
    if _sha256_file(preregistration_path) != PREREGISTRATION_SHA256:
        raise ValueError("Sealed Flux.1 calibration preregistration digest drifted.")
    sidecar_fields = preregistration_sidecar.read_text(encoding="utf-8").split()
    if not sidecar_fields or sidecar_fields[0] != PREREGISTRATION_SHA256:
        raise ValueError("Calibration preregistration SHA-256 sidecar is inconsistent.")

    protocol = load_yaml(config_path)
    expected_scope = {
        "model_name": MODEL_NAME,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pipeline_class": PIPELINE_CLASS,
        "prompt_ids": list(PROMPT_IDS),
        "seed": SEED,
        "num_inference_steps": 28,
        "embedded_guidance_scale": 3.5,
        "num_outputs_per_prompt": 1,
        "image_layout_by_prompt": IMAGE_LAYOUTS,
    }
    if protocol.get("schema_version") != 1:
        raise ValueError("Calibration config schema_version must be 1.")
    if protocol.get("calibration_id") != CALIBRATION_ID:
        raise ValueError("Calibration config ID drifted.")
    if protocol.get("benchmark") != finer.BENCHMARK_NAME:
        raise ValueError("Calibration source benchmark drifted.")
    if protocol.get("status") != "preregistered_before_generation":
        raise ValueError("Calibration was not marked preregistered before generation.")
    if protocol.get("scope") != expected_scope:
        raise ValueError("Calibration scope differs from the sealed 45-row protocol.")
    roles = protocol.get("roles")
    if not isinstance(roles, dict) or tuple(roles) != ROLE_ORDER:
        raise ValueError("Calibration roles or their frozen order drifted.")
    if tuple(float(value) for value in roles[ROLE_LADDER]["true_cfg_scales"]) != TRUE_CFG_SCALES:
        raise ValueError("Calibration true-CFG ladder drifted.")
    if roles[ROLE_CONTROL].get("negative_prompt") is not None:
        raise ValueError("Native calibration control must freeze negative_prompt: null.")
    if float(roles[ROLE_CONTROL].get("true_cfg_scale")) != 1.0:
        raise ValueError("Native calibration control must freeze true_cfg_scale=1.0.")
    if protocol.get("expected_rows") != {
        ROLE_OFFICIAL: 3,
        ROLE_CONTROL: 3,
        ROLE_LADDER: 39,
        "total": EXPECTED_ROWS,
    }:
        raise ValueError("Calibration expected-row contract drifted.")
    if tuple(float(value) for value in protocol["selection_rule"]["eligible_scales"]) != (
        ELIGIBLE_SCALES
    ):
        raise ValueError("Calibration eligibility ladder drifted.")
    output_root = Path(str(protocol.get("output_root", "")))
    if output_root.is_absolute() or output_root.as_posix() != (
        OUTPUT_ROOT_RELATIVE.as_posix()
    ):
        raise ValueError("Calibration output root drifted or became absolute.")

    inputs = {
        "calibration_config": _file_record(config_path),
        "sealed_preregistration": _file_record(preregistration_path),
        "sealed_preregistration_sidecar": _file_record(preregistration_sidecar),
    }
    if inputs != _expected_calibration_inputs(root):
        raise ValueError("Live calibration input digests differ from the sealed contract.")
    return protocol, inputs


def _baseline_jobs(root: Path, output_root: Path) -> dict[str, dict[str, Any]]:
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
            str(SEED),
            "--no-tensorboard",
        ]
    )
    source = finer.build_manifest(args, root)
    jobs = {str(job["prompt_id"]): job for job in source["jobs"]}
    if tuple(jobs) != PROMPT_IDS or len(jobs) != 3:
        raise RuntimeError("Official baseline builder did not return exactly three prompt rows.")
    return jobs


def _native_options(role: str, scale: float) -> dict[str, Any]:
    mode = (
        "explicit_none_control"
        if role == ROLE_CONTROL
        else "registered_negative_string"
    )
    return {
        "true_cfg_scale": float(scale),
        "calibration_negative_prompt_mode": mode,
        "calibration_id": CALIBRATION_ID,
        "calibration_config_sha256": CALIBRATION_CONFIG_SHA256,
        "calibration_role": role,
    }


def _row_path(output_root: Path, prompt_id: str, role: str, scale: float | None) -> Path:
    base = output_root / prompt_id / MODEL_NAME
    if role == ROLE_OFFICIAL:
        return base / "01_official_study_baseline" / "attempts" / "attempt_001"
    if role == ROLE_CONTROL:
        return (
            base
            / "02_native_no_negative_control"
            / "true_cfg_scale_1p00"
            / "attempts"
            / "attempt_001"
        )
    if role == ROLE_LADDER and scale is not None:
        return (
            base
            / "03_native_negative_scale_ladder"
            / f"true_cfg_scale_{_scale_label(scale)}"
            / "attempts"
            / "attempt_001"
        )
    raise ValueError(f"Invalid calibration path role={role!r}, scale={scale!r}.")


def _row_id(prompt_id: str, role: str, scale: float | None) -> str:
    suffix = role if scale is None else f"{role}__true_cfg_scale_{_scale_label(scale)}"
    return f"{prompt_id}__{MODEL_NAME}__calibration__{suffix}"


def _make_row(
    source_job: dict[str, Any],
    *,
    output_root: Path,
    calibration_inputs: dict[str, dict[str, str]],
    role: str,
    scale: float | None,
) -> dict[str, Any]:
    job = deepcopy(source_job)
    prompt_id = str(job["prompt_id"])
    job.update(
        {
            "schema_version": 4,
            "benchmark": CALIBRATION_ID,
            "stage": CALIBRATION_STAGE,
            "attempt": ATTEMPT,
            "variation": role,
            "variant": role if scale is None else f"{role}__{_scale_label(scale)}",
            "condition_id": _row_id(prompt_id, role, scale),
            "seed": SEED,
            "seed_scoped_output": False,
            "variation_dir": str(_row_path(output_root, prompt_id, role, scale).parents[1]),
            "output_dir": str(_row_path(output_root, prompt_id, role, scale)),
            "expected_media": True,
            "calibration_inputs": deepcopy(calibration_inputs),
            "calibration_config": calibration_inputs["calibration_config"]["path"],
            "sealed_preregistration": calibration_inputs["sealed_preregistration"]["path"],
            "sealed_preregistration_sidecar": calibration_inputs[
                "sealed_preregistration_sidecar"
            ]["path"],
            "calibration_row": {
                "schema_version": 1,
                "calibration_id": CALIBRATION_ID,
                "role": role,
                "execution_path": (
                    "GenerationRunner_FluxAdapter"
                    if role == ROLE_OFFICIAL
                    else "FluxPipeline_native_call"
                ),
                "prompt_id": prompt_id,
                "true_cfg_scale": scale,
                "negative_prompt_mode": (
                    "not_applied"
                    if role == ROLE_OFFICIAL
                    else (
                        "explicit_none_control"
                        if role == ROLE_CONTROL
                        else "registered_negative_string"
                    )
                ),
                "eligible_for_scale_selection": (
                    role == ROLE_LADDER and scale in ELIGIBLE_SCALES
                ),
                "blind_id": _blind_id(prompt_id, role, scale),
            },
        }
    )
    job["input_files"].update(deepcopy(calibration_inputs))
    if role == ROLE_OFFICIAL:
        job["variant_spec"] = {"kind": "baseline"}
        job["native_negative_prompt_options"] = {}
    else:
        assert scale is not None
        options = _native_options(role, scale)
        job["variant_spec"] = {
            "kind": "native_negative_prompt",
            "capability": "supported",
            "native_negative_prompt_options": deepcopy(options),
        }
        job["native_negative_prompt_options"] = deepcopy(options)
    return job


def build_calibration_manifest(root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    protocol, calibration_inputs = _load_protocol(root)
    output_root = (root / str(protocol["output_root"])).resolve()
    source_jobs = _baseline_jobs(root, output_root)
    jobs: list[dict[str, Any]] = []
    for prompt_id in PROMPT_IDS:
        source_job = source_jobs[prompt_id]
        jobs.append(
            _make_row(
                source_job,
                output_root=output_root,
                calibration_inputs=calibration_inputs,
                role=ROLE_OFFICIAL,
                scale=None,
            )
        )
        jobs.append(
            _make_row(
                source_job,
                output_root=output_root,
                calibration_inputs=calibration_inputs,
                role=ROLE_CONTROL,
                scale=1.0,
            )
        )
        jobs.extend(
            _make_row(
                source_job,
                output_root=output_root,
                calibration_inputs=calibration_inputs,
                role=ROLE_LADDER,
                scale=scale,
            )
            for scale in TRUE_CFG_SCALES
        )

    implementation = finer._provenance_from_container(jobs[0])
    manifest: dict[str, Any] = {
        "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
        "benchmark": CALIBRATION_ID,
        "calibration_id": CALIBRATION_ID,
        "stage": CALIBRATION_STAGE,
        "status": "frozen_before_generation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "attempt": ATTEMPT,
        "model_name": MODEL_NAME,
        "prompt_ids": list(PROMPT_IDS),
        "true_cfg_scales": list(TRUE_CFG_SCALES),
        "eligible_scales": list(ELIGIBLE_SCALES),
        "output_root": str(output_root),
        "num_jobs": len(jobs),
        "expected_media_jobs": len(jobs),
        "expected_not_supported_jobs": 0,
        "role_counts": deepcopy(EXPECTED_ROLE_COUNTS),
        "calibration_inputs": calibration_inputs,
        **finer._provenance_copy(implementation),
        "jobs": jobs,
    }
    manifest["manifest_sha256"] = finer.manifest_digest(manifest)
    validate_calibration_manifest(manifest, root=root, require_live_inputs=True)
    return manifest


def _expected_rows(output_root: Path) -> list[tuple[str, str, float | None, Path, str]]:
    rows: list[tuple[str, str, float | None, Path, str]] = []
    for prompt_id in PROMPT_IDS:
        for role, scales in (
            (ROLE_OFFICIAL, (None,)),
            (ROLE_CONTROL, (1.0,)),
            (ROLE_LADDER, TRUE_CFG_SCALES),
        ):
            for scale in scales:
                rows.append(
                    (
                        prompt_id,
                        role,
                        scale,
                        _row_path(output_root, prompt_id, role, scale),
                        _row_id(prompt_id, role, scale),
                    )
                )
    return rows


def validate_calibration_manifest(
    manifest: dict[str, Any],
    *,
    root: Path | None = None,
    require_live_inputs: bool,
) -> None:
    root = (root or project_root()).resolve()
    if require_live_inputs:
        protocol, calibration_inputs = _load_protocol(root)
        output_root_relative = Path(str(protocol["output_root"]))
        image_layouts = protocol["scope"]["image_layout_by_prompt"]
    else:
        calibration_inputs = _expected_calibration_inputs(root)
        output_root_relative = OUTPUT_ROOT_RELATIVE
        image_layouts = IMAGE_LAYOUTS
    if manifest.get("schema_version") != CALIBRATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Calibration manifest schema version is invalid.")
    expected_top = {
        "benchmark": CALIBRATION_ID,
        "calibration_id": CALIBRATION_ID,
        "stage": CALIBRATION_STAGE,
        "status": "frozen_before_generation",
        "seed": SEED,
        "attempt": ATTEMPT,
        "model_name": MODEL_NAME,
        "prompt_ids": list(PROMPT_IDS),
        "true_cfg_scales": list(TRUE_CFG_SCALES),
        "eligible_scales": list(ELIGIBLE_SCALES),
        "num_jobs": EXPECTED_ROWS,
        "expected_media_jobs": EXPECTED_ROWS,
        "expected_not_supported_jobs": 0,
        "role_counts": EXPECTED_ROLE_COUNTS,
        "calibration_inputs": calibration_inputs,
    }
    drift = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected_top.items()
        if manifest.get(key) != value
    }
    if drift:
        raise ValueError(f"Calibration manifest top-level contract drifted: {drift}")
    output_root = (root / output_root_relative).resolve()
    if Path(str(manifest.get("output_root"))).resolve() != output_root:
        raise ValueError("Calibration manifest output root drifted.")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != EXPECTED_ROWS:
        raise ValueError("Calibration manifest must contain exactly 45 ordered rows.")
    expected_rows = _expected_rows(output_root)
    role_counts = {role: 0 for role in ROLE_ORDER}
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    seen_blind_ids: set[str] = set()
    for index, (job, expected) in enumerate(zip(jobs, expected_rows, strict=True)):
        if not isinstance(job, dict):
            raise ValueError(f"Calibration job {index} is not an object.")
        prompt_id, role, scale, output_dir, condition_id = expected
        row = job.get("calibration_row")
        if not isinstance(row, dict):
            raise ValueError(f"Calibration job {index} lacks calibration_row provenance.")
        expected_row = {
            "schema_version": 1,
            "calibration_id": CALIBRATION_ID,
            "role": role,
            "execution_path": (
                "GenerationRunner_FluxAdapter"
                if role == ROLE_OFFICIAL
                else "FluxPipeline_native_call"
            ),
            "prompt_id": prompt_id,
            "true_cfg_scale": scale,
            "negative_prompt_mode": (
                "not_applied"
                if role == ROLE_OFFICIAL
                else (
                    "explicit_none_control"
                    if role == ROLE_CONTROL
                    else "registered_negative_string"
                )
            ),
            "eligible_for_scale_selection": role == ROLE_LADDER and scale in ELIGIBLE_SCALES,
            "blind_id": _blind_id(prompt_id, role, scale),
        }
        if row != expected_row:
            raise ValueError(f"Calibration job {index} row provenance drifted.")
        fixed_fields = {
            "schema_version": 4,
            "benchmark": CALIBRATION_ID,
            "stage": CALIBRATION_STAGE,
            "attempt": ATTEMPT,
            "prompt_id": prompt_id,
            "seed": SEED,
            "seed_scoped_output": False,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "condition_id": condition_id,
            "output_dir": str(output_dir),
            "expected_media": True,
            "expected_native_negative_support": True,
            "calibration_inputs": calibration_inputs,
            "calibration_config": calibration_inputs["calibration_config"]["path"],
            "sealed_preregistration": calibration_inputs["sealed_preregistration"]["path"],
            "sealed_preregistration_sidecar": calibration_inputs[
                "sealed_preregistration_sidecar"
            ]["path"],
        }
        bad = {
            key: {"expected": value, "actual": job.get(key)}
            for key, value in fixed_fields.items()
            if job.get(key) != value
        }
        if bad:
            raise ValueError(f"Calibration job {index} frozen field drift: {bad}")
        generation = job.get("generation")
        layout = image_layouts[prompt_id]
        expected_generation = {
            "task": "text_to_image",
            "num_inference_steps": 28,
            "height": int(layout["height"]),
            "width": int(layout["width"]),
            "guidance_scale": 3.5,
            "num_outputs_per_prompt": 1,
        }
        if generation != expected_generation:
            raise ValueError(f"Calibration job {index} generation contract drifted.")
        if role == ROLE_OFFICIAL:
            if job.get("variant_spec") != {"kind": "baseline"}:
                raise ValueError(f"Official calibration job {index} is not a baseline.")
            if job.get("native_negative_prompt_options") != {}:
                raise ValueError(f"Official calibration job {index} has native options.")
        else:
            assert scale is not None
            options = _native_options(role, scale)
            expected_variant = {
                "kind": "native_negative_prompt",
                "capability": "supported",
                "native_negative_prompt_options": options,
            }
            if job.get("variant_spec") != expected_variant:
                raise ValueError(f"Native calibration job {index} variant drifted.")
            if job.get("native_negative_prompt_options") != options:
                raise ValueError(f"Native calibration job {index} options drifted.")
        role_counts[role] += 1
        seen_paths.add(str(job["output_dir"]))
        seen_ids.add(str(job["condition_id"]))
        seen_blind_ids.add(str(row["blind_id"]))
        if require_live_inputs:
            finer._verify_job_input_files(job)
            finer._verify_job_semantic_snapshots(job)
    if role_counts != EXPECTED_ROLE_COUNTS:
        raise ValueError(f"Calibration role coverage drifted: {role_counts}")
    if not all(len(values) == EXPECTED_ROWS for values in (seen_paths, seen_ids, seen_blind_ids)):
        raise ValueError("Calibration paths, condition IDs, and blind IDs must all be unique.")
    expected_digest = manifest.get("manifest_sha256")
    actual_digest = finer.manifest_digest(manifest)
    if expected_digest != actual_digest:
        raise ValueError(
            f"Calibration manifest digest mismatch: expected={expected_digest}, actual={actual_digest}."
        )
    finer._manifest_provenance_consistency(manifest, require_live_protocol=require_live_inputs)
    if require_live_inputs:
        finer._verify_implementation_provenance(manifest, root)


def write_calibration_manifest_immutable(
    manifest: dict[str, Any], path: Path, root: Path | None = None
) -> None:
    root = (root or project_root()).resolve()
    validate_calibration_manifest(manifest, root=root, require_live_inputs=True)
    finer.write_manifest_immutable(manifest, path, root)
    validate_calibration_manifest(manifest, root=root, require_live_inputs=True)


def read_calibration_manifest(
    path: Path,
    root: Path | None = None,
    *,
    require_live_inputs: bool = True,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    resolved = path if path.is_absolute() else root / path
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    validate_calibration_manifest(
        manifest,
        root=root,
        require_live_inputs=require_live_inputs,
    )
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    sidecar_fields = (
        sidecar.read_text(encoding="utf-8").split() if sidecar.is_file() else []
    )
    if not sidecar_fields or sidecar_fields[0] != manifest["manifest_sha256"]:
        raise ValueError("Calibration manifest SHA-256 sidecar is absent or inconsistent.")
    if not isinstance(manifest.get("snapshot_bundle"), dict):
        raise ValueError("Calibration manifest lacks its immutable input snapshot bundle.")
    finer._verify_snapshot_bundle(manifest)
    return manifest


def read_calibration_manifest_for_audit(
    path: Path, root: Path | None = None
) -> dict[str, Any]:
    """Read a completed calibration from its archive after live-source repairs.

    This deliberately authenticates the immutable manifest, sidecar, and full
    content-addressed snapshot while skipping comparisons with current source
    and config files.  Generation dispatch never calls this function.
    """

    return read_calibration_manifest(path, root, require_live_inputs=False)


def execute_calibration_job(
    manifest: dict[str, Any], index: int, root: Path | None = None
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    validate_calibration_manifest(manifest, root=root, require_live_inputs=True)
    jobs = manifest["jobs"]
    if isinstance(index, bool) or index < 0 or index >= len(jobs):
        raise IndexError(f"Calibration index {index} is outside 0..{len(jobs) - 1}.")
    bound = finer._job_with_launch_manifest_binding(
        jobs[index], manifest, job_index=index
    )
    return finer.run_job(bound, root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-manifest")
    mode.add_argument("--manifest")
    parser.add_argument("--index", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = project_root()
    if args.write_manifest:
        if args.index is not None or args.validate_only:
            raise ValueError("Manifest construction cannot select an index or validate-only mode.")
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
            raise ValueError("--validate-only cannot also select an execution index.")
        print(
            json.dumps(
                {
                    "status": "strictly_validated",
                    "num_jobs": EXPECTED_ROWS,
                    "manifest_sha256": manifest["manifest_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.index is None:
        raise ValueError("Calibration execution requires one explicit --index.")
    print(json.dumps(execute_calibration_job(manifest, args.index, root), indent=2))


if __name__ == "__main__":
    main()
