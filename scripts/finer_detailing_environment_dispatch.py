#!/usr/bin/env python
"""Authenticate a finer-detailing job and enforce its execution environment.

The Slurm launchers invoke this script twice:

1. ``resolve`` runs in the main environment and reads the immutable manifest
   with the benchmark's strict generation reader.  Only the authenticated
   ``model_name`` selects an environment.
2. ``verify-job`` runs after activating that selected environment.  It reads
   the manifest again, verifies that the same job/model was selected, and
   checks the installed Diffusers VCS commit before generation starts.  LTX
   additionally reuses its adapter's pinned source-file hash contract.

There is deliberately no fallback environment.  An absent environment, an
unknown model, a changed manifest, or a dependency mismatch aborts the array
element before the model is loaded or an output directory is created.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping

from hierasafe_flow.adapters.hunyuan_video_adapter import HUNYUAN_DIFFUSERS_REVISION
from hierasafe_flow.adapters.ltx_adapter import (
    LTX_DIFFUSERS_REVISION,
    LTX_PIPELINE_SOURCE_SHA256,
    LTX_VIDEO_VAE_SOURCE_SHA256,
    _validate_diffusers_source_contract,
)
from hierasafe_flow.adapters.wan_adapter import (
    WAN_FTFY_VERSION,
    validate_wan_native_negative_prompt_cleaner,
)
from hierasafe_flow.benchmarks.finer_detailing_correction import (
    MODEL_NAMES,
    project_root,
    read_manifest,
)
from hierasafe_flow.benchmarks.slurm_tracking import publish_environment_preflight
from hierasafe_flow.evaluation.temporal_metrics import (
    validate_temporal_metric_contract,
    validate_temporal_metric_runtime_preflight,
)


MAIN_ENVIRONMENT = "safe_genai_conceptsteer"
LTX_ENVIRONMENT = "safe_genai_ltx23"
DIFFUSERS_REPOSITORY = "https://github.com/huggingface/diffusers.git"
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class EnvironmentContract:
    """One exact generation-environment contract."""

    name: str
    diffusers_revision: str
    model_names: frozenset[str]
    ltx_source_contract: bool = False


ENVIRONMENT_CONTRACTS: dict[str, EnvironmentContract] = {
    MAIN_ENVIRONMENT: EnvironmentContract(
        name=MAIN_ENVIRONMENT,
        diffusers_revision=HUNYUAN_DIFFUSERS_REVISION,
        # Keep this enumeration explicit: adding a benchmark model must fail
        # at module import until its dependency contract is reviewed.
        model_names=frozenset(
            {
                "cogvideox_5b",
                "cosmos3_super_text2image",
                "flux1_dev",
                "flux2_dev",
                "hunyuan_video",
                "ideogram4_nf4",
                "joyai_echo",
                "qwen_image",
                "qwen_image_2512",
                "sd35_large",
                "wan22_t2v_a14b",
            }
        ),
    ),
    LTX_ENVIRONMENT: EnvironmentContract(
        name=LTX_ENVIRONMENT,
        diffusers_revision=LTX_DIFFUSERS_REVISION,
        model_names=frozenset({"ltx_23"}),
        ltx_source_contract=True,
    ),
}
MODEL_ENVIRONMENTS: dict[str, EnvironmentContract] = {
    model_name: contract
    for contract in ENVIRONMENT_CONTRACTS.values()
    for model_name in contract.model_names
}

if set(MODEL_ENVIRONMENTS) != set(MODEL_NAMES):
    missing = sorted(set(MODEL_NAMES) - set(MODEL_ENVIRONMENTS))
    extra = sorted(set(MODEL_ENVIRONMENTS) - set(MODEL_NAMES))
    raise RuntimeError(
        "Finer-detailing environment dispatch does not exactly cover the canonical "
        f"model registry: missing={missing}, extra={extra}."
    )
if any(
    not _SHA1_RE.fullmatch(contract.diffusers_revision)
    for contract in ENVIRONMENT_CONTRACTS.values()
):
    raise RuntimeError("Every environment must pin Diffusers to one lowercase 40-hex commit.")


ManifestReader = Callable[[Path, Path], dict[str, Any]]


def contract_for_model(model_name: str) -> EnvironmentContract:
    """Resolve only canonical model IDs; never guess or fall back."""

    try:
        return MODEL_ENVIRONMENTS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"No authenticated environment contract exists for model {model_name!r}."
        ) from exc


def authenticated_manifest_job(
    manifest_path: Path,
    index: int,
    *,
    root: Path | None = None,
    manifest_reader: ManifestReader = read_manifest,
) -> tuple[dict[str, Any], dict[str, Any], EnvironmentContract]:
    """Read a strict immutable manifest and return it with one environment-bound job."""

    root = (root or project_root()).resolve()
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    manifest = manifest_reader(manifest_path.resolve(), root)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Authenticated finer-detailing manifest has no jobs list.")
    if isinstance(index, bool) or index < 0 or index >= len(jobs):
        raise IndexError(f"Job index {index} is outside 0..{len(jobs) - 1}.")
    job = jobs[index]
    if not isinstance(job, dict):
        raise ValueError(f"Authenticated manifest job {index} is not an object.")
    model_name = job.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"Authenticated manifest job {index} has no canonical model_name.")
    return manifest, job, contract_for_model(model_name)


def authenticated_job(
    manifest_path: Path,
    index: int,
    *,
    root: Path | None = None,
    manifest_reader: ManifestReader = read_manifest,
) -> tuple[dict[str, Any], EnvironmentContract]:
    """Compatibility wrapper returning one strictly authenticated job and contract."""

    _manifest, job, contract = authenticated_manifest_job(
        manifest_path,
        index,
        root=root,
        manifest_reader=manifest_reader,
    )
    return job, contract


def _distribution_direct_url(distribution: metadata.Distribution) -> dict[str, Any]:
    raw = distribution.read_text("direct_url.json")
    if not raw:
        raise RuntimeError(
            "Finer-detailing execution requires a VCS-installed Diffusers direct_url.json."
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Diffusers direct_url.json is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Diffusers direct_url.json must contain an object.")
    return payload


def validate_diffusers_install(
    contract: EnvironmentContract,
    *,
    distribution: metadata.Distribution | None = None,
) -> dict[str, Any]:
    """Verify the exact VCS dependency selected for one environment."""

    try:
        distribution = distribution or metadata.distribution("diffusers")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Environment {contract.name!r} does not contain Diffusers.") from exc
    direct_url = _distribution_direct_url(distribution)
    vcs_info = direct_url.get("vcs_info")
    installed_revision = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if installed_revision != contract.diffusers_revision:
        raise RuntimeError(
            f"Environment {contract.name!r} has the wrong Diffusers revision: expected "
            f"{contract.diffusers_revision}, got {installed_revision!r}."
        )
    repository = direct_url.get("url")
    if repository != DIFFUSERS_REPOSITORY:
        raise RuntimeError(
            f"Environment {contract.name!r} has an unexpected Diffusers repository: "
            f"expected {DIFFUSERS_REPOSITORY!r}, got {repository!r}."
        )
    dir_info = direct_url.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable") is True:
        raise RuntimeError(
            f"Environment {contract.name!r} uses an editable Diffusers checkout; "
            "the frozen experiment requires an immutable VCS wheel install."
        )
    return {
        "repository": repository,
        "revision": installed_revision,
        "version": distribution.version,
    }


def validate_runtime_distribution(
    package_name: str,
    expected_version: str,
    *,
    prefix: Path,
    distribution: metadata.Distribution | None = None,
) -> dict[str, str]:
    """Require one exact, non-shadowed distribution below the active environment."""

    try:
        distribution = distribution or metadata.distribution(package_name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Authenticated environment does not contain {package_name}=={expected_version}."
        ) from exc
    if distribution.version != expected_version:
        raise RuntimeError(
            f"Authenticated environment has the wrong {package_name} version: "
            f"expected {expected_version}, got {distribution.version!r}."
        )
    resolved_prefix = prefix.resolve()
    distribution_path = Path(distribution.locate_file("")).resolve()
    try:
        distribution_path.relative_to(resolved_prefix)
    except ValueError as exc:
        raise RuntimeError(
            f"Authenticated {package_name} distribution resolves outside CONDA_PREFIX: "
            f"{distribution_path} is not below {resolved_prefix}."
        ) from exc
    return {
        "package_name": package_name,
        "version": distribution.version,
        "distribution_path": str(distribution_path),
    }


def _validate_active_conda_environment(contract: EnvironmentContract) -> dict[str, str]:
    active_name = os.environ.get("CONDA_DEFAULT_ENV")
    if active_name != contract.name:
        raise RuntimeError(
            f"Wrong active conda environment: expected {contract.name!r}, got {active_name!r}."
        )
    raw_prefix = os.environ.get("CONDA_PREFIX")
    if not raw_prefix:
        raise RuntimeError("CONDA_PREFIX is absent after environment activation.")
    prefix = Path(raw_prefix).resolve()
    executable = Path(sys.executable).resolve()
    expected_executable = (prefix / "bin" / "python").resolve()
    if executable != expected_executable:
        raise RuntimeError(
            "Active Python executable does not belong to the authenticated conda "
            f"environment: expected {expected_executable}, got {executable}."
        )
    return {"name": active_name, "prefix": str(prefix), "python": str(executable)}


def _job_temporal_protocol(job: Mapping[str, Any], expected_key: str) -> Mapping[str, Any]:
    snapshot = job.get("temporal_protocol_snapshot")
    if not isinstance(snapshot, Mapping) or snapshot.get("config_key") != expected_key:
        raise RuntimeError(
            f"Authenticated {job.get('model_name')} job lacks {expected_key!r} provenance."
        )
    protocol = snapshot.get("protocol")
    if not isinstance(protocol, Mapping):
        raise RuntimeError(f"Authenticated job {expected_key!r} protocol is malformed.")
    return protocol


def validate_job_environment(
    job: Mapping[str, Any],
    contract: EnvironmentContract,
) -> dict[str, Any]:
    """Validate active environment, Diffusers commit, and model source contract."""

    model_name = str(job.get("model_name"))
    if contract_for_model(model_name) != contract:
        raise RuntimeError(
            f"Environment contract {contract.name!r} does not own model {model_name!r}."
        )
    active = _validate_active_conda_environment(contract)
    diffusers = validate_diffusers_install(contract)
    temporal_metric_contract = validate_temporal_metric_contract(
        project_root=project_root(),
        prefix=Path(active["prefix"]),
    )
    temporal_metric_runtime_preflight = validate_temporal_metric_runtime_preflight(
        project_root=project_root(),
        environment_name=contract.name,
        prefix=Path(active["prefix"]),
    )
    runtime_distributions = {
        "ftfy": validate_runtime_distribution(
            "ftfy",
            WAN_FTFY_VERSION,
            prefix=Path(active["prefix"]),
        ),
        **temporal_metric_contract["dependency_distributions"],
    }
    source_contract: dict[str, Any] | None = None
    wan_native_negative_prompt_cleaner: dict[str, Any] | None = None

    if model_name == "hunyuan_video":
        protocol = _job_temporal_protocol(job, "hunyuan_temporal_protocol")
        if protocol.get("diffusers_revision") != HUNYUAN_DIFFUSERS_REVISION:
            raise RuntimeError("Authenticated Hunyuan temporal protocol changed its Diffusers pin.")
    elif model_name == "ltx_23":
        protocol = _job_temporal_protocol(job, "ltx_temporal_protocol")
        expected_ltx_fields = {
            "diffusers_revision": LTX_DIFFUSERS_REVISION,
            "pipeline_source_sha256": LTX_PIPELINE_SOURCE_SHA256,
            "video_vae_source_sha256": LTX_VIDEO_VAE_SOURCE_SHA256,
        }
        drift = {
            key: {"expected": expected, "actual": protocol.get(key)}
            for key, expected in expected_ltx_fields.items()
            if protocol.get(key) != expected
        }
        if drift:
            raise RuntimeError(f"Authenticated LTX temporal source contract drifted: {drift}.")
        source_contract = _validate_diffusers_source_contract(protocol)

    variant_spec = job.get("variant_spec")
    if (
        model_name == "wan22_t2v_a14b"
        and isinstance(variant_spec, Mapping)
        and variant_spec.get("kind") == "native_negative_prompt"
    ):
        wan_native_negative_prompt_cleaner = validate_wan_native_negative_prompt_cleaner()

    result = {
        "schema_version": 2,
        "status": "verified_before_generation",
        "model_name": model_name,
        "environment": active,
        "diffusers": diffusers,
        "runtime_distributions": runtime_distributions,
        "temporal_metric_contract": temporal_metric_contract,
        "temporal_metric_runtime_preflight": temporal_metric_runtime_preflight,
        "source_contract": source_contract,
    }
    if wan_native_negative_prompt_cleaner is not None:
        result["wan_native_negative_prompt_cleaner"] = wan_native_negative_prompt_cleaner
    return result


def persist_validated_job_environment(
    *,
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
    job_index: int,
    validation: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Bind a successful runtime validation to its exact immutable attempt."""

    manifest_sha256 = manifest.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", manifest_sha256
    ) is None:
        raise ValueError("Authenticated manifest lacks a lowercase SHA-256 identity.")
    if isinstance(job_index, bool) or job_index < 0:
        raise ValueError("Authenticated manifest job index must be a non-negative integer.")
    model_name = job.get("model_name")
    condition_id = job.get("condition_id")
    if validation.get("model_name") != model_name:
        raise ValueError("Validated environment model identity differs from the manifest job.")
    if not isinstance(condition_id, str) or not condition_id:
        raise ValueError("Authenticated manifest job lacks a condition_id binding.")
    raw_output_dir = Path(str(job.get("output_dir", "")))
    if not str(job.get("output_dir", "")).strip():
        raise ValueError("Authenticated manifest job lacks an output_dir binding.")
    output_dir = raw_output_dir if raw_output_dir.is_absolute() else root / raw_output_dir
    payload = dict(validation)
    payload.update(
        {
            "manifest_sha256": manifest_sha256,
            "manifest_job_index": job_index,
            "condition_id": condition_id,
            "output_dir": str(output_dir.resolve(strict=False)),
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    publish_environment_preflight(output_dir, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("resolve", "verify-job"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", required=True)
        child.add_argument("--index", required=True, type=int)
        child.add_argument("--project-root", default=str(project_root()))
        reader_mode = child.add_mutually_exclusive_group()
        reader_mode.add_argument(
            "--flux1-native-negative-calibration",
            action="store_true",
            help=(
                "Authenticate the sealed 45-row Flux.1 calibration manifest with its "
                "dedicated strict reader. Ordinary manifests must omit this flag."
            ),
        )
        reader_mode.add_argument(
            "--flux1-native-negative-calibration-v2",
            action="store_true",
            help=(
                "Authenticate the selected-common-seed 45-row Flux.1 calibration-v2 "
                "manifest with its dedicated strict reader. Ordinary and v1 manifests "
                "must omit this flag."
            ),
        )
        reader_mode.add_argument(
            "--flux1-common-seed-v2",
            action="store_true",
            help=(
                "Authenticate one row through the exact committed eight-manifest "
                "Flux common-seed phase. Requires --phase-plan."
            ),
        )
        child.add_argument(
            "--phase-plan",
            help="Exact immutable phase plan required only by --flux1-common-seed-v2.",
        )
    verify = subparsers.choices["verify-job"]
    verify.add_argument("--expected-model", required=True)
    verify.add_argument("--expected-environment", required=True)
    verify.add_argument("--expected-diffusers-revision", required=True)

    install = subparsers.add_parser(
        "verify-install",
        help="Verify one freshly created environment without loading a model.",
    )
    install.add_argument("--environment", required=True, choices=tuple(ENVIRONMENT_CONTRACTS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-install":
        contract = ENVIRONMENT_CONTRACTS[args.environment]
        active = _validate_active_conda_environment(contract)
        result: dict[str, Any] = {
            "schema_version": 2,
            "status": "verified_install",
            "environment": active,
            "diffusers": validate_diffusers_install(contract),
            "runtime_distributions": {
                "ftfy": validate_runtime_distribution(
                    "ftfy",
                    WAN_FTFY_VERSION,
                    prefix=Path(active["prefix"]),
                )
            },
        }
        temporal_metric_contract = validate_temporal_metric_contract(
            project_root=project_root(),
            prefix=Path(active["prefix"]),
        )
        result["runtime_distributions"].update(
            temporal_metric_contract["dependency_distributions"]
        )
        result["temporal_metric_contract"] = temporal_metric_contract
        result["temporal_metric_runtime_preflight"] = (
            validate_temporal_metric_runtime_preflight(
                project_root=project_root(),
                environment_name=contract.name,
                prefix=Path(active["prefix"]),
            )
        )
        if contract.ltx_source_contract:
            result["source_contract"] = _validate_diffusers_source_contract(
                {
                    "diffusers_revision": LTX_DIFFUSERS_REVISION,
                    "pipeline_source_sha256": LTX_PIPELINE_SOURCE_SHA256,
                    "video_vae_source_sha256": LTX_VIDEO_VAE_SOURCE_SHA256,
                }
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    root = Path(args.project_root).resolve()
    if args.flux1_common_seed_v2 != bool(args.phase_plan):
        raise ValueError(
            "--flux1-common-seed-v2 and --phase-plan must be supplied together."
        )
    manifest_reader = read_manifest
    if args.flux1_native_negative_calibration:
        from hierasafe_flow.benchmarks.flux1_native_negative_calibration import (
            read_calibration_manifest,
        )

        manifest_reader = read_calibration_manifest
    elif args.flux1_native_negative_calibration_v2:
        from hierasafe_flow.benchmarks.flux1_native_negative_calibration_v2 import (
            read_calibration_manifest,
        )

        manifest_reader = read_calibration_manifest
    manifest, job, contract = authenticated_manifest_job(
        Path(args.manifest),
        args.index,
        root=root,
        manifest_reader=manifest_reader,
    )
    from hierasafe_flow.benchmarks.finer_detailing_smoke_launch import (
        reject_canonical_smoke_manifest_from_ordinary_dispatch,
    )

    reject_canonical_smoke_manifest_from_ordinary_dispatch(manifest, root)
    from hierasafe_flow.benchmarks.finer_detailing_qualification_launch import (
        reject_qualification_output_from_ordinary_dispatch,
    )

    reject_qualification_output_from_ordinary_dispatch(manifest, root=root)
    from hierasafe_flow.benchmarks.finer_detailing_campaign_launch import (
        reject_canonical_campaign_manifest_from_ordinary_dispatch,
    )

    reject_canonical_campaign_manifest_from_ordinary_dispatch(manifest, root)
    from hierasafe_flow.benchmarks.finer_detailing_correction import (
        FLUX_COMMON_SEED_STAGES,
        reject_reserved_flux_common_seed_outputs,
    )

    for candidate in manifest.get("jobs", ()):
        if isinstance(candidate, Mapping):
            reject_reserved_flux_common_seed_outputs(candidate, root)
    common_seed_authorization: dict[str, Any] | None = None
    if args.flux1_common_seed_v2:
        from hierasafe_flow.evaluation.flux1_common_seed_v2 import (
            STAGE as FLUX_COMMON_SEED_STAGE,
            build_common_seed_launch_authorization,
        )

        if job.get("stage") != FLUX_COMMON_SEED_STAGE:
            raise ValueError("Common-seed dispatch flag selected a non-common-stage job.")
        common_seed_authorization = build_common_seed_launch_authorization(
            phase_plan_path=Path(args.phase_plan),
            manifest_path=Path(args.manifest),
            job_index=args.index,
            root=root,
        )
    else:
        if job.get("stage") in FLUX_COMMON_SEED_STAGES:
            raise ValueError(
                "Common-seed jobs require the dedicated phase-bound dispatcher flag."
            )
    model_name = str(job["model_name"])
    if args.command == "resolve":
        # Tabs/newlines are forbidden in all emitted constants so the shell can
        # parse this record without eval or executing manifest-controlled text.
        fields = (contract.name, model_name, contract.diffusers_revision)
        if any("\t" in field or "\n" in field for field in fields):
            raise RuntimeError("Unsafe character in internal environment dispatch record.")
        print("\t".join(fields))
        return 0

    expected = (args.expected_environment, args.expected_model, args.expected_diffusers_revision)
    actual = (contract.name, model_name, contract.diffusers_revision)
    if expected != actual:
        raise RuntimeError(
            "Environment dispatch changed between authentication and preflight: "
            f"expected={expected}, authenticated={actual}."
        )
    validation = validate_job_environment(job, contract)
    if common_seed_authorization is not None:
        validation["common_seed_launch_authorization"] = common_seed_authorization
    persisted = persist_validated_job_environment(
        job=job,
        manifest=manifest,
        job_index=args.index,
        validation=validation,
        root=root,
    )
    print(json.dumps(persisted, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
