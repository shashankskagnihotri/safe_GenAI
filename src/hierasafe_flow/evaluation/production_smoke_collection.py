"""Fail-closed producers for production-smoke evidence bundles.

This module does not submit or launch generation jobs.  It either executes the
registered no-generation checks or reopens already completed smoke rows and
explicit human review ledgers.  Every output is assembled in a hidden sibling,
validated there using relocation-stable relative bindings and fsynced.  CephFS
does not provide ``renameat2(RENAME_NOREPLACE)`` here, so publication uses an
exclusive root claim, immutable no-replace hard links, and a self-authenticating
commit receipt linked last.  Readers reject partial or writable trees.
"""

from __future__ import annotations

from datetime import datetime, timezone
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import stat
import time
from typing import Any, Callable, Mapping
import uuid

from hierasafe_flow.benchmarks.finer_detailing_production_smoke import (
    EXACT_ONE_PATH_GATE,
    FINAL_CUMULATIVE_ADMISSION_GATE,
    FULL_PAIR_NON_REGRESSION_GATE,
    IDEOGRAM_CONDITIONING_GATE,
    NATIVE_BASELINE_GATE,
    NO_GENERATION_GATE,
    NO_GENERATION_PREFLIGHT,
    WAN_TRANSITION_GATE,
    _sha256_file,
    _builder_contract_sha256,
    canonical_sha256,
    paths_overlap,
    project_root,
    read_authenticated_document,
    require_external_artifact_path,
    validate_smoke_plan,
    write_authenticated_document,
)
from hierasafe_flow.evaluation.production_smoke import (
    EVIDENCE_INDEX_CONTRACT,
    EVIDENCE_INDEX_SCHEMA_VERSION,
    NO_GENERATION_CHECK_IDS,
    NO_GENERATION_RAW_EVIDENCE_CONTRACT,
    NO_GENERATION_RAW_EVIDENCE_SCHEMA_VERSION,
    NO_GENERATION_TEST_NODEIDS,
    REQUIRED_NON_REGRESSION_TESTS,
    _build_ideogram_conditioning_preflight_manifest,
    _selected_rows,
    evaluate_smoke_gate,
    no_generation_raw_evidence_digest,
    parse_non_regression_junit,
    read_smoke_gate_evaluation,
    write_smoke_evidence_index_immutable,
)
from hierasafe_flow.evaluation.temporal_metrics import (
    validate_temporal_metric_runtime_receipt,
)


SMOKE_EVIDENCE_BUNDLE_SCHEMA_VERSION = 1
SMOKE_EVIDENCE_BUNDLE_CONTRACT = "finer_detailing_smoke_evidence_bundle_v1"
BUNDLE_RECEIPT_FILENAME = "bundle_receipt.json"
EVIDENCE_INDEX_FILENAME = "evidence_index.json"
EVALUATION_FILENAME = "gate_evaluation.json"
ROW_GATE_CONTRACTS = frozenset(
    {
        WAN_TRANSITION_GATE,
        NATIVE_BASELINE_GATE,
        EXACT_ONE_PATH_GATE,
        FINAL_CUMULATIVE_ADMISSION_GATE,
    }
)

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bundle_receipt_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Short write while sealing smoke evidence.")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_new(path: Path, payload: Any) -> None:
    _write_new(path, _json_bytes(payload))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")


def _canonical_destination(raw: str | Path) -> Path:
    raw_text = os.fspath(raw)
    path = Path(raw_text)
    segments = raw_text.split(os.sep)[1:]
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in segments)
    ):
        raise ValueError("Evidence bundle destination must be one exact absolute canonical path.")
    lexical = path.absolute()
    if lexical != path or lexical.name in {"", ".", ".."}:
        raise ValueError("Evidence bundle destination contains a lexical alias.")
    _reject_symlink_components(lexical.parent, label="evidence bundle parent")
    if not lexical.parent.is_dir():
        raise FileNotFoundError(f"Evidence bundle parent is missing: {lexical.parent}")
    if os.path.lexists(lexical):
        raise FileExistsError(f"Refusing to replace occupied evidence bundle: {lexical}")
    if lexical.resolve(strict=False) != lexical:
        raise ValueError("Evidence bundle destination resolves through an alias.")
    return lexical


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Evidence staging tree contains a symlink: {path}")
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            raise ValueError(f"Evidence staging tree contains a special file: {path}")
    for directory in reversed(directories):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _chmod_tree(root: Path, *, files: int, directories: int) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            path.chmod(directories)
        elif path.is_file() and not path.is_symlink():
            path.chmod(files)
    root.chmod(directories)


def _cleanup_staging(path: Path) -> None:
    if not path.exists():
        return
    try:
        _chmod_tree(path, files=0o600, directories=0o700)
    except OSError:
        pass
    shutil.rmtree(path, ignore_errors=True)


def _remove_published_stage_links(path: Path) -> None:
    """Unlink a read-only stage without chmod'ing hard-linked canonical files."""

    if not path.exists():
        return
    directories = sorted(
        (member for member in path.rglob("*") if member.is_dir()),
        key=lambda member: len(member.parts),
        reverse=True,
    )
    path.chmod(0o700)
    for directory in directories:
        directory.chmod(0o700)
    shutil.rmtree(path, ignore_errors=True)


def _tree_hashes(root: Path, *, exclude_receipt: bool = False) -> dict[str, str]:
    hashes: dict[str, str] = {}
    inodes: set[tuple[int, int]] = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Evidence bundle contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Evidence bundle contains a special file: {path}")
        relative = path.relative_to(root).as_posix()
        if exclude_receipt and relative in {
            BUNDLE_RECEIPT_FILENAME,
            f"{BUNDLE_RECEIPT_FILENAME}.sha256",
        }:
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in inodes:
            raise ValueError("Evidence bundle contains a hard-linked file alias.")
        inodes.add(identity)
        hashes[relative] = _sha256_file(path)
    return hashes


def _write_bundle_receipt(
    stage: Path,
    *,
    destination: Path,
    gate_contract: str,
    subject_plan_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": SMOKE_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "contract": SMOKE_EVIDENCE_BUNDLE_CONTRACT,
        "created_at_utc": _utc_now(),
        "logical_destination": str(destination),
        "physical_staging_path": str(stage),
        "gate_contract": gate_contract,
        "subject_plan_sha256": subject_plan_sha256,
        "directories": sorted(
            path.relative_to(stage).as_posix()
            for path in stage.rglob("*")
            if path.is_dir()
        ),
        "files_sha256": _tree_hashes(stage),
    }
    return write_authenticated_document(
        payload,
        stage / BUNDLE_RECEIPT_FILENAME,
        digest_field="document_sha256",
        digest_function=bundle_receipt_digest,
    )


def _validate_bundle_receipt(
    root: Path,
    *,
    destination: Path,
    gate_contract: str,
    subject_plan_sha256: str,
    require_read_only: bool = True,
) -> dict[str, Any]:
    receipt = read_authenticated_document(
        root / BUNDLE_RECEIPT_FILENAME,
        digest_field="document_sha256",
        digest_function=bundle_receipt_digest,
        label="smoke evidence bundle receipt",
    )
    required = {
        "schema_version",
        "contract",
        "created_at_utc",
        "logical_destination",
        "physical_staging_path",
        "gate_contract",
        "subject_plan_sha256",
        "directories",
        "files_sha256",
        "document_sha256",
    }
    physical_staging = Path(str(receipt.get("physical_staging_path", "")))
    observed_directories = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()
    )
    if (
        set(receipt) != required
        or receipt["schema_version"] != SMOKE_EVIDENCE_BUNDLE_SCHEMA_VERSION
        or receipt["contract"] != SMOKE_EVIDENCE_BUNDLE_CONTRACT
        or receipt["logical_destination"] != str(destination)
        or receipt["gate_contract"] != gate_contract
        or receipt["subject_plan_sha256"] != subject_plan_sha256
        or not physical_staging.is_absolute()
        or physical_staging.parent != destination.parent
        or not physical_staging.name.startswith(f".{destination.name}.staging.")
        or root not in {destination, physical_staging}
        or receipt["directories"] != observed_directories
        or receipt["files_sha256"] != _tree_hashes(root, exclude_receipt=True)
    ):
        raise ValueError("Smoke evidence bundle receipt/path/hash contract drifted.")
    created = datetime.fromisoformat(str(receipt["created_at_utc"]).replace("Z", "+00:00"))
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("Smoke evidence bundle timestamp is not timezone-aware.")
    if require_read_only:
        for member in (root, *sorted(root.rglob("*"))):
            mode = member.stat(follow_symlinks=False).st_mode
            if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                raise ValueError(f"Committed smoke evidence remains writable: {member}")
    return receipt


def validate_committed_smoke_evidence_bundle_root(
    root: Path, *, gate_contract: str, subject_plan_sha256: str
) -> dict[str, Any]:
    """Public reader guard for a commit-last smoke evidence directory."""

    receipt_path = root / BUNDLE_RECEIPT_FILENAME
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Committed smoke evidence receipt is missing: {receipt_path}")
    try:
        preliminary = json.loads(receipt_path.read_text(encoding="utf-8"))
        destination = Path(str(preliminary["logical_destination"]))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Smoke evidence commit receipt cannot identify its logical root.") from exc
    return _validate_bundle_receipt(
        root,
        destination=destination,
        gate_contract=gate_contract,
        subject_plan_sha256=subject_plan_sha256,
        require_read_only=True,
    )


def _link_bundle_commit_last(stage: Path, destination: Path) -> None:
    """Claim and publish on CephFS without an overwrite-capable operation.

    A visible directory without the final receipt is deliberately incomplete;
    readers reject it.  Every immutable member is hard-linked with O_EXCL
    semantics, reauthenticated in its canonical location, and only then is the
    self-authenticating receipt linked as the commit marker.
    """

    directories = sorted(
        (path for path in stage.rglob("*") if path.is_dir()),
        key=lambda path: (len(path.relative_to(stage).parts), path.as_posix()),
    )
    for directory in directories:
        os.mkdir(destination / directory.relative_to(stage), 0o700)
    commit = stage / BUNDLE_RECEIPT_FILENAME
    members = sorted(path for path in stage.rglob("*") if path.is_file())
    if commit not in members:
        raise RuntimeError("Hidden evidence stage lacks its authenticated commit receipt.")
    for source in members:
        if source == commit:
            continue
        target = destination / source.relative_to(stage)
        os.link(source, target, follow_symlinks=False)
    receipt_payload = json.loads(commit.read_text(encoding="utf-8"))
    expected_hashes = receipt_payload.get("files_sha256")
    if not isinstance(expected_hashes, Mapping) or _tree_hashes(
        destination, exclude_receipt=True
    ) != dict(expected_hashes):
        raise ValueError("Canonical precommit files differ from the validated hidden stage.")
    if (destination / BUNDLE_RECEIPT_FILENAME).exists():
        raise FileExistsError("Evidence commit marker appeared before commit-last publication.")
    os.link(commit, destination / BUNDLE_RECEIPT_FILENAME, follow_symlinks=False)
    _fsync_tree(destination)
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        (destination / directory.relative_to(stage)).chmod(0o555)
    destination.chmod(0o555)
    _fsync_tree(destination)


BundleBuilder = Callable[[Path, Path], tuple[str, str, Callable[[Path], Any]]]


def _publish_atomic_bundle(destination: str | Path, builder: BundleBuilder) -> dict[str, Any]:
    destination_path = _canonical_destination(destination)
    stage = destination_path.parent / (
        f".{destination_path.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    )
    os.mkdir(stage, 0o700)
    published = False
    claimed = False
    try:
        gate_contract, subject_sha, reopen = builder(stage, destination_path)
        _write_bundle_receipt(
            stage,
            destination=destination_path,
            gate_contract=gate_contract,
            subject_plan_sha256=subject_sha,
        )
        _validate_bundle_receipt(
            stage,
            destination=destination_path,
            gate_contract=gate_contract,
            subject_plan_sha256=subject_sha,
            require_read_only=False,
        )
        _fsync_tree(stage)
        _chmod_tree(stage, files=0o444, directories=0o555)
        _fsync_tree(stage)
        _validate_bundle_receipt(
            stage,
            destination=destination_path,
            gate_contract=gate_contract,
            subject_plan_sha256=subject_sha,
            require_read_only=True,
        )
        reopen(stage)
        parent_fd = os.open(destination_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
            os.mkdir(destination_path, 0o700)
            claimed = True
            _link_bundle_commit_last(stage, destination_path)
            published = True
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        _validate_bundle_receipt(
            destination_path,
            destination=destination_path,
            gate_contract=gate_contract,
            subject_plan_sha256=subject_sha,
        )
        report = reopen(destination_path)
        return {
            "bundle": str(destination_path),
            "gate_contract": gate_contract,
            "subject_plan_sha256": subject_sha,
            "evaluation": report,
        }
    finally:
        if claimed and not published and destination_path.exists():
            _cleanup_staging(destination_path)
        if published:
            _remove_published_stage_links(stage)
        else:
            _cleanup_staging(stage)


def _run_process(
    command: list[str], *, root: Path, timeout_seconds: int, environment: Mapping[str, str] | None
) -> dict[str, Any]:
    started_at = _utc_now()
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=root,
        env=dict(environment) if environment is not None else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    duration = time.perf_counter() - start
    completed_at = _utc_now()
    if completed.returncode != 0:
        raise RuntimeError(
            f"Smoke evidence command failed (exit {completed.returncode}): {command!r}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Smoke evidence command returned an invalid measured duration.")
    return {
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "duration_seconds": duration,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _write_raw_process_evidence(
    stage: Path,
    *,
    ordinal: int,
    check_id: str,
    evidence_kind: str,
    command: list[str],
    execution: Mapping[str, Any],
    measurements: Mapping[str, Any],
) -> dict[str, str]:
    path = stage / "checks" / f"{ordinal:02d}_{check_id}.json"
    raw = write_authenticated_document(
        {
            "schema_version": NO_GENERATION_RAW_EVIDENCE_SCHEMA_VERSION,
            "contract": NO_GENERATION_RAW_EVIDENCE_CONTRACT,
            "check_id": check_id,
            "evidence_kind": evidence_kind,
            "started_at_utc": execution["started_at_utc"],
            "completed_at_utc": execution["completed_at_utc"],
            "command": command,
            "exit_code": execution["exit_code"],
            "duration_seconds": execution["duration_seconds"],
            "stdout": execution["stdout"],
            "stderr": execution["stderr"],
            "measurements": deepcopy(dict(measurements)),
        },
        path,
        digest_field="document_sha256",
        digest_function=no_generation_raw_evidence_digest,
    )
    return {
        "check_id": check_id,
        "evidence_path": path.relative_to(stage).as_posix(),
        "evidence_file_sha256": _sha256_file(path),
        "evidence_document_sha256": raw["document_sha256"],
    }


def collect_no_generation_evidence_bundle(
    *,
    subject_plan_path: str | Path,
    destination: str | Path,
    root: Path | None = None,
    timeout_seconds: int = 43_200,
) -> dict[str, Any]:
    """Execute and atomically seal the exact ten registered no-generation checks."""

    root = (root or project_root()).resolve()
    if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ValueError("No-generation command timeout must be a positive integer.")
    subject = validate_smoke_plan(
        subject_plan_path, root=root, expected_stage=NO_GENERATION_PREFLIGHT
    )
    generation_root = Path(subject.plan["output_root"]).resolve()
    destination_path = _canonical_destination(destination)
    require_external_artifact_path(destination_path, generation_root, "smoke evidence bundle")

    def build(stage: Path, _logical_destination: Path):
        from scripts.finer_detailing_environment_dispatch import (
            ENVIRONMENT_CONTRACTS,
            LTX_ENVIRONMENT,
            MAIN_ENVIRONMENT,
        )

        rows: list[dict[str, str]] = []
        base_environment = dict(os.environ)
        base_environment["PYTEST_ADDOPTS"] = ""
        for ordinal, check_id in enumerate(NO_GENERATION_CHECK_IDS):
            if check_id in {
                "main_environment_verified_install",
                "ltx_environment_verified_install",
            }:
                environment_name = (
                    MAIN_ENVIRONMENT
                    if check_id == "main_environment_verified_install"
                    else LTX_ENVIRONMENT
                )
                command = [
                    "conda",
                    "run",
                    "-n",
                    environment_name,
                    "python",
                    "scripts/finer_detailing_environment_dispatch.py",
                    "verify-install",
                    "--environment",
                    environment_name,
                ]
                execution = _run_process(
                    command,
                    root=root,
                    timeout_seconds=timeout_seconds,
                    environment=base_environment,
                )
                receipt = json.loads(execution["stdout"])
                measurements = {
                    "environment_name": environment_name,
                    "status": "verified_install",
                    "diffusers_revision": ENVIRONMENT_CONTRACTS[
                        environment_name
                    ].diffusers_revision,
                    "temporal_metric_status": "passed",
                }
                if (
                    receipt.get("status") != "verified_install"
                    or (receipt.get("environment") or {}).get("name") != environment_name
                    or (receipt.get("diffusers") or {}).get("revision")
                    != measurements["diffusers_revision"]
                    or (receipt.get("temporal_metric_contract") or {}).get("status") != "passed"
                ):
                    raise ValueError(f"Verified-install output drifted for {environment_name}.")
                runtime_preflight = receipt.get("temporal_metric_runtime_preflight")
                if not isinstance(runtime_preflight, Mapping):
                    raise ValueError(
                        f"Verified-install output lacks schema-2 numeric evidence for "
                        f"{environment_name}."
                    )
                validate_temporal_metric_runtime_receipt(
                    runtime_preflight,
                    expected_environment_name=environment_name,
                )
                kind = "verified_install"
            elif check_id == "peak_load_feasibility":
                detail = stage / "peak" / "all_models_h100_load_peak.json"
                detail.parent.mkdir(parents=True, exist_ok=True)
                script = (root / "scripts/measure_finer_detailing_model_load_peak.py").resolve()
                command = [
                    sys.executable,
                    str(script),
                    "--project-root",
                    str(root),
                    "--detail-output",
                    str(detail),
                    "--require-h100",
                ]
                execution = _run_process(
                    command,
                    root=root,
                    timeout_seconds=timeout_seconds,
                    environment=base_environment,
                )
                aggregate = json.loads(execution["stdout"])
                detail_payload = json.loads(detail.read_text(encoding="utf-8"))
                measurements = {
                    "aggregate": aggregate,
                    "detail_receipt": {
                        "path": detail.relative_to(stage).as_posix(),
                        "file_sha256": _sha256_file(detail),
                        "document_sha256": detail_payload["document_sha256"],
                    },
                }
                kind = "cuda_peak_measurement"
            else:
                nodeids = list(NO_GENERATION_TEST_NODEIDS[check_id])
                command = [sys.executable, "-m", "pytest", "-q", *nodeids]
                execution = _run_process(
                    command,
                    root=root,
                    timeout_seconds=timeout_seconds,
                    environment=base_environment,
                )
                source_paths = sorted(
                    {(root / nodeid.split("::", 1)[0]).resolve() for nodeid in nodeids}
                )
                measurements = {
                    "test_nodeids": nodeids,
                    "passed_count": len(nodeids),
                    "failed_count": 0,
                    "source_files_sha256": {
                        str(source): _sha256_file(source) for source in source_paths
                    },
                }
                kind = "pytest_execution"
            rows.append(
                _write_raw_process_evidence(
                    stage,
                    ordinal=ordinal,
                    check_id=check_id,
                    evidence_kind=kind,
                    command=command,
                    execution=execution,
                    measurements=measurements,
                )
            )
        report_path = stage / "no_generation_report.json"
        _write_json_new(
            report_path,
            {
                "schema_version": 1,
                "subject_plan_sha256": subject.digest,
                "implementation_files_sha256": subject.plan[
                    "implementation_files_sha256"
                ],
                "checks": rows,
            },
        )
        index_path = stage / EVIDENCE_INDEX_FILENAME
        write_smoke_evidence_index_immutable(
            {
                "schema_version": EVIDENCE_INDEX_SCHEMA_VERSION,
                "contract": EVIDENCE_INDEX_CONTRACT,
                "gate_contract": NO_GENERATION_GATE,
                "subject_plan_sha256": subject.digest,
                "row_evidence": {},
                "no_generation_report": {
                    "path": report_path.relative_to(stage).as_posix(),
                    "sha256": _sha256_file(report_path),
                },
                "ideogram_conditioning_report": None,
                "non_regression_report": None,
            },
            index_path,
            root=root,
        )
        evaluate_smoke_gate(
            NO_GENERATION_GATE,
            subject_plan_path=subject.path,
            evidence_index_path=index_path,
            output_path=stage / EVALUATION_FILENAME,
            root=root,
            bundle_relative_bindings=True,
        )

        def reopen(bundle: Path):
            return read_smoke_gate_evaluation(
                bundle / EVALUATION_FILENAME,
                subject=subject,
                expected_contract=NO_GENERATION_GATE,
                root=root,
            )

        return NO_GENERATION_GATE, subject.digest, reopen

    return _publish_atomic_bundle(destination_path, build)


def seal_ideogram_conditioning_evidence_bundle(
    *,
    subject_plan_path: str | Path,
    destination: str | Path,
    root: Path | None = None,
) -> dict[str, Any]:
    """Extract schema-3 provenance only from the exact executed Ideogram report."""

    root = (root or project_root()).resolve()
    subject = validate_smoke_plan(subject_plan_path, root=root)
    generation_root = Path(subject.plan["output_root"]).resolve()
    destination_path = _canonical_destination(destination)
    require_external_artifact_path(destination_path, generation_root, "smoke evidence bundle")
    manifest = _build_ideogram_conditioning_preflight_manifest(subject, root=root)
    job = manifest["jobs"][0]
    output = Path(str(job["output_dir"])).resolve()
    source_report = output / "sample_0000" / "report.json"
    source_result = output / "benchmark_job_result.json"
    for source, label in (
        (source_report, "executed Ideogram sample report"),
        (source_result, "executed Ideogram result"),
    ):
        _reject_symlink_components(source, label=label)
        if not source.is_file() or source.resolve() != source:
            raise FileNotFoundError(f"Exact {label} is missing or noncanonical: {source}")

    def build(stage: Path, _logical_destination: Path):
        # Read now and again through the evaluator.  The collector does not
        # accept a records argument and cannot synthesize branch provenance.
        report_payload = json.loads(source_report.read_text(encoding="utf-8"))
        result_payload = json.loads(source_result.read_text(encoding="utf-8"))
        conditioning = report_payload.get("conditioning_provenance")
        if not isinstance(conditioning, Mapping) or conditioning.get("schema_version") != 3:
            raise ValueError("Executed Ideogram report lacks schema-3 conditioning provenance.")
        embedded_job = result_payload.get("job")
        if not isinstance(embedded_job, Mapping):
            raise ValueError("Executed Ideogram result lacks its exact launch job.")
        executed_manifest_sha = embedded_job.get("launch_manifest_sha256")
        summary_path = stage / "ideogram_conditioning_report.json"
        _write_json_new(
            summary_path,
            {
                "schema_version": 2,
                "subject_plan_sha256": subject.digest,
                "implementation_files_sha256": subject.plan[
                    "implementation_files_sha256"
                ],
                "model_name": "ideogram4_nf4",
                "model_revision": job["model_revision"],
                "prompt_id": "03_empty_outdoor_mall",
                "max_sequence_length": 2048,
                "source_manifest": {
                    "executed_manifest_sha256": executed_manifest_sha,
                    "builder_contract_sha256": _builder_contract_sha256(manifest),
                    "manifest_job_index": 0,
                    "condition_id": job["condition_id"],
                },
                "source_sample_report": {
                    "path": str(source_report),
                    "file_sha256": _sha256_file(source_report),
                },
                "source_result": {
                    "path": str(source_result),
                    "file_sha256": _sha256_file(source_result),
                },
            },
        )
        index_path = stage / EVIDENCE_INDEX_FILENAME
        write_smoke_evidence_index_immutable(
            {
                "schema_version": EVIDENCE_INDEX_SCHEMA_VERSION,
                "contract": EVIDENCE_INDEX_CONTRACT,
                "gate_contract": IDEOGRAM_CONDITIONING_GATE,
                "subject_plan_sha256": subject.digest,
                "row_evidence": {},
                "no_generation_report": None,
                "ideogram_conditioning_report": {
                    "path": summary_path.relative_to(stage).as_posix(),
                    "sha256": _sha256_file(summary_path),
                },
                "non_regression_report": None,
            },
            index_path,
            root=root,
        )
        evaluate_smoke_gate(
            IDEOGRAM_CONDITIONING_GATE,
            subject_plan_path=subject.path,
            evidence_index_path=index_path,
            output_path=stage / EVALUATION_FILENAME,
            root=root,
            bundle_relative_bindings=True,
        )

        def reopen(bundle: Path):
            return read_smoke_gate_evaluation(
                bundle / EVALUATION_FILENAME,
                subject=subject,
                expected_contract=IDEOGRAM_CONDITIONING_GATE,
                root=root,
            )

        return IDEOGRAM_CONDITIONING_GATE, subject.digest, reopen

    return _publish_atomic_bundle(destination_path, build)


def collect_full_pair_non_regression_evidence_bundle(
    *,
    subject_plan_path: str | Path,
    destination: str | Path,
    root: Path | None = None,
    timeout_seconds: int = 1_800,
) -> dict[str, Any]:
    """Run the registered Shapley tests with JUnit capture, then live re-evaluate."""

    root = (root or project_root()).resolve()
    if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ValueError("Non-regression timeout must be a positive integer.")
    subject = validate_smoke_plan(subject_plan_path, root=root)
    generation_root = Path(subject.plan["output_root"]).resolve()
    destination_path = _canonical_destination(destination)
    require_external_artifact_path(destination_path, generation_root, "smoke evidence bundle")

    def build(stage: Path, _logical_destination: Path):
        junit = stage / "non_regression.junit.xml"
        command = [sys.executable, "-m", "pytest", "-q", *REQUIRED_NON_REGRESSION_TESTS]
        environment = dict(os.environ)
        environment["PYTEST_ADDOPTS"] = f"--junitxml={junit}"
        execution = _run_process(
            command,
            root=root,
            timeout_seconds=timeout_seconds,
            environment=environment,
        )
        if not junit.is_file():
            raise RuntimeError("Registered pytest execution did not emit its JUnit XML capture.")
        results = parse_non_regression_junit(junit)
        sources = {
            str((root / nodeid.split("::", 1)[0]).resolve()): _sha256_file(
                (root / nodeid.split("::", 1)[0]).resolve()
            )
            for nodeid in REQUIRED_NON_REGRESSION_TESTS
        }
        junit_binding = {
            "path": junit.relative_to(stage).as_posix(),
            "file_sha256": _sha256_file(junit),
        }
        canonical_output = {
            "command": command,
            "started_at_utc": execution["started_at_utc"],
            "completed_at_utc": execution["completed_at_utc"],
            "duration_seconds": execution["duration_seconds"],
            "exit_code": execution["exit_code"],
            "stdout": execution["stdout"],
            "stderr": execution["stderr"],
            "test_results": results,
            "source_files_sha256": sources,
            "junit_xml": junit_binding,
        }
        summary = {
            "schema_version": 2,
            "subject_plan_sha256": subject.digest,
            "implementation_files_sha256": subject.plan["implementation_files_sha256"],
            **canonical_output,
            "structured_output_sha256": canonical_sha256(canonical_output),
        }
        report_path = stage / "non_regression_report.json"
        _write_json_new(report_path, summary)
        index_path = stage / EVIDENCE_INDEX_FILENAME
        write_smoke_evidence_index_immutable(
            {
                "schema_version": EVIDENCE_INDEX_SCHEMA_VERSION,
                "contract": EVIDENCE_INDEX_CONTRACT,
                "gate_contract": FULL_PAIR_NON_REGRESSION_GATE,
                "subject_plan_sha256": subject.digest,
                "row_evidence": {},
                "no_generation_report": None,
                "ideogram_conditioning_report": None,
                "non_regression_report": {
                    "path": report_path.relative_to(stage).as_posix(),
                    "sha256": _sha256_file(report_path),
                },
            },
            index_path,
            root=root,
        )
        evaluate_smoke_gate(
            FULL_PAIR_NON_REGRESSION_GATE,
            subject_plan_path=subject.path,
            evidence_index_path=index_path,
            output_path=stage / EVALUATION_FILENAME,
            root=root,
            bundle_relative_bindings=True,
        )

        def reopen(bundle: Path):
            return read_smoke_gate_evaluation(
                bundle / EVALUATION_FILENAME,
                subject=subject,
                expected_contract=FULL_PAIR_NON_REGRESSION_GATE,
                root=root,
            )

        return FULL_PAIR_NON_REGRESSION_GATE, subject.digest, reopen

    return _publish_atomic_bundle(destination_path, build)


def _canonical_input_file(raw: str | Path, *, root: Path, label: str) -> Path:
    raw_text = os.fspath(raw)
    candidate = Path(raw_text)
    segments = raw_text.split(os.sep)
    if candidate.is_absolute():
        segments = segments[1:]
    if any(part in {"", ".", ".."} for part in segments):
        raise ValueError(f"{label} contains a lexical alias: {raw_text!r}")
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.absolute()
    _reject_symlink_components(candidate, label=label)
    if candidate.resolve(strict=False) != candidate or not candidate.is_file():
        raise ValueError(f"{label} is missing or noncanonical: {candidate}")
    return candidate


def seal_selected_row_evidence_bundle(
    contract: str,
    *,
    subject_plan_path: str | Path,
    manual_ledger_paths: Mapping[str, str | Path],
    destination: str | Path,
    root: Path | None = None,
) -> dict[str, Any]:
    """Seal selected row hashes plus caller-supplied human decisions atomically.

    No decision values are accepted on the command line and no ledger is
    synthesized.  Exact user-supplied ledger bytes are copied into the sealed
    bundle and then authenticated by the ordinary smoke evaluator.
    """

    if contract not in ROW_GATE_CONTRACTS:
        raise ValueError(f"{contract!r} is not a row-evidence smoke gate.")
    root = (root or project_root()).resolve()
    subject = validate_smoke_plan(subject_plan_path, root=root)
    rows = _selected_rows(contract, subject)
    media_conditions = [
        str(job["condition_id"]) for _owner, _role, _manifest, _index, job in rows
        if bool(job["expected_media"])
    ]
    if list(manual_ledger_paths) != media_conditions:
        raise ValueError(
            "Explicit manual-ledger keys/order must equal the exact selected media rows."
        )
    generation_root = Path(subject.plan["output_root"]).resolve()
    destination_path = _canonical_destination(destination)
    require_external_artifact_path(destination_path, generation_root, "smoke evidence bundle")
    inputs: dict[str, Path] = {}
    identities: set[tuple[int, int]] = set()
    for condition_id in media_conditions:
        source = _canonical_input_file(
            manual_ledger_paths[condition_id], root=root, label="manual review ledger"
        )
        require_external_artifact_path(source, generation_root, "manual review ledger")
        if paths_overlap(source, destination_path):
            raise ValueError("Manual review ledger may not alias its destination bundle.")
        stat = source.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in identities:
            raise ValueError("Two selected rows may not share one manual-ledger inode.")
        identities.add(identity)
        inputs[condition_id] = source

    def build(stage: Path, _logical_destination: Path):
        routing: dict[str, dict[str, Any]] = {}
        manual_ordinal = 0
        for _owner, _role, _manifest, _index, job in rows:
            condition_id = str(job["condition_id"])
            output = Path(str(job["output_dir"])).absolute()
            result = output / "benchmark_job_result.json"
            if not result.is_file():
                raise FileNotFoundError(f"Selected smoke result is missing: {result}")
            result_binding = {"path": str(result), "sha256": _sha256_file(result)}
            if bool(job["expected_media"]):
                media = output / "sample_0000" / (
                    "video_000.mp4"
                    if job["generation"]["task"] == "text_to_video"
                    else "image_000.png"
                )
                if not media.is_file():
                    raise FileNotFoundError(f"Selected smoke media is missing: {media}")
                source = inputs[condition_id]
                target = stage / "manual_ledgers" / f"{manual_ordinal:03d}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_new(target, source.read_bytes())
                if _sha256_file(target) != _sha256_file(source):
                    raise RuntimeError("Manual review ledger changed during exact byte copy.")
                routing[condition_id] = {
                    "result": result_binding,
                    "media": {"path": str(media), "sha256": _sha256_file(media)},
                    "manual_ledger": {
                        "path": target.relative_to(stage).as_posix(),
                        "sha256": _sha256_file(target),
                    },
                }
                manual_ordinal += 1
            else:
                routing[condition_id] = {"result": result_binding}
        if manual_ordinal != len(media_conditions):
            raise RuntimeError("Selected manual-ledger coverage drifted during bundle assembly.")
        index_path = stage / EVIDENCE_INDEX_FILENAME
        write_smoke_evidence_index_immutable(
            {
                "schema_version": EVIDENCE_INDEX_SCHEMA_VERSION,
                "contract": EVIDENCE_INDEX_CONTRACT,
                "gate_contract": contract,
                "subject_plan_sha256": subject.digest,
                "row_evidence": routing,
                "no_generation_report": None,
                "ideogram_conditioning_report": None,
                "non_regression_report": None,
            },
            index_path,
            root=root,
        )
        evaluate_smoke_gate(
            contract,
            subject_plan_path=subject.path,
            evidence_index_path=index_path,
            output_path=stage / EVALUATION_FILENAME,
            root=root,
            bundle_relative_bindings=True,
        )

        def reopen(bundle: Path):
            return read_smoke_gate_evaluation(
                bundle / EVALUATION_FILENAME,
                subject=subject,
                expected_contract=contract,
                root=root,
            )

        return contract, subject.digest, reopen

    return _publish_atomic_bundle(destination_path, build)
