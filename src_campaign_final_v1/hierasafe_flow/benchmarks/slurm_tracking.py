"""Immutable Slurm execution and submission provenance for benchmark jobs.

The generation manifest remains immutable.  Runtime scheduler identity is kept
in a separate file inside the attempt directory, while submission identity is
kept in an immutable registry created immediately after ``sbatch`` returns.
Together these records let the auditor distinguish an unsubmitted job from an
array element that is queued, running, or terminal in Slurm.
"""

from __future__ import annotations

import hashlib
import errno
import json
import os
import re
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hierasafe_flow.evaluation.temporal_metrics import (
    SEGMENTED_METRIC_PARAMETERS_SHA256,
    validate_temporal_metric_runtime_receipt,
)


EXECUTION_IDENTITY_FILENAME = "execution_identity.json"
ENVIRONMENT_PREFLIGHT_FILENAME = "environment_preflight.json"
ENVIRONMENT_PREFLIGHT_DIGEST_FILENAME = f"{ENVIRONMENT_PREFLIGHT_FILENAME}.sha256"
ENVIRONMENT_PREFLIGHT_REQUIRED_ENV = "HIERASAFE_REQUIRE_ENVIRONMENT_PREFLIGHT"
REGISTRY_SCHEMA_VERSION = 1
_ARRAY_ELEMENT_RE = re.compile(r"^(?P<start>[0-9]+)(?:-(?P<stop>[0-9]+)(?::(?P<step>[0-9]+))?)?$")
_SLURM_JOB_ID_RE = re.compile(r"^[0-9]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENT_PREFLIGHT_SCHEMA_VERSION = 2
_LEGACY_TEMPORAL_METRIC_SOURCE_SHA256 = (
    "4c0e757eed30b8f0a2dee85e9db912ae9d313dc12ac4b0ca50f8529a40d95ba8"
)
_LEGACY_TEMPORAL_METRIC_PARAMETERS_SHA256 = (
    "3f7ea400e3e5649598191a386bdce60bb19936a4301f47dfaba0d79891a1ef97"
)
_TEMPORAL_METRIC_SOURCE_PATH = "src/hierasafe_flow/evaluation/temporal_metrics.py"
_ENVIRONMENT_PREFLIGHT_V1_REQUIRED_KEYS = {
    "schema_version",
    "status",
    "model_name",
    "condition_id",
    "manifest_sha256",
    "manifest_job_index",
    "output_dir",
    "captured_at_utc",
    "environment",
    "diffusers",
    "runtime_distributions",
    "temporal_metric_contract",
    "source_contract",
}
_ENVIRONMENT_PREFLIGHT_V1_ALLOWED_KEYS = _ENVIRONMENT_PREFLIGHT_V1_REQUIRED_KEYS | {
    "wan_native_negative_prompt_cleaner",
    "common_seed_launch_authorization",
    "qualification_launch_authorization",
}
_ENVIRONMENT_PREFLIGHT_REQUIRED_KEYS = _ENVIRONMENT_PREFLIGHT_V1_REQUIRED_KEYS | {
    "temporal_metric_runtime_preflight"
}
_ENVIRONMENT_PREFLIGHT_ALLOWED_KEYS = _ENVIRONMENT_PREFLIGHT_V1_ALLOWED_KEYS | {
    "temporal_metric_runtime_preflight"
}


def capture_execution_identity(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture scheduler identity without adding runtime fields to a job dict."""

    values = os.environ if environ is None else environ
    slurm_job_id = _optional_text(values.get("SLURM_JOB_ID"))
    array_job_id = _optional_text(values.get("SLURM_ARRAY_JOB_ID"))
    array_task_id = _optional_text(values.get("SLURM_ARRAY_TASK_ID"))
    task_id = (
        f"{array_job_id}_{array_task_id}"
        if array_job_id is not None and array_task_id is not None
        else slurm_job_id
    )
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "SLURM_JOB_ID": slurm_job_id,
        "SLURM_ARRAY_JOB_ID": array_job_id,
        "SLURM_ARRAY_TASK_ID": array_task_id,
        "SLURM_JOB_NAME": _optional_text(values.get("SLURM_JOB_NAME")),
        "slurm_task_id": task_id,
    }


def write_execution_identity(
    output_dir: Path,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Write the immutable runtime identity once at attempt startup."""

    target = output_dir / EXECUTION_IDENTITY_FILENAME
    _atomic_write_new_json(target, capture_execution_identity(environ))
    return target


def publish_environment_preflight(
    output_dir: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one immutable, attempt-bound environment preflight.

    Validation happens before the attempt directory is created.  The JSON and
    its raw-file SHA-256 sidecar are then hard-link published without replacing
    either path.  If the two-file publication cannot complete, files created by
    this call are removed so a partial preflight cannot masquerade as evidence.
    """

    resolved_output = output_dir.resolve(strict=False)
    frozen = dict(payload)
    if frozen.get("schema_version") != _ENVIRONMENT_PREFLIGHT_SCHEMA_VERSION or isinstance(
        frozen.get("schema_version"), bool
    ):
        raise ValueError(
            "New environment preflight publication requires schema_version integer 2."
        )
    _validate_environment_preflight_payload(
        frozen,
        resolved_output,
        expected_job=None,
        allow_legacy_schema1=False,
    )
    encoded = (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode("utf-8")
    file_sha256 = hashlib.sha256(encoded).hexdigest()
    sidecar_bytes = (
        f"{file_sha256}  {ENVIRONMENT_PREFLIGHT_FILENAME}\n".encode("utf-8")
    )
    target = resolved_output / ENVIRONMENT_PREFLIGHT_FILENAME
    digest_target = resolved_output / ENVIRONMENT_PREFLIGHT_DIGEST_FILENAME

    if resolved_output.exists() and any(resolved_output.iterdir()):
        raise FileExistsError(
            "Refusing to publish an environment preflight into a non-empty attempt "
            f"directory: {resolved_output}"
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    created_output_dir = False
    if not resolved_output.exists():
        resolved_output.mkdir()
        created_output_dir = True

    temporary_json: Path | None = None
    temporary_digest: Path | None = None
    published: list[tuple[Path, Path]] = []
    try:
        temporary_json = _write_temporary_bytes(
            resolved_output,
            ENVIRONMENT_PREFLIGHT_FILENAME,
            encoded,
        )
        temporary_digest = _write_temporary_bytes(
            resolved_output,
            ENVIRONMENT_PREFLIGHT_DIGEST_FILENAME,
            sidecar_bytes,
        )
        for temporary, destination in (
            (temporary_json, target),
            (temporary_digest, digest_target),
        ):
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite immutable environment preflight file: {destination}"
                ) from exc
            published.append((destination, temporary))
        for destination, _temporary in published:
            destination.chmod(0o444)
        _fsync_directory(resolved_output)
    except Exception:
        for destination, temporary in reversed(published):
            try:
                if destination.stat().st_ino == temporary.stat().st_ino:
                    destination.unlink()
            except FileNotFoundError:
                pass
        if created_output_dir:
            try:
                resolved_output.rmdir()
            except OSError:
                pass
        raise
    finally:
        for temporary in (temporary_json, temporary_digest):
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    return {
        "path": str(target),
        "sha256": file_sha256,
        "digest_path": str(digest_target),
    }


def read_environment_preflight(
    output_dir: Path,
    *,
    expected_job: Mapping[str, Any] | None = None,
    expected_job_index: int | None = None,
) -> dict[str, Any]:
    """Authenticate a persisted preflight and optionally reopen its job binding."""

    resolved_output = output_dir.resolve(strict=False)
    target = resolved_output / ENVIRONMENT_PREFLIGHT_FILENAME
    digest_target = resolved_output / ENVIRONMENT_PREFLIGHT_DIGEST_FILENAME
    try:
        encoded = _read_immutable_single_link_file(
            target,
            label="environment preflight",
        )
        sidecar_encoded = _read_immutable_single_link_file(
            digest_target,
            label="environment preflight SHA-256 sidecar",
        )
        sidecar_fields = sidecar_encoded.decode("utf-8").split()
        payload = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot authenticate environment preflight in {resolved_output}: {exc}"
        ) from exc
    observed_sha256 = hashlib.sha256(encoded).hexdigest()
    if sidecar_fields != [observed_sha256, ENVIRONMENT_PREFLIGHT_FILENAME]:
        raise ValueError(
            f"Environment preflight SHA-256 sidecar is absent or inconsistent: {digest_target}"
        )
    if not isinstance(payload, dict):
        raise ValueError("Environment preflight must contain one JSON object.")
    _validate_environment_preflight_payload(
        payload,
        resolved_output,
        expected_job=expected_job,
        allow_legacy_schema1=expected_job is not None,
    )
    if expected_job is not None:
        raw_expected_output = str(expected_job.get("output_dir", "")).strip()
        if not raw_expected_output:
            raise ValueError("Expected generation job lacks an output_dir binding.")
        expected_output = Path(raw_expected_output).resolve(strict=False)
        expected = {
            "model_name": expected_job.get("model_name"),
            "condition_id": expected_job.get("condition_id"),
            "manifest_sha256": expected_job.get("launch_manifest_sha256"),
            "output_dir": str(expected_output),
        }
        drift = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if drift:
            raise ValueError(
                f"Environment preflight differs from the generation job binding: {drift}"
            )
    if expected_job_index is not None and payload["manifest_job_index"] != expected_job_index:
        raise ValueError(
            "Environment preflight manifest-job index differs from the expected binding: "
            f"expected={expected_job_index}, actual={payload['manifest_job_index']}."
        )
    return payload


def _read_immutable_single_link_file(path: Path, *, label: str) -> bytes:
    """Read one immutable evidence file through a stable, no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise ValueError(f"{label.capitalize()} must not be a symbolic link: {path}") from exc
        raise
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label.capitalize()} is not a regular file: {path}")
        if before.st_mode & 0o222:
            raise ValueError(f"{label.capitalize()} is writable: {path}")
        if before.st_nlink != 1:
            raise ValueError(
                f"{label.capitalize()} must have exactly one hard link; "
                f"found {before.st_nlink}: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
            raise ValueError(f"{label.capitalize()} changed while it was being read: {path}")
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"{label.capitalize()} path changed while it was read: {path}") from exc
        if any(getattr(before, field) != getattr(current, field) for field in stable_fields):
            raise ValueError(f"{label.capitalize()} inode was replaced while it was read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def environment_preflight_required(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Read the launcher's explicit preflight requirement without truthy guessing."""

    values = os.environ if environ is None else environ
    raw = values.get(ENVIRONMENT_PREFLIGHT_REQUIRED_ENV)
    if raw is None:
        return False
    if raw != "1":
        raise RuntimeError(
            f"{ENVIRONMENT_PREFLIGHT_REQUIRED_ENV} must be exactly '1' when present."
        )
    return True


def parse_array_spec(value: str) -> list[int]:
    """Expand the integer subset of a Slurm array specification.

    Comma-separated indices, inclusive ranges, range steps, and the optional
    ``%N`` concurrency suffix are accepted.  Slurm's filename expressions and
    arbitrary shell syntax are intentionally not accepted.
    """

    raw = value.strip()
    if not raw:
        raise ValueError("Array specification must not be empty.")
    subset, separator, concurrency = raw.partition("%")
    if separator:
        if not concurrency.isdigit() or int(concurrency) <= 0 or "%" in concurrency:
            raise ValueError(f"Invalid Slurm array concurrency suffix: {value!r}")
    indices: set[int] = set()
    for token in subset.split(","):
        match = _ARRAY_ELEMENT_RE.fullmatch(token.strip())
        if match is None:
            raise ValueError(f"Invalid Slurm array element: {token!r}")
        start = int(match.group("start"))
        stop_value = match.group("stop")
        if stop_value is None:
            indices.add(start)
            continue
        stop = int(stop_value)
        step = int(match.group("step") or 1)
        if step <= 0 or stop < start:
            raise ValueError(f"Invalid Slurm array range: {token!r}")
        indices.update(range(start, stop + 1, step))
    if not indices:
        raise ValueError("Array specification selected no indices.")
    return sorted(indices)


def build_submission_registry(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    slurm_array_job_id: str,
    array_spec: str,
    slurm_job_name: str | None = None,
) -> dict[str, Any]:
    """Build an immutable manifest-index to Slurm-array-task registry."""

    job_id = slurm_array_job_id.strip().split(";", 1)[0]
    if _SLURM_JOB_ID_RE.fullmatch(job_id) is None:
        raise ValueError(f"Slurm array job ID must be numeric, got {slurm_array_job_id!r}.")
    manifest_sha256 = str(manifest.get("manifest_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
        raise ValueError("Manifest is missing a lowercase SHA-256 identity.")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Manifest jobs must be a list.")
    indices = parse_array_spec(array_spec)
    out_of_range = [index for index in indices if index >= len(jobs)]
    if out_of_range:
        raise ValueError(
            f"Array indices exceed manifest job range 0..{len(jobs) - 1}: {out_of_range}"
        )
    registry: dict[str, Any] = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "benchmark": str(manifest.get("benchmark", "")),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve(strict=False)),
        "manifest_sha256": manifest_sha256,
        "slurm_array_job_id": job_id,
        "slurm_job_name": _optional_text(slurm_job_name),
        "array_spec": array_spec,
        "num_registered_tasks": len(indices),
        "submissions": [
            {
                "manifest_sha256": manifest_sha256,
                "job_index": index,
                "slurm_array_job_id": job_id,
                "slurm_array_task_id": index,
                "slurm_task_id": f"{job_id}_{index}",
            }
            for index in indices
        ],
    }
    registry["registry_sha256"] = submission_registry_digest(registry)
    return registry


def submission_registry_digest(registry: Mapping[str, Any]) -> str:
    canonical = dict(registry)
    canonical.pop("registry_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_submission_registry(registry: Mapping[str, Any], path: Path) -> None:
    """Atomically create a registry, refusing all overwrite attempts."""

    expected = submission_registry_digest(registry)
    if registry.get("registry_sha256") != expected:
        raise ValueError("Submission registry digest does not match its content.")
    _atomic_write_new_json(path, registry)


def read_submission_registry(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read submission registry {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Submission registry {path} must contain a JSON object.")
    if int(payload.get("schema_version", 0)) != REGISTRY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported submission registry schema in {path}.")
    expected = submission_registry_digest(payload)
    if payload.get("registry_sha256") != expected:
        raise ValueError(f"Submission registry digest mismatch: {path}")
    submissions = payload.get("submissions")
    if not isinstance(submissions, list) or len(submissions) != payload.get(
        "num_registered_tasks"
    ):
        raise ValueError(f"Submission registry task count mismatch: {path}")
    for position, item in enumerate(submissions):
        if not isinstance(item, dict):
            raise ValueError(f"Submission registry entry {position} is not an object: {path}")
        _validate_submission_entry(item, payload, position, path)
    return payload


def load_submission_lookup(
    registry_paths: Sequence[Path],
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    """Load registries and reject duplicate manifest/index ownership."""

    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for path in registry_paths:
        resolved = path.resolve(strict=False)
        registry = read_submission_registry(resolved)
        rows.append(
            {
                "path": str(resolved),
                "registry_sha256": registry["registry_sha256"],
                "manifest_sha256": registry["manifest_sha256"],
                "slurm_array_job_id": registry["slurm_array_job_id"],
                "array_spec": registry["array_spec"],
                "num_registered_tasks": registry["num_registered_tasks"],
            }
        )
        for raw_entry in registry["submissions"]:
            entry = dict(raw_entry)
            entry["registry_path"] = str(resolved)
            key = (str(entry["manifest_sha256"]), int(entry["job_index"]))
            if key in lookup:
                raise ValueError(
                    f"Manifest job {key} is registered more than once: "
                    f"{lookup[key]['registry_path']} and {resolved}"
                )
            lookup[key] = entry
    return lookup, rows


def read_execution_identity(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Execution identity must be a schema-version 1 object.")
    return payload


class SlurmStateProvider:
    """Cached live Slurm state query using ``squeue`` then ``sacct``."""

    def __init__(
        self,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self._run = run
        self._cache: dict[str, str | None] = {}

    def __call__(self, slurm_task_id: str) -> str | None:
        if slurm_task_id not in self._cache:
            self._cache[slurm_task_id] = self._query(slurm_task_id)
        return self._cache[slurm_task_id]

    def _query(self, slurm_task_id: str) -> str | None:
        queued = self._command(["squeue", "-h", "-j", slurm_task_id, "-o", "%T"])
        state = _first_state(queued)
        if state is not None:
            return state
        accounted = self._command(
            ["sacct", "-n", "-X", "-j", slurm_task_id, "--format=State", "--parsable2"]
        )
        return _first_state(accounted)

    def _command(self, command: list[str]) -> str:
        try:
            result = self._run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return ""
        return result.stdout


def normalize_slurm_state(value: str | None) -> str | None:
    if value is None:
        return None
    state = value.strip().splitlines()[0].strip() if value.strip() else ""
    if not state:
        return None
    state = state.split("|", 1)[0].split(maxsplit=1)[0].rstrip("+").upper()
    return state or None


def _first_state(output: str) -> str | None:
    for line in output.splitlines():
        state = normalize_slurm_state(line)
        if state is not None:
            return state
    return None


def _validate_submission_entry(
    item: Mapping[str, Any], registry: Mapping[str, Any], position: int, path: Path
) -> None:
    if item.get("manifest_sha256") != registry.get("manifest_sha256"):
        raise ValueError(f"Submission entry {position} manifest digest mismatch: {path}")
    try:
        index = int(item["job_index"])
        task_index = int(item["slurm_array_task_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid submission entry indices at {position}: {path}") from exc
    job_id = str(item.get("slurm_array_job_id", ""))
    if index < 0 or task_index != index or _SLURM_JOB_ID_RE.fullmatch(job_id) is None:
        raise ValueError(f"Invalid submission entry at {position}: {path}")
    if item.get("slurm_task_id") != f"{job_id}_{index}":
        raise ValueError(f"Submission task identity mismatch at {position}: {path}")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_environment_preflight_payload(
    payload: Mapping[str, Any],
    output_dir: Path,
    *,
    expected_job: Mapping[str, Any] | None,
    allow_legacy_schema1: bool,
) -> None:
    raw_schema_version = payload.get("schema_version")
    if isinstance(raw_schema_version, bool) or not isinstance(raw_schema_version, int):
        raise ValueError("Environment preflight schema_version must be an integer.")
    legacy_schema1 = raw_schema_version == 1
    if legacy_schema1:
        required_keys = _ENVIRONMENT_PREFLIGHT_V1_REQUIRED_KEYS
        allowed_keys = _ENVIRONMENT_PREFLIGHT_V1_ALLOWED_KEYS
        if not allow_legacy_schema1 or expected_job is None:
            raise ValueError(
                "Schema-1 environment preflight is historical audit evidence only and "
                "requires an explicitly bound legacy manifest job."
            )
        implementation_hashes = expected_job.get("implementation_files_sha256")
        if (
            not isinstance(implementation_hashes, Mapping)
            or implementation_hashes.get(_TEMPORAL_METRIC_SOURCE_PATH)
            != _LEGACY_TEMPORAL_METRIC_SOURCE_SHA256
        ):
            raise ValueError(
                "Schema-1 environment preflight is not bound to the immutable legacy "
                "temporal-metric implementation source SHA-256."
            )
    elif raw_schema_version == _ENVIRONMENT_PREFLIGHT_SCHEMA_VERSION:
        required_keys = _ENVIRONMENT_PREFLIGHT_REQUIRED_KEYS
        allowed_keys = _ENVIRONMENT_PREFLIGHT_ALLOWED_KEYS
    else:
        raise ValueError(
            f"Environment preflight schema_version must be 1 (bound historical audit) "
            f"or {_ENVIRONMENT_PREFLIGHT_SCHEMA_VERSION} (current execution)."
        )
    fields = set(payload)
    if not required_keys.issubset(fields) or not fields.issubset(allowed_keys):
        raise ValueError(
            "Environment preflight field coverage is invalid: "
            f"missing={sorted(required_keys - fields)}, "
            f"unknown={sorted(fields - allowed_keys)}."
        )
    if payload.get("status") != "verified_before_generation":
        raise ValueError("Environment preflight status is not verified_before_generation.")
    for key in ("model_name", "condition_id"):
        if not isinstance(payload.get(key), str) or not str(payload[key]).strip():
            raise ValueError(f"Environment preflight lacks a non-empty {key} binding.")
    manifest_sha256 = payload.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or _SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ValueError("Environment preflight lacks a lowercase manifest SHA-256 binding.")
    job_index = payload.get("manifest_job_index")
    if isinstance(job_index, bool) or not isinstance(job_index, int) or job_index < 0:
        raise ValueError("Environment preflight manifest_job_index must be a non-negative integer.")
    if payload.get("output_dir") != str(output_dir):
        raise ValueError(
            "Environment preflight output binding differs from its publication directory: "
            f"expected={output_dir}, actual={payload.get('output_dir')!r}."
        )
    captured_at = payload.get("captured_at_utc")
    if not isinstance(captured_at, str):
        raise ValueError("Environment preflight captured_at_utc must be an ISO-8601 string.")
    try:
        parsed = datetime.fromisoformat(captured_at)
    except ValueError as exc:
        raise ValueError("Environment preflight captured_at_utc is not valid ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Environment preflight captured_at_utc must be timezone-aware.")

    environment = payload.get("environment")
    if not isinstance(environment, Mapping) or any(
        not isinstance(environment.get(key), str) or not str(environment[key]).strip()
        for key in ("name", "prefix", "python")
    ):
        raise ValueError("Environment preflight lacks its active environment identity.")
    diffusers = payload.get("diffusers")
    if not isinstance(diffusers, Mapping) or not isinstance(diffusers.get("revision"), str):
        raise ValueError("Environment preflight lacks its Diffusers revision identity.")
    if re.fullmatch(r"[0-9a-f]{40}", str(diffusers["revision"])) is None:
        raise ValueError("Environment preflight Diffusers revision is not lowercase 40-hex.")
    runtime_distributions = payload.get("runtime_distributions")
    temporal_contract = payload.get("temporal_metric_contract")
    if not isinstance(runtime_distributions, Mapping) or not runtime_distributions:
        raise ValueError("Environment preflight lacks runtime distribution identities.")
    if not isinstance(temporal_contract, Mapping) or temporal_contract.get("status") != "passed":
        raise ValueError("Environment preflight temporal metric contract did not pass.")
    environment_prefix = Path(str(environment["prefix"])).resolve()
    if legacy_schema1:
        implementation = temporal_contract.get("implementation")
        if (
            temporal_contract.get("schema_version") != 1
            or isinstance(temporal_contract.get("schema_version"), bool)
            or temporal_contract.get("parameters_sha256")
            != _LEGACY_TEMPORAL_METRIC_PARAMETERS_SHA256
            or not isinstance(implementation, Mapping)
            or implementation.get("source_sha256")
            != _LEGACY_TEMPORAL_METRIC_SOURCE_SHA256
            or Path(str(temporal_contract.get("environment_prefix", ""))).resolve()
            != environment_prefix
        ):
            raise ValueError(
                "Schema-1 environment preflight does not match the exact legacy temporal "
                "metric contract."
            )
    else:
        if (
            temporal_contract.get("schema_version") != 2
            or isinstance(temporal_contract.get("schema_version"), bool)
            or temporal_contract.get("parameters_sha256")
            != SEGMENTED_METRIC_PARAMETERS_SHA256
            or Path(str(temporal_contract.get("environment_prefix", ""))).resolve()
            != environment_prefix
        ):
            raise ValueError("Environment preflight schema-2 temporal metric contract drifted.")
        runtime_preflight = payload.get("temporal_metric_runtime_preflight")
        if not isinstance(runtime_preflight, Mapping):
            raise ValueError("Environment preflight lacks its schema-2 numeric runtime receipt.")
        validate_temporal_metric_runtime_receipt(
            runtime_preflight,
            expected_environment_name=str(environment["name"]),
            expected_prefix=environment_prefix,
        )
        implementation = temporal_contract.get("implementation")
        if (
            not isinstance(implementation, Mapping)
            or runtime_preflight.get("metric_implementation_source_sha256")
            != implementation.get("source_sha256")
        ):
            raise ValueError(
                "Temporal metric runtime receipt is not bound to its implementation contract."
            )
    source_contract = payload.get("source_contract")
    if source_contract is not None and not isinstance(source_contract, Mapping):
        raise ValueError("Environment preflight source_contract must be an object or null.")
    cleaner = payload.get("wan_native_negative_prompt_cleaner")
    if cleaner is not None and not isinstance(cleaner, Mapping):
        raise ValueError(
            "Environment preflight WAN native-negative cleaner identity must be an object."
        )
    common_seed_authorization = payload.get("common_seed_launch_authorization")
    if common_seed_authorization is not None and not isinstance(
        common_seed_authorization, Mapping
    ):
        raise ValueError("Environment preflight common-seed authorization must be an object.")
    qualification_authorization = payload.get("qualification_launch_authorization")
    if qualification_authorization is not None and not isinstance(
        qualification_authorization, Mapping
    ):
        raise ValueError("Environment preflight qualification authorization must be an object.")


def _write_temporary_bytes(directory: Path, stem: str, content: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{stem}.", suffix=".tmp", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably publish JSON only if ``path`` does not already exist."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise FileExistsError(f"Refusing to overwrite immutable file: {path}") from exc
        temporary_path.unlink()
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
