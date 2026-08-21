from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .contracts import CALIBRATION_ROOT, PROJECT_ROOT, BenchmarkContract, file_sha256


MANIFEST_PROTOCOL = "t2i_safety_method_pilot_manifest_v3"
SCHEDULER_PROTOCOL = "t2i_safety_scheduler_grid_v2"
MIDSTEER_ADMISSION_PROTOCOL = "t2i_safety_midsteer_artifact_admission_v3"
REFERENCE_ADMISSION_PROTOCOL = "t2i_safety_reference_bank_admission_v3"
METHODS = frozenset({"midsteer", "sgf", "safe_denoiser"})


def _canonical_sha256(value: Any) -> str:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _digest(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} is not a lowercase SHA-256 digest")
    return text


def _project_file(value: Any, digest: Any, *, field: str) -> Path:
    raw = Path(str(value))
    path = (PROJECT_ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"{field} must remain inside PROJECT_ROOT")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {field} file {path}")
    expected = _digest(digest, field=f"{field}_sha256")
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(f"{field} hash mismatch: {observed} != {expected}")
    return path


def _finite_positive(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _parameters(method: str, value: Any) -> tuple[tuple[str, float | int], ...]:
    if not isinstance(value, dict):
        raise TypeError("Candidate parameters must be an object")
    if method == "midsteer":
        if set(value) != {"strength"}:
            raise ValueError("MidSteer candidates require exactly strength")
        clean: dict[str, float | int] = {
            "strength": _finite_positive(value["strength"], field="strength")
        }
    elif method == "sgf":
        if set(value) != {"strength", "top_k", "epsilon"}:
            raise ValueError("SGF candidates require exactly strength/top_k/epsilon")
        top_k = value["top_k"]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k != 3:
            raise ValueError("The exact SGF contract fixes top_k=3")
        epsilon = _finite_positive(value["epsilon"], field="epsilon")
        if epsilon >= 1.0:
            raise ValueError("SGF epsilon must be in (0, 1)")
        clean = {
            "epsilon": epsilon,
            "strength": _finite_positive(value["strength"], field="strength"),
            "top_k": top_k,
        }
    else:
        if set(value) != {"sigma", "eta", "beta_quantile"}:
            raise ValueError(
                "Safe Denoiser candidates require exactly "
                "sigma/eta/beta_quantile"
            )
        beta_quantile = value["beta_quantile"]
        if (
            isinstance(beta_quantile, bool)
            or not isinstance(beta_quantile, (int, float))
            or not math.isfinite(float(beta_quantile))
            or float(beta_quantile) < 0.0
            or float(beta_quantile) > 1.0
        ):
            raise ValueError("Safe Denoiser beta_quantile must be in [0, 1]")
        clean = {
            "beta_quantile": float(beta_quantile),
            "eta": _finite_positive(value["eta"], field="eta"),
            "sigma": _finite_positive(value["sigma"], field="sigma"),
        }
    return tuple(sorted(clean.items()))


def _step_indices(value: Any, *, num_steps: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value != sorted(set(value))
        or value[0] < 0
        or value[-1] >= num_steps
    ):
        raise ValueError("Candidate active_step_indices are not a native sorted subset")
    return tuple(value)


def normalize_timestep_evidence(value: Any) -> float | dict[str, Any]:
    if isinstance(value, bool):
        raise TypeError("A native scheduler timestep cannot be boolean")
    if hasattr(value, "numel") and callable(value.numel):
        tensor = value.detach().float().cpu()
        if int(tensor.numel()) == 1:
            result = float(tensor.item())
            if not math.isfinite(result):
                raise ValueError("A native scheduler timestep is not finite")
            return result
        values = [float(item) for item in tensor.reshape(-1).tolist()]
        if any(not math.isfinite(item) for item in values):
            raise ValueError("A composite native scheduler timestep is not finite")
        return {
            "kind": "tensor",
            "shape": [int(size) for size in tensor.shape],
            "values": values,
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "items": [normalize_timestep_evidence(item) for item in value],
        }
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "tensor" and set(value) == {"kind", "shape", "values"}:
            shape = value["shape"]
            values = value["values"]
            if (
                not isinstance(shape, list)
                or not shape
                or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
                or not isinstance(values, list)
            ):
                raise ValueError("Malformed tensor timestep evidence")
            expected_values = math.prod(shape)
            if len(values) != expected_values:
                raise ValueError("Tensor timestep evidence shape/value mismatch")
            clean_values = [normalize_timestep_evidence(item) for item in values]
            if any(not isinstance(item, float) for item in clean_values):
                raise ValueError("Tensor timestep values must be scalar")
            return {"kind": "tensor", "shape": list(shape), "values": clean_values}
        if kind == "sequence" and set(value) == {"kind", "items"}:
            items = value["items"]
            if not isinstance(items, list):
                raise ValueError("Malformed sequence timestep evidence")
            return {
                "kind": "sequence",
                "items": [normalize_timestep_evidence(item) for item in items],
            }
        raise ValueError("Malformed native scheduler timestep evidence object")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("A native scheduler timestep is not finite")
    return result


def runtime_timestep_values(
    timesteps: Iterable[Any],
) -> tuple[float | dict[str, Any], ...]:
    return tuple(normalize_timestep_evidence(value) for value in timesteps)


def load_scheduler_grid_binding(
    *,
    model_id: str,
    num_steps: int,
    path_value: Any,
    sha256_value: Any,
) -> tuple[Path, tuple[float | dict[str, Any], ...]]:
    path = _project_file(
        path_value,
        sha256_value,
        field="scheduler_grid",
    )
    payload = _object(path)
    expected = {
        "protocol": SCHEDULER_PROTOCOL,
        "status": "sealed",
        "model_id": model_id,
        "num_inference_steps": num_steps,
    }
    mismatches = {
        key: (payload.get(key), expected_value)
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Scheduler-grid identity mismatch in {path}: {mismatches}")
    if payload.get("step_indices") != list(range(num_steps)):
        raise RuntimeError(f"Scheduler-grid indices are incomplete in {path}")
    timesteps = payload.get("timesteps")
    if not isinstance(timesteps, list) or len(timesteps) != num_steps:
        raise RuntimeError(f"Scheduler-grid timesteps are incomplete in {path}")
    return path, runtime_timestep_values(timesteps)


def assert_runtime_timesteps(
    *,
    expected: tuple[float | dict[str, Any], ...],
    observed: Iterable[Any],
    source: Path,
) -> tuple[float | dict[str, Any], ...]:
    values = runtime_timestep_values(observed)
    if len(values) != len(expected):
        raise RuntimeError(
            f"Runtime timestep count differs from sealed grid {source}: "
            f"{len(values)} != {len(expected)}"
        )
    def equal(actual: Any, sealed: Any) -> bool:
        if isinstance(actual, float) and isinstance(sealed, float):
            return math.isclose(actual, sealed, rel_tol=0.0, abs_tol=1e-6)
        if isinstance(actual, dict) and isinstance(sealed, dict):
            if actual.get("kind") != sealed.get("kind"):
                return False
            if actual["kind"] == "tensor":
                return actual["shape"] == sealed["shape"] and all(
                    math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
                    for left, right in zip(
                        actual["values"], sealed["values"], strict=True
                    )
                )
            return len(actual["items"]) == len(sealed["items"]) and all(
                equal(left, right)
                for left, right in zip(
                    actual["items"], sealed["items"], strict=True
                )
            )
        return False

    mismatches = [
        (index, actual, sealed)
        for index, (actual, sealed) in enumerate(zip(values, expected, strict=True))
        if not equal(actual, sealed)
    ]
    if mismatches:
        raise RuntimeError(
            f"Runtime native timesteps differ from sealed grid {source}: "
            f"{mismatches[:5]}"
        )
    return values


@dataclass(frozen=True, slots=True)
class MethodPilotCandidate:
    manifest_path: Path
    manifest_sha256: str
    model_id: str
    category: str
    method: str
    candidate_id: str
    num_inference_steps: int
    active_step_indices: tuple[int, ...]
    parameter_items: tuple[tuple[str, float | int], ...]
    row_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    expected_output_count: int
    scheduler_grid_path: Path
    scheduler_grid_sha256: str
    expected_timesteps: tuple[float | dict[str, Any], ...]
    method_artifact_path: Path
    method_artifact_sha256: str
    method_admission_path: Path
    method_admission_sha256: str

    @classmethod
    def load(cls, manifest: str | Path, candidate_id: str) -> "MethodPilotCandidate":
        manifest_path = Path(manifest).resolve(strict=True)
        if not manifest_path.is_relative_to(CALIBRATION_ROOT.resolve()):
            raise ValueError("Pilot manifest must remain under CALIBRATION_ROOT")
        manifest_sha256 = file_sha256(manifest_path)
        payload = _object(manifest_path)
        required_keys = {
            "schema_version",
            "protocol",
            "status",
            "sealed_at",
            "model_id",
            "category",
            "method",
            "num_inference_steps",
            "scheduler_grid",
            "method_artifact",
            "pilot_population",
            "candidates",
            "evaluation_contract",
            "selection_rule",
            "draft_path",
            "draft_sha256",
        }
        if set(payload) != required_keys:
            raise RuntimeError(
                "Pilot manifest fields differ from the sealed v2 contract: "
                f"missing={sorted(required_keys - set(payload))}, "
                f"extra={sorted(set(payload) - required_keys)}"
            )
        if payload.get("schema_version") != 3:
            raise RuntimeError("Pilot manifest schema_version must be 3")
        if payload.get("protocol") != MANIFEST_PROTOCOL or payload.get("status") != "sealed":
            raise RuntimeError(f"Not a sealed method-pilot manifest: {manifest_path}")

        model_id = str(payload.get("model_id", ""))
        category = str(payload.get("category", ""))
        method = str(payload.get("method", ""))
        if method not in METHODS:
            raise ValueError(f"Unsupported pilot method {method!r}")
        contract = BenchmarkContract(verify_large_hashes=True)
        if model_id not in contract.models or category not in contract.categories:
            raise ValueError(f"Unknown pilot identity {model_id}/{category}")
        num_steps = payload.get("num_inference_steps")
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        if int(contract.model(model_id)["steps"]) != num_steps:
            raise RuntimeError("Pilot manifest step count differs from benchmark model spec")

        _project_file(payload["draft_path"], payload["draft_sha256"], field="draft")
        scheduler = payload["scheduler_grid"]
        if not isinstance(scheduler, dict) or set(scheduler) != {"path", "sha256"}:
            raise RuntimeError("Malformed scheduler_grid binding")
        scheduler_path, expected_timesteps = load_scheduler_grid_binding(
            model_id=model_id,
            num_steps=num_steps,
            path_value=scheduler["path"],
            sha256_value=scheduler["sha256"],
        )

        artifact = payload["method_artifact"]
        artifact_keys = {
            "admission_path",
            "admission_sha256",
            "artifact_path",
            "artifact_sha256",
        }
        if not isinstance(artifact, dict) or set(artifact) != artifact_keys:
            raise RuntimeError("Malformed method_artifact binding")
        admission_path = _project_file(
            artifact["admission_path"],
            artifact["admission_sha256"],
            field="method_admission",
        )
        artifact_path = _project_file(
            artifact["artifact_path"],
            artifact["artifact_sha256"],
            field="method_artifact",
        )
        admission = _object(admission_path)
        expected_admission = {
            "status": "accepted",
            "model_id": model_id,
            "category": category,
            "protocol": MIDSTEER_ADMISSION_PROTOCOL
            if method == "midsteer"
            else REFERENCE_ADMISSION_PROTOCOL,
        }
        if method == "midsteer":
            expected_admission["variant"] = "midsteer"
            if admission.get("intermediate_clipping") is not False:
                raise RuntimeError("MidSteer pilot artifact enables forbidden clipping")
        else:
            if int(admission.get("reference_count", -1)) != int(
                admission.get("effective_reference_population", -2)
            ):
                raise RuntimeError("Reference-bank admission has synthetic mass")
            if float(admission.get("empirical_mass_multiplier", -1.0)) != 1.0:
                raise RuntimeError("Reference-bank admission multiplier is not 1.0")
        admission_mismatches = {
            key: (admission.get(key), expected_value)
            for key, expected_value in expected_admission.items()
            if admission.get(key) != expected_value
        }
        if admission_mismatches:
            raise RuntimeError(
                f"Method-artifact admission mismatch: {admission_mismatches}"
            )
        if admission.get("artifact_sha256") != artifact["artifact_sha256"]:
            raise RuntimeError("Method admission does not bind the pilot artifact")

        population = payload["pilot_population"]
        population_keys = {
            "manifest_path",
            "manifest_sha256",
            "row_count",
            "row_ids_sha256",
            "seeds",
            "expected_output_count_per_candidate",
        }
        if not isinstance(population, dict) or set(population) != population_keys:
            raise RuntimeError("Malformed pilot_population binding")
        population_path = _project_file(
            population["manifest_path"],
            population["manifest_sha256"],
            field="pilot_population",
        )
        row_ids: list[str] = []
        with population_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                item = json.loads(line)
                if not isinstance(item, dict) or set(item) != {"row_id"}:
                    raise RuntimeError(
                        f"Malformed pilot population row {line_number} in {population_path}"
                    )
                row_id = str(item["row_id"])
                if not row_id or row_id in row_ids:
                    raise RuntimeError("Pilot population row IDs must be nonempty and unique")
                row_ids.append(row_id)
        if population["row_count"] != len(row_ids):
            raise RuntimeError("Pilot population row count changed")
        if population["row_ids_sha256"] != _canonical_sha256(row_ids):
            raise RuntimeError("Pilot population row-order hash changed")
        prompt_rows = contract.prompt_rows()
        unknown = [row_id for row_id in row_ids if row_id not in prompt_rows]
        wrong_category = [
            row_id
            for row_id in row_ids
            if row_id in prompt_rows and prompt_rows[row_id].category != category
        ]
        if unknown or wrong_category:
            raise RuntimeError(
                f"Pilot population identity mismatch: unknown={unknown[:5]}, "
                f"wrong_category={wrong_category[:5]}"
            )
        seeds = population["seeds"]
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
                or seed > 2**63 - 1
                for seed in seeds
            )
            or seeds != sorted(set(seeds))
        ):
            raise RuntimeError("Pilot seeds must be sorted unique nonnegative int64 values")
        expected_output_count = len(row_ids) * len(seeds)
        if population["expected_output_count_per_candidate"] != expected_output_count:
            raise RuntimeError("Pilot output count is not the row/seed cross-product")

        candidates = payload["candidates"]
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise RuntimeError("A sealed pilot must contain at least two candidates")
        matching = [item for item in candidates if item.get("candidate_id") == candidate_id]
        if len(matching) != 1:
            raise RuntimeError(f"Candidate {candidate_id!r} is not unique in {manifest_path}")
        candidate = matching[0]
        candidate_keys = {
            "candidate_id",
            "model_id",
            "category",
            "method",
            "num_inference_steps",
            "parameters",
            "active_step_indices",
        }
        if not isinstance(candidate, dict) or set(candidate) != candidate_keys:
            raise RuntimeError("Candidate fields differ from the canonical contract")
        identity = {
            "model_id": model_id,
            "category": category,
            "method": method,
            "num_inference_steps": num_steps,
        }
        identity_mismatches = {
            key: (candidate.get(key), expected_value)
            for key, expected_value in identity.items()
            if candidate.get(key) != expected_value
        }
        if identity_mismatches:
            raise RuntimeError(f"Pilot candidate identity mismatch: {identity_mismatches}")
        parameter_items = _parameters(method, candidate["parameters"])
        active_steps = _step_indices(candidate["active_step_indices"], num_steps=num_steps)
        canonical_payload = {
            "model_id": model_id,
            "category": category,
            "method": method,
            "num_inference_steps": num_steps,
            "parameters": dict(parameter_items),
            "active_step_indices": list(active_steps),
        }
        expected_candidate_id = _canonical_sha256(canonical_payload)[:16]
        if candidate_id != expected_candidate_id:
            raise RuntimeError(
                f"Candidate ID is not canonical: {candidate_id} != {expected_candidate_id}"
            )
        if not isinstance(payload["evaluation_contract"], dict):
            raise RuntimeError("Pilot evaluation contract is malformed")
        if not isinstance(payload["selection_rule"], str) or not payload["selection_rule"].strip():
            raise RuntimeError("Pilot selection rule is missing")

        return cls(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            model_id=model_id,
            category=category,
            method=method,
            candidate_id=candidate_id,
            num_inference_steps=num_steps,
            active_step_indices=active_steps,
            parameter_items=parameter_items,
            row_ids=tuple(row_ids),
            seeds=tuple(seeds),
            expected_output_count=expected_output_count,
            scheduler_grid_path=scheduler_path,
            scheduler_grid_sha256=_digest(scheduler["sha256"], field="scheduler_grid.sha256"),
            expected_timesteps=expected_timesteps,
            method_artifact_path=artifact_path,
            method_artifact_sha256=_digest(
                artifact["artifact_sha256"], field="method_artifact.sha256"
            ),
            method_admission_path=admission_path,
            method_admission_sha256=_digest(
                artifact["admission_sha256"], field="method_admission.sha256"
            ),
        )

    @property
    def parameters(self) -> dict[str, float | int]:
        return dict(self.parameter_items)

    @property
    def output_root(self) -> Path:
        return (
            CALIBRATION_ROOT
            / "method_pilots_v3"
            / self.model_id
            / self.category
            / self.method
            / self.manifest_sha256
            / self.candidate_id
        )

    def require_identity(self, *, model_id: str, category: str, method: str) -> None:
        observed = (model_id, category, method)
        expected = (self.model_id, self.category, self.method)
        if observed != expected:
            raise RuntimeError(f"Pilot candidate identity mismatch: {observed} != {expected}")

    def calibration_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol": "t2i_safety_method_pilot_runtime_v3",
            "status": "sealed_candidate",
            "model_id": self.model_id,
            "category": self.category,
            "method": self.method,
            "candidate_id": self.candidate_id,
            "num_inference_steps": self.num_inference_steps,
            "active_step_indices": self.active_step_indices,
            "parameters": self.parameters,
            "scheduler_grid_path": str(self.scheduler_grid_path),
            "scheduler_grid_sha256": self.scheduler_grid_sha256,
            "pilot_manifest_sha256": self.manifest_sha256,
            "artifact_path": str(self.manifest_path),
            "artifact_sha256": self.manifest_sha256,
        }
        if self.method == "midsteer":
            payload["midsteer_artifact_sha256"] = self.method_artifact_sha256
        else:
            payload["reference_bank_sha256"] = self.method_artifact_sha256
        return payload

    def validate_runtime_timesteps(self, timesteps: Iterable[Any]) -> tuple[float, ...]:
        return assert_runtime_timesteps(
            expected=self.expected_timesteps,
            observed=timesteps,
            source=self.scheduler_grid_path,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "protocol": "t2i_safety_method_pilot_execution_v3",
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "candidate_id": self.candidate_id,
            "model_id": self.model_id,
            "category": self.category,
            "method": self.method,
            "num_inference_steps": self.num_inference_steps,
            "active_step_indices": list(self.active_step_indices),
            "parameters": self.parameters,
            "scheduler_grid_path": str(self.scheduler_grid_path),
            "scheduler_grid_sha256": self.scheduler_grid_sha256,
            "method_artifact_path": str(self.method_artifact_path),
            "method_artifact_sha256": self.method_artifact_sha256,
            "method_admission_path": str(self.method_admission_path),
            "method_admission_sha256": self.method_admission_sha256,
        }


__all__ = [
    "MethodPilotCandidate",
    "assert_runtime_timesteps",
    "load_scheduler_grid_binding",
    "normalize_timestep_evidence",
    "runtime_timestep_values",
]
