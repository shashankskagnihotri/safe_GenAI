#!/usr/bin/env python
"""Create provenance-bound plots for the finer-detailing study.

With no fresh-bundle flag this preserves the historical Phase-A v3 execution and
machine-diagnostic reproducer.  The opt-in fresh path is fail-closed over the exact
396-row qualification, 288-row target-blind ladder, 36 seed selections, and 504-row
selected-seed final campaign.  It keeps structural/media, objective temporal,
human semantic/preservation/gender, runtime, ablation, and Shapley evidence in
separate ledgers; it never infers a semantic outcome from completion or diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


STATUS_ORDER = ["completed", "not_supported", "failed", "incomplete", "pending", "running"]
STATUS_COLORS = {
    "completed": "#2E8B57",
    "not_supported": "#64748B",
    "failed": "#C2413B",
    "incomplete": "#D97706",
    "pending": "#4F46E5",
    "running": "#0891B2",
}

# The new evidence path deliberately freezes the study axes locally.  Plotting must
# fail if a future input silently adds/drops a model, prompt, condition, or concept
# pair; it must never normalize an incomplete campaign into a plausible-looking rate.
FRESH_EVIDENCE_CONTRACT = "finer_detailing_plot_evidence_bundle_v1"
FRESH_PUBLICATION_CONTRACT = "finer_detailing_fresh_plot_bundle_publication_v1"
FRESH_PUBLICATION_MANIFEST_COUNTS = {
    "qualification": 8,
    "seed_ladder": 8,
    "final": 72,
}
FRESH_PUBLICATION_REGISTRY_COUNT = 88
FRESH_PUBLICATION_BUNDLE_KEYS = {
    "schema_version",
    "contract",
    "publication_contract",
    "cohorts",
    "selection_cohort_commit",
    "seed_selections",
    "submission_registries",
    "raw_ledger_sources",
    "evidence_ledgers",
    "document_sha256",
}
LEDGER_CONTRACTS = {
    "objective_temporal": "finer_detailing_objective_temporal_ledger_v1",
    "manual_semantic": "finer_detailing_manual_semantic_ledger_v1",
    "shapley_diagnostics": "finer_detailing_shapley_diagnostics_ledger_v1",
}
PROMPT_IDS = (
    "01_sad_young_girl",
    "02_angry_old_man",
    "03_empty_outdoor_mall",
)
IMAGE_MODELS = (
    "cosmos3_super_text2image",
    "flux1_dev",
    "flux2_dev",
    "ideogram4_nf4",
    "qwen_image",
    "qwen_image_2512",
    "sd35_large",
)
VIDEO_MODELS = (
    "cogvideox_5b",
    "hunyuan_video",
    "joyai_echo",
    "ltx_23",
    "wan22_t2v_a14b",
)
MODEL_NAMES = (*IMAGE_MODELS, *VIDEO_MODELS)
NATIVE_NEGATIVE_UNSUPPORTED_MODELS = {
    "cosmos3_super_text2image",
    "flux2_dev",
    "ideogram4_nf4",
    "joyai_echo",
    "ltx_23",
}
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
QUALIFICATION_PAIR_BY_PROMPT = {
    "01_sad_young_girl": "facial_affect_negative_to_happy",
    "02_angry_old_man": "facial_affect_negative_to_happy",
    "03_empty_outdoor_mall": "vertical_circulation_escalators_to_marble_stairs",
}
EXPECTED_COHORT_COUNTS = {
    "qualification": {"logical": 396, "media": 369, "unsupported": 27},
    "seed_ladder": {"logical": 288, "media": 288, "unsupported": 0},
    "final": {"logical": 504, "media": 489, "unsupported": 15},
}
FRESH_DEFAULT_OUTPUT_FORBIDDEN = (
    "debugging/plots/finer_detailing_correction/phase_a_terminal_20260720_v3"
)
PLOTTING_PYTHON_VERSION = "3.12.9"
PLOTTING_DEPENDENCY_VERSIONS = {
    "contourpy": "1.3.3",
    "cycler": "0.12.1",
    "fonttools": "4.62.1",
    "kiwisolver": "1.5.0",
    "matplotlib": "3.10.9",
    "numpy": "2.0.1",
    "packaging": "24.2",
    "pandas": "2.2.3",
    "pillow": "11.1.0",
    "pyparsing": "3.3.2",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2025.2",
    "seaborn": "0.13.2",
    "six": "1.17.0",
    "tzdata": "2025.2",
}
DEFAULT_FULL_VIDEO_DIRS = (
    "debugging/audits/finer_detailing_20260718/qualification_completed_video",
    "debugging/audits/finer_detailing_20260719/qualification_completed_video",
    "debugging/audits/finer_detailing_20260719/qualification_completed_video_b",
    "debugging/audits/finer_detailing_20260719/qualification_completed_video_c",
    "debugging/audits/finer_detailing_20260720/qualification_completed_video_terminal",
)
DEFAULT_MOTION_DIRS = (
    "debugging/audits/finer_detailing_20260718/qualification_prompt3_motion",
    "debugging/audits/finer_detailing_20260719/qualification_prompt3_motion",
    "debugging/audits/finer_detailing_20260719/qualification_prompt3_motion_b",
    "debugging/audits/finer_detailing_20260719/qualification_prompt3_motion_c",
    "debugging/audits/finer_detailing_20260720/qualification_prompt3_motion_terminal",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plotting_environment_provenance() -> dict[str, Any]:
    """Require and record the exact publication-plotting dependency closure."""

    requirements_path = Path(__file__).resolve().parents[1] / "requirements-plotting.txt"
    if not requirements_path.is_file():
        raise FileNotFoundError(
            f"Dedicated plotting requirements file is missing: {requirements_path}"
        )
    declared: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        requirements_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError(
                f"Plotting requirement line {line_number} is not an exact pin: {raw_line!r}."
            )
        name, version = line.split("==", maxsplit=1)
        normalized_name = name.strip().casefold().replace("_", "-")
        if not normalized_name or not version.strip() or normalized_name in declared:
            raise ValueError(
                f"Plotting requirement line {line_number} is empty or duplicated: {raw_line!r}."
            )
        declared[normalized_name] = version.strip()
    if declared != PLOTTING_DEPENDENCY_VERSIONS:
        raise RuntimeError(
            "requirements-plotting.txt differs from the plotter's frozen dependency contract: "
            f"declared={declared}, expected={PLOTTING_DEPENDENCY_VERSIONS}."
        )
    observed = {
        distribution: importlib_metadata.version(distribution)
        for distribution in PLOTTING_DEPENDENCY_VERSIONS
    }
    mismatches = {
        distribution: {
            "expected": PLOTTING_DEPENDENCY_VERSIONS[distribution],
            "observed": observed[distribution],
        }
        for distribution in PLOTTING_DEPENDENCY_VERSIONS
        if observed[distribution] != PLOTTING_DEPENDENCY_VERSIONS[distribution]
    }
    if mismatches:
        raise RuntimeError(
            "Plotting dependency versions differ from requirements-plotting.txt: "
            f"{mismatches}."
        )
    python_version = platform.python_version()
    if python_version != PLOTTING_PYTHON_VERSION:
        raise RuntimeError(
            "Publication plotting requires the verified Python version: "
            f"expected={PLOTTING_PYTHON_VERSION}, observed={python_version}."
        )
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": python_version,
        "matplotlib_backend": str(matplotlib.get_backend()),
        "requirements_path": str(requirements_path),
        "requirements_sha256": _sha256(requirements_path),
        "required_versions": dict(PLOTTING_DEPENDENCY_VERSIONS),
        "observed_versions": observed,
        "exact_match_verified": True,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _condition_kind(row: Mapping[str, Any]) -> str:
    variation = str(row.get("variation", ""))
    if variation == "01_baseline":
        return "baseline"
    if variation == "02_negative_prompt":
        return "negative"
    if variation == "03_concept_steering":
        return "concept_full"
    if variation == "04_shapley_concept_steering":
        return "shapley_full"
    if variation == "05_concept_steering_single_pair":
        return "concept_single"
    if variation == "06_shapley_concept_steering_single_pair":
        return "shapley_single_v1"
    return variation or "unknown"


def load_latest_jobs(audit_path: Path) -> pd.DataFrame:
    payload = _read_json(audit_path)
    jobs = payload.get("latest_jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"Audit {audit_path} has no latest_jobs list.")
    frame = pd.DataFrame(jobs)
    required = {"condition_id", "model_name", "prompt_id", "status", "task", "variation"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Audit {audit_path} lacks required columns: {missing}.")
    if frame["condition_id"].duplicated().any():
        duplicate = frame.loc[frame["condition_id"].duplicated(), "condition_id"].tolist()
        raise ValueError(f"Audit contains duplicate latest condition IDs: {duplicate[:5]}.")
    frame["condition_kind"] = frame.apply(_condition_kind, axis=1)
    return frame


def attach_timings(jobs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in jobs.to_dict(orient="records"):
        output_dir = row.get("output_dir")
        timing_path = Path(str(output_dir)) / "experiment_timing.json" if output_dir else None
        if timing_path is None or not timing_path.is_file():
            continue
        timing = _read_json(timing_path)
        wall_seconds = timing.get("wall_seconds")
        if not isinstance(wall_seconds, (int, float)) or isinstance(wall_seconds, bool):
            raise ValueError(f"Timing file {timing_path} has invalid wall_seconds.")
        records.append(
            {
                "condition_id": row["condition_id"],
                "prompt_id": row["prompt_id"],
                "model_name": row["model_name"],
                "task": row["task"],
                "status": row["status"],
                "condition_kind": row["condition_kind"],
                "seed": row.get("seed"),
                "wall_seconds": float(wall_seconds),
                "timing_status": timing.get("status"),
                "timing_path": str(timing_path.resolve()),
                "timing_sha256": _sha256(timing_path),
            }
        )
    return pd.DataFrame(records)


def _collect_unique_documents(
    directories: Iterable[Path],
    pattern: str,
) -> list[tuple[Path, Mapping[str, Any]]]:
    by_condition: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"Required evidence directory does not exist: {directory}")
        for path in sorted(directory.glob(pattern)):
            payload = _read_json(path)
            condition = payload.get("condition")
            condition_id = condition.get("condition_id") if isinstance(condition, Mapping) else None
            if not isinstance(condition_id, str) or not condition_id:
                raise ValueError(f"Evidence document {path} lacks condition.condition_id.")
            if condition_id in by_condition:
                previous = by_condition[condition_id][0]
                raise ValueError(
                    f"Evidence roots overlap for {condition_id}: {previous} and {path}."
                )
            by_condition[condition_id] = (path, payload)
    return [by_condition[key] for key in sorted(by_condition)]


def load_full_video_evidence(directories: Iterable[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path, payload in _collect_unique_documents(directories, "*__full_video.json"):
        condition = payload["condition"]
        terminal = payload.get("terminal_window_diagnostics") or {}
        terminal_freeze = terminal.get("freeze_diagnostics") or {}
        records.append(
            {
                "condition_id": condition["condition_id"],
                "prompt_id": condition["prompt_id"],
                "model_name": condition["model_name"],
                "seed": condition["seed"],
                "condition_kind": _condition_kind(condition),
                "evidence_contract_pass": bool(payload.get("evidence_contract_pass")),
                "structural_contract_pass": bool(payload.get("structural_contract_pass")),
                "frozen_candidate": bool(
                    (payload.get("freeze_diagnostics") or {}).get("frozen_candidate")
                ),
                "terminal_frozen_candidate": bool(terminal_freeze.get("frozen_candidate")),
                "near_periodic_candidate": bool(
                    (payload.get("long_lag_repetition_diagnostics") or {}).get(
                        "near_periodic_candidate"
                    )
                ),
                "cut_candidate": int(
                    (payload.get("cut_diagnostics") or {}).get("candidate_count", 0)
                )
                > 0,
                "cadence_candidate": bool(
                    (payload.get("cadence_diagnostics") or {}).get(
                        "alternating_cadence_anomaly_candidate"
                    )
                ),
                "evidence_path": str(path.resolve()),
                "evidence_sha256": _sha256(path),
            }
        )
    return pd.DataFrame(records)


def load_motion_evidence(directories: Iterable[Path]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path, payload in _collect_unique_documents(directories, "*__prompt3_motion.json"):
        condition = payload["condition"]
        assessment = payload.get("condition_assessment") or {}
        global_motion = payload.get("global_background_motion") or {}
        records.append(
            {
                "condition_id": condition["condition_id"],
                "model_name": condition["model_name"],
                "seed": condition["seed"],
                "condition_kind": _condition_kind(condition),
                "motion_requirement_status": assessment.get("motion_requirement_status"),
                "observed_motion_evidence": assessment.get("observed_motion_evidence"),
                "image_plane_path_length_px": float(
                    global_motion.get("total_image_plane_path_length_px", np.nan)
                ),
                "physical_camera_motion_proven": bool(
                    global_motion.get("physical_camera_motion_proven", False)
                ),
                "evidence_path": str(path.resolve()),
                "evidence_sha256": _sha256(path),
            }
        )
    return pd.DataFrame(records)


@dataclass(frozen=True)
class FreshPlotData:
    """Fully reconciled fresh-study evidence; no field is inferred from scheduler state."""

    rows: pd.DataFrame
    selections: pd.DataFrame
    objective_temporal: pd.DataFrame
    manual_semantic: pd.DataFrame
    manual_pairs: pd.DataFrame
    shapley_diagnostics: pd.DataFrame
    provenance: Mapping[str, Any]

    def snapshot(self) -> dict[str, Any]:
        """Return deterministic plot-data facts suitable for regression snapshots."""

        cohort_counts: dict[str, dict[str, int]] = {}
        for cohort in EXPECTED_COHORT_COUNTS:
            subset = self.rows[self.rows["cohort"] == cohort]
            cohort_counts[cohort] = {
                "logical": int(len(subset)),
                "media": int(subset["media_valid"].sum()),
                "unsupported": int((subset["outcome"] == "truthful_unsupported").sum()),
            }
        final = self.rows[self.rows["cohort"] == "final"]
        exact_one = final[final["ablation"] == "exact_one"]
        return {
            "cohorts": cohort_counts,
            "seed_selection_count": int(len(self.selections)),
            "selected_seed_rows": int(
                self.selections[["prompt_id", "model_name"]].drop_duplicates().shape[0]
            ),
            "final_model_prompt_rows": int(
                final[["prompt_id", "model_name"]].drop_duplicates().shape[0]
            ),
            "final_exact_one_media": int(exact_one["media_valid"].sum()),
            "final_ordinary_exact_one_media": int(
                ((exact_one["method"] == "conceptsteer") & exact_one["media_valid"]).sum()
            ),
            "final_shapley_exact_one_media": int(
                ((exact_one["method"] == "shapley") & exact_one["media_valid"]).sum()
            ),
            "objective_temporal_rows": int(len(self.objective_temporal)),
            "manual_semantic_rows": int(len(self.manual_semantic)),
            "manual_pair_rows": int(len(self.manual_pairs)),
            "shapley_diagnostic_rows": int(len(self.shapley_diagnostics)),
        }


def _canonical_digest(payload: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest: {digest!r}.")
    return digest


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be Boolean; got {value!r}.")
    return value


def _require_number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be numeric; got {value!r}.")
    number = float(value)
    if not np.isfinite(number) or number < minimum:
        raise ValueError(f"{label} must be finite and >= {minimum}; got {number!r}.")
    return number


def _resolve_input_path(value: Any, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    lexical = Path(os.path.abspath(path))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Fresh evidence path traverses a forbidden symlink: {cursor}.")
    return lexical.resolve()


def _validate_digest_sidecar(path: Path, digest: str) -> Path:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise FileNotFoundError(f"Immutable document sidecar is missing: {sidecar}")
    expected = f"{digest}  {path.name}"
    observed = sidecar.read_text(encoding="utf-8").strip()
    if observed != expected:
        raise ValueError(
            f"Immutable sidecar mismatch for {path}: expected={expected!r}, observed={observed!r}."
        )
    return sidecar


def _validate_raw_digest_sidecar(
    path: Path, binding: Mapping[str, Any], root: Path, label: str
) -> Path:
    raw_sidecar = _resolve_input_path(binding.get("raw_sidecar_path"), root)
    expected_path = path.with_suffix(path.suffix + ".raw.sha256")
    if raw_sidecar != expected_path or not raw_sidecar.is_file() or raw_sidecar.is_symlink():
        raise ValueError(f"{label} raw-file digest sidecar is missing/stale: {raw_sidecar}.")
    file_sha = _require_sha256(binding.get("file_sha256"), f"{label}.file_sha256")
    if raw_sidecar.read_text(encoding="utf-8").split() != [file_sha, path.name]:
        raise ValueError(f"{label} raw-file digest sidecar does not authenticate {path}.")
    declared_sidecar_sha = _require_sha256(
        binding.get("raw_sidecar_sha256"), f"{label}.raw_sidecar_sha256"
    )
    if _sha256(raw_sidecar) != declared_sidecar_sha:
        raise ValueError(f"{label} raw-file digest sidecar changed: {raw_sidecar}.")
    return raw_sidecar


def _validate_production_publication_tree(bundle_path: Path) -> None:
    """Reject every partial, aliased, or writable commit-last publication."""

    publication_root = bundle_path.parent
    if bundle_path.name != "fresh_bundle.json" or publication_root.is_symlink():
        raise ValueError("Production fresh evidence must use its canonical commit path.")
    expected_directories = {Path("ledgers")}
    expected_files = {
        Path("fresh_bundle.json"),
        Path("fresh_bundle.json.sha256"),
        Path("fresh_bundle.json.raw.sha256"),
    }
    for name in LEDGER_CONTRACTS:
        relative = Path("ledgers") / f"{name}.json"
        expected_files.update(
            {
                relative,
                relative.with_suffix(relative.suffix + ".sha256"),
                relative.with_suffix(relative.suffix + ".raw.sha256"),
            }
        )
    observed_directories: set[Path] = set()
    observed_files: set[Path] = set()
    for member in publication_root.rglob("*"):
        relative = member.relative_to(publication_root)
        observed = member.lstat()
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"Production fresh evidence contains a symlink: {member}.")
        if stat.S_ISDIR(observed.st_mode):
            observed_directories.add(relative)
        elif stat.S_ISREG(observed.st_mode):
            observed_files.add(relative)
        else:
            raise ValueError(
                f"Production fresh evidence contains a non-regular member: {member}."
            )
        if stat.S_IMODE(observed.st_mode) & 0o222:
            raise ValueError(f"Production fresh evidence contains a writable member: {member}.")
    root_mode = publication_root.lstat().st_mode
    if not stat.S_ISDIR(root_mode) or stat.S_IMODE(root_mode) & 0o222:
        raise ValueError("Production fresh evidence is an uncommitted writable claim.")
    if (
        observed_directories != expected_directories
        or observed_files != expected_files
    ):
        raise ValueError(
            "Production fresh evidence tree is partial or has unexpected members: "
            f"directories={sorted(map(str, observed_directories))}, "
            f"files={sorted(map(str, observed_files))}."
        )


def _load_publication_registries(
    bindings: Any, root: Path
) -> tuple[
    dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]],
    list[dict[str, Any]],
]:
    if not isinstance(bindings, list) or len(bindings) != FRESH_PUBLICATION_REGISTRY_COUNT:
        raise ValueError(
            "Production fresh bundle must bind exactly 88 immutable submission registries."
        )
    expected_keys = {
        "path",
        "file_sha256",
        "registry_sha256",
        "manifest_path",
        "manifest_sha256",
        "slurm_array_job_id",
        "num_registered_tasks",
    }
    lookup: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    provenance: list[dict[str, Any]] = []
    paths: set[Path] = set()
    manifest_digests: set[str] = set()
    for index, raw_binding in enumerate(bindings):
        if not isinstance(raw_binding, Mapping) or set(raw_binding) != expected_keys:
            raise ValueError(f"Submission registry binding {index} has the wrong fields.")
        binding = dict(raw_binding)
        path = _resolve_input_path(binding["path"], root)
        if path in paths or path.is_symlink() or not path.is_file():
            raise ValueError(f"Submission registry path is missing/duplicated/symlinked: {path}.")
        paths.add(path)
        file_sha = _require_sha256(binding["file_sha256"], "registry file_sha256")
        if _sha256(path) != file_sha:
            raise ValueError(f"Submission registry raw bytes changed: {path}.")
        registry = _read_json(path)
        if not isinstance(registry, dict) or registry.get("schema_version") != 1:
            raise ValueError(f"Submission registry schema is invalid: {path}.")
        registry_sha = _require_sha256(
            binding["registry_sha256"], "registry registry_sha256"
        )
        if (
            registry.get("registry_sha256") != registry_sha
            or _canonical_digest(registry, "registry_sha256") != registry_sha
        ):
            raise ValueError(f"Submission registry canonical digest changed: {path}.")
        manifest_sha = _require_sha256(
            binding["manifest_sha256"], "registry manifest_sha256"
        )
        manifest_path = _resolve_input_path(binding["manifest_path"], root)
        if (
            registry.get("manifest_sha256") != manifest_sha
            or Path(str(registry.get("manifest_path", ""))).resolve() != manifest_path
            or manifest_sha in manifest_digests
        ):
            raise ValueError(f"Submission registry has a stale/duplicated manifest binding: {path}.")
        manifest_digests.add(manifest_sha)
        job_id = str(binding["slurm_array_job_id"])
        task_count = binding["num_registered_tasks"]
        submissions = registry.get("submissions")
        if (
            registry.get("slurm_array_job_id") != job_id
            or not isinstance(task_count, int)
            or isinstance(task_count, bool)
            or task_count <= 0
            or registry.get("num_registered_tasks") != task_count
            or not isinstance(submissions, list)
            or len(submissions) != task_count
            or [entry.get("job_index") for entry in submissions]
            != list(range(task_count))
        ):
            raise ValueError(f"Submission registry does not cover one exact full array: {path}.")
        record = {
            **binding,
            "path": str(path),
            "manifest_path": str(manifest_path),
        }
        provenance.append(record)
        for task_index, entry in enumerate(submissions):
            if not isinstance(entry, Mapping) or (
                entry.get("manifest_sha256") != manifest_sha
                or entry.get("slurm_array_job_id") != job_id
                or entry.get("slurm_array_task_id") != task_index
                or entry.get("slurm_task_id") != f"{job_id}_{task_index}"
            ):
                raise ValueError(f"Submission registry task identity is invalid: {path}.")
            key = (manifest_sha, task_index)
            if key in lookup:
                raise ValueError(f"Submission registry task ownership is duplicated: {key}.")
            lookup[key] = (dict(entry), record)
    return lookup, sorted(provenance, key=lambda row: row["path"])


def _load_bound_document(
    binding: Mapping[str, Any],
    root: Path,
    *,
    digest_field: str,
    label: str,
    contract_field: str | None = None,
    contract_value: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if not isinstance(binding, Mapping):
        raise ValueError(f"{label} binding must be a mapping.")
    path = _resolve_input_path(binding.get("path"), root)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    expected_file_sha = _require_sha256(binding.get("file_sha256"), f"{label}.file_sha256")
    actual_file_sha = _sha256(path)
    if actual_file_sha != expected_file_sha:
        raise ValueError(
            f"{label} byte hash changed: expected={expected_file_sha}, actual={actual_file_sha}."
        )
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    expected_document_sha = _require_sha256(
        binding.get(digest_field), f"{label}.{digest_field}"
    )
    declared_document_sha = _require_sha256(
        payload.get(digest_field), f"{label} payload {digest_field}"
    )
    actual_document_sha = _canonical_digest(payload, digest_field)
    if not (
        expected_document_sha == declared_document_sha == actual_document_sha
    ):
        raise ValueError(
            f"{label} canonical digest mismatch: binding={expected_document_sha}, "
            f"declared={declared_document_sha}, actual={actual_document_sha}."
        )
    sidecar = _validate_digest_sidecar(path, actual_document_sha)
    if contract_field is not None and payload.get(contract_field) != contract_value:
        raise ValueError(
            f"{label} contract mismatch: expected {contract_field}={contract_value!r}, "
            f"got {payload.get(contract_field)!r}."
        )
    provenance = {
        "path": str(path),
        "file_sha256": actual_file_sha,
        digest_field: actual_document_sha,
        "sidecar_path": str(sidecar),
        "sidecar_sha256": _sha256(sidecar),
    }
    return path, payload, provenance


def _condition_family(job: Mapping[str, Any]) -> tuple[str, str, str, tuple[str, ...]]:
    prompt_id = str(job.get("prompt_id", ""))
    if prompt_id not in PAIR_IDS_BY_PROMPT:
        raise ValueError(f"Fresh job has unknown prompt_id: {prompt_id!r}.")
    variant = job.get("variant_spec")
    if not isinstance(variant, Mapping):
        raise ValueError(f"Fresh job {job.get('condition_id')} lacks variant_spec.")
    kind = str(variant.get("kind", ""))
    active = tuple(str(value) for value in variant.get("active_pair_ids", ()))
    if len(active) != len(set(active)) or not set(active) <= set(PAIR_IDS_BY_PROMPT[prompt_id]):
        raise ValueError(
            f"Fresh job {job.get('condition_id')} has invalid active concept pairs: {active}."
        )
    if kind == "baseline":
        if active:
            raise ValueError("Baseline job cannot claim active concept pairs.")
        return "baseline", "baseline", "none", active
    if kind == "native_negative_prompt":
        if active:
            raise ValueError("Native-negative job cannot claim positive active concept pairs.")
        return "native_negative", "native_negative", "none", active
    if kind not in {"conceptsteer", "shapley_concept_steering"}:
        raise ValueError(f"Unknown fresh-study variant kind: {kind!r}.")
    method = "conceptsteer" if kind == "conceptsteer" else "shapley"
    selection = str(variant.get("pair_selection", ""))
    if selection == "full":
        if active != PAIR_IDS_BY_PROMPT[prompt_id]:
            raise ValueError(
                f"Full {method} job must freeze all five prompt pairs in registered order."
            )
        return f"{method}_full", method, "full", active
    if selection == "single":
        if len(active) != 1:
            raise ValueError(f"Exact-one {method} job must freeze exactly one active pair.")
        return f"{method}_single__{active[0]}", method, "exact_one", active
    raise ValueError(f"Steering job has invalid pair_selection={selection!r}.")


def _expected_unsupported(job: Mapping[str, Any], family: str) -> bool:
    return family == "native_negative" and str(job.get("model_name")) in (
        NATIVE_NEGATIVE_UNSUPPORTED_MODELS
    )


def _recorded_path(value: Any, output_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = output_dir / path
    return path.resolve()


def _validate_media_result(
    *,
    job: Mapping[str, Any],
    result: Mapping[str, Any],
    output_dir: Path,
    expected_unsupported: bool,
) -> tuple[bool, str | None, str | None]:
    media_files = sorted(
        path.resolve()
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".mp4"}
    )
    if expected_unsupported:
        variant = job.get("variant_spec")
        if (
            not isinstance(variant, Mapping)
            or variant.get("capability") != "not_supported"
            or not str(variant.get("reason", "")).strip()
        ):
            raise ValueError(
                "Truthful unsupported job lacks its manifest-bound capability/reason receipt."
            )
        if result.get("status") != "not_supported":
            raise ValueError("Expected native-negative unsupported row did not resolve unsupported.")
        if media_files:
            raise ValueError(f"Truthful unsupported row contains media: {media_files}.")
        if result.get("media_validation") is not None or result.get("validated_media_paths") != []:
            raise ValueError("Truthful unsupported result must bind null validation and zero media.")
        if result.get("reason") != variant["reason"]:
            raise ValueError("Truthful unsupported result differs from its registered reason.")
        return False, None, None

    variant = job.get("variant_spec")
    if (
        isinstance(variant, Mapping)
        and variant.get("kind") == "native_negative_prompt"
        and variant.get("capability") != "supported"
    ):
        raise ValueError("Supported native-negative row lacks its registered capability.")

    if result.get("status") != "completed":
        raise ValueError(
            f"Expected media row is not completed: status={result.get('status')!r}."
        )
    generation = job.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("Fresh job lacks generation contract.")
    task = str(generation.get("task", ""))
    relative = (
        Path("sample_0000/image_000.png")
        if task == "text_to_image"
        else Path("sample_0000/video_000.mp4")
    )
    if task not in {"text_to_image", "text_to_video"}:
        raise ValueError(f"Unsupported generation task in fresh evidence: {task!r}.")
    expected_path = (output_dir / relative).resolve()
    if media_files != [expected_path]:
        raise ValueError(
            f"Completed row media set mismatch: expected={[expected_path]}, found={media_files}."
        )
    validation = result.get("media_validation")
    if not isinstance(validation, Mapping) or validation.get("decode_verified") is not True:
        raise ValueError("Completed row lacks a decode-verified media validation object.")
    paths = result.get("validated_media_paths")
    if not isinstance(paths, list) or len(paths) != 1:
        raise ValueError("Completed row must contain exactly one validated_media_paths entry.")
    if _recorded_path(paths[0], output_dir) != expected_path or _recorded_path(
        validation.get("path"), output_dir
    ) != expected_path:
        raise ValueError("Completed result paths do not resolve to the exact expected media.")
    actual_sha = _sha256(expected_path)
    if validation.get("sha256") != actual_sha:
        raise ValueError("Completed result media SHA-256 differs from the reopened media.")
    if int(validation.get("size_bytes", -1)) != expected_path.stat().st_size:
        raise ValueError("Completed result media size differs from the reopened media.")
    if int(validation.get("width", -1)) != int(generation.get("width", -2)) or int(
        validation.get("height", -1)
    ) != int(generation.get("height", -2)):
        raise ValueError("Completed result dimensions differ from the frozen job.")
    if task == "text_to_image":
        if validation.get("media_type") != "image/png":
            raise ValueError("Image result does not identify PNG media.")
    else:
        exact_video_contract = {
            "media_type": "video/mp4",
            "codec_name": "h264",
            "frame_count": 240,
        }
        if any(validation.get(key) != value for key, value in exact_video_contract.items()):
            raise ValueError("Video result violates the exact MP4/H.264/240-frame contract.")
        if abs(float(validation.get("fps", -1.0)) - 16.0) > 1.0e-6 or abs(
            float(validation.get("duration_seconds", -1.0)) - 15.0
        ) > 1.0e-6:
            raise ValueError("Video result violates exact 16-fps/15-second timing.")
    return True, str(expected_path), actual_sha


def _load_result_binding(
    *,
    cohort: str,
    row_binding: Mapping[str, Any],
    manifests: Mapping[str, tuple[Path, Mapping[str, Any]]],
    root: Path,
    registry_lookup: Mapping[
        tuple[str, int], tuple[Mapping[str, Any], Mapping[str, Any]]
    ]
    | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_sha = _require_sha256(
        row_binding.get("manifest_sha256"), f"{cohort} row manifest_sha256"
    )
    if manifest_sha not in manifests:
        raise ValueError(f"{cohort} row references an unbound manifest {manifest_sha}.")
    manifest_path, manifest = manifests[manifest_sha]
    job_index = row_binding.get("manifest_job_index")
    if not isinstance(job_index, int) or isinstance(job_index, bool):
        raise ValueError(f"{cohort} manifest_job_index must be an integer.")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not 0 <= job_index < len(jobs):
        raise ValueError(f"{cohort} manifest_job_index {job_index} is out of range.")
    job = jobs[job_index]
    if not isinstance(job, Mapping):
        raise ValueError(f"{cohort} manifest job {job_index} is not a mapping.")
    model_name = str(job.get("model_name", ""))
    prompt_id = str(job.get("prompt_id", ""))
    if model_name not in MODEL_NAMES or prompt_id not in PROMPT_IDS:
        raise ValueError(f"{cohort} job has an unregistered model/prompt axis.")
    task = str((job.get("generation") or {}).get("task", ""))
    expected_task = "text_to_image" if model_name in IMAGE_MODELS else "text_to_video"
    if task != expected_task:
        raise ValueError(f"{model_name} has task {task!r}, expected {expected_task!r}.")
    family, method, ablation, active_pairs = _condition_family(job)
    expected_unsupported = _expected_unsupported(job, family)
    if job.get("expected_media") is not (not expected_unsupported):
        raise ValueError(
            f"{cohort} job expected_media disagrees with registered support topology."
        )
    # Generation never runs the bare manifest row.  The dispatcher adds the
    # immutable manifest identity and exact array index before writing the
    # startup/result artifacts.  Reconstruct that launch-bound job here rather
    # than accepting either an unbound result or caller-supplied launch fields
    # in the manifest itself.
    if "launch_manifest_sha256" in job or "launch_manifest_job_index" in job:
        raise ValueError("Immutable manifest rows must not contain runtime launch bindings.")
    launch_bound_job = dict(job)
    launch_bound_job["launch_manifest_sha256"] = manifest_sha
    launch_bound_job["launch_manifest_job_index"] = job_index

    execution_provenance: dict[str, Any] | None = None
    registry_provenance: dict[str, Any] | None = None
    if registry_lookup is not None:
        registry_source = registry_lookup.get((manifest_sha, job_index))
        if registry_source is None:
            raise ValueError(
                f"{cohort} row has no exact immutable registry task owner: "
                f"{manifest_sha}/{job_index}."
            )
        registry_entry, registry_record = registry_source
        if row_binding.get("submission_registry") != registry_record:
            raise ValueError(f"{cohort} row registry receipt differs from its live registry.")
        identity_binding = row_binding.get("execution_identity")
        if not isinstance(identity_binding, Mapping) or set(identity_binding) != {
            "path",
            "sha256",
            "slurm_task_id",
            "registry_path",
            "registry_sha256",
        }:
            raise ValueError(f"{cohort} row lacks its exact execution-identity receipt.")
        output_dir_for_identity = Path(str(job.get("output_dir", ""))).expanduser().resolve()
        identity_path = _resolve_input_path(identity_binding["path"], root)
        expected_identity_path = output_dir_for_identity / "execution_identity.json"
        identity_sha = _require_sha256(
            identity_binding["sha256"], f"{cohort} execution identity SHA-256"
        )
        if (
            identity_path != expected_identity_path
            or identity_path.is_symlink()
            or not identity_path.is_file()
            or _sha256(identity_path) != identity_sha
        ):
            raise ValueError(f"{cohort} execution identity is missing/stale: {identity_path}.")
        identity = _read_json(identity_path)
        if (
            not isinstance(identity, Mapping)
            or identity.get("schema_version") != 1
            or identity.get("SLURM_ARRAY_JOB_ID")
            != str(registry_entry["slurm_array_job_id"])
            or identity.get("SLURM_ARRAY_TASK_ID")
            != str(registry_entry["slurm_array_task_id"])
            or identity.get("slurm_task_id") != registry_entry["slurm_task_id"]
            or identity_binding["slurm_task_id"] != registry_entry["slurm_task_id"]
            or _resolve_input_path(identity_binding["registry_path"], root)
            != Path(str(registry_record["path"])).resolve()
            or identity_binding["registry_sha256"] != registry_record["registry_sha256"]
        ):
            raise ValueError(f"{cohort} execution identity differs from its registry task.")
        _parse_timestamp(identity.get("captured_at_utc"), f"{cohort} execution identity")
        execution_provenance = {
            "path": str(identity_path),
            "sha256": identity_sha,
            "slurm_task_id": registry_entry["slurm_task_id"],
        }
        registry_provenance = dict(registry_record)

    result_binding = row_binding.get("result")
    if not isinstance(result_binding, Mapping):
        raise ValueError(f"{cohort} row lacks its immutable result binding.")
    result_path = _resolve_input_path(result_binding.get("path"), root)
    output_dir = Path(str(job.get("output_dir", ""))).expanduser().resolve()
    expected_result_path = (output_dir / "benchmark_job_result.json").resolve()
    if result_path != expected_result_path or not result_path.is_file():
        raise ValueError(
            f"{cohort} result path is missing/stale: expected={expected_result_path}, "
            f"bound={result_path}."
        )
    expected_result_sha = _require_sha256(
        result_binding.get("sha256"), f"{cohort} result SHA-256"
    )
    actual_result_sha = _sha256(result_path)
    if expected_result_sha != actual_result_sha:
        raise ValueError(
            f"{cohort} result changed: expected={expected_result_sha}, actual={actual_result_sha}."
        )
    result = _read_json(result_path)
    if not isinstance(result, Mapping) or result.get("job") != launch_bound_job:
        raise ValueError(
            f"{cohort} result embeds a stale/different job for manifest index {job_index}."
        )
    media_valid, media_path, media_sha = _validate_media_result(
        job=job,
        result=result,
        output_dir=output_dir,
        expected_unsupported=expected_unsupported,
    )

    timing_binding = row_binding.get("timing")
    if not isinstance(timing_binding, Mapping):
        raise ValueError(f"{cohort} row lacks its immutable timing binding.")
    timing_path = _resolve_input_path(timing_binding.get("path"), root)
    expected_timing_path = (output_dir / "experiment_timing.json").resolve()
    if timing_path != expected_timing_path or not timing_path.is_file():
        raise ValueError(f"{cohort} timing path is missing/stale: {timing_path}.")
    timing_sha = _require_sha256(timing_binding.get("sha256"), f"{cohort} timing SHA-256")
    if _sha256(timing_path) != timing_sha:
        raise ValueError(f"{cohort} timing file changed: {timing_path}.")
    timing = _read_json(timing_path)
    wall_seconds = _require_number(timing.get("wall_seconds"), f"{cohort} wall_seconds")
    if timing.get("status") != result.get("status"):
        raise ValueError(f"{cohort} timing status differs from result status.")

    condition_id = str(job.get("condition_id", ""))
    if not condition_id:
        raise ValueError(f"{cohort} job lacks condition_id.")
    seed = job.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{cohort} job seed must be an integer.")
    row_key = f"{cohort}|{manifest_sha}|{job_index}"
    record = {
        "row_key": row_key,
        "cohort": cohort,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "manifest_job_index": job_index,
        "manifest_created_at_utc": manifest.get("created_at_utc"),
        "result_path": str(result_path),
        "result_sha256": actual_result_sha,
        "timing_path": str(timing_path),
        "timing_sha256": timing_sha,
        "condition_id": condition_id,
        "model_name": model_name,
        "prompt_id": prompt_id,
        "seed": seed,
        "task": task,
        "condition_family": family,
        "method": method,
        "ablation": ablation,
        "active_pair_ids": active_pairs,
        "active_pair_count": len(active_pairs),
        "status": str(result.get("status")),
        "outcome": "media_validated" if media_valid else "truthful_unsupported",
        "media_valid": media_valid,
        "media_path": media_path,
        "media_sha256": media_sha,
        "wall_seconds": wall_seconds,
    }
    provenance = {
        "cohort": cohort,
        "row_key": row_key,
        "result_path": str(result_path),
        "result_sha256": actual_result_sha,
        "timing_path": str(timing_path),
        "timing_sha256": timing_sha,
        "media_path": media_path,
        "media_sha256": media_sha,
        "submission_registry": registry_provenance,
        "execution_identity": execution_provenance,
    }
    return record, provenance


def _load_cohort(
    *,
    cohort: str,
    descriptor: Mapping[str, Any],
    root: Path,
    registry_lookup: Mapping[
        tuple[str, int], tuple[Mapping[str, Any], Mapping[str, Any]]
    ]
    | None = None,
    expected_manifest_count: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if descriptor.get("expected_counts") != EXPECTED_COHORT_COUNTS[cohort]:
        raise ValueError(
            f"{cohort} descriptor must declare exact counts "
            f"{EXPECTED_COHORT_COUNTS[cohort]}."
        )
    manifest_bindings = descriptor.get("manifests")
    if not isinstance(manifest_bindings, list) or not manifest_bindings:
        raise ValueError(f"{cohort} descriptor has no immutable manifest bindings.")
    if expected_manifest_count is not None and len(manifest_bindings) != expected_manifest_count:
        raise ValueError(
            f"{cohort} production descriptor must bind exactly "
            f"{expected_manifest_count} manifests; got {len(manifest_bindings)}."
        )
    manifests: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    manifest_provenance: list[dict[str, Any]] = []
    manifest_paths: set[Path] = set()
    for index, binding in enumerate(manifest_bindings):
        path, payload, provenance = _load_bound_document(
            binding,
            root,
            digest_field="manifest_sha256",
            label=f"{cohort} manifest[{index}]",
        )
        manifest_sha = str(payload["manifest_sha256"])
        if manifest_sha in manifests or path in manifest_paths:
            raise ValueError(f"{cohort} contains a duplicate manifest binding: {path}.")
        jobs = payload.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            raise ValueError(f"{cohort} manifest has no jobs: {path}.")
        manifests[manifest_sha] = (path, payload)
        manifest_paths.add(path)
        manifest_provenance.append(provenance)

    row_bindings = descriptor.get("rows")
    if not isinstance(row_bindings, list):
        raise ValueError(f"{cohort} descriptor has no row bindings list.")
    records: list[dict[str, Any]] = []
    row_provenance: list[dict[str, Any]] = []
    for row_binding in row_bindings:
        if not isinstance(row_binding, Mapping):
            raise ValueError(f"{cohort} row binding must be a mapping.")
        record, provenance = _load_result_binding(
            cohort=cohort,
            row_binding=row_binding,
            manifests=manifests,
            root=root,
            registry_lookup=registry_lookup,
        )
        records.append(record)
        row_provenance.append(provenance)
    frame = pd.DataFrame(records)
    if len(frame) != EXPECTED_COHORT_COUNTS[cohort]["logical"]:
        raise ValueError(
            f"{cohort} row count is {len(frame)}, expected "
            f"{EXPECTED_COHORT_COUNTS[cohort]['logical']}."
        )
    for column in ("row_key", "condition_id", "result_path", "timing_path"):
        duplicate = frame[frame[column].duplicated(keep=False)]
        if not duplicate.empty:
            values = sorted(set(duplicate[column].astype(str)))
            raise ValueError(f"{cohort} contains duplicate {column} values: {values[:5]}.")
    media_paths = frame.loc[frame["media_valid"], "media_path"]
    if media_paths.duplicated().any():
        raise ValueError(f"{cohort} reuses one media file for multiple logical rows.")
    used_manifests = set(frame["manifest_sha256"])
    if used_manifests != set(manifests):
        raise ValueError(
            f"{cohort} binds manifests that contribute no selected row: "
            f"{sorted(set(manifests) - used_manifests)}."
        )
    if registry_lookup is not None:
        for manifest_sha, (manifest_path, manifest) in manifests.items():
            first = registry_lookup.get((manifest_sha, 0))
            if first is None or Path(str(first[1]["manifest_path"])).resolve() != manifest_path:
                raise ValueError(
                    f"{cohort} registry identifies another manifest path: {manifest_path}."
                )
            if len(manifest["jobs"]) != int(first[1]["num_registered_tasks"]):
                raise ValueError(
                    f"{cohort} registry task count differs from manifest {manifest_path}."
                )
    return frame, {
        "manifests": sorted(manifest_provenance, key=lambda row: row["path"]),
        "rows": sorted(row_provenance, key=lambda row: row["row_key"]),
    }


def _qualification_families(prompt_id: str) -> tuple[str, ...]:
    pair_id = QUALIFICATION_PAIR_BY_PROMPT[prompt_id]
    return (
        "baseline",
        "native_negative",
        "conceptsteer_full",
        "shapley_full",
        f"conceptsteer_single__{pair_id}",
        f"shapley_single__{pair_id}",
    )


def _final_families(prompt_id: str) -> tuple[str, ...]:
    pairs = PAIR_IDS_BY_PROMPT[prompt_id]
    return (
        "baseline",
        "native_negative",
        "conceptsteer_full",
        "shapley_full",
        *(f"conceptsteer_single__{pair_id}" for pair_id in pairs),
        *(f"shapley_single__{pair_id}" for pair_id in pairs),
    )


def _counter_difference(
    expected: Counter[tuple[Any, ...]], actual: Counter[tuple[Any, ...]]
) -> str:
    missing = list((expected - actual).elements())
    extra = list((actual - expected).elements())
    return f"missing={missing[:8]}, extra={extra[:8]}"


def _validate_cohort_topology(frame: pd.DataFrame, cohort: str) -> None:
    actual_counts = {
        "logical": int(len(frame)),
        "media": int(frame["media_valid"].sum()),
        "unsupported": int((frame["outcome"] == "truthful_unsupported").sum()),
    }
    if actual_counts != EXPECTED_COHORT_COUNTS[cohort]:
        raise ValueError(
            f"{cohort} terminal arithmetic mismatch: expected={EXPECTED_COHORT_COUNTS[cohort]}, "
            f"actual={actual_counts}."
        )
    expected: Counter[tuple[Any, ...]] = Counter()
    if cohort == "qualification":
        for model_name in MODEL_NAMES:
            seeds = (0,) if model_name in IMAGE_MODELS else (0, 1, 2)
            for prompt_id in PROMPT_IDS:
                for seed in seeds:
                    for family in _qualification_families(prompt_id):
                        expected[(model_name, prompt_id, seed, family)] += 1
        actual = Counter(
            zip(
                frame["model_name"],
                frame["prompt_id"],
                frame["seed"],
                frame["condition_family"],
                strict=True,
            )
        )
    elif cohort == "seed_ladder":
        for model_name in MODEL_NAMES:
            for prompt_id in PROMPT_IDS:
                for seed in range(8):
                    expected[(model_name, prompt_id, seed, "baseline")] += 1
        actual = Counter(
            zip(
                frame["model_name"],
                frame["prompt_id"],
                frame["seed"],
                frame["condition_family"],
                strict=True,
            )
        )
    else:
        for model_name in MODEL_NAMES:
            for prompt_id in PROMPT_IDS:
                for family in _final_families(prompt_id):
                    expected[(model_name, prompt_id, family)] += 1
        actual = Counter(
            zip(
                frame["model_name"],
                frame["prompt_id"],
                frame["condition_family"],
                strict=True,
            )
        )
    if actual != expected:
        raise ValueError(f"{cohort} axis topology mismatch: {_counter_difference(expected, actual)}.")
    expected_unsupported = frame.apply(
        lambda row: row["condition_family"] == "native_negative"
        and row["model_name"] in NATIVE_NEGATIVE_UNSUPPORTED_MODELS,
        axis=1,
    )
    observed_unsupported = frame["outcome"] == "truthful_unsupported"
    if not observed_unsupported.equals(expected_unsupported):
        raise ValueError(f"{cohort} unsupported rows differ from the registered support matrix.")
    if cohort == "final":
        exact_one = frame[frame["ablation"] == "exact_one"]
        if len(exact_one) != 360 or not exact_one["media_valid"].all():
            raise ValueError(
                "Final exact-one subset must be exactly 360 validated media rows, not an "
                "additional cohort."
            )


def _parse_timestamp(value: Any, label: str) -> datetime:
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not an ISO-8601 timestamp: {text!r}.") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include an explicit timezone.")
    return parsed


def _reject_target_informed_candidate(value: Any, path: str = "candidate_rows") -> None:
    forbidden_fragments = (
        "active_target",
        "target_achievement",
        "steering_result",
        "shapley",
        "final_condition",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if any(fragment in normalized for fragment in forbidden_fragments):
                raise ValueError(
                    f"Seed selection contains target/steering-informed field {path}.{key}."
                )
            _reject_target_informed_candidate(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_target_informed_candidate(child, f"{path}[{index}]")


def _load_seed_selections(
    *,
    bindings: Any,
    ladder: pd.DataFrame,
    final: pd.DataFrame,
    root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not isinstance(bindings, list) or len(bindings) != 36:
        raise ValueError("Fresh bundle must bind exactly 36 immutable seed-selection documents.")
    ladder_by_axis_seed = {
        (row.prompt_id, row.model_name, int(row.seed)): row
        for row in ladder.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for index, binding in enumerate(bindings):
        _, payload, source = _load_bound_document(
            binding,
            root,
            digest_field="document_sha256",
            label=f"seed selection[{index}]",
            contract_field="selection",
            contract_value="finer_detailing_target_blind_seed_v1",
        )
        if payload.get("schema_version") != 1:
            raise ValueError("Seed selection schema_version must be 1.")
        prompt_id = str(payload.get("prompt_id", ""))
        model_name = str(payload.get("model_name", ""))
        if prompt_id not in PROMPT_IDS or model_name not in MODEL_NAMES:
            raise ValueError("Seed selection identifies an unregistered prompt/model row.")
        candidates = payload.get("candidate_rows")
        if not isinstance(candidates, list) or len(candidates) != 8:
            raise ValueError("Seed selection must bind exactly eight ordered candidate rows.")
        _reject_target_informed_candidate(candidates)
        if [candidate.get("seed") for candidate in candidates] != list(range(8)):
            raise ValueError("Seed-selection candidate order must be exactly seeds 0..7.")
        for candidate in candidates:
            seed = int(candidate["seed"])
            ladder_row = ladder_by_axis_seed.get((prompt_id, model_name, seed))
            if ladder_row is None:
                raise ValueError("Seed selection references a missing ladder candidate.")
            if (
                candidate.get("manifest_sha256") != ladder_row.manifest_sha256
                or candidate.get("manifest_job_index") != ladder_row.manifest_job_index
                or candidate.get("result_sha256") != ladder_row.result_sha256
            ):
                raise ValueError("Seed selection candidate binding differs from ladder evidence.")
        eligible = payload.get("eligible_seed_ids")
        if (
            not isinstance(eligible, list)
            or not eligible
            or any(not isinstance(seed, int) or seed not in range(8) for seed in eligible)
            or eligible != sorted(set(eligible))
        ):
            raise ValueError("Seed selection eligible_seed_ids must be a non-empty sorted subset.")
        selected_seed = payload.get("selected_seed")
        if not isinstance(selected_seed, int) or selected_seed not in eligible:
            raise ValueError("Seed selection selected_seed must be one eligible seed.")
        attestation = payload.get("attestation")
        if not isinstance(attestation, Mapping) or attestation.get(
            "no_steering_or_target_evidence_inspected"
        ) is not True:
            raise ValueError("Seed selection lacks its target-blind reviewer attestation.")
        selected_at = _parse_timestamp(payload.get("selected_at_utc"), "selected_at_utc")
        records.append(
            {
                "prompt_id": prompt_id,
                "model_name": model_name,
                "selected_seed": selected_seed,
                "eligible_seed_count": len(eligible),
                "selected_at_utc": selected_at.isoformat(),
                "selection_path": source["path"],
                "selection_file_sha256": source["file_sha256"],
                "selection_document_sha256": source["document_sha256"],
            }
        )
        provenance.append(source)
    selection_frame = pd.DataFrame(records)
    if selection_frame[["prompt_id", "model_name"]].duplicated().any():
        raise ValueError("Seed selections duplicate one or more prompt/model rows.")
    expected_axes = {(prompt, model) for prompt in PROMPT_IDS for model in MODEL_NAMES}
    observed_axes = set(
        selection_frame[["prompt_id", "model_name"]].itertuples(index=False, name=None)
    )
    if observed_axes != expected_axes:
        raise ValueError("Seed selections omit one or more prompt/model rows.")
    selected_by_axis = {
        (row.prompt_id, row.model_name): (int(row.selected_seed), row.selected_at_utc)
        for row in selection_frame.itertuples(index=False)
    }
    for row in final.itertuples(index=False):
        selected_seed, selected_at_text = selected_by_axis[(row.prompt_id, row.model_name)]
        if int(row.seed) != selected_seed:
            raise ValueError(
                f"Final row {row.condition_id} uses seed {row.seed}, selected {selected_seed}."
            )
        manifest_created = _parse_timestamp(
            row.manifest_created_at_utc, f"final manifest timestamp for {row.condition_id}"
        )
        selected_at = _parse_timestamp(selected_at_text, "selected_at_utc")
        if manifest_created < selected_at:
            raise ValueError(
                f"Final manifest for {row.condition_id} predates immutable seed selection."
            )
    return selection_frame.sort_values(["model_name", "prompt_id"]).reset_index(drop=True), sorted(
        provenance, key=lambda row: row["path"]
    )


def _load_ledger_document(
    *,
    name: str,
    binding: Any,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    _, payload, provenance = _load_bound_document(
        binding,
        root,
        digest_field="document_sha256",
        label=f"{name} ledger",
        contract_field="evidence",
        contract_value=LEDGER_CONTRACTS[name],
    )
    if payload.get("schema_version") != 1:
        raise ValueError(f"{name} ledger schema_version must be 1.")
    raw_sources = payload.get("source_artifacts")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"{name} ledger must bind at least one raw source artifact.")
    source_hashes: set[str] = set()
    source_provenance: list[dict[str, Any]] = []
    source_paths: set[Path] = set()
    for index, source in enumerate(raw_sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"{name} source_artifacts[{index}] must be a mapping.")
        path = _resolve_input_path(source.get("path"), root)
        digest = _require_sha256(source.get("sha256"), f"{name} source artifact SHA-256")
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"{name} raw source artifact is missing or changed: {path}.")
        if path in source_paths or digest in source_hashes:
            raise ValueError(f"{name} ledger duplicates a raw source artifact: {path}.")
        source_paths.add(path)
        source_hashes.add(digest)
        source_provenance.append(
            {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{name} ledger is missing its rows list.")
    raw_document_provenance: dict[str, Any] | None = None
    raw_document_binding = payload.get("raw_source_document")
    if raw_document_binding is not None:
        _, _, raw_document_provenance = _load_bound_document(
            raw_document_binding,
            root,
            digest_field="document_sha256",
            label=f"{name} raw source document",
        )
        if raw_sources != [
            {
                "path": raw_document_provenance["path"],
                "sha256": raw_document_provenance["file_sha256"],
            }
        ]:
            raise ValueError(
                f"{name} source_artifacts differs from its canonical raw document."
            )
    provenance = {
        **provenance,
        "source_artifacts": source_provenance,
        "raw_source_document": raw_document_provenance,
    }
    return payload, provenance, source_hashes


def _evidence_identity(
    evidence_row: Mapping[str, Any],
    lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
    *,
    ledger_name: str,
    source_hashes: set[str],
) -> tuple[tuple[str, str, int], Mapping[str, Any]]:
    cohort = str(evidence_row.get("cohort", ""))
    manifest_sha = _require_sha256(
        evidence_row.get("manifest_sha256"), f"{ledger_name} manifest_sha256"
    )
    job_index = evidence_row.get("manifest_job_index")
    if not isinstance(job_index, int) or isinstance(job_index, bool):
        raise ValueError(f"{ledger_name} manifest_job_index must be an integer.")
    key = (cohort, manifest_sha, job_index)
    source = lookup.get(key)
    if source is None:
        raise ValueError(f"{ledger_name} row references unknown campaign evidence: {key}.")
    result_sha = _require_sha256(
        evidence_row.get("result_sha256"), f"{ledger_name} result_sha256"
    )
    if result_sha != source["result_sha256"]:
        raise ValueError(f"{ledger_name} row is stale relative to its bound result.")
    source_sha = _require_sha256(
        evidence_row.get("source_sha256"), f"{ledger_name} source_sha256"
    )
    if source_sha not in source_hashes:
        raise ValueError(f"{ledger_name} row references an unbound raw evidence artifact.")
    return key, source


def _load_objective_temporal_ledger(
    *,
    binding: Any,
    root: Path,
    lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload, provenance, source_hashes = _load_ledger_document(
        name="objective_temporal", binding=binding, root=root
    )
    expected = {
        key
        for key, row in lookup.items()
        if row["media_valid"] and row["task"] == "text_to_video"
    }
    seen: set[tuple[str, str, int]] = set()
    records: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    for evidence_row in payload["rows"]:
        if not isinstance(evidence_row, Mapping):
            raise ValueError("objective_temporal ledger row must be a mapping.")
        key, source = _evidence_identity(
            evidence_row,
            lookup,
            ledger_name="objective_temporal",
            source_hashes=source_hashes,
        )
        if key in seen:
            raise ValueError(f"objective_temporal ledger duplicates row {key}.")
        if key not in expected:
            raise ValueError("objective_temporal ledger includes a non-video/non-media row.")
        seen.add(key)
        source_sha = str(evidence_row["source_sha256"])
        used_sources.add(source_sha)
        record = {
            "row_key": source["row_key"],
            "cohort": source["cohort"],
            "condition_id": source["condition_id"],
            "model_name": source["model_name"],
            "prompt_id": source["prompt_id"],
            "condition_family": source["condition_family"],
            "method": source["method"],
            "ablation": source["ablation"],
            "automatic_gate_pass": _require_bool(
                evidence_row.get("automatic_gate_pass"),
                "objective_temporal.automatic_gate_pass",
            ),
            "freeze_candidate": _require_bool(
                evidence_row.get("freeze_candidate"),
                "objective_temporal.freeze_candidate",
            ),
            "near_periodic_candidate": _require_bool(
                evidence_row.get("near_periodic_candidate"),
                "objective_temporal.near_periodic_candidate",
            ),
            "cut_candidate": _require_bool(
                evidence_row.get("cut_candidate"), "objective_temporal.cut_candidate"
            ),
            "cadence_candidate": _require_bool(
                evidence_row.get("cadence_candidate"),
                "objective_temporal.cadence_candidate",
            ),
            "source_sha256": source_sha,
        }
        motion = evidence_row.get("prompt3_motion_gate_pass")
        if source["prompt_id"] == "03_empty_outdoor_mall":
            record["prompt3_motion_gate_pass"] = _require_bool(
                motion, "objective_temporal.prompt3_motion_gate_pass"
            )
        elif motion is not None:
            raise ValueError("Prompt-3 motion gate must be null for person prompts.")
        else:
            record["prompt3_motion_gate_pass"] = None
        records.append(record)
    if seen != expected:
        raise ValueError(
            "objective_temporal coverage is incomplete: "
            f"missing={list(expected - seen)[:8]}, extra={list(seen - expected)[:8]}."
        )
    if used_sources != source_hashes:
        raise ValueError("objective_temporal ledger binds unused source artifacts.")
    return pd.DataFrame(records).sort_values("row_key").reset_index(drop=True), provenance


def _load_manual_semantic_ledger(
    *,
    binding: Any,
    root: Path,
    lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    payload, provenance, source_hashes = _load_ledger_document(
        name="manual_semantic", binding=binding, root=root
    )
    expected = {key for key, row in lookup.items() if row["media_valid"]}
    seen: set[tuple[str, str, int]] = set()
    records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    for evidence_row in payload["rows"]:
        if not isinstance(evidence_row, Mapping):
            raise ValueError("manual_semantic ledger row must be a mapping.")
        key, source = _evidence_identity(
            evidence_row,
            lookup,
            ledger_name="manual_semantic",
            source_hashes=source_hashes,
        )
        if key in seen:
            raise ValueError(f"manual_semantic ledger duplicates row {key}.")
        if key not in expected:
            raise ValueError("manual_semantic ledger includes a non-media row.")
        seen.add(key)
        source_sha = str(evidence_row["source_sha256"])
        used_sources.add(source_sha)
        active_pairs = tuple(source["active_pair_ids"])
        pair_outcomes = evidence_row.get("active_pair_results")
        if not isinstance(pair_outcomes, Mapping) or set(pair_outcomes) != set(active_pairs):
            raise ValueError(
                f"manual_semantic active-pair coverage differs from {source['row_key']}."
            )
        gender = evidence_row.get("gender_preserved")
        if source["prompt_id"] in {"01_sad_young_girl", "02_angry_old_man"}:
            gender = _require_bool(gender, "manual_semantic.gender_preserved")
        elif gender is not None:
            raise ValueError("gender_preserved must be null for the non-person mall prompt.")
        negative_suppression = evidence_row.get("native_negative_source_suppression")
        if source["method"] == "native_negative":
            negative_suppression = _require_bool(
                negative_suppression,
                "manual_semantic.native_negative_source_suppression",
            )
        elif negative_suppression is not None:
            raise ValueError(
                "native_negative_source_suppression must be null outside native negative rows."
            )
        record = {
            "row_key": source["row_key"],
            "cohort": source["cohort"],
            "condition_id": source["condition_id"],
            "model_name": source["model_name"],
            "prompt_id": source["prompt_id"],
            "task": source["task"],
            "condition_family": source["condition_family"],
            "method": source["method"],
            "ablation": source["ablation"],
            "semantic_success": _require_bool(
                evidence_row.get("semantic_success"), "manual_semantic.semantic_success"
            ),
            "source_fidelity": _require_bool(
                evidence_row.get("source_fidelity"), "manual_semantic.source_fidelity"
            ),
            "non_target_preservation": _require_bool(
                evidence_row.get("non_target_preservation"),
                "manual_semantic.non_target_preservation",
            ),
            "gender_preserved": gender,
            "native_negative_source_suppression": negative_suppression,
            "source_sha256": source_sha,
        }
        records.append(record)
        for pair_id in active_pairs:
            outcome = pair_outcomes[pair_id]
            if not isinstance(outcome, Mapping):
                raise ValueError("manual_semantic pair outcome must be a mapping.")
            pair_records.append(
                {
                    "row_key": source["row_key"],
                    "cohort": source["cohort"],
                    "condition_id": source["condition_id"],
                    "model_name": source["model_name"],
                    "prompt_id": source["prompt_id"],
                    "task": source["task"],
                    "method": source["method"],
                    "ablation": source["ablation"],
                    "pair_id": pair_id,
                    "target_achieved": _require_bool(
                        outcome.get("target_achieved"),
                        f"manual_semantic.{pair_id}.target_achieved",
                    ),
                    "selective_without_collateral_change": _require_bool(
                        outcome.get("selective_without_collateral_change"),
                        f"manual_semantic.{pair_id}.selective_without_collateral_change",
                    ),
                    "source_sha256": source_sha,
                }
            )
    if seen != expected:
        raise ValueError(
            "manual_semantic coverage is incomplete: "
            f"missing={list(expected - seen)[:8]}, extra={list(seen - expected)[:8]}."
        )
    if used_sources != source_hashes:
        raise ValueError("manual_semantic ledger binds unused source artifacts.")
    manual = pd.DataFrame(records).sort_values("row_key").reset_index(drop=True)
    pairs = pd.DataFrame(pair_records)
    if not pairs.empty:
        pairs = pairs.sort_values(["row_key", "pair_id"]).reset_index(drop=True)
    return manual, pairs, provenance


def _load_shapley_diagnostics_ledger(
    *,
    binding: Any,
    root: Path,
    lookup: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload, provenance, source_hashes = _load_ledger_document(
        name="shapley_diagnostics", binding=binding, root=root
    )
    expected = {
        key
        for key, row in lookup.items()
        if row["media_valid"] and row["method"] == "shapley"
    }
    seen: set[tuple[str, str, int]] = set()
    records: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    for evidence_row in payload["rows"]:
        if not isinstance(evidence_row, Mapping):
            raise ValueError("shapley_diagnostics ledger row must be a mapping.")
        key, source = _evidence_identity(
            evidence_row,
            lookup,
            ledger_name="shapley_diagnostics",
            source_hashes=source_hashes,
        )
        if key in seen:
            raise ValueError(f"shapley_diagnostics ledger duplicates row {key}.")
        if key not in expected:
            raise ValueError("shapley_diagnostics ledger includes a non-Shapley row.")
        seen.add(key)
        if evidence_row.get("protocol_version") != 2 or evidence_row.get(
            "trace_schema_version"
        ) != 2:
            raise ValueError("Shapley evidence must use repaired protocol/trace version 2.")
        accepted_fraction = _require_number(
            evidence_row.get("accepted_step_fraction"),
            "shapley_diagnostics.accepted_step_fraction",
        )
        coordinate_fraction = _require_number(
            evidence_row.get("selected_coordinate_fraction"),
            "shapley_diagnostics.selected_coordinate_fraction",
        )
        if accepted_fraction > 1.0 or coordinate_fraction > 1.0:
            raise ValueError("Shapley fractions must lie in [0, 1].")
        violation_count = evidence_row.get("non_target_violation_count")
        if not isinstance(violation_count, int) or isinstance(violation_count, bool) or (
            violation_count < 0
        ):
            raise ValueError("Shapley non_target_violation_count must be a non-negative integer.")
        source_sha = str(evidence_row["source_sha256"])
        used_sources.add(source_sha)
        records.append(
            {
                "row_key": source["row_key"],
                "cohort": source["cohort"],
                "condition_id": source["condition_id"],
                "model_name": source["model_name"],
                "prompt_id": source["prompt_id"],
                "task": source["task"],
                "condition_family": source["condition_family"],
                "ablation": source["ablation"],
                "validation_pass": _require_bool(
                    evidence_row.get("validation_pass"),
                    "shapley_diagnostics.validation_pass",
                ),
                "accepted_step_fraction": accepted_fraction,
                "aggregate_score_decrease": _require_number(
                    evidence_row.get("aggregate_score_decrease"),
                    "shapley_diagnostics.aggregate_score_decrease",
                ),
                "selected_coordinate_fraction": coordinate_fraction,
                "max_ci_half_width": _require_number(
                    evidence_row.get("max_ci_half_width"),
                    "shapley_diagnostics.max_ci_half_width",
                ),
                "non_target_violation_count": violation_count,
                "source_sha256": source_sha,
            }
        )
    if seen != expected:
        raise ValueError(
            "shapley_diagnostics coverage is incomplete: "
            f"missing={list(expected - seen)[:8]}, extra={list(seen - expected)[:8]}."
        )
    if used_sources != source_hashes:
        raise ValueError("shapley_diagnostics ledger binds unused source artifacts.")
    return pd.DataFrame(records).sort_values("row_key").reset_index(drop=True), provenance


def load_fresh_plot_data(bundle_path: Path, root: Path) -> FreshPlotData:
    """Reconcile every fresh-study input before permitting any plot output."""

    root = root.resolve()
    supplied_bundle_path = Path(bundle_path).expanduser()
    lexical_bundle_path = Path(os.path.abspath(supplied_bundle_path))
    cursor = Path(lexical_bundle_path.anchor)
    for part in lexical_bundle_path.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(
                f"Fresh evidence commit path traverses a forbidden symlink: {cursor}."
            )
    bundle_path = lexical_bundle_path.resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Fresh evidence bundle does not exist: {bundle_path}")
    bundle = _read_json(bundle_path)
    if not isinstance(bundle, dict):
        raise ValueError("Fresh evidence bundle must be a JSON object.")
    if bundle.get("schema_version") != 1 or bundle.get("contract") != FRESH_EVIDENCE_CONTRACT:
        raise ValueError("Fresh evidence bundle schema/contract mismatch.")
    production_publication = bundle.get("publication_contract") is not None
    if production_publication and (
        bundle.get("publication_contract") != FRESH_PUBLICATION_CONTRACT
        or set(bundle) != FRESH_PUBLICATION_BUNDLE_KEYS
    ):
        raise ValueError("Production fresh evidence publication schema/contract mismatch.")
    if production_publication:
        _validate_production_publication_tree(bundle_path)
    bundle_digest = _require_sha256(bundle.get("document_sha256"), "bundle.document_sha256")
    actual_bundle_digest = _canonical_digest(bundle, "document_sha256")
    if actual_bundle_digest != bundle_digest:
        raise ValueError(
            f"Fresh evidence bundle digest mismatch: declared={bundle_digest}, "
            f"actual={actual_bundle_digest}."
        )
    bundle_sidecar = _validate_digest_sidecar(bundle_path, bundle_digest)
    if production_publication:
        bundle_raw_sidecar = bundle_path.with_suffix(bundle_path.suffix + ".raw.sha256")
        bundle_file_sha = _sha256(bundle_path)
        if (
            not bundle_raw_sidecar.is_file()
            or bundle_raw_sidecar.is_symlink()
            or bundle_raw_sidecar.read_text(encoding="utf-8").split()
            != [bundle_file_sha, bundle_path.name]
        ):
            raise ValueError("Production fresh bundle raw-file digest sidecar is inconsistent.")
        registry_lookup, registry_provenance = _load_publication_registries(
            bundle.get("submission_registries"), root
        )
    else:
        registry_lookup = None
        registry_provenance = []
    cohorts = bundle.get("cohorts")
    if not isinstance(cohorts, Mapping) or set(cohorts) != set(EXPECTED_COHORT_COUNTS):
        raise ValueError("Fresh bundle must contain exactly qualification, seed_ladder, and final.")
    frames: dict[str, pd.DataFrame] = {}
    cohort_provenance: dict[str, Any] = {}
    for cohort in EXPECTED_COHORT_COUNTS:
        descriptor = cohorts[cohort]
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Fresh cohort descriptor {cohort!r} must be a mapping.")
        frames[cohort], cohort_provenance[cohort] = _load_cohort(
            cohort=cohort,
            descriptor=descriptor,
            root=root,
            registry_lookup=registry_lookup,
            expected_manifest_count=(
                FRESH_PUBLICATION_MANIFEST_COUNTS[cohort]
                if production_publication
                else None
            ),
        )
        _validate_cohort_topology(frames[cohort], cohort)
    rows = pd.concat([frames[name] for name in EXPECTED_COHORT_COUNTS], ignore_index=True)
    if rows["result_path"].duplicated().any() or rows["media_path"].dropna().duplicated().any():
        raise ValueError("Scientific cohorts share result/media files and are not disjoint.")
    if registry_lookup is not None:
        used_registry_tasks = set(
            zip(rows["manifest_sha256"], rows["manifest_job_index"], strict=True)
        )
        if used_registry_tasks != set(registry_lookup):
            raise ValueError(
                "Production registries differ from the exact 1,188 campaign-row union."
            )
    selections, selection_provenance = _load_seed_selections(
        bindings=bundle.get("seed_selections"),
        ladder=frames["seed_ladder"],
        final=frames["final"],
        root=root,
    )
    selection_commit_provenance: dict[str, Any] | None = None
    if production_publication:
        cohort_binding = bundle.get("selection_cohort_commit")
        required_cohort_fields = {
            "contract",
            "commit_path",
            "commit_sha256",
            "commit_file_sha256",
            "selection_paths",
            "selection_count",
        }
        if not isinstance(cohort_binding, Mapping) or set(cohort_binding) != required_cohort_fields:
            raise ValueError("Production bundle lacks its exact selection-cohort commit binding.")
        commit_path = _resolve_input_path(cohort_binding["commit_path"], root)
        commit_sha = _require_sha256(
            cohort_binding["commit_sha256"], "selection cohort commit_sha256"
        )
        commit_file_sha = _require_sha256(
            cohort_binding["commit_file_sha256"], "selection cohort commit_file_sha256"
        )
        commit = _read_json(commit_path)
        raw_selection_paths = cohort_binding["selection_paths"]
        if not isinstance(raw_selection_paths, list):
            raise ValueError("Selection-cohort commit binding has no exact member paths.")
        committed_paths = tuple(_resolve_input_path(path, root) for path in raw_selection_paths)
        bound_paths = tuple(
            _resolve_input_path(binding.get("path"), root)
            for binding in bundle.get("seed_selections", [])
            if isinstance(binding, Mapping)
        )
        if (
            cohort_binding["contract"]
            != "finer_detailing_atomic_target_blind_selection_cohort_v1"
            or commit.get("commit_sha256") != commit_sha
            or _sha256(commit_path) != commit_file_sha
            or cohort_binding["selection_count"] != 36
            or len(committed_paths) != 36
            or len(set(committed_paths)) != 36
            or [str(path) for path in committed_paths] != raw_selection_paths
            or set(committed_paths) != set(bound_paths)
            or len(bound_paths) != 36
        ):
            raise ValueError("Selection-cohort commit/file/member binding changed or is incomplete.")
        selection_commit_provenance = {
            "contract": cohort_binding["contract"],
            "commit_path": str(commit_path),
            "commit_sha256": commit_sha,
            "commit_file_sha256": commit_file_sha,
            "selection_paths": [str(path) for path in committed_paths],
        }
    ledger_bindings = bundle.get("evidence_ledgers")
    if not isinstance(ledger_bindings, Mapping) or set(ledger_bindings) != set(LEDGER_CONTRACTS):
        raise ValueError("Fresh bundle must bind all three domain-separated evidence ledgers.")
    # Atomically published ledgers live beside the bundle and use portable
    # bundle-relative paths.  All campaign/raw sources remain absolute or
    # project-root relative and are still resolved against ``root``.
    ledger_root = bundle_path.parent if production_publication else root
    if production_publication:
        for name, binding in ledger_bindings.items():
            if not isinstance(binding, Mapping):
                raise ValueError(f"{name} ledger publication binding must be a mapping.")
            ledger_path = _resolve_input_path(binding.get("path"), ledger_root)
            _validate_raw_digest_sidecar(
                ledger_path, binding, ledger_root, f"{name} ledger"
            )
    lookup = {
        (str(row["cohort"]), str(row["manifest_sha256"]), int(row["manifest_job_index"])): row
        for row in rows.to_dict(orient="records")
    }
    objective, objective_provenance = _load_objective_temporal_ledger(
        binding=ledger_bindings["objective_temporal"], root=ledger_root, lookup=lookup
    )
    manual, manual_pairs, manual_provenance = _load_manual_semantic_ledger(
        binding=ledger_bindings["manual_semantic"], root=ledger_root, lookup=lookup
    )
    shapley, shapley_provenance = _load_shapley_diagnostics_ledger(
        binding=ledger_bindings["shapley_diagnostics"], root=ledger_root, lookup=lookup
    )
    if production_publication:
        raw_sources = bundle.get("raw_ledger_sources")
        if not isinstance(raw_sources, Mapping) or set(raw_sources) != set(LEDGER_CONTRACTS):
            raise ValueError("Production bundle must bind exactly three raw ledger sources.")
        observed_raw = {
            "objective_temporal": objective_provenance["raw_source_document"],
            "manual_semantic": manual_provenance["raw_source_document"],
            "shapley_diagnostics": shapley_provenance["raw_source_document"],
        }
        if any(value is None for value in observed_raw.values()) or dict(raw_sources) != observed_raw:
            raise ValueError(
                "Production bundle raw-ledger bindings differ from normalized ledger sources."
            )
    rows = rows.sort_values(
        ["cohort", "model_name", "prompt_id", "condition_family", "seed"]
    ).reset_index(drop=True)
    provenance = {
        "bundle": {
            "path": str(bundle_path),
            "file_sha256": _sha256(bundle_path),
            "document_sha256": bundle_digest,
            "sidecar_path": str(bundle_sidecar),
            "sidecar_sha256": _sha256(bundle_sidecar),
        },
        "cohorts": cohort_provenance,
        "seed_selections": selection_provenance,
        "selection_cohort_commit": selection_commit_provenance,
        "submission_registries": registry_provenance,
        "evidence_ledgers": {
            "objective_temporal": objective_provenance,
            "manual_semantic": manual_provenance,
            "shapley_diagnostics": shapley_provenance,
        },
    }
    data = FreshPlotData(
        rows=rows,
        selections=selections,
        objective_temporal=objective,
        manual_semantic=manual,
        manual_pairs=manual_pairs,
        shapley_diagnostics=shapley,
        provenance=provenance,
    )
    if data.snapshot()["cohorts"] != EXPECTED_COHORT_COUNTS:
        raise RuntimeError("Internal fresh plot-data snapshot arithmetic changed unexpectedly.")
    return data


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(
        paths[1],
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None, "Creator": "hierasafe-flow"},
    )
    plt.close(fig)
    return paths


def plot_status(jobs: pd.DataFrame, output_dir: Path) -> list[Path]:
    counts = (
        jobs.groupby(["model_name", "status"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATUS_ORDER, fill_value=0)
        .sort_index()
    )
    counts.to_csv(output_dir / "phase_a_status_by_model.csv")
    fig, ax = plt.subplots(figsize=(15, 6.5))
    counts.plot(
        kind="bar",
        stacked=True,
        color=[STATUS_COLORS[name] for name in counts.columns],
        ax=ax,
        width=0.82,
    )
    ax.set_title("Phase-A qualification status by model (330 logical rows)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Logical experiment count")
    ax.legend(
        title="Latest status",
        ncol=1,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
    )
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    return _save_figure(fig, output_dir, "phase_a_status_by_model")


def plot_resolution_heatmap(jobs: pd.DataFrame, output_dir: Path) -> list[Path]:
    jobs = jobs.copy()
    jobs["valid_resolution"] = jobs["status"].isin(["completed", "not_supported"])
    matrix = jobs.pivot_table(
        index="model_name",
        columns="prompt_id",
        values="valid_resolution",
        aggfunc="mean",
        fill_value=0.0,
    ).sort_index()
    prompt_labels = {
        "01_sad_young_girl": "P01\nGirl",
        "02_angry_old_man": "P02\nMan",
        "03_empty_outdoor_mall": "P03\nMall",
    }
    matrix = matrix.rename(columns=prompt_labels)
    matrix.to_csv(output_dir / "phase_a_valid_resolution_fraction.csv")
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="YlGn",
        linewidths=0.5,
        cbar_kws={"label": "Completed media or truthful unsupported fraction"},
        ax=ax,
    )
    ax.set_title("Phase-A valid resolution fraction by prompt and model")
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Model")
    fig.tight_layout()
    return _save_figure(fig, output_dir, "phase_a_valid_resolution_heatmap")


def plot_runtime(timings: pd.DataFrame, output_dir: Path) -> list[Path]:
    timings.to_csv(output_dir / "phase_a_runtime_records.csv", index=False)
    plotted = timings[timings["status"].isin(["completed", "failed"])].copy()
    if plotted.empty:
        raise ValueError("No completed/failed runtime records are available.")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    for ax, task in zip(axes, ["text_to_image", "text_to_video"], strict=True):
        subset = plotted[plotted["task"] == task]
        if subset.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=subset,
            x="model_name",
            y="wall_seconds",
            hue="status",
            hue_order=["completed", "failed"],
            palette=STATUS_COLORS,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            data=subset,
            x="model_name",
            y="wall_seconds",
            hue="status",
            hue_order=["completed", "failed"],
            palette=STATUS_COLORS,
            dodge=True,
            alpha=0.55,
            size=3,
            legend=False,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_title(task.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Wall time (seconds, log scale)")
        ax.tick_params(axis="x", rotation=30)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:2], labels[:2], title="Result status", frameon=False)
    fig.suptitle("Phase-A runtime distribution (failures shown separately)", y=1.01)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "phase_a_runtime_distribution")


def plot_temporal_quality(video: pd.DataFrame, output_dir: Path) -> list[Path]:
    video.to_csv(output_dir / "phase_a_full_video_diagnostics.csv", index=False)
    metrics = [
        "frozen_candidate",
        "terminal_frozen_candidate",
        "near_periodic_candidate",
        "cut_candidate",
        "cadence_candidate",
    ]
    long = video.melt(
        id_vars=["condition_id", "model_name", "condition_kind"],
        value_vars=metrics,
        var_name="diagnostic",
        value_name="flagged",
    )
    rates = (
        long.groupby(["model_name", "diagnostic"], observed=True)["flagged"]
        .mean()
        .reset_index()
    )
    rates.to_csv(output_dir / "phase_a_temporal_quality_rates.csv", index=False)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sns.barplot(data=rates, x="model_name", y="flagged", hue="diagnostic", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(f"Machine temporal-diagnostic rates across {len(video)} completed videos")
    ax.set_xlabel("Video model")
    ax.set_ylabel("Fraction flagged")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Diagnostic (candidate, not semantic verdict)", frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "phase_a_temporal_quality_rates")


def plot_prompt3_motion(motion: pd.DataFrame, output_dir: Path) -> list[Path]:
    motion.to_csv(output_dir / "phase_a_prompt3_motion_records.csv", index=False)
    plotted = motion.copy()
    plotted["image_plane_path_plus_one_px"] = plotted["image_plane_path_length_px"] + 1.0
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sns.stripplot(
        data=plotted,
        x="model_name",
        y="image_plane_path_plus_one_px",
        hue="condition_kind",
        dodge=True,
        jitter=0.18,
        size=6,
        alpha=0.75,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_title("Prompt-3 global image-plane path diagnostic")
    ax.set_xlabel("Video model")
    ax.set_ylabel("Cumulative median-flow path + 1 px (log scale)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Condition", frameon=False, ncol=2)
    ax.text(
        0.01,
        0.01,
        "Image-plane flow is a diagnostic; it does not prove physical camera or escalator motion.",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )
    fig.tight_layout()
    return _save_figure(fig, output_dir, "phase_a_prompt3_motion_diagnostics")


def _condition_slot(row: Mapping[str, Any]) -> str:
    family = str(row["condition_family"])
    fixed = {
        "baseline": "01 baseline",
        "native_negative": "02 native negative",
        "conceptsteer_full": "03 ordinary full",
        "shapley_full": "04 Shapley full",
    }
    if family in fixed:
        return fixed[family]
    pair_id = family.split("__", maxsplit=1)[1]
    pair_index = PAIR_IDS_BY_PROMPT[str(row["prompt_id"])].index(pair_id) + 1
    if family.startswith("conceptsteer_single__"):
        return f"{4 + pair_index:02d} ordinary exact-one {pair_index}"
    if family.startswith("shapley_single__"):
        return f"{9 + pair_index:02d} Shapley exact-one {pair_index}"
    raise ValueError(f"Cannot assign final condition slot for {family!r}.")


def _fresh_export_tables(data: FreshPlotData, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    rows = data.rows.copy()
    rows["active_pair_ids"] = rows["active_pair_ids"].map(
        lambda values: json.dumps(list(values), separators=(",", ":"))
    )
    path = output_dir / "all_reconciled_campaign_rows.csv"
    rows.to_csv(path, index=False)
    paths.append(path)

    structural = (
        data.rows.groupby(["cohort", "outcome"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=list(EXPECTED_COHORT_COUNTS), fill_value=0)
        .reindex(columns=["media_validated", "truthful_unsupported"], fill_value=0)
    )
    structural["logical"] = structural.sum(axis=1)
    path = output_dir / "structural_media_cohort_counts.csv"
    structural.to_csv(path)
    paths.append(path)

    final = data.rows[data.rows["cohort"] == "final"].copy()
    final["condition_slot"] = final.apply(_condition_slot, axis=1)
    final["pair_id"] = final["active_pair_ids"].map(
        lambda values: values[0] if len(values) == 1 else ""
    )
    final = final.sort_values(["model_name", "prompt_id", "condition_slot"])
    coverage_columns = [
        "model_name",
        "prompt_id",
        "seed",
        "condition_slot",
        "condition_family",
        "method",
        "ablation",
        "pair_id",
        "condition_id",
        "status",
        "outcome",
        "media_path",
        "media_sha256",
        "manifest_sha256",
        "manifest_job_index",
        "result_sha256",
    ]
    path = output_dir / "final_model_prompt_condition_coverage.csv"
    final[coverage_columns].to_csv(path, index=False)
    paths.append(path)
    path = output_dir / "final_exact_one_360_media.csv"
    final[final["ablation"] == "exact_one"][coverage_columns].to_csv(path, index=False)
    paths.append(path)

    path = output_dir / "target_blind_seed_selections_36.csv"
    data.selections.to_csv(path, index=False)
    paths.append(path)
    path = output_dir / "objective_temporal_records.csv"
    data.objective_temporal.to_csv(path, index=False)
    paths.append(path)
    path = output_dir / "manual_semantic_preservation_gender_records.csv"
    data.manual_semantic.to_csv(path, index=False)
    paths.append(path)
    path = output_dir / "manual_active_pair_outcomes.csv"
    data.manual_pairs.to_csv(path, index=False)
    paths.append(path)
    path = output_dir / "shapley_protocol2_diagnostics.csv"
    data.shapley_diagnostics.to_csv(path, index=False)
    paths.append(path)
    path = output_dir / "runtime_records.csv"
    data.rows[
        [
            "row_key",
            "cohort",
            "model_name",
            "prompt_id",
            "task",
            "condition_family",
            "method",
            "ablation",
            "status",
            "wall_seconds",
            "timing_path",
            "timing_sha256",
        ]
    ].to_csv(path, index=False)
    paths.append(path)
    snapshot_path = output_dir / "plot_data_snapshot.json"
    _write_json(snapshot_path, data.snapshot())
    paths.append(snapshot_path)
    provenance_path = output_dir / "input_provenance.json"
    _write_json(provenance_path, data.provenance)
    paths.append(provenance_path)
    return paths


def plot_fresh_structural_coverage(data: FreshPlotData, output_dir: Path) -> list[Path]:
    counts = (
        data.rows.groupby(["cohort", "outcome"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=list(EXPECTED_COHORT_COUNTS), fill_value=0)
        .reindex(columns=["media_validated", "truthful_unsupported"], fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 5.8))
    counts.plot(
        kind="bar",
        stacked=True,
        color=["#2E8B57", "#64748B"],
        ax=ax,
        width=0.68,
    )
    for index, cohort in enumerate(counts.index):
        total = int(counts.loc[cohort].sum())
        ax.text(index, total + 5, str(total), ha="center", va="bottom", fontsize=10)
    ax.set_title("Fail-closed structural/media reconciliation by scientific cohort")
    ax.set_xlabel("Cohort (counts are not interchangeable)")
    ax.set_ylabel("Logical rows")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(
        ["Validated media", "Truthful native-negative unsupported"],
        title="Structural outcome (not semantic success)",
        frameon=False,
    )
    ax.set_ylim(0, max(counts.sum(axis=1)) * 1.14)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "fresh_structural_media_cohort_counts")


def plot_final_condition_coverage(data: FreshPlotData, output_dir: Path) -> list[Path]:
    final = data.rows[data.rows["cohort"] == "final"].copy()
    final["condition_slot"] = final.apply(_condition_slot, axis=1)
    final["axis"] = final["model_name"] + " | " + final["prompt_id"].str[:2]
    final["structural_code"] = np.where(final["media_valid"], 1.0, 0.0)
    matrix = final.pivot(index="axis", columns="condition_slot", values="structural_code")
    matrix = matrix.sort_index().reindex(sorted(matrix.columns), axis=1)
    if matrix.shape != (36, 14) or matrix.isna().any().any():
        raise RuntimeError(f"Final condition matrix changed shape or has gaps: {matrix.shape}.")
    fig, ax = plt.subplots(figsize=(19, 14))
    sns.heatmap(
        matrix,
        vmin=0,
        vmax=1,
        cmap=sns.color_palette(["#64748B", "#2E8B57"], as_cmap=True),
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "0 = truthful unsupported, 1 = validated media"},
        ax=ax,
    )
    ax.set_title(
        "Final selected-seed model × prompt × condition structural coverage\n"
        "(504 logical rows; exact-one cells are the embedded 360-media subset)"
    )
    ax.set_xlabel("Registered condition slot")
    ax.set_ylabel("Model | prompt")
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "final_model_prompt_condition_coverage")


def plot_objective_temporal_evidence(data: FreshPlotData, output_dir: Path) -> list[Path]:
    temporal = data.objective_temporal.copy()
    metrics = [
        "automatic_gate_pass",
        "freeze_candidate",
        "near_periodic_candidate",
        "cut_candidate",
        "cadence_candidate",
    ]
    long = temporal.melt(
        id_vars=["cohort", "model_name", "condition_family"],
        value_vars=metrics,
        var_name="objective_metric",
        value_name="fraction",
    )
    rates = (
        long.groupby(["cohort", "model_name", "objective_metric"], observed=True)["fraction"]
        .mean()
        .reset_index()
    )
    rates.to_csv(output_dir / "objective_temporal_rates.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(15, 16), sharex=True)
    for ax, cohort in zip(axes, EXPECTED_COHORT_COUNTS, strict=True):
        subset = rates[rates["cohort"] == cohort]
        sns.barplot(
            data=subset,
            x="model_name",
            y="fraction",
            hue="objective_metric",
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_title(cohort.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Fraction")
        ax.legend(
            title="Objective gate/candidate flag\n(not semantic motion)",
            frameon=False,
            ncol=3,
        )
    axes[-1].tick_params(axis="x", rotation=25)
    fig.suptitle("Objective temporal gates and candidate diagnostics", y=1.005)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "objective_temporal_evidence")


def plot_manual_semantic_evidence(data: FreshPlotData, output_dir: Path) -> list[Path]:
    manual = data.manual_semantic[data.manual_semantic["cohort"] == "final"].copy()
    metrics = ["semantic_success", "source_fidelity", "non_target_preservation"]
    long = manual.melt(
        id_vars=["model_name", "prompt_id", "method", "ablation"],
        value_vars=metrics,
        var_name="manual_metric",
        value_name="passed",
    )
    rates = (
        long.groupby(["method", "ablation", "manual_metric"], observed=True)["passed"]
        .mean()
        .reset_index()
    )
    rates.to_csv(output_dir / "final_manual_semantic_preservation_rates.csv", index=False)
    person = manual[manual["prompt_id"].isin(["01_sad_young_girl", "02_angry_old_man"])]
    gender = (
        person.groupby(
            ["model_name", "prompt_id", "method", "ablation"], observed=True
        )["gender_preserved"]
        .mean()
        .reset_index()
    )
    gender.to_csv(output_dir / "final_manual_gender_preservation_rates.csv", index=False)
    fig, axes = plt.subplots(3, 1, figsize=(17, 18))
    rates["method_ablation"] = rates["method"] + " | " + rates["ablation"]
    sns.barplot(
        data=rates,
        x="method_ablation",
        y="passed",
        hue="manual_metric",
        ax=axes[0],
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Final manual semantic and preservation decisions")
    axes[0].set_xlabel("Method | ablation")
    axes[0].set_ylabel("Human-reviewed pass fraction")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(title="Manual metric", frameon=False)
    gender["method_ablation"] = gender["method"] + " | " + gender["ablation"]
    for ax, prompt_id in zip(
        axes[1:], ["01_sad_young_girl", "02_angry_old_man"], strict=True
    ):
        subset = gender[gender["prompt_id"] == prompt_id]
        sns.barplot(
            data=subset,
            x="model_name",
            y="gender_preserved",
            hue="method_ablation",
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_title(
            f"{prompt_id}: gender preservation by method and ablation "
            "(manual evidence only)"
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Human-reviewed preservation fraction")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Method | ablation", frameon=False, ncol=2)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "manual_semantic_preservation_gender")


def plot_method_ablation_evidence(data: FreshPlotData, output_dir: Path) -> list[Path]:
    pairs = data.manual_pairs[data.manual_pairs["cohort"] == "final"].copy()
    rates = (
        pairs.groupby(["method", "ablation", "pair_id"], observed=True)[
            ["target_achieved", "selective_without_collateral_change"]
        ]
        .mean()
        .reset_index()
    )
    rates.to_csv(output_dir / "final_method_ablation_pair_rates.csv", index=False)
    long = rates.melt(
        id_vars=["method", "ablation", "pair_id"],
        value_vars=["target_achieved", "selective_without_collateral_change"],
        var_name="manual_pair_metric",
        value_name="fraction",
    )
    long["method_ablation"] = long["method"] + " | " + long["ablation"]
    fig, axes = plt.subplots(1, 2, figsize=(19, 7), sharey=True)
    for ax, metric in zip(
        axes,
        ["target_achieved", "selective_without_collateral_change"],
        strict=True,
    ):
        subset = long[long["manual_pair_metric"] == metric]
        sns.barplot(
            data=subset,
            x="pair_id",
            y="fraction",
            hue="method_ablation",
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Active concept pair")
        ax.set_ylabel("Human-reviewed fraction")
        ax.tick_params(axis="x", rotation=70, labelsize=8)
        ax.legend(title="Method | ablation", frameon=False)
    fig.suptitle("Final full-versus-exact-one intervention outcomes", y=1.01)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "final_method_ablation_pair_outcomes")


def plot_fresh_runtime(data: FreshPlotData, output_dir: Path) -> list[Path]:
    rows = data.rows.copy()
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    for ax, task in zip(axes, ["text_to_image", "text_to_video"], strict=True):
        subset = rows[rows["task"] == task]
        sns.boxplot(
            data=subset,
            x="model_name",
            y="wall_seconds",
            hue="cohort",
            showfliers=False,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_title(task.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Wall time (seconds, log scale)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="Scientific cohort", frameon=False)
    fig.suptitle("Bound per-attempt runtime distributions", y=1.01)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "fresh_bound_runtime_distributions")


def plot_shapley_diagnostics(data: FreshPlotData, output_dir: Path) -> list[Path]:
    diagnostics = data.shapley_diagnostics.copy()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    sns.boxplot(
        data=diagnostics,
        x="cohort",
        y="accepted_step_fraction",
        hue="ablation",
        ax=axes[0, 0],
    )
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Accepted step fraction")
    sns.boxplot(
        data=diagnostics,
        x="cohort",
        y="aggregate_score_decrease",
        hue="ablation",
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Aggregate score decrease")
    sns.boxplot(
        data=diagnostics,
        x="cohort",
        y="selected_coordinate_fraction",
        hue="ablation",
        ax=axes[1, 0],
    )
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Selected coordinate fraction")
    summary = (
        diagnostics.groupby(["cohort", "ablation"], observed=True)
        .agg(
            validation_pass_fraction=("validation_pass", "mean"),
            non_target_violation_mean=("non_target_violation_count", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "shapley_protocol2_summary.csv", index=False)
    sns.barplot(
        data=summary,
        x="cohort",
        y="validation_pass_fraction",
        hue="ablation",
        ax=axes[1, 1],
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Protocol-2 trace validation fraction")
    for ax in axes.flat:
        ax.set_xlabel("")
        ax.legend(title="Full / exact-one", frameon=False)
    fig.suptitle("Repaired Shapley protocol-2 diagnostics (not semantic outcomes)", y=1.01)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "shapley_protocol2_diagnostics")


def _run_fresh_evidence(
    *,
    root: Path,
    bundle_path: Path,
    output_dir: Path,
    plotting_environment: Mapping[str, Any] | None = None,
) -> int:
    # Reconcile and rehash every source before creating even a temporary plot tree.
    plotting_environment = (
        dict(plotting_environment)
        if plotting_environment is not None
        else _plotting_environment_provenance()
    )
    data = load_fresh_plot_data(bundle_path, root)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite fresh plot output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        sns.set_theme(style="whitegrid", context="notebook")
        tables = _fresh_export_tables(data, temporary)
        figures: list[Path] = []
        figures.extend(plot_fresh_structural_coverage(data, temporary))
        figures.extend(plot_final_condition_coverage(data, temporary))
        figures.extend(plot_objective_temporal_evidence(data, temporary))
        figures.extend(plot_manual_semantic_evidence(data, temporary))
        figures.extend(plot_method_ablation_evidence(data, temporary))
        figures.extend(plot_fresh_runtime(data, temporary))
        figures.extend(plot_shapley_diagnostics(data, temporary))
        script_path = Path(__file__).resolve()
        produced = sorted(path for path in temporary.iterdir() if path.is_file())
        table_files = [
            path for path in produced if path.suffix.lower() in {".csv", ".json"}
        ]
        metadata = {
            "schema_version": 2,
            "scope": "fresh_qualification_seed_ladder_and_selected_seed_final_evidence",
            "plot_script": str(script_path),
            "plot_script_sha256": _sha256(script_path),
            "plotting_environment": plotting_environment,
            "input_bundle": data.provenance["bundle"],
            "plot_data_snapshot": data.snapshot(),
            "figure_count": len(figures) // 2,
            "figure_file_count": len(figures),
            "primary_export_file_count": len(tables),
            "table_and_data_file_count": len(table_files),
            "artifacts": [
                {
                    "path": path.name,
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in produced
            ],
            "limitations": [
                "Validated scheduler/result/media state is structural evidence, never semantic success.",
                "Objective temporal gates and candidate flags do not establish physical or semantic "
                "motion plausibility; those judgments remain in the manual ledger.",
                "Semantic, preservation, and gender rates derive only from the immutable manual "
                "ledger and never from completion status or machine diagnostics.",
                "Shapley numerical/trace validation is not equivalent to target achievement or "
                "selective semantic success.",
                "Qualification, target-blind ladder, seed selections, and final rows are distinct "
                "scientific products and are not pooled as replicates.",
                "The 360 exact-one media are an embedded subset of the 489 final media and must not "
                "be added to the final denominator again.",
            ],
        }
        _write_json(temporary / "plot_manifest.json", metadata)
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "figure_count": len(figures) // 2,
                "plot_manifest": str(output_dir / "plot_manifest.json"),
                "plot_data_snapshot": data.snapshot(),
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--audit-json",
        default=(
            "debugging/audits/finer_detailing_20260719/"
            "qualification_phase_a_live_20260720_103750_rerun.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=FRESH_DEFAULT_OUTPUT_FORBIDDEN,
    )
    parser.add_argument("--full-video-dir", action="append", default=[])
    parser.add_argument("--motion-dir", action="append", default=[])
    parser.add_argument(
        "--fresh-evidence-bundle",
        help=(
            "Opt in to the fail-closed fresh 396/288/36/504 evidence path. The bundle and "
            "all bound manifests, results, timings, media, selections, and ledgers must be "
            "complete and immutable before any output is created."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    audit_path = (root / args.audit_json).resolve()
    output_dir = (root / args.output_dir).resolve()
    plotting_environment = _plotting_environment_provenance()
    if args.fresh_evidence_bundle:
        if args.output_dir == FRESH_DEFAULT_OUTPUT_FORBIDDEN:
            raise ValueError(
                "Fresh evidence plotting requires an explicit new --output-dir; the Phase-A v3 "
                "reproducer path is reserved."
            )
        bundle_path = (root / args.fresh_evidence_bundle).resolve()
        return _run_fresh_evidence(
            root=root,
            bundle_path=bundle_path,
            output_dir=output_dir,
            plotting_environment=plotting_environment,
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    full_video_dirs = [
        (root / value).resolve()
        for value in (args.full_video_dir or DEFAULT_FULL_VIDEO_DIRS)
    ]
    motion_dirs = [
        (root / value).resolve() for value in (args.motion_dir or DEFAULT_MOTION_DIRS)
    ]

    sns.set_theme(style="whitegrid", context="notebook")
    jobs = load_latest_jobs(audit_path)
    timings = attach_timings(jobs)
    video = load_full_video_evidence(full_video_dirs)
    motion = load_motion_evidence(motion_dirs)

    artifacts: list[Path] = []
    artifacts.extend(plot_status(jobs, output_dir))
    artifacts.extend(plot_resolution_heatmap(jobs, output_dir))
    artifacts.extend(plot_runtime(timings, output_dir))
    artifacts.extend(plot_temporal_quality(video, output_dir))
    artifacts.extend(plot_prompt3_motion(motion, output_dir))

    script_path = Path(__file__).resolve()
    produced_files = sorted(path for path in output_dir.iterdir() if path.is_file())
    metadata = {
        "schema_version": 1,
        "scope": "phase_a_execution_and_machine_diagnostics_only",
        "plot_script": str(script_path),
        "plot_script_sha256": _sha256(script_path),
        "plotting_environment": plotting_environment,
        "audit_json": str(audit_path),
        "audit_json_sha256": _sha256(audit_path),
        "logical_job_count": int(len(jobs)),
        "timing_record_count": int(len(timings)),
        "full_video_evidence_count": int(len(video)),
        "prompt3_motion_evidence_count": int(len(motion)),
        "full_video_directories": [str(path) for path in full_video_dirs],
        "motion_directories": [str(path) for path in motion_dirs],
        "figure_count": len(artifacts) // 2,
        "figure_file_count": len(artifacts),
        "artifacts": [
            {"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in produced_files
        ],
        "limitations": [
            "Machine diagnostics are candidates and do not establish semantic success.",
            "Image-plane flow does not prove physical camera, escalator, or stair motion.",
            "No semantic steering, preservation, identity, or gender rate is plotted because the "
            "historical media lack one unified structured human-review ledger.",
            "Old Shapley outputs use the invalidated protocol-v1 intervention and are not efficacy "
            "evidence for repaired Shapley protocol 2.",
        ],
    }
    metadata_path = output_dir / "plot_manifest.json"
    _write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "figure_count": len(artifacts) // 2,
                "figure_file_count": len(artifacts),
                "full_video_evidence_count": len(video),
                "prompt3_motion_evidence_count": len(motion),
                "plot_manifest": str(metadata_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
