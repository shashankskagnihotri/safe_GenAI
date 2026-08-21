"""Exact topology controls for the final finer-detailing campaign.

This module deliberately separates *preview* from *production*.  Preview
builders may materialize manifests while a configured text-to-video route is
still an explicitly authorized pilot.  Every publication entry point and the
production validator reject such a route.

The scientific topology is immutable:

* eight target-blind, baseline-only seed manifests (seeds 0..7), each with all
  36 prompt/model axes, under the canonical qualification output root;
* exactly one canonical target-blind selection record for every axis; and
* two final manifests per axis: an eight-row standard family and a six-row
  Shapley family.  Their union is 504 logical conditions, 489 media, 15
  truthful native-negative unsupported declarations, and 360 exact-one media.

Generation semantics remain owned by ``finer_detailing_correction``.  The
builders below call that benchmark's parser and manifest builder instead of
reconstructing jobs.  Publication uses its no-overwrite immutable manifest
writer inside a private sibling tree, authenticates the complete cohort, then
publishes immutable members with no-replace hard links and the cohort commit
last.  Selection
publication remains owned by
``target_blind_seed_selection.write_selection_record_immutable``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hierasafe_flow.benchmarks.finer_detailing_correction import (
    BENCHMARK_NAME,
    MODEL_NAMES,
    NATIVE_NEGATIVE_UNSUPPORTED_REASONS,
    PAIR_IDS_BY_PROMPT,
    PROMPT_IDS,
    SHAPLEY_CONFIG,
    SHAPLEY_PROVENANCE,
    build_manifest as build_benchmark_manifest,
    build_parser as build_benchmark_parser,
    is_flux1_job_v3,
    manifest_digest,
    read_manifest_for_audit,
    reopen_completed_flux1_output_v3,
    write_manifest_immutable,
)
from hierasafe_flow.evaluation import target_blind_seed_selection as target_blind
from hierasafe_flow.evaluation import flux1_dual_view_jobs_v3 as flux_v3
from hierasafe_flow.evaluation import finer_detailing_selection_cohort as selection_cohort


CAMPAIGN_SCHEMA_VERSION = 1
CAMPAIGN_CONTRACT = "finer_detailing_production_campaign_topology_v1"
FLUX1_COMPLETED_CAMPAIGN_CONTRACT = "finer_detailing_flux1_completed_ladder_final_receipt_v3"

_METHOD_KIND_ALIASES: dict[str, str] = {}


def _normalize_variant_kind(kind: str) -> str:
    return _METHOD_KIND_ALIASES.get(kind, kind)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


SEEDS = tuple(range(8))
AXES = tuple((prompt_id, model_name) for prompt_id in PROMPT_IDS for model_name in MODEL_NAMES)

VIDEO_MODELS = frozenset(
    {
        "cogvideox_5b",
        "hunyuan_video",
        "joyai_echo",
        "ltx_23",
        "wan22_t2v_a14b",
    }
)
TASK_BY_MODEL = {
    model_name: ("text_to_video" if model_name in VIDEO_MODELS else "text_to_image")
    for model_name in MODEL_NAMES
}

LADDER_OUTPUT_ROOT_RELATIVE = target_blind.CANDIDATE_ROOT_RELATIVE
FINAL_OUTPUT_ROOT_RELATIVE = Path("outputs/finer_detailing_correction_selected_seed")
LADDER_MANIFEST_ROOT_RELATIVE = Path("debugging/manifests/finer_detailing_seed_qualification_v1")
FINAL_MANIFEST_ROOT_RELATIVE = Path(
    "debugging/manifests/finer_detailing_correction_selected_seed_v1"
)

LADDER_VARIATIONS = ("01_baseline",)
STANDARD_VARIATIONS = (
    "01_baseline",
    "02_negative_prompt",
    "03_concept_steering",
    "05_concept_steering_single_pair",
)
SHAPLEY_VARIATIONS = (
    "04_shapley_concept_steering",
    "06_shapley_concept_steering_single_pair",
)
FINAL_FAMILIES = ("standard", "shapley")

EXPECTED_LADDER_MANIFESTS = 8
EXPECTED_LADDER_ROWS_PER_MANIFEST = 36
EXPECTED_LADDER_MEDIA = 288
EXPECTED_SELECTION_RECORDS = 36
EXPECTED_FINAL_MANIFESTS = 72
EXPECTED_STANDARD_ROWS_PER_AXIS = 8
EXPECTED_SHAPLEY_ROWS_PER_AXIS = 6
EXPECTED_FINAL_LOGICAL = 504
EXPECTED_FINAL_MEDIA = 489
EXPECTED_FINAL_UNSUPPORTED = 15
EXPECTED_FINAL_EXACT_ONE_MEDIA = 360

COHORT_COMMIT_FILENAME = "campaign_cohort_commit.json"
COHORT_COMMIT_CONTRACT = "finer_detailing_atomic_campaign_cohort_v1"


ManifestReader = Callable[[Path, Path], Mapping[str, Any]]
SelectionReader = Callable[[Path, Path], Mapping[str, Any]]
AxisFinalValidator = Callable[..., Mapping[str, Any]]


def ladder_manifest_path(root: str | Path, seed: int) -> Path:
    """Return the only admitted manifest path for one ladder seed."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise ValueError("Ladder seed must be one exact integer in 0..7.")
    return (
        Path(root).expanduser().resolve() / LADDER_MANIFEST_ROOT_RELATIVE / f"seed_{seed:08d}.json"
    )


def final_manifest_path(root: str | Path, prompt_id: str, model_name: str, family: str) -> Path:
    """Return the only admitted final manifest path for an axis/family."""

    _validate_axis(prompt_id, model_name)
    if family not in FINAL_FAMILIES:
        raise ValueError(f"Unknown final manifest family: {family!r}.")
    return (
        Path(root).expanduser().resolve()
        / FINAL_MANIFEST_ROOT_RELATIVE
        / f"{prompt_id}__{model_name}__{family}.json"
    )


def canonical_ladder_manifest_paths(root: str | Path) -> tuple[Path, ...]:
    return tuple(ladder_manifest_path(root, seed) for seed in SEEDS)


def canonical_selection_paths(root: str | Path) -> tuple[Path, ...]:
    return tuple(
        target_blind.selection_output_path(root, prompt_id, model_name)
        for prompt_id, model_name in AXES
    )


def canonical_final_manifest_paths(root: str | Path) -> tuple[Path, ...]:
    return tuple(
        final_manifest_path(root, prompt_id, model_name, family)
        for prompt_id, model_name in AXES
        for family in FINAL_FAMILIES
    )


def _validate_axis(prompt_id: str, model_name: str) -> None:
    if (prompt_id, model_name) not in AXES:
        raise ValueError(
            f"Unknown finer-detailing prompt/model axis: {prompt_id!r}/{model_name!r}."
        )


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Campaign path escapes the project root: {value!r}.")
    return resolved


def _resolve_exact_path_set(
    supplied: Sequence[str | Path], expected: Sequence[Path], *, label: str, root: Path
) -> tuple[Path, ...]:
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
        raise ValueError(f"{label} paths must be a sequence.")
    resolved = tuple(_resolve_path(path, root) for path in supplied)
    duplicates = sorted(str(path) for path, count in Counter(resolved).items() if count != 1)
    if duplicates:
        raise ValueError(f"{label} paths contain duplicates: {duplicates}.")
    expected_tuple = tuple(path.resolve() for path in expected)
    missing = sorted(str(path) for path in set(expected_tuple) - set(resolved))
    extra = sorted(str(path) for path in set(resolved) - set(expected_tuple))
    if missing or extra or len(resolved) != len(expected_tuple):
        raise ValueError(
            f"{label} paths differ from the exact canonical set: missing={missing}, extra={extra}."
        )
    # Canonical order prevents caller order from influencing validation output.
    return expected_tuple


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a timezone-aware ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not ISO-8601: {value!r}.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include an explicit timezone.")
    return parsed


def _read_selection(path: Path, root: Path) -> Mapping[str, Any]:
    return target_blind.read_selection_record(path, root=root)


def _read_committed_selection_cohort(
    root: Path,
) -> tuple[
    dict[tuple[str, str], Mapping[str, Any]],
    tuple[Path, ...],
    dict[str, Any],
]:
    """Reopen the only selection cohort authorized for production construction.

    The selection-cohort reader authenticates the physical commit-last tree,
    every member and its bound ladder.  These additional exact-shape checks
    keep this production boundary fail-closed if that reader's return contract
    ever drifts.  Callers receive a compact commit proof so a publication can
    bind and compare the same cohort at every precommit revalidation point.
    """

    snapshot = selection_cohort.read_selection_cohort(root=root)
    expected_paths = canonical_selection_paths(root)
    expected_path_strings = [str(path) for path in expected_paths]
    expected_commit_path = selection_cohort.selection_cohort_commit_path(root)
    selections = snapshot.get("selections")
    commit_sha256 = snapshot.get("commit_sha256")
    commit_file_sha256 = snapshot.get("commit_file_sha256")
    if (
        snapshot.get("status") != "valid"
        or snapshot.get("contract") != selection_cohort.COHORT_CONTRACT
        or snapshot.get("selection_paths") != expected_path_strings
        or snapshot.get("selection_count") != EXPECTED_SELECTION_RECORDS
        or snapshot.get("candidate_bindings") != EXPECTED_LADDER_MEDIA
        or str(snapshot.get("commit_path", "")) != str(expected_commit_path)
        or not isinstance(commit_sha256, str)
        or len(commit_sha256) != 64
        or any(character not in "0123456789abcdef" for character in commit_sha256)
        or not isinstance(commit_file_sha256, str)
        or len(commit_file_sha256) != 64
        or any(character not in "0123456789abcdef" for character in commit_file_sha256)
        or not isinstance(selections, Mapping)
        or set(selections) != set(AXES)
        or any(not isinstance(selections[axis], Mapping) for axis in AXES)
    ):
        raise ValueError(
            "Production final campaign requires the exact committed canonical 36-selection cohort."
        )
    loaded = {axis: deepcopy(dict(selections[axis])) for axis in AXES}
    axis_by_path = dict(zip(expected_paths, AXES, strict=True))

    def committed_reader(path: Path, _root: Path) -> Mapping[str, Any]:
        return deepcopy(dict(loaded[axis_by_path[_resolve_path(path, root)]]))

    loaded, _ = _load_selections(
        expected_paths,
        root=root,
        selection_reader=committed_reader,
    )
    selections_sha256 = _canonical_sha256(
        [
            {
                "prompt_id": prompt_id,
                "model_name": model_name,
                "selection": loaded[(prompt_id, model_name)],
            }
            for prompt_id, model_name in AXES
        ]
    )
    proof = {
        "contract": selection_cohort.COHORT_CONTRACT,
        "commit_path": str(expected_commit_path),
        "commit_sha256": commit_sha256,
        "commit_file_sha256": commit_file_sha256,
        "selections_sha256": selections_sha256,
        "selection_count": EXPECTED_SELECTION_RECORDS,
        "candidate_bindings": EXPECTED_LADDER_MEDIA,
    }
    return loaded, expected_paths, proof


def _selection_mapping_reader(
    selections: Mapping[tuple[str, str], Mapping[str, Any]], root: Path
) -> SelectionReader:
    """Return a private reader over an already authenticated cohort snapshot."""

    def reader(path: Path, _root: Path) -> Mapping[str, Any]:
        return deepcopy(dict(selections[_axis_for_selection_path(path, root)]))

    return reader


def _mapping_manifest_reader(
    manifests: Mapping[Path, Mapping[str, Any]], root: Path
) -> ManifestReader:
    indexed = {_resolve_path(path, root): value for path, value in manifests.items()}

    def reader(path: Path, _root: Path) -> Mapping[str, Any]:
        resolved = _resolve_path(path, root)
        if resolved not in indexed:
            raise FileNotFoundError(f"No planned manifest exists at {resolved}.")
        return deepcopy(indexed[resolved])

    return reader


def _manifest_header(
    manifest: Mapping[str, Any], *, path: Path, expected_jobs: int
) -> list[Mapping[str, Any]]:
    if manifest.get("benchmark") != BENCHMARK_NAME:
        raise ValueError(f"Manifest benchmark mismatch: {path}.")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != expected_jobs:
        raise ValueError(
            f"Manifest {path} must contain exactly {expected_jobs} jobs; "
            f"got {len(jobs) if isinstance(jobs, list) else 'non-list'}."
        )
    if manifest.get("num_jobs") != expected_jobs:
        raise ValueError(f"Manifest {path} top-level job count is inconsistent.")
    digest = manifest.get("manifest_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"Manifest {path} lacks a canonical SHA-256 binding.")
    _parse_timestamp(manifest.get("created_at_utc"), f"manifest {path} created_at_utc")
    return jobs


def _validate_counts(
    manifest: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]], *, path: Path
) -> tuple[int, int]:
    media = sum(job.get("expected_media") is True for job in jobs)
    unsupported = len(jobs) - media
    if any(job.get("expected_media") not in {True, False} for job in jobs):
        raise ValueError(f"Manifest {path} contains a non-boolean expected_media value.")
    if (
        manifest.get("expected_media_jobs") != media
        or manifest.get("expected_not_supported_jobs") != unsupported
    ):
        raise ValueError(f"Manifest {path} media/unsupported totals are inconsistent.")
    return media, unsupported


def _validate_temporal_routes(
    manifest: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]], *, path: Path, production: bool
) -> int:
    pilot_jobs = 0
    for job in jobs:
        model_name = str(job.get("model_name", ""))
        expected_task = TASK_BY_MODEL.get(model_name)
        task = str((job.get("generation") or {}).get("task", ""))
        if task != expected_task:
            raise ValueError(
                f"Job {job.get('condition_id')} has task {task!r}; expected {expected_task!r}."
            )
        snapshot = job.get("temporal_protocol_snapshot")
        if expected_task == "text_to_image":
            if snapshot is not None or "temporal_pilot_authorized" in job:
                raise ValueError(f"Image job {job.get('condition_id')} claims a temporal route.")
            continue
        if not isinstance(snapshot, Mapping):
            raise ValueError(
                f"Video job {job.get('condition_id')} lacks a temporal route snapshot."
            )
        qualification = snapshot.get("qualification")
        if qualification not in {"pilot", "production"}:
            raise ValueError(
                f"Video job {job.get('condition_id')} has unknown temporal qualification "
                f"{qualification!r}."
            )
        authorized = job.get("temporal_pilot_authorized")
        if qualification == "pilot":
            pilot_jobs += 1
            if authorized is not True:
                raise ValueError(
                    f"Pilot video job {job.get('condition_id')} lacks explicit authorization."
                )
        elif authorized is not False:
            raise ValueError(
                f"Production video job {job.get('condition_id')} has inconsistent pilot status."
            )
    if manifest.get("unvalidated_temporal_pilot_jobs") != pilot_jobs or manifest.get(
        "allows_unvalidated_temporal_pilot"
    ) is not bool(pilot_jobs):
        raise ValueError(f"Manifest {path} temporal pilot totals are inconsistent.")
    if production and pilot_jobs:
        raise ValueError(f"Production campaign rejects {pilot_jobs} pilot temporal jobs in {path}.")
    return pilot_jobs


def _validate_flux1_routes(
    jobs: Sequence[Mapping[str, Any]], *, production: bool
) -> tuple[str, dict[str, dict[str, str]]] | None:
    """Require one exact shared-v3 route/binding for every FLUX row in a slice."""

    observed_mode: str | None = None
    observed_inputs: dict[str, dict[str, str]] | None = None
    for job in jobs:
        if job.get("model_name") != "flux1_dev":
            continue
        route = job.get("flux1_dual_view_route_v3")
        if not isinstance(route, Mapping):
            raise ValueError("Campaign FLUX row lacks its shared-v3 route.")
        mode = str(route.get("mode", ""))
        if mode not in {flux_v3.MODE_PREVIEW, flux_v3.MODE_EXECUTION}:
            raise ValueError("Campaign FLUX route mode is invalid.")
        if production and mode != flux_v3.MODE_EXECUTION:
            raise ValueError("Production campaign rejects a preview FLUX row.")
        inputs = route.get("protocol_inputs")
        expected_roles = flux_v3.expected_flux1_protocol_input_roles_v3(mode)
        if not isinstance(inputs, Mapping) or set(inputs) != set(expected_roles):
            raise ValueError("Campaign FLUX protocol-input role set drifted.")
        normalized: dict[str, dict[str, str]] = {}
        for role, record in inputs.items():
            if (
                not isinstance(record, Mapping)
                or set(record) != {"path", "sha256"}
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
            ):
                raise ValueError("Campaign FLUX protocol binding is malformed.")
            normalized[str(role)] = {
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
            }
        if observed_mode is None:
            observed_mode = mode
            observed_inputs = normalized
        elif observed_mode != mode or observed_inputs != normalized:
            raise ValueError("Campaign FLUX rows do not share one exact route/binding.")
    if observed_mode is None or observed_inputs is None:
        return None
    return observed_mode, observed_inputs


def _merge_flux1_binding(
    current: tuple[str, dict[str, dict[str, str]]] | None,
    incoming: tuple[str, dict[str, dict[str, str]]] | None,
) -> tuple[str, dict[str, dict[str, str]]] | None:
    if incoming is None:
        return current
    if current is not None and current != incoming:
        raise ValueError("Campaign manifests do not share one exact FLUX execution binding.")
    return incoming


def _flux1_binding_proof(
    binding: tuple[str, dict[str, dict[str, str]]] | None,
) -> dict[str, Any] | None:
    if binding is None:
        return None
    mode, inputs = binding
    return {
        "route_mode": mode,
        "protocol_input_count": len(inputs),
        "protocol_inputs_sha256": _canonical_sha256(inputs),
        "acceptance_receipt": deepcopy(inputs.get("native_equivalence_receipt")),
    }


def _job_variant_key(job: Mapping[str, Any]) -> tuple[str, str | None, str | None]:
    spec = job.get("variant_spec")
    if not isinstance(spec, Mapping):
        raise ValueError(f"Job {job.get('condition_id')} lacks variant_spec.")
    kind = str(spec.get("kind", ""))
    pair_selection = spec.get("pair_selection")
    active = spec.get("active_pair_ids")
    pair_id: str | None = None
    if pair_selection == "single":
        if not isinstance(active, (list, tuple)) or len(active) != 1:
            raise ValueError(
                f"Exact-one job {job.get('condition_id')} must activate exactly one pair."
            )
        pair_id = str(active[0])
    elif pair_selection == "full":
        if not isinstance(active, (list, tuple)) or len(active) != 5:
            raise ValueError(f"Full job {job.get('condition_id')} must activate five pairs.")
    elif pair_selection is not None:
        raise ValueError(f"Job {job.get('condition_id')} has unknown pair selection.")
    return kind, pair_selection, pair_id


def _expected_family_keys(prompt_id: str, family: str) -> set[tuple[str, str | None, str | None]]:
    pairs = PAIR_IDS_BY_PROMPT[prompt_id]
    if family == "standard":
        return {
            ("baseline", None, None),
            ("native_negative_prompt", None, None),
            ("conceptsteer", "full", None),
            *{("conceptsteer", "single", pair_id) for pair_id in pairs},
        }
    if family == "shapley":
        return {
            ("shapley_concept_steering", "full", None),
            *{("shapley_concept_steering", "single", pair_id) for pair_id in pairs},
        }
    raise ValueError(f"Unknown final manifest family: {family!r}.")


def _expected_variation_for_key(key: tuple[str, str | None, str | None]) -> str:
    kind, pair_selection, _ = key
    kind = _normalize_variant_kind(kind)
    if kind == "baseline":
        return "01_baseline"
    if kind == "native_negative_prompt":
        return "02_negative_prompt"
    if kind == "conceptsteer":
        return (
            "05_concept_steering_single_pair"
            if pair_selection == "single"
            else "03_concept_steering"
        )
    if kind == "shapley_concept_steering":
        return (
            "06_shapley_concept_steering_single_pair"
            if pair_selection == "single"
            else "04_shapley_concept_steering"
        )
    raise ValueError(f"Unknown final variant key: {key}.")


def _expected_variant_for_key(key: tuple[str, str | None, str | None]) -> str:
    kind, pair_selection, pair_id = key
    kind = _normalize_variant_kind(kind)
    if kind == "baseline":
        return "01_baseline"
    if kind == "native_negative_prompt":
        return "02_negative_prompt"
    if kind == "conceptsteer":
        return (
            f"conceptsteer_single__{pair_id}" if pair_selection == "single" else "conceptsteer_full"
        )
    if kind == "shapley_concept_steering":
        return (
            f"shapley_concept_steering_single__{pair_id}"
            if pair_selection == "single"
            else "shapley_concept_steering_full"
        )
    raise ValueError(f"Unknown final variant key: {key}.")


def _validate_variant_spec(
    job: Mapping[str, Any],
    *,
    key: tuple[str, str | None, str | None],
    prompt_id: str,
    model_name: str,
) -> None:
    spec = job["variant_spec"]
    assert isinstance(spec, Mapping)
    kind, pair_selection, pair_id = key
    kind = _normalize_variant_kind(kind)
    if kind == "baseline":
        if spec != {"kind": "baseline"}:
            raise ValueError(f"Baseline job {job.get('condition_id')} has a noncanonical spec.")
        return
    if kind == "native_negative_prompt":
        unsupported = model_name in NATIVE_NEGATIVE_UNSUPPORTED_REASONS
        expected_capability = "not_supported" if unsupported else "supported"
        if spec.get("capability") != expected_capability:
            raise ValueError(
                f"Native-negative job {job.get('condition_id')} has the wrong capability."
            )
        if unsupported and spec.get("reason") != NATIVE_NEGATIVE_UNSUPPORTED_REASONS[model_name]:
            raise ValueError(
                f"Native-negative job {job.get('condition_id')} has the wrong unsupported reason."
            )
        return
    active = spec.get("active_pair_ids")
    if pair_selection == "full" and tuple(active or ()) != tuple(PAIR_IDS_BY_PROMPT[prompt_id]):
        raise ValueError(
            f"Full job {job.get('condition_id')} does not activate the prompt's exact five pairs."
        )
    if pair_selection == "single" and tuple(active or ()) != (pair_id,):
        raise ValueError(
            f"Exact-one job {job.get('condition_id')} activates the wrong concept pair."
        )
    if kind == "shapley_concept_steering":
        if (
            spec.get("shapley") != SHAPLEY_CONFIG
            or spec.get("shapley_provenance") != SHAPLEY_PROVENANCE
        ):
            raise ValueError(
                f"Shapley job {job.get('condition_id')} lacks the exact frozen Shapley protocol."
            )
    elif "shapley" in spec or "shapley_provenance" in spec:
        raise ValueError(f"Ordinary job {job.get('condition_id')} falsely claims Shapley protocol.")


def _validate_seed_attempt_path(
    output_dir: Path, *, seed: int, attempt: int, condition_id: str
) -> None:
    expected_suffix = (
        f"seed_{seed:08d}",
        "attempts",
        f"attempt_{attempt:03d}",
    )
    if tuple(output_dir.parts[-3:]) != expected_suffix:
        raise ValueError(
            f"Output {output_dir} lacks the exact seed/attempt suffix {expected_suffix}."
        )
    if not condition_id.endswith(f"__seed_{seed:08d}"):
        raise ValueError(f"Seed-scoped condition {condition_id!r} lacks its exact seed suffix.")


def _validate_unique_union(jobs: Sequence[Mapping[str, Any]], *, label: str) -> None:
    condition_ids = [str(job.get("condition_id", "")) for job in jobs]
    if any(not value for value in condition_ids) or len(condition_ids) != len(set(condition_ids)):
        raise ValueError(f"{label} condition IDs are missing or duplicated.")
    output_dirs = [str(Path(str(job.get("output_dir", ""))).resolve()) for job in jobs]
    if any(value == str(Path("").resolve()) for value in output_dirs) or len(output_dirs) != len(
        set(output_dirs)
    ):
        raise ValueError(f"{label} output directories are missing or duplicated.")


def _validate_seed_ladder(
    manifest_paths: Sequence[str | Path],
    *,
    root: str | Path,
    manifest_reader: ManifestReader,
    production: bool,
) -> dict[str, Any]:
    resolved_root = Path(root).expanduser().resolve()
    paths = _resolve_exact_path_set(
        manifest_paths,
        canonical_ladder_manifest_paths(resolved_root),
        label="Seed-ladder manifest",
        root=resolved_root,
    )
    output_root = (resolved_root / LADDER_OUTPUT_ROOT_RELATIVE).resolve()
    all_jobs: list[Mapping[str, Any]] = []
    pilots = 0
    digests: set[str] = set()
    flux1_binding: tuple[str, dict[str, dict[str, str]]] | None = None
    for seed, path in zip(SEEDS, paths, strict=True):
        manifest = manifest_reader(path, resolved_root)
        jobs = _manifest_header(
            manifest, path=path, expected_jobs=EXPECTED_LADDER_ROWS_PER_MANIFEST
        )
        if (
            manifest.get("seed") != seed
            or manifest.get("attempt") != seed + 1
            or manifest.get("seed_scoped_output") is not True
            or manifest.get("models") != list(MODEL_NAMES)
            or manifest.get("prompt_ids") != list(PROMPT_IDS)
            or manifest.get("variation_groups") != list(LADDER_VARIATIONS)
            or _resolve_path(str(manifest.get("output_root", "")), resolved_root) != output_root
        ):
            raise ValueError(
                f"Seed-ladder manifest {path} violates seed/attempt/scope/axis/root topology."
            )
        if manifest["manifest_sha256"] in digests:
            raise ValueError("Seed-ladder manifests contain a duplicated manifest digest.")
        digests.add(str(manifest["manifest_sha256"]))
        observed_axes: set[tuple[str, str]] = set()
        for job in jobs:
            axis = (str(job.get("prompt_id", "")), str(job.get("model_name", "")))
            if axis not in AXES or axis in observed_axes:
                raise ValueError(f"Seed {seed} ladder axes are missing or duplicated: {axis}.")
            observed_axes.add(axis)
            output_dir = _resolve_path(str(job.get("output_dir", "")), resolved_root)
            if (
                job.get("seed") != seed
                or job.get("attempt") != seed + 1
                or job.get("seed_scoped_output") is not True
                or job.get("variation") != "01_baseline"
                or job.get("variant") != "01_baseline"
                or job.get("variant_spec") != {"kind": "baseline"}
                or job.get("expected_media") is not True
                or output_root not in output_dir.parents
            ):
                raise ValueError(
                    f"Seed-ladder job {job.get('condition_id')} is not an exact canonical baseline."
                )
            _validate_seed_attempt_path(
                output_dir,
                seed=seed,
                attempt=seed + 1,
                condition_id=str(job.get("condition_id", "")),
            )
        if observed_axes != set(AXES):
            raise ValueError(f"Seed {seed} ladder manifest does not cover all 36 axes.")
        media, unsupported = _validate_counts(manifest, jobs, path=path)
        if (media, unsupported) != (36, 0):
            raise ValueError(f"Seed {seed} ladder must be 36 media and zero unsupported rows.")
        pilots += _validate_temporal_routes(manifest, jobs, path=path, production=production)
        flux1_binding = _merge_flux1_binding(
            flux1_binding,
            _validate_flux1_routes(jobs, production=production),
        )
        all_jobs.extend(jobs)
    _validate_unique_union(all_jobs, label="Seed-ladder union")
    if len(all_jobs) != EXPECTED_LADDER_MEDIA:
        raise ValueError("Seed-ladder union is not exactly 288 baseline media rows.")
    if production and (flux1_binding is None or flux1_binding[0] != flux_v3.MODE_EXECUTION):
        raise ValueError("Production seed ladder lacks one exact FLUX execution binding.")
    if production:
        representative = next(job for job in all_jobs if job.get("model_name") == "flux1_dev")
        flux_v3.validate_flux1_job_v3(
            representative,
            project_root=resolved_root,
            mode=flux_v3.MODE_EXECUTION,
        )
    return {
        "manifest_count": len(paths),
        "axis_count": len(AXES),
        "logical_rows": len(all_jobs),
        "media_rows": len(all_jobs),
        "unsupported_rows": 0,
        "pilot_temporal_rows": pilots,
        "status": "production_ready" if pilots == 0 else "preview_only_pilot_routes",
        "flux1_execution_binding": _flux1_binding_proof(flux1_binding),
    }


def validate_seed_ladder_preview(
    manifest_paths: Sequence[str | Path],
    *,
    root: str | Path,
    manifest_reader: ManifestReader = read_manifest_for_audit,
) -> dict[str, Any]:
    """Validate exact ladder topology while admitting explicit pilot routes."""

    return _validate_seed_ladder(
        manifest_paths, root=root, manifest_reader=manifest_reader, production=False
    )


def validate_production_seed_ladder(
    manifest_paths: Sequence[str | Path],
    *,
    root: str | Path,
    manifest_reader: ManifestReader = read_manifest_for_audit,
) -> dict[str, Any]:
    """Validate a launchable ladder; any temporal pilot fails closed."""

    return _validate_seed_ladder(
        manifest_paths, root=root, manifest_reader=manifest_reader, production=True
    )


def _load_selections(
    selection_paths: Sequence[str | Path],
    *,
    root: Path,
    selection_reader: SelectionReader,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], tuple[Path, ...]]:
    paths = _resolve_exact_path_set(
        selection_paths,
        canonical_selection_paths(root),
        label="Target-blind selection",
        root=root,
    )
    selections: dict[tuple[str, str], Mapping[str, Any]] = {}
    for axis, path in zip(AXES, paths, strict=True):
        payload = selection_reader(path, root)
        prompt_id, model_name = axis
        selected_seed = payload.get("selected_seed")
        if (
            payload.get("selection") != target_blind.SELECTION_NAME
            or payload.get("benchmark") != BENCHMARK_NAME
            or payload.get("prompt_id") != prompt_id
            or payload.get("model_name") != model_name
            or payload.get("task") != TASK_BY_MODEL[model_name]
            or isinstance(selected_seed, bool)
            or not isinstance(selected_seed, int)
            or selected_seed not in SEEDS
        ):
            raise ValueError(f"Selection record {path} does not identify its canonical axis/seed.")
        _parse_timestamp(payload.get("selected_at_utc"), f"selection {path} selected_at_utc")
        if axis in selections:
            raise ValueError(f"Target-blind selection axis is duplicated: {axis}.")
        selections[axis] = payload
    if set(selections) != set(AXES):
        raise ValueError("Target-blind selections do not cover exactly all 36 axes.")
    return selections, paths


def _validate_selection_ladder_bindings(
    selections: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    root: Path,
    manifest_reader: ManifestReader,
) -> dict[str, int]:
    """Bind every one of the 36 selections to the same canonical 8 manifests."""

    ladder_manifests = {
        seed: manifest_reader(ladder_manifest_path(root, seed), root) for seed in SEEDS
    }
    bound_rows = 0
    for axis in AXES:
        rows = selections[axis].get("candidate_rows")
        if not isinstance(rows, list) or len(rows) != len(SEEDS):
            raise ValueError(f"Selection {axis} must bind exactly eight ladder candidate rows.")
        if [row.get("seed") for row in rows if isinstance(row, Mapping)] != list(SEEDS):
            raise ValueError(f"Selection {axis} ladder rows are not ordered seeds 0..7.")
        for seed, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"Selection {axis} seed {seed} row is not a mapping.")
            bindings = row.get("artifact_bindings")
            source = bindings.get("source_manifest") if isinstance(bindings, Mapping) else None
            expected_path = ladder_manifest_path(root, seed)
            manifest = ladder_manifests[seed]
            job_index = row.get("manifest_job_index")
            jobs = manifest.get("jobs")
            if (
                not isinstance(source, Mapping)
                or _resolve_path(str(source.get("path", "")), root) != expected_path
                or source.get("manifest_sha256") != manifest.get("manifest_sha256")
                or row.get("manifest_sha256") != manifest.get("manifest_sha256")
                or isinstance(job_index, bool)
                or not isinstance(job_index, int)
                or not isinstance(jobs, list)
                or not 0 <= job_index < len(jobs)
            ):
                raise ValueError(
                    f"Selection {axis} seed {seed} does not bind the canonical ladder manifest."
                )
            job = jobs[job_index]
            if (
                not isinstance(job, Mapping)
                or (job.get("prompt_id"), job.get("model_name")) != axis
                or job.get("seed") != seed
                or job.get("variant_spec") != {"kind": "baseline"}
            ):
                raise ValueError(
                    f"Selection {axis} seed {seed} binds the wrong canonical ladder job."
                )
            bound_rows += 1
    if bound_rows != EXPECTED_LADDER_MEDIA:
        raise ValueError("Selection-to-ladder binding is not exactly 36 axes by eight seeds.")
    return {"record_count": len(selections), "candidate_bindings": bound_rows}


def _validate_final_campaign(
    final_manifest_paths: Sequence[str | Path],
    selection_paths: Sequence[str | Path],
    *,
    root: str | Path,
    manifest_reader: ManifestReader,
    selection_reader: SelectionReader,
    axis_final_validator: AxisFinalValidator,
    production: bool,
    preloaded_selections: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_root = Path(root).expanduser().resolve()
    paths = _resolve_exact_path_set(
        final_manifest_paths,
        canonical_final_manifest_paths(resolved_root),
        label="Final manifest",
        root=resolved_root,
    )
    if preloaded_selections is None:
        selections, _ = _load_selections(
            selection_paths, root=resolved_root, selection_reader=selection_reader
        )
    else:
        _resolve_exact_path_set(
            selection_paths,
            canonical_selection_paths(resolved_root),
            label="Target-blind selection",
            root=resolved_root,
        )
        if set(preloaded_selections) != set(AXES):
            raise ValueError("Preloaded target-blind selections do not cover exactly all axes.")
        selections = dict(preloaded_selections)
    expected_root = (resolved_root / FINAL_OUTPUT_ROOT_RELATIVE).resolve()
    all_jobs: list[Mapping[str, Any]] = []
    pilots = 0
    manifest_digests: set[str] = set()
    flux1_binding: tuple[str, dict[str, dict[str, str]]] | None = None
    axis_paths: dict[tuple[str, str], list[Path]] = {axis: [] for axis in AXES}
    path_index = 0
    for axis in AXES:
        prompt_id, model_name = axis
        selected_seed = selections[axis]["selected_seed"]
        for family in FINAL_FAMILIES:
            path = paths[path_index]
            path_index += 1
            axis_paths[axis].append(path)
            manifest = manifest_reader(path, resolved_root)
            expected_rows = (
                EXPECTED_STANDARD_ROWS_PER_AXIS
                if family == "standard"
                else EXPECTED_SHAPLEY_ROWS_PER_AXIS
            )
            jobs = _manifest_header(manifest, path=path, expected_jobs=expected_rows)
            digest = str(manifest["manifest_sha256"])
            if digest in manifest_digests:
                raise ValueError("Final campaign contains a duplicated manifest digest.")
            manifest_digests.add(digest)
            expected_variations = (
                list(STANDARD_VARIATIONS) if family == "standard" else list(SHAPLEY_VARIATIONS)
            )
            if (
                manifest.get("seed") != selected_seed
                or manifest.get("attempt") != 1
                or manifest.get("seed_scoped_output") is not True
                or manifest.get("models") != [model_name]
                or manifest.get("prompt_ids") != [prompt_id]
                or manifest.get("variation_groups") != expected_variations
                or _resolve_path(str(manifest.get("output_root", "")), resolved_root)
                != expected_root
            ):
                raise ValueError(
                    f"Final {family} manifest {path} violates selected-seed/attempt/axis/root topology."
                )
            if manifest.get("selected_single_pair_ids_by_prompt") != {
                prompt_id: list(PAIR_IDS_BY_PROMPT[prompt_id])
            }:
                raise ValueError(
                    f"Final {family} manifest {path} does not freeze all five exact-one pairs."
                )
            observed: set[tuple[str, str | None, str | None]] = set()
            for job in jobs:
                key = _job_variant_key(job)
                if key in observed:
                    raise ValueError(f"Final {family} manifest duplicates variant {key}.")
                observed.add(key)
                output_dir = _resolve_path(str(job.get("output_dir", "")), resolved_root)
                expected_media = not (
                    key[0] == "native_negative_prompt"
                    and model_name in NATIVE_NEGATIVE_UNSUPPORTED_REASONS
                )
                if (
                    job.get("prompt_id") != prompt_id
                    or job.get("model_name") != model_name
                    or job.get("seed") != selected_seed
                    or job.get("attempt") != 1
                    or job.get("seed_scoped_output") is not True
                    or job.get("variation") != _expected_variation_for_key(key)
                    or job.get("variant") != _expected_variant_for_key(key)
                    or job.get("expected_media") is not expected_media
                    or expected_root not in output_dir.parents
                ):
                    raise ValueError(
                        f"Final job {job.get('condition_id')} violates its canonical axis/variant/media contract."
                    )
                _validate_variant_spec(
                    job,
                    key=key,
                    prompt_id=prompt_id,
                    model_name=model_name,
                )
                _validate_seed_attempt_path(
                    output_dir,
                    seed=selected_seed,
                    attempt=1,
                    condition_id=str(job.get("condition_id", "")),
                )
            expected_keys = _expected_family_keys(prompt_id, family)
            if observed != expected_keys:
                raise ValueError(
                    f"Final {family} family is wrong for {prompt_id}/{model_name}: "
                    f"missing={sorted(expected_keys - observed)}, "
                    f"unknown={sorted(observed - expected_keys)}."
                )
            _validate_counts(manifest, jobs, path=path)
            pilots += _validate_temporal_routes(manifest, jobs, path=path, production=production)
            flux1_binding = _merge_flux1_binding(
                flux1_binding,
                _validate_flux1_routes(jobs, production=production),
            )
            all_jobs.extend(jobs)

    # This is the strengthened target-blind per-axis boundary.  Passing all 72
    # paths at once would allow an aggregate validator to hide a partial axis;
    # every immutable selection is therefore checked against exactly its two
    # canonical 8+6 manifests.
    for axis in AXES:
        result = axis_final_validator(
            selections[axis],
            final_manifest_paths=axis_paths[axis],
            root=resolved_root,
            manifest_reader=manifest_reader,
        )
        if (
            not isinstance(result, Mapping)
            or result.get("matching_final_manifests") != 2
            or result.get("matching_final_jobs") != 14
        ):
            raise ValueError(f"Per-axis final validator did not authenticate {axis}: {result!r}.")

    _validate_unique_union(all_jobs, label="Final campaign union")
    logical = len(all_jobs)
    media = sum(job.get("expected_media") is True for job in all_jobs)
    unsupported = logical - media
    exact_one = sum(
        _job_variant_key(job)[1] == "single" and job.get("expected_media") is True
        for job in all_jobs
    )
    if (
        logical != EXPECTED_FINAL_LOGICAL
        or media != EXPECTED_FINAL_MEDIA
        or unsupported != EXPECTED_FINAL_UNSUPPORTED
        or exact_one != EXPECTED_FINAL_EXACT_ONE_MEDIA
    ):
        raise ValueError(
            "Final campaign totals differ from 504 logical / 489 media / 15 unsupported / "
            f"360 exact-one: got {logical}/{media}/{unsupported}/{exact_one}."
        )
    if production and (flux1_binding is None or flux1_binding[0] != flux_v3.MODE_EXECUTION):
        raise ValueError("Production final campaign lacks one exact FLUX execution binding.")
    if production:
        representative = next(job for job in all_jobs if job.get("model_name") == "flux1_dev")
        flux_v3.validate_flux1_job_v3(
            representative,
            project_root=resolved_root,
            mode=flux_v3.MODE_EXECUTION,
        )
    return {
        "manifest_count": len(paths),
        "selection_count": len(selections),
        "axis_count": len(AXES),
        "standard_rows": len(AXES) * EXPECTED_STANDARD_ROWS_PER_AXIS,
        "shapley_rows": len(AXES) * EXPECTED_SHAPLEY_ROWS_PER_AXIS,
        "logical_rows": logical,
        "media_rows": media,
        "unsupported_rows": unsupported,
        "exact_one_media_rows": exact_one,
        "pilot_temporal_rows": pilots,
        "status": "production_ready" if pilots == 0 else "preview_only_pilot_routes",
        "flux1_execution_binding": _flux1_binding_proof(flux1_binding),
    }


def validate_final_campaign_preview(
    final_manifest_paths: Sequence[str | Path],
    selection_paths: Sequence[str | Path],
    *,
    root: str | Path,
    manifest_reader: ManifestReader = read_manifest_for_audit,
    selection_reader: SelectionReader = _read_selection,
    axis_final_validator: AxisFinalValidator = target_blind.validate_final_manifests,
) -> dict[str, Any]:
    """Validate final topology while admitting explicitly authorized pilots."""

    return _validate_final_campaign(
        final_manifest_paths,
        selection_paths,
        root=root,
        manifest_reader=manifest_reader,
        selection_reader=selection_reader,
        axis_final_validator=axis_final_validator,
        production=False,
    )


def validate_production_final_campaign(
    final_manifest_paths: Sequence[str | Path],
    selection_paths: Sequence[str | Path],
    *,
    root: str | Path,
    manifest_reader: ManifestReader = read_manifest_for_audit,
    selection_reader: SelectionReader = _read_selection,
    axis_final_validator: AxisFinalValidator = target_blind.validate_final_manifests,
) -> dict[str, Any]:
    """Validate the exact selected-seed final campaign; pilots fail closed."""

    return _validate_final_campaign(
        final_manifest_paths,
        selection_paths,
        root=root,
        manifest_reader=manifest_reader,
        selection_reader=selection_reader,
        axis_final_validator=axis_final_validator,
        production=True,
    )


def _validate_production_final_against_committed_selection_cohort(
    final_manifest_paths: Sequence[str | Path],
    *,
    root: Path,
    manifest_reader: ManifestReader,
) -> dict[str, Any]:
    """Authenticate final manifests against the live committed selection tree."""

    selections, selection_paths, selection_proof = _read_committed_selection_cohort(root)
    result = _validate_final_campaign(
        final_manifest_paths,
        selection_paths,
        root=root,
        manifest_reader=manifest_reader,
        selection_reader=_selection_mapping_reader(selections, root),
        axis_final_validator=target_blind.validate_final_manifests,
        production=True,
        preloaded_selections=selections,
    )
    result["selection_cohort_binding"] = selection_proof
    return result


def _validate_complete_campaign(
    ladder_manifest_paths: Sequence[str | Path],
    selection_paths: Sequence[str | Path],
    final_manifest_paths: Sequence[str | Path],
    *,
    root: str | Path,
    manifest_reader: ManifestReader,
    selection_reader: SelectionReader,
    axis_final_validator: AxisFinalValidator,
    production: bool,
) -> dict[str, Any]:
    ladder = _validate_seed_ladder(
        ladder_manifest_paths,
        root=root,
        manifest_reader=manifest_reader,
        production=production,
    )
    resolved_root = Path(root).expanduser().resolve()
    loaded_selections, _ = _load_selections(
        selection_paths, root=resolved_root, selection_reader=selection_reader
    )
    selection_bindings = _validate_selection_ladder_bindings(
        loaded_selections,
        root=resolved_root,
        manifest_reader=manifest_reader,
    )
    final = _validate_final_campaign(
        final_manifest_paths,
        selection_paths,
        root=root,
        manifest_reader=manifest_reader,
        selection_reader=selection_reader,
        axis_final_validator=axis_final_validator,
        production=production,
        preloaded_selections=loaded_selections,
    )
    pilots = ladder["pilot_temporal_rows"] + final["pilot_temporal_rows"]
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "contract": CAMPAIGN_CONTRACT,
        "status": "production_validated" if pilots == 0 else "preview_only_pilot_routes",
        "ladder": ladder,
        "selections": {
            **selection_bindings,
            "axis_count": len(AXES),
        },
        "final": final,
        "campaign_totals": {
            "ladder_media": EXPECTED_LADDER_MEDIA,
            "selection_records": EXPECTED_SELECTION_RECORDS,
            "final_logical": EXPECTED_FINAL_LOGICAL,
            "final_media": EXPECTED_FINAL_MEDIA,
            "final_unsupported": EXPECTED_FINAL_UNSUPPORTED,
            "final_exact_one_media": EXPECTED_FINAL_EXACT_ONE_MEDIA,
        },
    }


def validate_campaign_preview(
    ladder_manifest_paths: Sequence[str | Path],
    selection_paths: Sequence[str | Path],
    final_manifest_paths: Sequence[str | Path],
    *,
    root: str | Path,
    manifest_reader: ManifestReader = read_manifest_for_audit,
    selection_reader: SelectionReader = _read_selection,
    axis_final_validator: AxisFinalValidator = target_blind.validate_final_manifests,
) -> dict[str, Any]:
    """Validate all topology while allowing explicit temporal pilot manifests."""

    return _validate_complete_campaign(
        ladder_manifest_paths,
        selection_paths,
        final_manifest_paths,
        root=root,
        manifest_reader=manifest_reader,
        selection_reader=selection_reader,
        axis_final_validator=axis_final_validator,
        production=False,
    )


def validate_production_campaign(
    ladder_manifest_paths: Sequence[str | Path],
    selection_paths: Sequence[str | Path],
    final_manifest_paths: Sequence[str | Path],
    *,
    root: str | Path,
    manifest_reader: ManifestReader = read_manifest_for_audit,
    selection_reader: SelectionReader = _read_selection,
    axis_final_validator: AxisFinalValidator = target_blind.validate_final_manifests,
) -> dict[str, Any]:
    """Fail-closed validation of the complete launchable production campaign."""

    return _validate_complete_campaign(
        ladder_manifest_paths,
        selection_paths,
        final_manifest_paths,
        root=root,
        manifest_reader=manifest_reader,
        selection_reader=selection_reader,
        axis_final_validator=axis_final_validator,
        production=True,
    )


def validate_completed_flux1_production_campaign_v3(
    ladder_manifest_paths: Sequence[str | Path],
    final_manifest_paths: Sequence[str | Path],
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Authenticate all 24 ladder and 42 final completed FLUX-v3 rows."""

    resolved_root = Path(root).expanduser().resolve()
    cohorts = (
        (
            "seed_ladder",
            _resolve_exact_path_set(
                ladder_manifest_paths,
                canonical_ladder_manifest_paths(resolved_root),
                label="Completed FLUX-v3 ladder manifest",
                root=resolved_root,
            ),
        ),
        (
            "final",
            _resolve_exact_path_set(
                final_manifest_paths,
                canonical_final_manifest_paths(resolved_root),
                label="Completed FLUX-v3 final manifest",
                root=resolved_root,
            ),
        ),
    )
    rows: list[dict[str, Any]] = []
    counts = {"seed_ladder": 0, "final": 0}
    for cohort, paths in cohorts:
        for manifest_path in paths:
            manifest = read_manifest_for_audit(manifest_path, resolved_root)
            manifest_sha256 = str(manifest["manifest_sha256"])
            for index, raw_job in enumerate(manifest["jobs"]):
                if not is_flux1_job_v3(raw_job):
                    continue
                if raw_job.get("expected_media") is not True:
                    raise ValueError("Completed campaign FLUX-v3 row is not a media row.")
                exact_job = {
                    **raw_job,
                    "launch_manifest_sha256": manifest_sha256,
                    "launch_manifest_job_index": index,
                }
                result_path = Path(str(raw_job["output_dir"])) / "benchmark_job_result.json"
                independently_bound_result_sha256 = _sha256_file(result_path)
                reopened = reopen_completed_flux1_output_v3(
                    exact_job,
                    root=resolved_root,
                    manifest_path=manifest_path,
                    manifest_sha256=manifest_sha256,
                    manifest_job_index=index,
                    result_path=result_path,
                )
                if reopened["result_sha256"] != independently_bound_result_sha256:
                    raise ValueError("Campaign FLUX-v3 result changed during strict reopening.")
                rows.append(
                    {
                        "cohort": cohort,
                        "manifest_path": str(manifest_path),
                        "manifest_sha256": manifest_sha256,
                        "manifest_job_index": index,
                        "condition_id": raw_job["condition_id"],
                        "job_sha256": reopened["job_sha256"],
                        "result_sha256": reopened["result_sha256"],
                        "runtime_validation_sha256": hashlib.sha256(
                            json.dumps(
                                reopened["runtime_validation"],
                                ensure_ascii=False,
                                allow_nan=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest(),
                        "evidence_bindings": deepcopy(reopened["evidence_bindings"]),
                    }
                )
                counts[cohort] += 1
    if counts != {"seed_ladder": 24, "final": 42} or len(rows) != 66:
        raise ValueError(
            "Completed FLUX-v3 campaign must contain ladder=24, final=42, total=66 rows."
        )
    identities = {
        (row["cohort"], row["manifest_sha256"], row["manifest_job_index"]) for row in rows
    }
    if len(identities) != 66:
        raise ValueError("Completed FLUX-v3 campaign reuses a manifest row identity.")
    receipt = {
        "schema_version": 3,
        "contract": FLUX1_COMPLETED_CAMPAIGN_CONTRACT,
        "benchmark": BENCHMARK_NAME,
        "status": "passed",
        "counts": counts,
        "rows": rows,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return receipt


def _benchmark_args(
    *,
    output_root: Path,
    attempt: int,
    models: Sequence[str],
    prompt_ids: Sequence[str],
    variations: Sequence[str],
    seed: int,
    allow_pilot: bool,
) -> Any:
    argv = [
        "--output-root",
        str(output_root),
        "--attempt",
        str(attempt),
        "--models",
        ",".join(models),
        "--prompt-ids",
        ",".join(prompt_ids),
        "--variations",
        ",".join(variations),
        "--pair-ids",
        "all",
        "--seed",
        str(seed),
        "--seed-scoped-output",
    ]
    if allow_pilot:
        argv.append("--allow-unvalidated-temporal-pilot")
    return build_benchmark_parser().parse_args(argv)


def _build_seed_ladder(
    root: str | Path,
    *,
    allow_pilot: bool,
    flux1_execution_protocol_inputs: Mapping[str, Any] | None,
) -> dict[Path, dict[str, Any]]:
    resolved_root = Path(root).expanduser().resolve()
    manifests: dict[Path, dict[str, Any]] = {}
    for seed in SEEDS:
        args = _benchmark_args(
            output_root=LADDER_OUTPUT_ROOT_RELATIVE,
            attempt=seed + 1,
            models=MODEL_NAMES,
            prompt_ids=PROMPT_IDS,
            variations=LADDER_VARIATIONS,
            seed=seed,
            allow_pilot=allow_pilot,
        )
        manifests[ladder_manifest_path(resolved_root, seed)] = build_benchmark_manifest(
            args,
            resolved_root,
            **(
                {
                    "flux1_route_context": flux_v3.MODE_EXECUTION,
                    "flux1_protocol_inputs": flux1_execution_protocol_inputs,
                }
                if flux1_execution_protocol_inputs is not None
                else {}
            ),
        )
    reader = _mapping_manifest_reader(manifests, resolved_root)
    validator = validate_seed_ladder_preview if allow_pilot else validate_production_seed_ladder
    validator(tuple(manifests), root=resolved_root, manifest_reader=reader)
    return manifests


def build_seed_ladder_preview(root: str | Path) -> dict[Path, dict[str, Any]]:
    """Build (without writing) the exact ladder, admitting configured pilots."""

    return _build_seed_ladder(
        root,
        allow_pilot=True,
        flux1_execution_protocol_inputs=None,
    )


def build_production_seed_ladder(
    root: str | Path,
    *,
    flux1_native_equivalence_acceptance_receipt_path: str | Path,
) -> dict[Path, dict[str, Any]]:
    """Build (without writing) a launchable ladder; pilot configs fail closed."""

    resolved_root = Path(root).expanduser().resolve()
    protocol_inputs = flux_v3.load_flux1_execution_protocol_inputs_v3(
        flux1_native_equivalence_acceptance_receipt_path,
        project_root=resolved_root,
    )
    return _build_seed_ladder(
        resolved_root,
        allow_pilot=False,
        flux1_execution_protocol_inputs=protocol_inputs,
    )


def _build_final_campaign(
    selection_paths: Sequence[str | Path],
    *,
    root: str | Path,
    selection_reader: SelectionReader,
    allow_pilot: bool,
    flux1_execution_protocol_inputs: Mapping[str, Any] | None,
) -> dict[Path, dict[str, Any]]:
    resolved_root = Path(root).expanduser().resolve()
    selections, canonical_paths = _load_selections(
        selection_paths, root=resolved_root, selection_reader=selection_reader
    )
    manifests: dict[Path, dict[str, Any]] = {}
    for prompt_id, model_name in AXES:
        seed = int(selections[(prompt_id, model_name)]["selected_seed"])
        for family, variations in (
            ("standard", STANDARD_VARIATIONS),
            ("shapley", SHAPLEY_VARIATIONS),
        ):
            args = _benchmark_args(
                output_root=FINAL_OUTPUT_ROOT_RELATIVE,
                attempt=1,
                models=(model_name,),
                prompt_ids=(prompt_id,),
                variations=variations,
                seed=seed,
                allow_pilot=allow_pilot,
            )
            flux_kwargs: dict[str, Any] = {}
            if model_name == "flux1_dev" and flux1_execution_protocol_inputs is not None:
                flux_kwargs = {
                    "flux1_route_context": flux_v3.MODE_EXECUTION,
                    "flux1_protocol_inputs": flux1_execution_protocol_inputs,
                }
            manifests[final_manifest_path(resolved_root, prompt_id, model_name, family)] = (
                build_benchmark_manifest(args, resolved_root, **flux_kwargs)
            )
    reader = _mapping_manifest_reader(manifests, resolved_root)
    validator = (
        validate_final_campaign_preview if allow_pilot else validate_production_final_campaign
    )
    validator(
        tuple(manifests),
        canonical_paths,
        root=resolved_root,
        manifest_reader=reader,
        selection_reader=lambda path, _root: selections[
            _axis_for_selection_path(path, resolved_root)
        ],
    )
    return manifests


def _axis_for_selection_path(path: str | Path, root: Path) -> tuple[str, str]:
    resolved = _resolve_path(path, root)
    for axis, expected in zip(AXES, canonical_selection_paths(root), strict=True):
        if resolved == expected:
            return axis
    raise ValueError(f"Unknown canonical selection path: {resolved}.")


def build_final_campaign_preview(
    selection_paths: Sequence[str | Path],
    *,
    root: str | Path,
    selection_reader: SelectionReader = _read_selection,
) -> dict[Path, dict[str, Any]]:
    """Build (without writing) the exact 72-manifest final preview."""

    return _build_final_campaign(
        selection_paths,
        root=root,
        selection_reader=selection_reader,
        allow_pilot=True,
        flux1_execution_protocol_inputs=None,
    )


def build_production_final_campaign(
    *,
    root: str | Path,
    flux1_native_equivalence_acceptance_receipt_path: str | Path,
) -> dict[Path, dict[str, Any]]:
    """Build from the committed 36-selection cohort; pilots fail closed.

    Unlike the explicit preview builder, this production entry point admits no
    caller-supplied selection paths or reader.  It reopens the canonical atomic
    cohort before any manifest construction and again after validation, so an
    absent, uncommitted, stale, or changed decision set cannot drive a final
    campaign.
    """

    resolved_root = Path(root).expanduser().resolve()
    selections, selection_paths, selection_proof = _read_committed_selection_cohort(resolved_root)
    protocol_inputs = flux_v3.load_flux1_execution_protocol_inputs_v3(
        flux1_native_equivalence_acceptance_receipt_path,
        project_root=resolved_root,
    )
    manifests = _build_final_campaign(
        selection_paths,
        root=resolved_root,
        selection_reader=_selection_mapping_reader(selections, resolved_root),
        allow_pilot=False,
        flux1_execution_protocol_inputs=protocol_inputs,
    )
    _, reopened_paths, reopened_proof = _read_committed_selection_cohort(resolved_root)
    if reopened_paths != selection_paths or reopened_proof != selection_proof:
        raise RuntimeError("Committed selection cohort changed while final manifests were built.")
    return manifests


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_document_sha256(payload: Mapping[str, Any], *, digest_field: str) -> str:
    canonical = deepcopy(dict(payload))
    canonical.pop(digest_field, None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _existing_component_is_symlink(path: Path, *, root: Path) -> bool:
    if path != root and root not in path.parents:
        return True
    current = path
    while True:
        if current.is_symlink():
            return True
        if current == root:
            return False
        current = current.parent


def _strict_manifest_mapping(
    manifests: Mapping[Path, Mapping[str, Any]],
    expected_paths: Sequence[Path],
    *,
    root: Path,
) -> dict[Path, Mapping[str, Any]]:
    """Reject missing, extra, relative, resolved-alias, and symlinked keys."""

    if not isinstance(manifests, Mapping):
        raise ValueError("Campaign manifests must be an exact canonical path mapping.")
    expected = tuple(path.resolve() for path in expected_paths)
    expected_set = set(expected)
    indexed: dict[Path, Mapping[str, Any]] = {}
    invalid: list[str] = []
    for supplied, payload in manifests.items():
        try:
            lexical = Path(supplied).expanduser()
        except TypeError:
            invalid.append(repr(supplied))
            continue
        if (
            not lexical.is_absolute()
            or lexical not in expected_set
            or lexical.resolve(strict=False) != lexical
            or _existing_component_is_symlink(lexical.parent, root=root)
        ):
            invalid.append(str(lexical))
            continue
        if not isinstance(payload, Mapping):
            raise ValueError(f"Campaign manifest payload at {lexical} is not a mapping.")
        indexed[lexical] = payload
    missing = sorted(str(path) for path in expected_set - set(indexed))
    extra = sorted(str(path) for path in set(indexed) - expected_set)
    if invalid or missing or extra or len(indexed) != len(expected):
        raise ValueError(
            "Campaign manifest mapping keys differ from the exact canonical set: "
            f"missing={missing}, extra={extra}, invalid_aliases={sorted(invalid)}."
        )
    return {path: indexed[path] for path in expected}


def _planned_output_attempt_paths(
    indexed: Mapping[Path, Mapping[str, Any]], *, root: Path
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for manifest in indexed.values():
        jobs = manifest.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError("Campaign manifest jobs must be a list before publication.")
        for job in jobs:
            if not isinstance(job, Mapping):
                raise ValueError("Campaign publication encountered a non-mapping job.")
            raw = Path(str(job.get("output_dir", ""))).expanduser()
            if not raw.is_absolute():
                raise ValueError("Campaign output attempt paths must be absolute.")
            resolved = raw.resolve(strict=False)
            if resolved != raw or _existing_component_is_symlink(raw.parent, root=root):
                raise ValueError(f"Campaign output attempt path is an alias/symlink: {raw}.")
            paths.append(raw)
    if len(paths) != len(set(paths)):
        raise ValueError("Campaign output attempt paths are not globally unique.")
    return tuple(paths)


def _preflight_fresh_output_attempts(paths: Sequence[Path]) -> None:
    occupied = sorted(str(path) for path in paths if path.exists() or path.is_symlink())
    if occupied:
        raise FileExistsError(
            f"Refusing campaign publication because output attempts already exist: {occupied}."
        )


def _preflight_canonical_cohort_root(cohort_root: Path, *, root: Path) -> None:
    if _existing_component_is_symlink(cohort_root.parent, root=root):
        raise ValueError("Canonical campaign publication path contains a symlink component.")
    if cohort_root.exists() or cohort_root.is_symlink():
        raise FileExistsError(
            f"Refusing to replace an immutable campaign cohort root: {cohort_root}."
        )


def _path_identity(path: Path) -> tuple[int, int]:
    observed = path.lstat()
    return observed.st_dev, observed.st_ino


def _same_path_identity(path: Path, identity: tuple[int, int]) -> bool:
    try:
        return not path.is_symlink() and _path_identity(path) == identity
    except FileNotFoundError:
        return False


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hard_link_new(source: Path, destination: Path) -> tuple[int, int]:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Campaign publication source is not a physical file: {source}.")
    source_identity = _path_identity(source)
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to replace campaign publication member: {destination}."
        ) from exc
    except BaseException:
        if _same_path_identity(destination, source_identity):
            destination.unlink()
        raise
    descriptor = os.open(destination, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        observed = os.fstat(descriptor)
        if (observed.st_dev, observed.st_ino) != source_identity:
            raise RuntimeError("Campaign hard-link identity changed during publication.")
        if observed.st_mode & 0o222:
            raise RuntimeError("Campaign hard-link member is writable before publication.")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return source_identity


def _claim_staging_root(cohort_root: Path, *, root: Path) -> tuple[Path, tuple[int, int]]:
    cohort_root.parent.mkdir(parents=True, exist_ok=True)
    _preflight_canonical_cohort_root(cohort_root, root=root)
    staging = Path(tempfile.mkdtemp(prefix=f".{cohort_root.name}.staging-", dir=cohort_root.parent))
    observed = staging.lstat()
    return staging, (observed.st_dev, observed.st_ino)


def _cleanup_owned_staging(staging: Path, identity: tuple[int, int]) -> None:
    """Delete only the exact private inode claimed by this publisher."""

    try:
        observed = staging.lstat()
    except FileNotFoundError:
        return
    if staging.is_symlink() or (observed.st_dev, observed.st_ino) != identity:
        return
    descendants = list(staging.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        return
    staging.chmod(0o700)
    for directory in (path for path in descendants if path.is_dir()):
        directory.chmod(0o700)
    for path in sorted(descendants, key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    staging.rmdir()


def _write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_snapshot_storage(
    manifest: Mapping[str, Any], *, physical_manifest: Path, logical_manifest: Path
) -> Path:
    descriptor = manifest.get("snapshot_bundle")
    jobs = manifest.get("jobs")
    if not isinstance(descriptor, Mapping) or not isinstance(jobs, list):
        raise ValueError("Staged campaign manifest lacks its immutable snapshot descriptor.")
    logical_snapshot = Path(f"{logical_manifest}.snapshot")
    physical_snapshot = Path(f"{physical_manifest}.snapshot")
    physical_index = physical_snapshot / "index.json"
    if (
        descriptor.get("root_path") != str(logical_snapshot)
        or descriptor.get("index_path") != str(logical_snapshot / "index.json")
        or any(job.get("snapshot_bundle") != descriptor for job in jobs)
        or not physical_snapshot.is_dir()
        or not physical_index.is_file()
    ):
        raise ValueError("Staged campaign snapshot does not bind its logical canonical path.")
    if any(path.is_symlink() for path in physical_snapshot.rglob("*")):
        raise ValueError("Staged campaign snapshot contains a symlink.")
    if _sha256_file(physical_index) != descriptor.get("index_sha256"):
        raise ValueError("Staged campaign snapshot index digest is inconsistent.")
    index = json.loads(physical_index.read_text(encoding="utf-8"))
    if (
        not isinstance(index, Mapping)
        or index.get("benchmark") != manifest.get("benchmark")
        or (
            "implementation_files" in manifest
            and index.get("implementation_files") != manifest.get("implementation_files")
        )
        or (
            "implementation_files_sha256" in manifest
            and index.get("implementation_files_sha256")
            != manifest.get("implementation_files_sha256")
        )
    ):
        raise ValueError("Staged campaign snapshot index differs from the manifest.")
    objects = index.get("objects")
    if not isinstance(objects, Mapping) or len(objects) != descriptor.get("num_objects"):
        raise ValueError("Staged campaign snapshot object count is inconsistent.")
    resolved_snapshot = physical_snapshot.resolve()
    expected_snapshot_files = {Path("index.json")}
    expected_snapshot_directories = {Path("objects"), Path("objects/sha256")}
    for digest, record in objects.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(record, Mapping)
        ):
            raise ValueError("Staged campaign snapshot contains a malformed object record.")
        relative = Path(str(record.get("path", "")))
        expected_snapshot_files.add(relative)
        expected_snapshot_directories.update(relative.parents)
        object_path = (physical_snapshot / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or resolved_snapshot not in object_path.parents
            or object_path.is_symlink()
            or not object_path.is_file()
            or _sha256_file(object_path) != digest
            or object_path.stat().st_size != record.get("size_bytes")
        ):
            raise ValueError("Staged campaign snapshot object authentication failed.")
    expected_snapshot_directories.discard(Path("."))
    actual_snapshot_files = {
        path.relative_to(physical_snapshot)
        for path in physical_snapshot.rglob("*")
        if path.is_file()
    }
    actual_snapshot_directories = {
        path.relative_to(physical_snapshot)
        for path in physical_snapshot.rglob("*")
        if path.is_dir()
    }
    if (
        actual_snapshot_files != expected_snapshot_files
        or actual_snapshot_directories != expected_snapshot_directories
    ):
        raise ValueError("Staged campaign snapshot contains unindexed files/directories.")
    input_index = index.get("input_files")
    if not isinstance(input_index, Mapping):
        raise ValueError("Staged campaign snapshot lacks its input-file index.")
    for job in jobs:
        input_files = job.get("input_files")
        if input_files is None:
            continue
        if not isinstance(input_files, Mapping):
            raise ValueError("Staged campaign job input_files is malformed.")
        for record in input_files.values():
            if not isinstance(record, Mapping):
                raise ValueError("Staged campaign input binding is malformed.")
            original = str(Path(str(record.get("path", ""))).resolve())
            frozen = input_index.get(original)
            if not isinstance(frozen, Mapping) or frozen.get("sha256") != record.get("sha256"):
                raise ValueError("Staged campaign snapshot omitted a frozen input binding.")
    return physical_index


def _authenticate_staged_manifest(
    *,
    physical_path: Path,
    logical_path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if physical_path.is_symlink() or not physical_path.is_file():
        raise FileNotFoundError(f"Staged campaign manifest is absent: {physical_path}.")
    loaded = json.loads(physical_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or loaded != dict(expected):
        raise ValueError("Reopened staged campaign manifest differs from writer output.")
    digest = manifest_digest(loaded)
    if loaded.get("manifest_sha256") != digest:
        raise ValueError("Reopened staged campaign manifest digest is inconsistent.")
    sidecar = physical_path.with_suffix(physical_path.suffix + ".sha256")
    if sidecar.is_symlink() or sidecar.read_text(encoding="utf-8").split() != [
        digest,
        physical_path.name,
    ]:
        raise ValueError("Reopened staged campaign manifest sidecar is inconsistent.")
    _verify_snapshot_storage(
        loaded,
        physical_manifest=physical_path,
        logical_manifest=logical_path,
    )
    return loaded


def _cohort_counts(kind: str, validation: Mapping[str, Any]) -> dict[str, int]:
    if kind == "seed_ladder":
        expected = {
            "manifest_count": EXPECTED_LADDER_MANIFESTS,
            "logical_rows": EXPECTED_LADDER_MEDIA,
            "media_rows": EXPECTED_LADDER_MEDIA,
            "unsupported_rows": 0,
            "exact_one_media_rows": 0,
        }
    else:
        expected = {
            "manifest_count": EXPECTED_FINAL_MANIFESTS,
            "logical_rows": EXPECTED_FINAL_LOGICAL,
            "media_rows": EXPECTED_FINAL_MEDIA,
            "unsupported_rows": EXPECTED_FINAL_UNSUPPORTED,
            "exact_one_media_rows": EXPECTED_FINAL_EXACT_ONE_MEDIA,
        }
    observed = {
        key: int(validation.get(key, 0))
        for key in ("manifest_count", "logical_rows", "media_rows", "unsupported_rows")
    }
    observed["exact_one_media_rows"] = int(validation.get("exact_one_media_rows", 0))
    if observed != expected:
        raise RuntimeError(
            f"Campaign cohort validation/count mismatch: expected={expected}, observed={observed}."
        )
    return expected


def _build_cohort_commit(
    *,
    kind: str,
    cohort_root: Path,
    staging_root: Path,
    logical_paths: Sequence[Path],
    manifests: Mapping[Path, Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    for index, logical in enumerate(logical_paths):
        physical = staging_root / logical.name
        sidecar = physical.with_suffix(physical.suffix + ".sha256")
        snapshot_index = Path(f"{physical}.snapshot") / "index.json"
        members.append(
            {
                "index": index,
                "manifest_path": str(logical),
                "manifest_sha256": manifests[logical]["manifest_sha256"],
                "manifest_file_sha256": _sha256_file(physical),
                "manifest_sidecar_path": str(logical.with_suffix(logical.suffix + ".sha256")),
                "manifest_sidecar_file_sha256": _sha256_file(sidecar),
                "snapshot_index_path": str(Path(f"{logical}.snapshot") / "index.json"),
                "snapshot_index_sha256": manifests[logical]["snapshot_bundle"]["index_sha256"],
                "snapshot_index_file_sha256": _sha256_file(snapshot_index),
            }
        )
    counts = _cohort_counts(kind, validation)
    if counts["manifest_count"] != len(members):
        raise RuntimeError("Campaign cohort commit member count drifted.")
    commit: dict[str, Any] = {
        "schema_version": 1,
        "contract": COHORT_COMMIT_CONTRACT,
        "campaign_contract": CAMPAIGN_CONTRACT,
        "cohort_kind": kind,
        "status": "complete_before_any_launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_root": str(cohort_root),
        "counts": counts,
        "members": members,
    }
    commit["members_sha256"] = hashlib.sha256(
        json.dumps(members, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    commit["commit_sha256"] = _canonical_document_sha256(commit, digest_field="commit_sha256")
    return commit


def _authenticate_commit(path: Path, commit: Mapping[str, Any]) -> None:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    loaded_members = loaded.get("members") if isinstance(loaded, Mapping) else None
    members_sha256 = (
        hashlib.sha256(
            json.dumps(loaded_members, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if isinstance(loaded_members, list)
        else None
    )
    if (
        loaded != dict(commit)
        or loaded.get("members_sha256") != members_sha256
        or loaded.get("commit_sha256")
        != _canonical_document_sha256(loaded, digest_field="commit_sha256")
        or path.with_suffix(path.suffix + ".sha256").read_text(encoding="utf-8").split()
        != [_sha256_file(path), path.name]
    ):
        raise RuntimeError("Staged campaign cohort commit failed byte authentication.")


def _write_and_authenticate_commit(staging_root: Path, commit: Mapping[str, Any]) -> None:
    path = staging_root / COHORT_COMMIT_FILENAME
    serialized = json.dumps(dict(commit), indent=2, sort_keys=True) + "\n"
    _write_text_new(path, serialized)
    _write_text_new(
        path.with_suffix(path.suffix + ".sha256"),
        f"{_sha256_file(path)}  {path.name}\n",
    )
    _authenticate_commit(path, commit)


def _verify_staging_top_level(staging_root: Path, logical_paths: Sequence[Path]) -> None:
    expected = {COHORT_COMMIT_FILENAME, f"{COHORT_COMMIT_FILENAME}.sha256"}
    for logical in logical_paths:
        expected.update(
            {
                logical.name,
                f"{logical.name}.sha256",
                f"{logical.name}.snapshot",
            }
        )
    actual = {path.name for path in staging_root.iterdir()}
    if actual != expected:
        raise RuntimeError(
            "Campaign staging top-level differs from the exact cohort: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
        )


def _freeze_tree(staging_root: Path) -> None:
    paths = list(staging_root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Campaign staging tree contains a symlink.")
    for path in paths:
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in sorted(
        (path for path in paths if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    staging_root.chmod(0o555)
    descriptor = os.open(staging_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_uncommitted_campaign_claim(
    cohort_root: Path,
    *,
    root_identity: tuple[int, int],
    directory_identities: Mapping[Path, tuple[int, int]],
    linked_identities: Mapping[Path, tuple[int, int]],
) -> None:
    """Remove only publisher-owned links from a caught unadmitted claim."""

    if _same_path_identity(cohort_root, root_identity):
        cohort_root.chmod(0o700)
    for directory, identity in directory_identities.items():
        if _same_path_identity(directory, identity):
            directory.chmod(0o700)
    for destination, identity in reversed(tuple(linked_identities.items())):
        if _same_path_identity(destination, identity):
            destination.unlink()
    for directory, identity in sorted(
        directory_identities.items(),
        key=lambda item: len(item[0].parts),
        reverse=True,
    ):
        if directory == cohort_root or not _same_path_identity(directory, identity):
            continue
        try:
            directory.rmdir()
        except OSError:
            pass
    if _same_path_identity(cohort_root, root_identity):
        try:
            cohort_root.rmdir()
        except OSError:
            pass


def _validate_campaign_precommit_tree(
    *,
    cohort_root: Path,
    staging_root: Path,
    logical_paths: Sequence[Path],
    staged: Mapping[Path, Mapping[str, Any]],
    commit: Mapping[str, Any],
    validate: Callable[[ManifestReader], Mapping[str, Any]],
    validation: Mapping[str, Any],
    root_identity: tuple[int, int],
    directory_identities: Mapping[Path, tuple[int, int]],
    linked_identities: Mapping[Path, tuple[int, int]],
    root: Path,
) -> None:
    if not _same_path_identity(cohort_root, root_identity):
        raise RuntimeError("Campaign cohort root identity changed before commit.")
    for directory, identity in directory_identities.items():
        if not _same_path_identity(directory, identity):
            raise RuntimeError("Campaign cohort directory identity changed before commit.")
    for destination, identity in linked_identities.items():
        if not _same_path_identity(destination, identity) or destination.stat().st_mode & 0o222:
            raise RuntimeError("Campaign cohort member changed before commit.")
    expected_files = {
        path.relative_to(staging_root)
        for path in staging_root.rglob("*")
        if path.is_file() and path.name != COHORT_COMMIT_FILENAME
    }
    actual_files = {
        path.relative_to(cohort_root) for path in cohort_root.rglob("*") if path.is_file()
    }
    expected_directories = {
        path.relative_to(staging_root) for path in staging_root.rglob("*") if path.is_dir()
    }
    actual_directories = {
        path.relative_to(cohort_root) for path in cohort_root.rglob("*") if path.is_dir()
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise RuntimeError("Campaign cohort precommit tree is incomplete or contains extras.")
    sidecar_name = f"{COHORT_COMMIT_FILENAME}.sha256"
    if (cohort_root / sidecar_name).read_bytes() != (staging_root / sidecar_name).read_bytes():
        raise RuntimeError("Campaign cohort commit sidecar changed before commit.")
    reopened: dict[Path, dict[str, Any]] = {}
    for logical in logical_paths:
        reopened[logical] = _authenticate_staged_manifest(
            physical_path=logical,
            logical_path=logical,
            expected=staged[logical],
        )
    result = validate(_mapping_manifest_reader(reopened, root))
    if result != validation:
        raise RuntimeError("Canonical campaign precommit topology changed from staging.")
    _authenticate_commit(staging_root / COHORT_COMMIT_FILENAME, commit)


def _freeze_published_campaign_directories(
    cohort_root: Path,
    *,
    root_identity: tuple[int, int],
    directory_identities: Mapping[Path, tuple[int, int]],
) -> None:
    for directory, identity in sorted(
        directory_identities.items(),
        key=lambda item: len(item[0].parts),
        reverse=True,
    ):
        if not _same_path_identity(directory, identity):
            raise RuntimeError("Campaign cohort directory changed before final freeze.")
        directory.chmod(0o555)
        _fsync_directory(directory)
    if not _same_path_identity(cohort_root, root_identity):
        raise RuntimeError("Campaign cohort root changed before final freeze.")
    cohort_root.chmod(0o555)
    _fsync_directory(cohort_root)


def _publish_campaign_commit_last(
    *,
    staging_root: Path,
    cohort_root: Path,
    logical_paths: Sequence[Path],
    staged: Mapping[Path, Mapping[str, Any]],
    commit: Mapping[str, Any],
    validate: Callable[[ManifestReader], Mapping[str, Any]],
    validation: Mapping[str, Any],
    output_paths: Sequence[Path],
    root: Path,
    staging_identity: tuple[int, int],
) -> None:
    """Hard-link a complete immutable tree and expose its commit file last."""

    _verify_staging_top_level(staging_root, logical_paths)
    descendants = list(staging_root.rglob("*"))
    if any(path.is_symlink() for path in descendants) or any(
        path.is_file() and path.stat().st_mode & 0o222 for path in descendants
    ):
        raise ValueError("Frozen campaign staging tree is writable or contains symlinks.")
    _authenticate_commit(staging_root / COHORT_COMMIT_FILENAME, commit)
    _preflight_canonical_cohort_root(cohort_root, root=root)
    try:
        cohort_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FileExistsError(f"Competing campaign publisher claimed {cohort_root}.") from exc
    root_identity = _path_identity(cohort_root)
    directory_identities: dict[Path, tuple[int, int]] = {}
    linked_identities: dict[Path, tuple[int, int]] = {}
    admitted = False
    try:
        source_directories = sorted(
            (path for path in descendants if path.is_dir()),
            key=lambda path: len(path.parts),
        )
        for source in source_directories:
            relative = source.relative_to(staging_root)
            destination = cohort_root / relative
            parent_identity = (
                root_identity
                if destination.parent == cohort_root
                else directory_identities.get(destination.parent)
            )
            if parent_identity is None or not _same_path_identity(
                destination.parent, parent_identity
            ):
                raise RuntimeError("Campaign destination parent changed during directory claim.")
            destination.mkdir(mode=0o700)
            directory_identities[destination] = _path_identity(destination)

        commit_sidecar = staging_root / f"{COHORT_COMMIT_FILENAME}.sha256"
        sidecar_target = cohort_root / commit_sidecar.name
        if not _same_path_identity(cohort_root, root_identity):
            raise RuntimeError("Campaign cohort root changed before sidecar publication.")
        linked_identities[sidecar_target] = _hard_link_new(commit_sidecar, sidecar_target)
        source_files = sorted(
            (
                path
                for path in descendants
                if path.is_file() and path.name not in {COHORT_COMMIT_FILENAME, commit_sidecar.name}
            ),
            key=lambda path: path.relative_to(staging_root).as_posix(),
        )
        for source in source_files:
            relative = source.relative_to(staging_root)
            destination = cohort_root / relative
            parent_identity = (
                root_identity
                if destination.parent == cohort_root
                else directory_identities.get(destination.parent)
            )
            if parent_identity is None or not _same_path_identity(
                destination.parent, parent_identity
            ):
                raise RuntimeError("Campaign destination parent changed during member linking.")
            linked_identities[destination] = _hard_link_new(source, destination)
        for directory in sorted(
            directory_identities,
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(cohort_root)
        _validate_campaign_precommit_tree(
            cohort_root=cohort_root,
            staging_root=staging_root,
            logical_paths=logical_paths,
            staged=staged,
            commit=commit,
            validate=validate,
            validation=validation,
            root_identity=root_identity,
            directory_identities=directory_identities,
            linked_identities=linked_identities,
            root=root,
        )
        _preflight_fresh_output_attempts(output_paths)
        if not _same_path_identity(cohort_root, root_identity):
            raise RuntimeError("Campaign cohort root changed before commit publication.")
        commit_target = cohort_root / COHORT_COMMIT_FILENAME
        commit_source = staging_root / COHORT_COMMIT_FILENAME
        commit_identity = _path_identity(commit_source)
        try:
            linked_identities[commit_target] = _hard_link_new(commit_source, commit_target)
        except BaseException:
            if _same_path_identity(commit_target, commit_identity):
                linked_identities[commit_target] = commit_identity
            raise
        _fsync_directory(cohort_root)
        _freeze_published_campaign_directories(
            cohort_root,
            root_identity=root_identity,
            directory_identities=directory_identities,
        )
        _authenticate_commit(commit_target, commit)
        _verify_staging_top_level(cohort_root, logical_paths)
        if cohort_root.stat().st_mode & 0o222 or any(
            path.stat().st_mode & 0o222 for path in cohort_root.rglob("*")
        ):
            raise RuntimeError("Published campaign cohort tree contains writable entries.")
        reopened = {
            logical: _authenticate_staged_manifest(
                physical_path=logical,
                logical_path=logical,
                expected=staged[logical],
            )
            for logical in logical_paths
        }
        if validate(_mapping_manifest_reader(reopened, root)) != validation:
            raise RuntimeError("Published campaign cohort topology failed reauthentication.")
        admitted = True
        _cleanup_owned_staging(staging_root, staging_identity)
    except BaseException:
        if not admitted:
            _cleanup_uncommitted_campaign_claim(
                cohort_root,
                root_identity=root_identity,
                directory_identities=directory_identities,
                linked_identities=linked_identities,
            )
        raise


def _publish_campaign_cohort(
    manifests: Mapping[Path, Mapping[str, Any]],
    *,
    paths: Sequence[Path],
    kind: str,
    root: Path,
    validate: Callable[[ManifestReader], Mapping[str, Any]],
) -> tuple[Path, ...]:
    logical_paths = tuple(paths)
    indexed = _strict_manifest_mapping(manifests, logical_paths, root=root)
    for path, manifest in indexed.items():
        if manifest_digest(dict(manifest)) != manifest.get("manifest_sha256"):
            raise ValueError(f"Campaign input manifest digest is inconsistent: {path}.")
    validation = validate(_mapping_manifest_reader(indexed, root))
    output_paths = _planned_output_attempt_paths(indexed, root=root)
    _preflight_fresh_output_attempts(output_paths)
    cohort_root = logical_paths[0].parent
    if any(path.parent != cohort_root for path in logical_paths):
        raise RuntimeError("Campaign publication paths do not share one cohort root.")
    staging, identity = _claim_staging_root(cohort_root, root=root)
    try:
        staged: dict[Path, dict[str, Any]] = {}
        for logical in logical_paths:
            physical = staging / logical.name
            candidate = deepcopy(dict(indexed[logical]))
            write_manifest_immutable(
                candidate,
                physical,
                root,
                logical_publication_path=logical,
                allow_atomic_campaign_cohort_staging=True,
            )
            staged[logical] = _authenticate_staged_manifest(
                physical_path=physical,
                logical_path=logical,
                expected=candidate,
            )

        # Re-run the complete 288/504 topology validator from reopened bytes,
        # rather than trusting either the caller mapping or writer-side state.
        reopened_validation = validate(_mapping_manifest_reader(staged, root))
        if reopened_validation != validation:
            raise RuntimeError("Reopened campaign topology differs from preflight validation.")
        commit = _build_cohort_commit(
            kind=kind,
            cohort_root=cohort_root,
            staging_root=staging,
            logical_paths=logical_paths,
            manifests=staged,
            validation=reopened_validation,
        )
        _write_and_authenticate_commit(staging, commit)

        # A final late-byte pass detects corruption after an earlier member was
        # written and proves every commit member before the single public rename.
        for logical in logical_paths:
            loaded = _authenticate_staged_manifest(
                physical_path=staging / logical.name,
                logical_path=logical,
                expected=staged[logical],
            )
            member = next(
                member for member in commit["members"] if member["manifest_path"] == str(logical)
            )
            physical = staging / logical.name
            if (
                loaded["manifest_sha256"] != member["manifest_sha256"]
                or _sha256_file(physical) != member["manifest_file_sha256"]
                or _sha256_file(physical.with_suffix(physical.suffix + ".sha256"))
                != member["manifest_sidecar_file_sha256"]
                or _sha256_file(Path(f"{physical}.snapshot") / "index.json")
                != member["snapshot_index_file_sha256"]
            ):
                raise RuntimeError("Campaign commit member binding changed before publication.")
        _authenticate_commit(staging / COHORT_COMMIT_FILENAME, commit)
        _verify_staging_top_level(staging, logical_paths)
        _preflight_fresh_output_attempts(output_paths)
        _preflight_canonical_cohort_root(cohort_root, root=root)
        _freeze_tree(staging)
        _publish_campaign_commit_last(
            staging_root=staging,
            cohort_root=cohort_root,
            logical_paths=logical_paths,
            staged=staged,
            commit=commit,
            validate=validate,
            validation=reopened_validation,
            output_paths=output_paths,
            root=root,
            staging_identity=identity,
        )
    except BaseException:
        _cleanup_owned_staging(staging, identity)
        raise
    return logical_paths


def publish_production_seed_ladder(
    manifests: Mapping[Path, Mapping[str, Any]], *, root: str | Path
) -> tuple[Path, ...]:
    """Atomically publish the exact eight-manifest production seed ladder."""

    resolved_root = Path(root).expanduser().resolve()
    paths = canonical_ladder_manifest_paths(resolved_root)
    return _publish_campaign_cohort(
        manifests,
        paths=paths,
        kind="seed_ladder",
        root=resolved_root,
        validate=lambda reader: validate_production_seed_ladder(
            paths, root=resolved_root, manifest_reader=reader
        ),
    )


def publish_production_final_campaign(
    manifests: Mapping[Path, Mapping[str, Any]],
    *,
    root: str | Path,
) -> tuple[Path, ...]:
    """Publish 72 final manifests only against the live committed selections.

    The validation callback reopens the atomic selection cohort at every
    caller, staging, precommit, and post-publication check.  Its commit and
    ordered semantic digest are part of the returned validation, so the
    campaign transaction aborts if that authority disappears or changes.
    """

    resolved_root = Path(root).expanduser().resolve()
    paths = canonical_final_manifest_paths(resolved_root)
    return _publish_campaign_cohort(
        manifests,
        paths=paths,
        kind="selected_seed_final",
        root=resolved_root,
        validate=lambda reader: _validate_production_final_against_committed_selection_cohort(
            paths,
            root=resolved_root,
            manifest_reader=reader,
        ),
    )


__all__ = [
    "AXES",
    "CAMPAIGN_CONTRACT",
    "CAMPAIGN_SCHEMA_VERSION",
    "COHORT_COMMIT_CONTRACT",
    "COHORT_COMMIT_FILENAME",
    "EXPECTED_FINAL_EXACT_ONE_MEDIA",
    "EXPECTED_FINAL_LOGICAL",
    "EXPECTED_FINAL_MANIFESTS",
    "EXPECTED_FINAL_MEDIA",
    "EXPECTED_FINAL_UNSUPPORTED",
    "EXPECTED_LADDER_MEDIA",
    "EXPECTED_SELECTION_RECORDS",
    "FINAL_MANIFEST_ROOT_RELATIVE",
    "FINAL_OUTPUT_ROOT_RELATIVE",
    "FLUX1_COMPLETED_CAMPAIGN_CONTRACT",
    "LADDER_MANIFEST_ROOT_RELATIVE",
    "LADDER_OUTPUT_ROOT_RELATIVE",
    "SEEDS",
    "build_final_campaign_preview",
    "build_production_final_campaign",
    "build_production_seed_ladder",
    "build_seed_ladder_preview",
    "canonical_final_manifest_paths",
    "canonical_ladder_manifest_paths",
    "canonical_selection_paths",
    "final_manifest_path",
    "ladder_manifest_path",
    "publish_production_final_campaign",
    "publish_production_seed_ladder",
    "validate_campaign_preview",
    "validate_completed_flux1_production_campaign_v3",
    "validate_final_campaign_preview",
    "validate_production_campaign",
    "validate_production_final_campaign",
    "validate_production_seed_ladder",
    "validate_seed_ladder_preview",
]
