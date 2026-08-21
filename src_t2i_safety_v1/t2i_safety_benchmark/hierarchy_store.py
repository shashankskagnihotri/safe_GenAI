from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .contracts import (
    CALIBRATION_ROOT,
    PROMPTS_SHA256,
    WORK_ROOT,
    canonical_sha256,
    file_sha256,
)


HIERARCHY_PROTOCOL = "t2i_safety_concept_hierarchy_admission_v2"
FAIRNESS_OVERLAY_PROTOCOL = "t2i_safety_fairness_overlay_admission_v2"
HIERARCHY_ROW_COUNT = 2686
FAIRNESS_ROW_COUNT = 236
HIERARCHY_ROOT = WORK_ROOT / "concept_hierarchy_v2"
SEALED_HIERARCHY = HIERARCHY_ROOT / "SEALED_HIERARCHY.jsonl"
HIERARCHY_ADMISSION = HIERARCHY_ROOT / "ADMISSION.json"
FAIRNESS_OVERLAY_ROOT = CALIBRATION_ROOT / "fairness_overlays_v2"

BASE_REQUIRED_FIELDS = frozenset(
    {
        "row_id",
        "row_sha256",
        "prompt_sha256",
        "category",
        "primary_subtheme_id",
        "secondary_subtheme_ids",
        "target_change",
        "preserved_invariants",
        "unsafe_concept",
        "neutral_concept",
        "safe_sibling_concept",
        "uncertainty",
    }
)
OVERLAY_REQUIRED_FIELDS = frozenset(
    {
        "row_id",
        "row_sha256",
        "prompt_sha256",
        "model_id",
        "base_record_sha256",
        "fairness_profile_sha256",
        "unsafe_concept",
        "neutral_concept",
        "safe_sibling_concept",
        "preserved_invariants",
    }
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing sealed hierarchy evidence {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Hierarchy evidence is not a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing sealed hierarchy manifest {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise RuntimeError(f"Blank hierarchy line {line_number} in {path}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(
                f"Hierarchy line {line_number} is not an object in {path}"
            )
        records.append(value)
    return records


def _require_sha(value: Any, *, field: str, path: Path) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise RuntimeError(f"{field} is not a lowercase SHA-256 digest in {path}")
    return text


def _require_git_revision(value: Any, *, field: str, path: Path) -> str:
    text = str(value)
    if len(text) != 40 or any(c not in "0123456789abcdef" for c in text):
        raise RuntimeError(f"{field} is not a Git commit revision in {path}")
    return text


def _require_endpoint(record: dict[str, Any], field: str, *, path: Path) -> str:
    value = record.get(field)
    if not isinstance(value, str) or len(value.split()) < 3:
        raise RuntimeError(
            f"Hierarchy field {field} is not a concrete visual concept "
            f"for {record.get('row_id')} in {path}"
        )
    return value.strip()


def _validate_base_record(record: dict[str, Any], *, path: Path) -> None:
    missing = BASE_REQUIRED_FIELDS - set(record)
    if missing:
        raise RuntimeError(
            f"Hierarchy record {record.get('row_id')} lacks {sorted(missing)}"
        )
    _require_sha(record["row_sha256"], field="row_sha256", path=path)
    _require_sha(record["prompt_sha256"], field="prompt_sha256", path=path)
    for field in ("unsafe_concept", "neutral_concept", "safe_sibling_concept"):
        _require_endpoint(record, field, path=path)
    if not isinstance(record["secondary_subtheme_ids"], list):
        raise RuntimeError(
            f"secondary_subtheme_ids is not a list for {record['row_id']}"
        )
    if (
        not isinstance(record["preserved_invariants"], list)
        or not record["preserved_invariants"]
    ):
        raise RuntimeError(
            f"preserved_invariants is empty for {record['row_id']}"
        )


@lru_cache(maxsize=1)
def _base_store() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    admission = _load_json(HIERARCHY_ADMISSION)
    if admission.get("protocol") != HIERARCHY_PROTOCOL:
        raise RuntimeError(
            f"Hierarchy admission protocol mismatch in {HIERARCHY_ADMISSION}"
        )
    if admission.get("status") != "passed":
        raise RuntimeError("The dataset-wide hierarchy was not admitted")
    if int(admission.get("row_count", -1)) != HIERARCHY_ROW_COUNT:
        raise RuntimeError("The dataset-wide hierarchy is not complete")
    if admission.get("prompt_manifest_sha256") != PROMPTS_SHA256:
        raise RuntimeError("The hierarchy is bound to a different prompt release")
    if admission.get("independent_verification_status") != "passed":
        raise RuntimeError("The hierarchy lacks independent admission")
    _require_sha(
        admission.get("taxonomy_sha256"),
        field="taxonomy_sha256",
        path=HIERARCHY_ADMISSION,
    )
    _require_git_revision(
        admission.get("independent_verifier_revision"),
        field="independent_verifier_revision",
        path=HIERARCHY_ADMISSION,
    )
    manifest_sha = file_sha256(SEALED_HIERARCHY)
    if admission.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("The sealed hierarchy hash does not match admission")

    records = _load_jsonl(SEALED_HIERARCHY)
    if len(records) != HIERARCHY_ROW_COUNT:
        raise RuntimeError(
            f"Expected {HIERARCHY_ROW_COUNT} hierarchy rows, observed {len(records)}"
        )
    store: dict[str, dict[str, Any]] = {}
    for record in records:
        _validate_base_record(record, path=SEALED_HIERARCHY)
        row_id = str(record["row_id"])
        if row_id in store:
            raise RuntimeError(f"Duplicate hierarchy row {row_id}")
        store[row_id] = record
    return store, {
        "protocol": HIERARCHY_PROTOCOL,
        "manifest_path": str(SEALED_HIERARCHY),
        "manifest_sha256": manifest_sha,
        "admission_path": str(HIERARCHY_ADMISSION),
        "admission_sha256": file_sha256(HIERARCHY_ADMISSION),
        "taxonomy_sha256": admission["taxonomy_sha256"],
        "prompt_manifest_sha256": PROMPTS_SHA256,
        "independent_verifier_revision": admission[
            "independent_verifier_revision"
        ],
    }


@lru_cache(maxsize=None)
def _fairness_overlay(
    model_id: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    model_root = FAIRNESS_OVERLAY_ROOT / model_id
    manifest = model_root / "SEALED_OVERLAY.jsonl"
    admission_path = model_root / "ADMISSION.json"
    admission = _load_json(admission_path)
    if admission.get("protocol") != FAIRNESS_OVERLAY_PROTOCOL:
        raise RuntimeError(f"Fairness overlay protocol mismatch in {admission_path}")
    if admission.get("status") != "passed":
        raise RuntimeError(f"Fairness overlay for {model_id} was not admitted")
    if admission.get("model_id") != model_id:
        raise RuntimeError(f"Fairness overlay identity mismatch for {model_id}")
    if int(admission.get("row_count", -1)) != FAIRNESS_ROW_COUNT:
        raise RuntimeError(f"Fairness overlay for {model_id} is incomplete")
    base_records, base_provenance = _base_store()
    if (
        admission.get("base_hierarchy_sha256")
        != base_provenance["manifest_sha256"]
    ):
        raise RuntimeError(
            f"Fairness overlay for {model_id} is bound to another hierarchy"
        )
    profile_sha = _require_sha(
        admission.get("fairness_profile_sha256"),
        field="fairness_profile_sha256",
        path=admission_path,
    )
    _require_sha(
        admission.get("visual_admission_sha256"),
        field="visual_admission_sha256",
        path=admission_path,
    )
    manifest_sha = file_sha256(manifest)
    if admission.get("manifest_sha256") != manifest_sha:
        raise RuntimeError(
            f"Fairness overlay hash does not match admission for {model_id}"
        )

    records = _load_jsonl(manifest)
    if len(records) != FAIRNESS_ROW_COUNT:
        raise RuntimeError(
            f"Expected {FAIRNESS_ROW_COUNT} fairness overlays for {model_id}, "
            f"observed {len(records)}"
        )
    store: dict[str, dict[str, Any]] = {}
    for record in records:
        missing = OVERLAY_REQUIRED_FIELDS - set(record)
        if missing:
            raise RuntimeError(
                f"Fairness overlay {record.get('row_id')} lacks {sorted(missing)}"
            )
        row_id = str(record["row_id"])
        if row_id in store:
            raise RuntimeError(f"Duplicate fairness overlay row {row_id}")
        if record["model_id"] != model_id:
            raise RuntimeError(f"Fairness overlay model mismatch for {row_id}")
        base = base_records.get(row_id)
        if base is None:
            raise RuntimeError(f"Fairness overlay references unknown row {row_id}")
        if record["base_record_sha256"] != canonical_sha256(base):
            raise RuntimeError(f"Fairness overlay base hash mismatch for {row_id}")
        if record["fairness_profile_sha256"] != profile_sha:
            raise RuntimeError(f"Fairness profile hash mismatch for {row_id}")
        for field in (
            "unsafe_concept",
            "neutral_concept",
            "safe_sibling_concept",
        ):
            _require_endpoint(record, field, path=manifest)
        store[row_id] = record
    return store, {
        "protocol": FAIRNESS_OVERLAY_PROTOCOL,
        "model_id": model_id,
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha,
        "admission_path": str(admission_path),
        "admission_sha256": file_sha256(admission_path),
        "fairness_profile_sha256": profile_sha,
        "visual_admission_sha256": admission["visual_admission_sha256"],
        "base_hierarchy_sha256": base_provenance["manifest_sha256"],
    }


def load_sealed_concept_pair(*, model_id: str, row: Any) -> dict[str, Any]:
    base_records, base_provenance = _base_store()
    row_id = str(row.row_id)
    base = base_records.get(row_id)
    if base is None:
        raise RuntimeError(f"No sealed hierarchy record exists for {row_id}")
    if (
        base["row_sha256"] != row.row_sha256
        or base["prompt_sha256"] != row.prompt_sha256
        or base["category"] != row.category
    ):
        raise RuntimeError(f"Hierarchy prompt binding mismatch for {row_id}")

    record = base
    overlay_provenance: dict[str, Any] | None = None
    if row.domain == "fairness":
        overlays, overlay_provenance = _fairness_overlay(model_id)
        record = overlays.get(row_id)
        if record is None:
            raise RuntimeError(
                f"No sealed fairness overlay exists for {model_id}/{row_id}"
            )

    record_sha = canonical_sha256(record)
    steering_action = "skip" if base["uncertainty"] == "no_action" else "apply"
    return {
        "unsafe_concept": _require_endpoint(
            record,
            "unsafe_concept",
            path=SEALED_HIERARCHY,
        ),
        "safe_concept": _require_endpoint(
            record,
            "safe_sibling_concept",
            path=SEALED_HIERARCHY,
        ),
        "neutral_concept": _require_endpoint(
            record,
            "neutral_concept",
            path=SEALED_HIERARCHY,
        ),
        "steering_action": steering_action,
        "provenance": {
            "row_id": row_id,
            "record_sha256": record_sha,
            "record_source": "model_fairness_overlay"
            if overlay_provenance is not None
            else "dataset_wide_hierarchy",
            "base_record_sha256": canonical_sha256(base),
            "base": base_provenance,
            "fairness_overlay": overlay_provenance,
            "target_change": record.get("target_change"),
            "preserved_invariants": record["preserved_invariants"],
            "primary_subtheme_id": base["primary_subtheme_id"],
            "secondary_subtheme_ids": base["secondary_subtheme_ids"],
            "uncertainty": base["uncertainty"],
            "steering_action": steering_action,
        },
    }


def validate_hierarchy_admission(model_id: str) -> dict[str, Any]:
    _, base_provenance = _base_store()
    _, overlay_provenance = _fairness_overlay(model_id)
    return {
        "model_id": model_id,
        "base": base_provenance,
        "fairness_overlay": overlay_provenance,
    }


__all__ = ["load_sealed_concept_pair", "validate_hierarchy_admission"]
