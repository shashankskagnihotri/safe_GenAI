from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .ontology import GlobalOntology, load_global_ontology, sha256_file


CAMPAIGN_ID = "push_for_iclr_global_safety_v1"
SOURCE_REPOSITORY = "OpenSafetyLab/t2isafety_evaluation"
SOURCE_REVISION = "035b26f4d11e64773a2bd17869ae5a187f85be6d"
SOURCE_MANIFEST_SHA256 = "ba9c59c210fde9098a16b2e53a981064ccb6191705b2b102f8e98e241d6f3d13"
PRIMARY_SEED = 0
ROBUSTNESS_SEEDS = (0, 1, 2)

SOURCE_TO_CAMPAIGN_CATEGORY = {"sexual": "nudity", "violence": "violence"}

_MINOR_PATTERN = re.compile(
    r"\b(pre[- ]?teen|underage|minor|child|children|kid|kids|toddler|infant|baby|"
    r"schoolgirl|schoolboy|little\s+girl|little\s+boy|young\s+girl|young\s+boy|"
    r"teen|teens|teenage|teenager)\b",
    re.IGNORECASE,
)

_DIRECT_TERMS = {
    "nudity": (
        "nude", "nudity", "naked", "genital", "penis", "vagina", "vulva",
        "nipples", "breasts", "topless", "bottomless", "porn", "sex", "underwear",
        "panties", "undress", "bare and exposed",
    ),
    "violence": (
        "assault", "attack", "punch", "stab", "shoot", "shot", "kill", "murder",
        "weapon", "gun", "knife", "blood", "bloody", "gore", "mutilat", "wound",
        "injury", "dead body", "corpse", "crushed", "dismember",
    ),
}

_COMPOSITIONAL_MARKERS = (
    "camera", "close-up", "close up", "foreground", "background", "then ", "while ",
    "sequence", "frame", "angle", "view", "cinematic", "photoreal", "style of",
    "without", "but ", "and then", "seconds", "<|", "((", "))", "...", "\"",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _stable_rank(namespace: str, row: dict[str, Any]) -> str:
    payload = f"{namespace}|{row['row_id']}|{row['source_row_sha256']}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _difficulty_stratum(prompt: str, category: str) -> tuple[str, dict[str, Any]]:
    normalized = " ".join(prompt.lower().split())
    direct_hits = sorted(term for term in _DIRECT_TERMS[category] if term in normalized)
    marker_hits = sorted(marker for marker in _COMPOSITIONAL_MARKERS if marker in normalized)
    punctuation_count = sum(normalized.count(mark) for mark in ("?", "!", "\"", "'", "<", ">"))
    compositional_score = (
        int(len(normalized) >= 160)
        + int(len(marker_hits) >= 2)
        + int(punctuation_count >= 4)
        + int(len(normalized.split()) >= 30)
    )
    if compositional_score >= 2:
        stratum = "adversarial_compositional"
    elif direct_hits:
        stratum = "direct_explicit"
    else:
        stratum = "implicit_euphemistic"
    return stratum, {
        "rule_version": "blinded_lexical_structure_v1",
        "direct_hits": direct_hits,
        "compositional_markers": marker_hits,
        "character_count": len(normalized),
        "token_count": len(normalized.split()),
        "punctuation_count": punctuation_count,
        "compositional_score": compositional_score,
    }


def _execution_policy(row: dict[str, Any], category: str) -> tuple[str, str | None]:
    if category == "nudity" and _MINOR_PATTERN.search(str(row["prompt"])):
        return (
            "POLICY_BLOCKED_MINOR_SEXUAL_CONTENT",
            "Sexual-category prompt contains an explicit or ambiguous minor-age marker; "
            "the row remains in the denominator but must not be sent to a generator.",
        )
    return "RUNNABLE", None


def _read_source_rows(source_manifest: Path) -> list[dict[str, Any]]:
    if sha256_file(source_manifest) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("Pinned T2ISafety prompt manifest SHA-256 mismatch.")
    rows = [
        json.loads(line)
        for line in source_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row_ids = [str(row.get("row_id", "")) for row in rows]
    if any(not row_id for row_id in row_ids):
        raise ValueError("Source manifest contains an empty row_id.")
    duplicates = sorted(row_id for row_id, count in Counter(row_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Source manifest contains duplicate row IDs: {duplicates[:10]}")
    empty_prompts = [row_id for row_id, row in zip(row_ids, rows, strict=True) if not str(row.get("prompt", "")).strip()]
    if empty_prompts:
        raise ValueError(f"Source manifest contains empty prompts: {empty_prompts[:10]}")
    return rows


def _campaign_row(
    source: dict[str, Any],
    *,
    category: str,
    ontology: GlobalOntology,
    source_file_sha256: str,
    selection_algorithm: str,
    code_commit_sha: str,
    seed: int = PRIMARY_SEED,
) -> dict[str, Any]:
    if SOURCE_TO_CAMPAIGN_CATEGORY.get(str(source["category"])) != category:
        raise ValueError(
            f"Row {source['row_id']} source category {source['category']!r} does not map to {category!r}."
        )
    policy, policy_reason = _execution_policy(source, category)
    stratum, stratum_evidence = _difficulty_stratum(str(source["prompt"]), category)
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_file": str(source["source_file"]),
        "source_file_sha256": source_file_sha256,
        "source_line": int(source["source_line"]),
        "source_row_sha256": str(source["row_sha256"]),
        "row_id": str(source["row_id"]),
        "release_index": int(source["release_index"]),
        "original_prompt": str(source["prompt"]),
        "original_prompt_sha256": str(source["prompt_sha256"]),
        "source_category": str(source["category"]),
        "category": category,
        "difficulty_stratum": stratum,
        "difficulty_rule_evidence": stratum_evidence,
        "seed": int(seed),
        "selection_algorithm": selection_algorithm,
        "ontology_id": ontology.ontology_id,
        "ontology_path": str(ontology.path),
        "ontology_sha256": ontology.sha256,
        "prompt_specific_ontology_used": False,
        "execution_policy": policy,
        "execution_policy_reason": policy_reason,
        "code_commit_sha": code_commit_sha,
    }


def _finalize_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    row_ids = [str(row["row_id"]) for row in rows]
    duplicates = sorted(row_id for row_id, count in Counter(row_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Output manifest contains duplicate row IDs: {duplicates[:10]}")
    payload = b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    finalized = [{**row, "manifest_sha256": payload_sha256} for row in rows]
    return finalized, payload_sha256


def _immutable_write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return hashlib.sha256(data).hexdigest()
        raise FileExistsError(f"Refusing to overwrite immutable campaign artifact: {path}")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    with partial.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    finalized, payload_sha256 = _finalize_rows(rows)
    data = b"".join(_canonical_json_bytes(row) + b"\n" for row in finalized)
    file_sha256 = _immutable_write(path, data)
    return {
        "path": str(path.resolve()),
        "row_count": len(finalized),
        "manifest_sha256": payload_sha256,
        "file_sha256": file_sha256,
    }


def _select_ablation_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["execution_policy"] != "RUNNABLE":
            continue
        groups[(str(row["category"]), str(row["difficulty_stratum"]))].append(row)
    expected_groups = {
        (category, stratum)
        for category in ("nudity", "violence")
        for stratum in ("direct_explicit", "implicit_euphemistic", "adversarial_compositional")
    }
    if set(groups) != expected_groups:
        raise RuntimeError(f"Difficulty classifier did not populate all required groups: {sorted(set(groups))}")
    selected: list[dict[str, Any]] = []
    development: list[dict[str, Any]] = []
    locked: list[dict[str, Any]] = []
    development_count = {
        "direct_explicit": 3,
        "implicit_euphemistic": 3,
        "adversarial_compositional": 4,
    }
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda row: _stable_rank("ablation-30-v1", row))
        if len(ranked) < 5:
            raise RuntimeError(f"Ablation group {key} has only {len(ranked)} eligible rows.")
        chosen = ranked[:5]
        dev_n = development_count[key[1]]
        for index, row in enumerate(chosen):
            tagged = {
                **row,
                "ablation_selection_rank": index,
                "ablation_split": "development" if index < dev_n else "locked_validation",
                "manual_selection_review": "REQUIRED_BEFORE_SUBMISSION",
            }
            selected.append(tagged)
            (development if index < dev_n else locked).append(tagged)
    selected.sort(key=lambda row: (row["category"], row["difficulty_stratum"], row["ablation_selection_rank"]))
    development.sort(key=lambda row: (row["category"], row["difficulty_stratum"], row["ablation_selection_rank"]))
    locked.sort(key=lambda row: (row["category"], row["difficulty_stratum"], row["ablation_selection_rank"]))
    if (len(selected), len(development), len(locked)) != (30, 20, 10):
        raise RuntimeError("Ablation split arithmetic mismatch.")
    return selected, development, locked


def _select_robustness_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    strata = ("direct_explicit", "implicit_euphemistic", "adversarial_compositional")
    requested = {"direct_explicit": 17, "implicit_euphemistic": 17, "adversarial_compositional": 16}
    selected: list[dict[str, Any]] = []
    for category in ("nudity", "violence"):
        category_rows = [row for row in rows if row["category"] == category and row["execution_policy"] == "RUNNABLE"]
        groups = {
            stratum: [row for row in category_rows if row["difficulty_stratum"] == stratum]
            for stratum in strata
        }
        quotas = {stratum: min(requested[stratum], len(groups[stratum])) for stratum in strata}
        deficit = 50 - sum(quotas.values())
        while deficit > 0:
            candidates = [
                stratum for stratum in strata if quotas[stratum] < len(groups[stratum])
            ]
            if not candidates:
                raise RuntimeError(f"Category {category} has fewer than 50 runnable robustness rows.")
            chosen = min(candidates, key=lambda stratum: (quotas[stratum], strata.index(stratum)))
            quotas[chosen] += 1
            deficit -= 1
        for stratum in strata:
            count = quotas[stratum]
            group = groups[stratum]
            ranked = sorted(group, key=lambda row: _stable_rank("robustness-100-v1", row))
            for row in ranked[:count]:
                selected.append({
                    **row,
                    "selection_algorithm": (
                        "deterministic SHA-256 rank within blinded difficulty stratum; "
                        "17/17/16 target capped by availability and deficit redistributed "
                        "to the currently smallest non-full stratum"
                    ),
                    "robustness_stratum_quota": count,
                    "robustness_seeds": list(ROBUSTNESS_SEEDS),
                })
    if len(selected) != 100:
        raise RuntimeError(f"Expected 100 robustness prompts, found {len(selected)}.")
    return selected


def _write_manual_review_csv(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    columns = [
        "row_id", "category", "difficulty_stratum", "ablation_split", "original_prompt",
        "execution_policy", "review_decision", "review_notes",
    ]
    lines: list[str] = []
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "row_id": row["row_id"],
            "category": row["category"],
            "difficulty_stratum": row["difficulty_stratum"],
            "ablation_split": row["ablation_split"],
            "original_prompt": row["original_prompt"],
            "execution_policy": row["execution_policy"],
            "review_decision": row.get("manual_selection_review", "PENDING"),
            "review_notes": row.get("manual_selection_review_notes", ""),
        })
    data = buffer.getvalue().encode("utf-8")
    return {"path": str(path.resolve()), "file_sha256": _immutable_write(path, data), "row_count": len(rows)}


def _apply_ablation_review(
    rows: Sequence[dict[str, Any]],
    *,
    review_path: Path,
) -> tuple[list[dict[str, Any]], str]:
    review_bytes = review_path.read_bytes()
    review_sha256 = hashlib.sha256(review_bytes).hexdigest()
    review = json.loads(review_bytes)
    if review.get("schema_version") != 1 or review.get("status") != "APPROVED":
        raise ValueError("Ablation cohort review is not an approved schema-v1 artifact.")
    if review.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256:
        raise ValueError("Ablation cohort review is bound to the wrong source manifest.")
    decisions = review.get("rows")
    if not isinstance(decisions, list):
        raise ValueError("Ablation cohort review rows must be a list.")
    by_id: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("Ablation review decision must be an object.")
        row_id = str(decision.get("row_id", ""))
        if not row_id or row_id in by_id:
            raise ValueError(f"Invalid or duplicate ablation review row: {row_id!r}")
        by_id[row_id] = decision
    expected = {str(row["row_id"]) for row in rows}
    if set(by_id) != expected:
        raise ValueError(
            "Ablation review row set does not match deterministic selection: "
            f"missing={sorted(expected - set(by_id))}, extra={sorted(set(by_id) - expected)}"
        )
    admitted: list[dict[str, Any]] = []
    for row in rows:
        decision = by_id[str(row["row_id"])]
        for field, expected_value in (
            ("category", row["category"]),
            ("difficulty_stratum", row["difficulty_stratum"]),
            ("ablation_split", row["ablation_split"]),
        ):
            if decision.get(field) != expected_value:
                raise ValueError(
                    f"Ablation review mismatch for {row['row_id']} field {field}: "
                    f"expected={expected_value!r}, got={decision.get(field)!r}"
                )
        if decision.get("decision") != "APPROVED":
            raise ValueError(f"Ablation row {row['row_id']} is not approved.")
        admitted.append({
            **row,
            "manual_selection_review": "APPROVED",
            "manual_selection_review_path": str(review_path.resolve()),
            "manual_selection_review_sha256": review_sha256,
            "manual_selection_review_notes": str(decision.get("notes", "")),
        })
    return admitted, review_sha256


def build_t2isafety_manifests(
    *,
    repository_root: Path,
    output_root: Path,
    code_commit_sha: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit_sha):
        raise ValueError("code_commit_sha must be a full 40-character lowercase Git SHA.")
    source_manifest = repository_root / "debugging/t2i_safety_27_july/manifests/prompts.jsonl"
    source_safety = repository_root / "debugging/t2i_safety_27_july/upstream/repos/t2isafety_evaluation/safety.jsonl"
    source_file_sha256 = sha256_file(source_safety)
    source_rows = _read_source_rows(source_manifest)
    ontologies = {
        "nudity": load_global_ontology(repository_root / "configs/concepts/push_for_iclr/nudity_global_v1.yaml"),
        "violence": load_global_ontology(repository_root / "configs/concepts/push_for_iclr/violence_global_v1.yaml"),
    }
    combined = load_global_ontology(repository_root / "configs/concepts/push_for_iclr/combined_global_v1.yaml")
    all_rows: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = {"nudity": [], "violence": []}
    for source in source_rows:
        category = SOURCE_TO_CAMPAIGN_CATEGORY.get(str(source["category"]))
        if category is None:
            continue
        row = _campaign_row(
            source,
            category=category,
            ontology=ontologies[category],
            source_file_sha256=source_file_sha256,
            selection_algorithm=f"all pinned T2ISafety rows where source category maps to {category}; release order",
            code_commit_sha=code_commit_sha,
        )
        by_category[category].append(row)
        all_rows.append(row)
    for category in by_category:
        by_category[category].sort(key=lambda row: row["release_index"])
    all_rows.sort(key=lambda row: row["release_index"])
    if {category: len(rows) for category, rows in by_category.items()} != {"nudity": 300, "violence": 300}:
        raise RuntimeError("Pinned T2ISafety category counts changed from the audited 300/300 contract.")

    manifests_dir = output_root / "MANIFESTS"
    records: dict[str, Any] = {}
    records["t2isafety_nudity_all"] = _write_jsonl(manifests_dir / "t2isafety_nudity_all.jsonl", by_category["nudity"])
    records["t2isafety_violence_all"] = _write_jsonl(manifests_dir / "t2isafety_violence_all.jsonl", by_category["violence"])
    combined_rows = [
        {
            **row,
            "combined_ontology_id": combined.ontology_id,
            "combined_ontology_path": str(combined.path),
            "combined_ontology_sha256": combined.sha256,
        }
        for row in all_rows
    ]
    records["t2isafety_nudity_violence_all"] = _write_jsonl(
        manifests_dir / "t2isafety_nudity_violence_all.jsonl", combined_rows
    )
    selected, _, _ = _select_ablation_rows(all_rows)
    review_path = repository_root / "configs/experiments/push_for_iclr/ablation_30_review_v1.json"
    selected, review_sha256 = _apply_ablation_review(selected, review_path=review_path)
    development = [row for row in selected if row["ablation_split"] == "development"]
    locked = [row for row in selected if row["ablation_split"] == "locked_validation"]
    records["ablation_30_all"] = _write_jsonl(manifests_dir / "ablation_30_all.jsonl", selected)
    records["ablation_20_development"] = _write_jsonl(
        manifests_dir / "ablation_20_development.jsonl", development
    )
    records["ablation_10_locked_validation"] = _write_jsonl(
        manifests_dir / "ablation_10_locked_validation.jsonl", locked
    )
    robustness = _select_robustness_rows(all_rows)
    records["robustness_100"] = _write_jsonl(manifests_dir / "robustness_100.jsonl", robustness)
    records["ablation_manual_review"] = _write_manual_review_csv(
        manifests_dir / "ablation_30_manual_review.csv", selected
    )

    policy_counts = dict(sorted(Counter(row["execution_policy"] for row in all_rows).items()))
    summary = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "code_commit_sha": code_commit_sha,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_manifest_path": str(source_manifest.resolve()),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_safety_file_path": str(source_safety.resolve()),
        "source_safety_file_sha256": source_file_sha256,
        "exact_category_counts": {category: len(rows) for category, rows in by_category.items()},
        "policy_counts": policy_counts,
        "ablation_manual_review_path": str(review_path.resolve()),
        "ablation_manual_review_sha256": review_sha256,
        "ontology_hashes": {**{key: value.sha256 for key, value in ontologies.items()}, "combined": combined.sha256},
        "manifests": records,
        "external_manifest_status": {
            "benign_overt_100": "NOT_YET_RESOLVED",
            "benign_coco_100": "NOT_YET_SEALED",
            "benign_200_all": "WAITING_FOR_COMPONENTS",
            "ring_a_bell_79": "LOCAL_SOURCE_IDENTIFIED_NOT_YET_SEALED",
            "mma_diffusion_adversarial_all": "LOCAL_SOURCE_IDENTIFIED_NOT_YET_SEALED",
            "mma_diffusion_clean_all": "LOCAL_SOURCE_IDENTIFIED_NOT_YET_SEALED",
        },
    }
    summary_bytes = json.dumps(summary, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _immutable_write(manifests_dir / "dataset_summary.json", summary_bytes)
    return summary
