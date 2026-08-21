"""Deterministic, provenance-complete external benchmark manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


HEX40 = re.compile(r"^[0-9a-f]{40}$")
WORD = re.compile(r"[A-Za-z0-9']+")


class ExternalManifestError(RuntimeError):
    """Raised when a pinned source or manifest contract is violated."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExternalManifestError(message)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    _require(isinstance(value, dict), f"Expected YAML mapping: {path}")
    return value


def _stable_rank(seed: str, *parts: object) -> str:
    material = "\x1f".join([seed, *(str(part) for part in parts)])
    return _sha256_text(material)


def _normalise_prompt(value: object, *, source: str) -> str:
    _require(isinstance(value, str), f"Non-string prompt in {source}")
    prompt = " ".join(value.split())
    _require(bool(prompt), f"Empty prompt in {source}")
    return prompt


def _verify_file(path: Path, expected_sha256: str) -> str:
    _require(path.is_file(), f"Pinned source file is missing: {path}")
    observed = _sha256_file(path)
    _require(
        observed == expected_sha256,
        f"SHA-256 mismatch for {path}: expected {expected_sha256}, observed {observed}",
    )
    return observed


def _ontology_record(
    config: Mapping[str, Any], ontology_name: str, repository_root: Path
) -> dict[str, Any]:
    spec = config["ontologies"][ontology_name]
    path = repository_root / spec["path"]
    _verify_file(path, spec["sha256"])
    return {
        "ontology_id": spec["id"],
        "ontology_path": spec["path"],
        "ontology_sha256": spec["sha256"],
        "prompt_specific_ontology": False,
    }


def _base_row(
    *,
    config: Mapping[str, Any],
    code_commit: str,
    benchmark: str,
    split: str,
    source_repository: str,
    source_revision: str,
    source_file: str,
    source_file_sha256: str,
    source_row_id: str,
    prompt: str,
    category: str,
    safety_expectation: str,
    ontology: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    selection_algorithm: str,
) -> dict[str, Any]:
    return {
        "manifest_schema_version": "push-for-iclr.prompt-row.v1",
        "campaign_id": config["campaign_id"],
        "benchmark": benchmark,
        "split": split,
        "row_id": f"{benchmark}:{source_row_id}",
        "source_repository": source_repository,
        "source_revision": source_revision,
        "source_file": source_file,
        "source_file_sha256": source_file_sha256,
        "source_row_id": source_row_id,
        "source_row_sha256": _sha256_text(_canonical_json(source_payload)),
        "prompt": prompt,
        "prompt_sha256": _sha256_text(prompt),
        "category": category,
        "safety_expectation": safety_expectation,
        "generation_seed": int(config["generation_seed"]),
        "selection_algorithm": selection_algorithm,
        "selection_seed": config["selection_seed"],
        "execution_policy": "generate_exact_prompt_no_rewrite_no_fallback",
        "code_commit": code_commit,
        **ontology,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _ring_rows(
    config: Mapping[str, Any], code_commit: str, safe_denoiser_root: Path, repository_root: Path
) -> list[dict[str, Any]]:
    spec = config["ring_a_bell"]
    source = safe_denoiser_root / spec["relative_path"]
    _verify_file(source, spec["sha256"])
    records = _read_csv(source)
    _require(len(records) == spec["expected_rows"], "Ring-A-Bell row-count mismatch")
    ontology = _ontology_record(config, spec["ontology"], repository_root)
    rows = []
    for index, record in enumerate(records):
        prompt = _normalise_prompt(record.get(spec["prompt_column"]), source=f"Ring row {index}")
        clean = _normalise_prompt(
            record.get(spec["clean_reference_column"]), source=f"Ring clean row {index}"
        )
        row = _base_row(
            config=config,
            code_commit=code_commit,
            benchmark="ring_a_bell",
            split="nudity_adversarial",
            source_repository=spec["repository"],
            source_revision=spec["revision"],
            source_file=spec["relative_path"],
            source_file_sha256=spec["sha256"],
            source_row_id=f"row_{index + 1:04d}",
            prompt=prompt,
            category=spec["category"],
            safety_expectation="unsafe_target",
            ontology=ontology,
            source_payload=record,
            selection_algorithm="all_source_rows_in_original_order_v1",
        )
        row["clean_reference_prompt"] = clean
        row["clean_reference_prompt_sha256"] = _sha256_text(clean)
        rows.append(row)
    return rows


def _mma_rows(
    config: Mapping[str, Any], code_commit: str, safe_denoiser_root: Path, repository_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = config["mma"]
    source = safe_denoiser_root / spec["relative_path"]
    _verify_file(source, spec["sha256"])
    records = _read_csv(source)
    _require(len(records) == spec["expected_rows"], "MMA row-count mismatch")
    ontology = _ontology_record(config, spec["ontology"], repository_root)
    adversarial_rows: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        source_row_id = f"row_{index + 1:04d}"
        target = _normalise_prompt(
            record.get(spec["target_prompt_column"]), source=f"MMA target row {index}"
        )
        sanitized = _normalise_prompt(
            record.get(spec["sanitized_prompt_column"]), source=f"MMA sanitized row {index}"
        )
        adversarial = _normalise_prompt(
            record.get(spec["adversarial_prompt_column"]), source=f"MMA adversarial row {index}"
        )
        clean = _normalise_prompt(
            record.get(spec["clean_prompt_column"]), source=f"MMA clean row {index}"
        )
        shared = {
            "target_prompt": target,
            "target_prompt_sha256": _sha256_text(target),
            "sanitized_adversarial_prompt": sanitized,
            "sanitized_adversarial_prompt_sha256": _sha256_text(sanitized),
            "paired_source_row_id": source_row_id,
        }
        adversarial_row = _base_row(
            config=config,
            code_commit=code_commit,
            benchmark="mma_adversarial",
            split="nudity_adversarial",
            source_repository=spec["repository"],
            source_revision=spec["revision"],
            source_file=spec["relative_path"],
            source_file_sha256=spec["sha256"],
            source_row_id=source_row_id,
            prompt=adversarial,
            category=spec["category"],
            safety_expectation="unsafe_target",
            ontology=ontology,
            source_payload=record,
            selection_algorithm="all_source_rows_in_original_order_v1",
        )
        adversarial_row.update(shared)
        adversarial_row["paired_clean_prompt_sha256"] = _sha256_text(clean)
        clean_row = _base_row(
            config=config,
            code_commit=code_commit,
            benchmark="mma_clean",
            split="clean_counterpart",
            source_repository=spec["repository"],
            source_revision=spec["revision"],
            source_file=spec["relative_path"],
            source_file_sha256=spec["sha256"],
            source_row_id=source_row_id,
            prompt=clean,
            category="benign_general",
            safety_expectation="benign",
            ontology=ontology,
            source_payload=record,
            selection_algorithm="all_source_rows_in_original_order_v1",
        )
        clean_row.update(shared)
        clean_row["paired_adversarial_prompt_sha256"] = _sha256_text(adversarial)
        adversarial_rows.append(adversarial_row)
        clean_rows.append(clean_row)
    return adversarial_rows, clean_rows


def _overt_rows(
    config: Mapping[str, Any], code_commit: str, overt_root: Path, repository_root: Path
) -> list[dict[str, Any]]:
    spec = config["overt"]
    quota = int(spec["quota_per_category"])
    selected: list[dict[str, Any]] = []
    for source_category, category_spec in spec["categories"].items():
        source = overt_root / category_spec["relative_path"]
        _verify_file(source, category_spec["sha256"])
        with source.open("r", encoding="utf-8") as handle:
            groups = json.load(handle)
        _require(isinstance(groups, list), f"OVERT source is not a list: {source}")
        _require(
            len(groups) == category_spec["expected_groups"],
            f"OVERT group-count mismatch for {source_category}",
        )
        ontology = _ontology_record(config, category_spec["ontology"], repository_root)
        candidates: list[tuple[str, dict[str, Any]]] = []
        for group_index, group in enumerate(groups):
            _require(isinstance(group, dict), f"Malformed OVERT group {source_category}:{group_index}")
            image_prompts = group.get("image_prompts")
            _require(isinstance(image_prompts, list), "OVERT image_prompts must be a list")
            for prompt_index, value in enumerate(image_prompts):
                source_row_id = f"{source_category}:group_{group_index + 1:04d}:prompt_{prompt_index + 1:03d}"
                prompt = _normalise_prompt(value, source=f"OVERT {source_row_id}")
                source_payload = {
                    "seed_prompt": group.get("seed_prompt"),
                    "image_prompt": value,
                    "group_index": group_index,
                    "prompt_index": prompt_index,
                }
                row = _base_row(
                    config=config,
                    code_commit=code_commit,
                    benchmark="overt_sensitive_benign",
                    split=source_category,
                    source_repository=spec["repository"],
                    source_revision=spec["revision"],
                    source_file=category_spec["relative_path"],
                    source_file_sha256=category_spec["sha256"],
                    source_row_id=source_row_id,
                    prompt=prompt,
                    category=category_spec["mapped_category"],
                    safety_expectation="benign_sensitive",
                    ontology=ontology,
                    source_payload=source_payload,
                    selection_algorithm=spec["selection"]["algorithm"],
                )
                row["source_benchmark_category"] = source_category
                row["source_seed_prompt"] = group.get("seed_prompt")
                rank = _stable_rank(config["selection_seed"], "overt", source_row_id, prompt)
                row["selection_rank_sha256"] = rank
                candidates.append((rank, row))
        candidates.sort(key=lambda pair: (pair[0], pair[1]["row_id"]))
        _require(len(candidates) >= quota, f"Insufficient OVERT {source_category} candidates")
        selected.extend(row for _, row in candidates[:quota])
    selected.sort(key=lambda row: (row["category"], row["selection_rank_sha256"], row["row_id"]))
    return selected


def _contains_excluded_word(prompt: str, excluded: set[str]) -> bool:
    return any(token.casefold() in excluded for token in WORD.findall(prompt))


def _coco_rows(
    config: Mapping[str, Any],
    code_commit: str,
    hf_cache_root: Path,
    repository_root: Path,
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ExternalManifestError("pyarrow is required to seal the pinned COCO manifest") from exc

    spec = config["coco"]
    cache_repo = hf_cache_root / "datasets--phiyodr--coco2017"
    source = cache_repo / "snapshots" / spec["revision"] / spec["relative_path"]
    _verify_file(source, spec["sha256"])
    table = pq.read_table(source, columns=["image_id", "ids", "captions"])
    _require(table.num_rows == spec["expected_source_rows"], "COCO source row-count mismatch")
    selection = spec["selection"]
    excluded = {str(word).casefold() for word in selection["exclude_if_word_present"]}
    minimum_words = int(selection["minimum_words"])
    maximum_words = int(selection["maximum_words"])
    ontology = _ontology_record(config, spec["ontology"], repository_root)
    candidates: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for source_index, record in enumerate(table.to_pylist()):
        image_id = int(record["image_id"])
        captions = record["captions"]
        caption_ids = record["ids"]
        _require(len(captions) == len(caption_ids), f"COCO caption/id mismatch for {image_id}")
        per_image: list[tuple[str, int, int, str]] = []
        for caption_index, (caption_id, value) in enumerate(zip(caption_ids, captions)):
            prompt = _normalise_prompt(value, source=f"COCO image {image_id} caption {caption_index}")
            words = WORD.findall(prompt)
            if not minimum_words <= len(words) <= maximum_words:
                continue
            if _contains_excluded_word(prompt, excluded):
                continue
            rank = _stable_rank(
                config["selection_seed"], "coco-caption", image_id, caption_id, prompt
            )
            per_image.append((rank, int(caption_id), caption_index, prompt))
        if not per_image:
            continue
        _, caption_id, caption_index, prompt = min(per_image)
        dedupe_key = prompt.casefold() if selection["casefold_deduplicate"] else prompt
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        source_row_id = f"image_{image_id}:caption_{caption_id}"
        source_payload = {
            "source_index": source_index,
            "image_id": image_id,
            "caption_id": caption_id,
            "caption_index": caption_index,
            "caption": prompt,
        }
        row = _base_row(
            config=config,
            code_commit=code_commit,
            benchmark="coco_benign",
            split=spec["split"],
            source_repository=spec["repository"],
            source_revision=spec["revision"],
            source_file=spec["relative_path"],
            source_file_sha256=spec["sha256"],
            source_row_id=source_row_id,
            prompt=prompt,
            category="benign_general",
            safety_expectation="benign",
            ontology=ontology,
            source_payload=source_payload,
            selection_algorithm=selection["algorithm"],
        )
        row["coco_image_id"] = image_id
        row["coco_caption_id"] = caption_id
        rank = _stable_rank(config["selection_seed"], "coco-image", source_row_id, prompt)
        row["selection_rank_sha256"] = rank
        candidates.append((rank, row))
    candidates.sort(key=lambda pair: (pair[0], pair[1]["row_id"]))
    quota = int(spec["quota"])
    _require(len(candidates) >= quota, "Insufficient ordinary COCO caption candidates")
    return [row for _, row in candidates[:quota]]


def _validate_rows(rows: Sequence[Mapping[str, Any]], expected_count: int, name: str) -> None:
    _require(len(rows) == expected_count, f"{name}: expected {expected_count}, got {len(rows)}")
    row_ids = [row["row_id"] for row in rows]
    _require(len(row_ids) == len(set(row_ids)), f"{name}: duplicate row_id")
    for row in rows:
        _require(row["prompt_sha256"] == _sha256_text(row["prompt"]), f"{name}: prompt hash mismatch")
        _require(row["prompt_specific_ontology"] is False, f"{name}: prompt-specific ontology found")
        _require(row["code_commit"] and HEX40.fullmatch(row["code_commit"]), f"{name}: bad commit")


def _write_immutable(path: Path, payload: bytes) -> str:
    _require(not path.exists(), f"Refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    _require(not partial.exists(), f"Stale partial artifact exists: {partial}")
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return _sha256_bytes(payload)


def _seal_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload_rows = [dict(row) for row in rows]
    payload = "".join(f"{_canonical_json(row)}\n" for row in payload_rows).encode("utf-8")
    manifest_sha256 = _sha256_bytes(payload)
    sealed_rows = [{**row, "manifest_sha256": manifest_sha256} for row in payload_rows]
    physical = "".join(f"{_canonical_json(row)}\n" for row in sealed_rows).encode("utf-8")
    file_sha256 = _write_immutable(path, physical)
    return {
        "path": str(path),
        "row_count": len(rows),
        "manifest_sha256": manifest_sha256,
        "file_sha256": file_sha256,
    }


def _combined_benign_rows(
    overt_rows: Sequence[Mapping[str, Any]], coco_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    combined = []
    for row in [*overt_rows, *coco_rows]:
        copied = dict(row)
        copied["parent_benchmark"] = copied["benchmark"]
        copied["benchmark"] = "benign_overrefusal"
        copied["row_id"] = f"benign_overrefusal:{row['row_id']}"
        copied["split"] = "overt" if row["benchmark"] == "overt_sensitive_benign" else "coco"
        combined.append(copied)
    return combined


def build_external_manifests(
    *,
    config_path: Path,
    output_root: Path,
    code_commit: str,
    repository_root: Path,
    safe_denoiser_root: Path,
    overt_root: Path,
    hf_cache_root: Path,
) -> dict[str, Any]:
    """Build all pinned external manifests, failing closed on every mismatch."""
    _require(bool(HEX40.fullmatch(code_commit)), "code_commit must be a full lowercase Git SHA")
    config = _load_yaml(config_path)
    _require(config.get("schema_version") == "push-for-iclr.external-benchmarks.v1", "Bad config schema")
    config_sha256 = _sha256_file(config_path)

    ring = _ring_rows(config, code_commit, safe_denoiser_root, repository_root)
    mma_adversarial, mma_clean = _mma_rows(
        config, code_commit, safe_denoiser_root, repository_root
    )
    overt = _overt_rows(config, code_commit, overt_root, repository_root)
    coco = _coco_rows(config, code_commit, hf_cache_root, repository_root)
    benign_combined = _combined_benign_rows(overt, coco)

    expected = {
        "ring_a_bell": int(config["ring_a_bell"]["expected_rows"]),
        "mma_adversarial": int(config["mma"]["expected_rows"]),
        "mma_clean": int(config["mma"]["expected_rows"]),
        "overt": 2 * int(config["overt"]["quota_per_category"]),
        "coco": int(config["coco"]["quota"]),
        "benign_combined": 2 * int(config["coco"]["quota"]),
    }
    collections = {
        "ring_a_bell": ring,
        "mma_adversarial": mma_adversarial,
        "mma_clean": mma_clean,
        "overt": overt,
        "coco": coco,
        "benign_combined": benign_combined,
    }
    for name, rows in collections.items():
        _validate_rows(rows, expected[name], name)

    output_names = config["output_manifests"]
    sealed = {
        name: _seal_jsonl(output_root / output_names[name], rows)
        for name, rows in collections.items()
    }
    summary = {
        "schema_version": "push-for-iclr.external-manifest-summary.v1",
        "campaign_id": config["campaign_id"],
        "code_commit": code_commit,
        "config_path": str(config_path.relative_to(repository_root)),
        "config_sha256": config_sha256,
        "selection_seed": config["selection_seed"],
        "generation_seed": config["generation_seed"],
        "manifests": sealed,
        "source_revisions": {
            "safe_denoiser": config["mma"]["revision"],
            "overt": config["overt"]["revision"],
            "coco": config["coco"]["revision"],
        },
        "selection_contracts": {
            "ring_a_bell": "all 79 source rows, original order",
            "mma": "all 1000 source rows, adversarial and clean kept as row-aligned paired manifests",
            "overt": "50 sexual-content plus 50 violence prompts by deterministic SHA-256 rank",
            "coco": "one non-sensitive caption per image, then 100 images by deterministic SHA-256 rank",
        },
    }
    summary_payload = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    summary["summary_file_sha256"] = _write_immutable(
        output_root / output_names["summary"], summary_payload
    )
    return summary
