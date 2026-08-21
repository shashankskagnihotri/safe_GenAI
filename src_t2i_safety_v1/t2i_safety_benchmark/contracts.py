from __future__ import annotations

import hashlib
import json
import os
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = PROJECT_ROOT / "debugging" / "t2i_safety_27_july"
SPEC_PATH = WORK_ROOT / "benchmark_spec.yaml"
PROMPTS_PATH = WORK_ROOT / "manifests" / "prompts.jsonl"
MATRIX_PATH = WORK_ROOT / "manifests" / "matrix.jsonl"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "full_t2i_safety"
CALIBRATION_ROOT = WORK_ROOT / "calibration"

BENCHMARK_ID = "t2isafety_full_release_2026_07_27"
PROMPTS_SHA256 = "ba9c59c210fde9098a16b2e53a981064ccb6191705b2b102f8e98e241d6f3d13"
MATRIX_SHA256 = "c1cd0936b85e222c24fda0ba93b8ecb50eae21fa75f27925153d6709476cbf94"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class PromptRow:
    schema_version: int
    row_id: str
    row_sha256: str
    prompt: str
    prompt_sha256: str
    category: str
    domain: str
    release_index: int
    seed: int
    source_file: str
    source_line: int
    source_repo_revision: str
    duplicate_count: int
    duplicate_occurrence: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PromptRow":
        return cls(**value)


@dataclass(frozen=True)
class MatrixCell:
    schema_version: int
    cell_id: str
    cell_sha256: str
    model_id: str
    variant: str
    row_id: str
    prompt_row_sha256: str
    attempt: int
    output_dir: str
    status: str
    reason: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MatrixCell":
        return cls(**value)


class BenchmarkContract:
    def __init__(self, *, verify_large_hashes: bool = True) -> None:
        with SPEC_PATH.open("r", encoding="utf-8") as handle:
            self.spec = yaml.safe_load(handle)
        if self.spec.get("benchmark_id") != BENCHMARK_ID:
            raise RuntimeError("The benchmark specification identity changed.")
        if Path(self.spec.get("output_root", "")).as_posix() != "outputs/full_t2i_safety":
            raise RuntimeError("The benchmark output root changed.")
        if verify_large_hashes:
            observed_prompts = file_sha256(PROMPTS_PATH)
            observed_matrix = file_sha256(MATRIX_PATH)
            if observed_prompts != PROMPTS_SHA256:
                raise RuntimeError(
                    f"Prompt manifest hash mismatch: {observed_prompts} != {PROMPTS_SHA256}"
                )
            if observed_matrix != MATRIX_SHA256:
                raise RuntimeError(
                    f"Matrix manifest hash mismatch: {observed_matrix} != {MATRIX_SHA256}"
                )
        self.models = {str(item["id"]): dict(item) for item in self.spec["models"]}
        self.categories = {
            str(category): dict(value)
            for category, value in self.spec["categories"].items()
        }
        self.variants = tuple(str(value) for value in self.spec["variants"])

    def model(self, model_id: str) -> dict[str, Any]:
        try:
            return dict(self.models[model_id])
        except KeyError as exc:
            raise ValueError(
                f"Unknown model {model_id!r}; expected one of {sorted(self.models)}"
            ) from exc

    def category(self, category: str) -> dict[str, Any]:
        try:
            return dict(self.categories[category])
        except KeyError as exc:
            raise ValueError(
                f"Unknown category {category!r}; expected one of {sorted(self.categories)}"
            ) from exc

    def prompt_rows(self) -> dict[str, PromptRow]:
        rows: dict[str, PromptRow] = {}
        with PROMPTS_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = PromptRow.from_dict(json.loads(line))
                    if row.row_id in rows:
                        raise RuntimeError(f"Duplicate prompt row id {row.row_id}")
                    rows[row.row_id] = row
        if len(rows) != 2686:
            raise RuntimeError(f"Expected 2686 prompt rows, observed {len(rows)}")
        return rows

    def cells(
        self,
        *,
        model_id: str,
        variant: str,
        shard_index: int = 0,
        num_shards: int = 1,
        row_ids: set[str] | None = None,
        include_unsupported: bool = False,
    ) -> list[MatrixCell]:
        self.model(model_id)
        if variant not in self.variants:
            raise ValueError(f"Unknown variant {variant!r}")
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError("Invalid shard coordinates.")
        prompts = self.prompt_rows()
        selected: list[MatrixCell] = []
        with MATRIX_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if value["model_id"] != model_id or value["variant"] != variant:
                    continue
                cell = MatrixCell.from_dict(value)
                if row_ids is not None and cell.row_id not in row_ids:
                    continue
                row = prompts[cell.row_id]
                if row.release_index % num_shards != shard_index:
                    continue
                if cell.prompt_row_sha256 != row.row_sha256:
                    raise RuntimeError(f"Cell {cell.cell_id} is bound to the wrong prompt row.")
                if cell.status != "planned" and not include_unsupported:
                    continue
                expected = OUTPUT_ROOT / model_id / variant / row.domain / row.category / row.row_id / "attempt_001"
                if Path(cell.output_dir) != expected:
                    raise RuntimeError(
                        f"Cell {cell.cell_id} has unexpected output path {cell.output_dir}"
                    )
                selected.append(cell)
        if row_ids is not None:
            planned_rows = {cell.row_id for cell in selected}
            missing = row_ids - planned_rows
            if missing and not include_unsupported:
                raise RuntimeError(
                    f"Requested rows are absent or unsupported for {model_id}/{variant}: "
                    f"{sorted(missing)}"
                )
        return selected


@contextmanager
def staged_attempt(final_directory: str | Path) -> Iterator[Path | None]:
    final_path = Path(final_directory)
    success = final_path / "_SUCCESS.json"
    if success.is_file():
        yield None
        return
    if final_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite incomplete or foreign attempt directory {final_path}"
        )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{final_path.name}.staging-",
            dir=final_path.parent,
        )
    )
    try:
        yield staging
        if not (staging / "image.png").is_file():
            raise RuntimeError(f"Staged attempt has no image: {staging}")
        if not (staging / "metadata.json").is_file():
            raise RuntimeError(f"Staged attempt has no metadata: {staging}")
        os.replace(staging, final_path)
    except Exception as exc:
        atomic_json(
            staging / "_FAILURE.json",
            {
                "schema_version": 1,
                "status": "failed",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "execution": execution_identity(),
            },
        )
        raise


def execution_identity() -> dict[str, Any]:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
    }
