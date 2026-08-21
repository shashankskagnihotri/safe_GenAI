from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .contracts import PROJECT_ROOT, file_sha256, staged_attempt
from .pilot_context import MethodPilotCandidate
from .runner import ShardRunner


def _canonical_line(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _write_immutable(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"Refusing to replace immutable pilot artifact {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.staging-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise FileExistsError(
                    f"Concurrent writer produced different pilot artifact {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


class MethodPilotRunner(ShardRunner):
    def __init__(self, candidate: MethodPilotCandidate) -> None:
        self.candidate = candidate
        super().__init__(
            model_id=candidate.model_id,
            variant=candidate.method,
            shard_index=0,
            num_shards=1,
            row_ids=set(candidate.row_ids),
            midsteer_strength=None,
            sgf_strength=0.03,
            safe_sigma=None,
            safe_scale=None,
            pilot_candidate=candidate,
        )
        cells_by_row = {cell.row_id: cell for cell in self.cells}
        if set(cells_by_row) != set(candidate.row_ids):
            raise RuntimeError("Pilot matrix selection differs from sealed population")
        self.cells = [cells_by_row[row_id] for row_id in candidate.row_ids]

    def _validated_existing_record(
        self,
        *,
        final: Path,
        row_id: str,
        seed: int,
    ) -> dict[str, Any]:
        image_path = final / "image.png"
        metadata_path = final / "metadata.json"
        success_path = final / "_SUCCESS.json"
        for path in (image_path, metadata_path, success_path):
            if not path.is_file():
                raise RuntimeError(f"Completed pilot attempt lacks {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        success = json.loads(success_path.read_text(encoding="utf-8"))
        image_sha256 = file_sha256(image_path)
        if success.get("image_sha256") != image_sha256:
            raise RuntimeError(f"Pilot success marker does not bind {image_path}")
        if metadata.get("image", {}).get("sha256") != image_sha256:
            raise RuntimeError(f"Pilot metadata does not bind {image_path}")
        if metadata.get("seed") != seed:
            raise RuntimeError(f"Pilot metadata seed mismatch in {metadata_path}")
        if metadata.get("prompt_row", {}).get("row_id") != row_id:
            raise RuntimeError(f"Pilot metadata row mismatch in {metadata_path}")
        if metadata.get("calibration_pilot") != self.candidate.provenance():
            raise RuntimeError(f"Pilot metadata candidate mismatch in {metadata_path}")
        return {
            "status": "success",
            "candidate_id": self.candidate.candidate_id,
            "row_id": row_id,
            "seed": seed,
            "image_path": _project_relative(image_path),
            "image_sha256": image_sha256,
        }

    def run(self) -> list[dict[str, Any]]:
        if not self.cells:
            raise RuntimeError("Sealed pilot population selected no runnable cells")
        self.adapter.load()
        records: list[dict[str, Any]] = []
        for cell in self.cells:
            row = self.prompts[cell.row_id]
            if row.category != self.candidate.category:
                raise RuntimeError("Pilot row escaped its sealed category")
            for seed in self.candidate.seeds:
                final = (
                    self.candidate.output_root
                    / row.row_id
                    / f"seed_{seed}"
                    / "attempt_001"
                )
                with staged_attempt(final) as staging:
                    if staging is not None:
                        self._run_cell(
                            cell=cell,
                            row=row,
                            staging=staging,
                            seed_override=seed,
                        )
                records.append(
                    self._validated_existing_record(
                        final=final,
                        row_id=row.row_id,
                        seed=seed,
                    )
                )
        if len(records) != self.candidate.expected_output_count:
            raise RuntimeError("Pilot output count differs from sealed cross-product")
        output_text = "".join(_canonical_line(record) + "\n" for record in records)
        _write_immutable(self.candidate.output_root / "outputs.jsonl", output_text)
        _write_immutable(
            self.candidate.output_root / "PROVENANCE.json",
            _canonical_line(
                {
                    **self.candidate.provenance(),
                    "status": "complete",
                    "output_count": len(records),
                    "outputs_path": _project_relative(
                        self.candidate.output_root / "outputs.jsonl"
                    ),
                    "outputs_sha256": file_sha256(
                        self.candidate.output_root / "outputs.jsonl"
                    ),
                }
            )
            + "\n",
        )
        return records


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run one sealed T2ISafety method-calibration candidate."
    )
    value.add_argument("--manifest", required=True)
    value.add_argument("--candidate-id", required=True)
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    candidate = MethodPilotCandidate.load(args.manifest, args.candidate_id)
    MethodPilotRunner(candidate).run()


if __name__ == "__main__":
    main()
