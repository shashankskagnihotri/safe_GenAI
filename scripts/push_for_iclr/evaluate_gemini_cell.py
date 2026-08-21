#!/usr/bin/env python3
"""Run two independent Gemini safety/fidelity judges and required adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hierasafe_flow.campaigns.push_for_iclr.judges import (  # noqa: E402
    FidelityAdjudicationResult,
    FidelityJudgeResult,
    JudgeContractError,
    SafetyAdjudicationResult,
    SafetyJudgeResult,
    adjudication_prompt,
    call_gemini_structured,
    canonical_agreement_result,
    fidelity_requires_adjudication,
    render_fidelity_prompt,
    render_safety_prompt,
    safety_requires_adjudication,
)
from hierasafe_flow.campaigns.push_for_iclr.ablation_manifests import (  # noqa: E402
    sha256_file,
    validate_sealed_job_manifest,
)


ResultT = TypeVar("ResultT", bound=BaseModel)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise JudgeContractError(f"Refusing to overwrite evaluation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with partial.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def _load_call(path: Path, schema: type[ResultT]) -> ResultT | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    return schema.model_validate(value["parsed"])


def _make_call(image: Path, prompt: str, schema: type[ResultT], path: Path) -> ResultT:
    existing = _load_call(path, schema)
    if existing is not None:
        return existing
    parsed, raw = call_gemini_structured(image_path=image, prompt_text=prompt, schema=schema)
    _atomic_json(path, {"parsed": parsed.model_dump(mode="json"), "raw": raw})
    return parsed


def _judge_pair(
    *,
    image: Path,
    prompt: str,
    output_dir: Path,
    prefix: str,
    schema: type[ResultT],
    adjudication_schema: type[ResultT],
    trigger: Callable[[ResultT, ResultT], list[str]],
) -> None:
    final_path = output_dir / f"{prefix}_final.json"
    if final_path.exists():
        return
    call_1 = _make_call(image, prompt, schema, output_dir / f"{prefix}_call_1.json")
    call_2 = _make_call(image, prompt, schema, output_dir / f"{prefix}_call_2.json")
    reasons = trigger(call_1, call_2)
    adjudication_path = output_dir / f"{prefix}_adjudication.json"
    if reasons:
        adjudicator_prompt = adjudication_prompt(prompt, call_1, call_2, reasons)
        adjudicated = _make_call(
            image, adjudicator_prompt, adjudication_schema, adjudication_path
        )
        final = adjudicated
        policy = "third_call_adjudication"
    else:
        final, policy = canonical_agreement_result(call_1, call_2)
        _atomic_json(
            adjudication_path,
            {"performed": False, "reason": "no_adjudication_trigger", "trigger_reasons": []},
        )
    _atomic_json(
        final_path,
        {
            "parsed": final.model_dump(mode="json"),
            "resolution_policy": policy,
            "adjudication_trigger_reasons": reasons,
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "judge_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-file-sha256")
    parser.add_argument("--index", type=int)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != args.code_commit:
        parser.error("evaluation worktree HEAD differs from --code-commit")
    if args.cell is not None:
        if args.manifest is not None or args.index is not None:
            parser.error("--cell cannot be combined with --manifest/--index")
        cell = args.cell.resolve()
    else:
        if args.manifest is None or args.index is None or not args.manifest_file_sha256:
            parser.error("--manifest, --manifest-file-sha256, and --index are required together")
        manifest = args.manifest.resolve()
        if sha256_file(manifest) != args.manifest_file_sha256:
            parser.error("generation job manifest file hash changed")
        rows, _ = validate_sealed_job_manifest(manifest)
        if not 0 <= args.index < len(rows) or rows[args.index]["job_index"] != args.index:
            parser.error("evaluation index does not match generation manifest")
        cell = (
            REPOSITORY_ROOT
            / "outputs/PUSH_FOR_ICLR"
            / rows[args.index]["expected_output_relative_path"]
        ).resolve()
    try:
        if not (cell / "_SUCCESS").exists():
            raise JudgeContractError(f"Cell is not complete: {cell}")
        image = cell / "image.png"
        metadata = json.loads((cell / "metadata.json").read_text())
        original_prompt = metadata["original_prompt"]
        category = metadata["job"]["category"]
        output = cell / "evaluations/gemini"
        safety_prompt = render_safety_prompt(original_prompt, category)
        fidelity_prompt = render_fidelity_prompt(original_prompt, category)
        _judge_pair(
            image=image,
            prompt=safety_prompt,
            output_dir=output,
            prefix="safety",
            schema=SafetyJudgeResult,
            adjudication_schema=SafetyAdjudicationResult,
            trigger=safety_requires_adjudication,
        )
        _judge_pair(
            image=image,
            prompt=fidelity_prompt,
            output_dir=output,
            prefix="fidelity",
            schema=FidelityJudgeResult,
            adjudication_schema=FidelityAdjudicationResult,
            trigger=fidelity_requires_adjudication,
        )
        print(f"GEMINI_EVALUATION_COMPLETE {cell}")
        return 0
    except BaseException as exc:
        failure = cell / "evaluations/gemini/_FAILURE.json"
        if not failure.exists():
            try:
                _atomic_json(
                    failure,
                    {
                        "status": "FAILED_EVALUATION",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            except BaseException:
                traceback.print_exc()
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
