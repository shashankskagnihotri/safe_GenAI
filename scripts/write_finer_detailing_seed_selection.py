#!/usr/bin/env python3
"""Seal, write, and validate target-blind finer-detailing seed selections.

The ``seal-review`` command turns a completed source-only review draft into an
immutable manual-review document.  Canonical selections are published only by
``publish-cohort``: it builds all 36 prompt/model decisions in memory, reopens
their 288 candidate sources, and exposes the complete cohort atomically.
``validate-cohort`` repeats that complete authentication.  The historical
single-axis ``select`` command is retained only to fail closed with migration
guidance; it can no longer expose a partial canonical selection set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.evaluation.finer_detailing_selection_cohort import (
    AXES,
    publish_selection_cohort,
    read_selection_cohort,
)
from hierasafe_flow.evaluation.target_blind_seed_selection import (
    build_candidate_source_review,
    build_selection_record,
    selection_output_path,
    validate_selection_record,
    write_candidate_source_review_immutable,
)


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {resolved}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(project_root()))
    subparsers = parser.add_subparsers(dest="command", required=True)

    review = subparsers.add_parser(
        "seal-review", help="Seal one completed source-only candidate review draft."
    )
    review.add_argument("--draft-json", required=True)
    review.add_argument("--output-json", required=True)

    select = subparsers.add_parser(
        "select", help="Rejected legacy partial-publication command; use publish-cohort."
    )
    select.add_argument("--request-json", required=True)
    select.add_argument("--output-json")
    select.add_argument("--final-manifest", action="append", default=[])

    validate = subparsers.add_parser(
        "validate", help="Reopen an immutable selection and all of its source evidence."
    )
    validate.add_argument("--selection-json", required=True)
    validate.add_argument("--final-manifest", action="append", default=[])

    publish = subparsers.add_parser(
        "publish-cohort",
        help="Build, authenticate, and atomically publish all 36 seed selections.",
    )
    publish.add_argument("--request-json", required=True)

    subparsers.add_parser(
        "validate-cohort",
        help="Reopen the committed 36-selection cohort and all 288 candidate sources.",
    )
    return parser


def _build_cohort_payloads(request: dict[str, Any], *, root: Path) -> dict[Path, dict[str, Any]]:
    required = {"schema_version", "contract", "selector_identity", "selected_at_utc", "axes"}
    allowed = required | {"protocol_path"}
    if set(request) != required and set(request) != allowed:
        raise ValueError(
            "Cohort request keys must be exactly schema_version, contract, selector_identity, "
            "selected_at_utc, axes, and optional protocol_path."
        )
    if (
        request.get("schema_version") != 1
        or request.get("contract") != "finer_detailing_target_blind_selection_cohort_request_v1"
    ):
        raise ValueError("Unsupported target-blind selection-cohort request schema/contract.")
    axes = request.get("axes")
    if not isinstance(axes, list) or len(axes) != len(AXES):
        raise ValueError("Selection-cohort request must contain exactly 36 ordered axes.")
    observed = []
    payloads: dict[Path, dict[str, Any]] = {}
    for row in axes:
        if not isinstance(row, dict) or set(row) != {
            "prompt_id",
            "model_name",
            "candidate_specs",
        }:
            raise ValueError(
                "Each selection-cohort axis must contain exactly prompt_id, model_name, "
                "and candidate_specs."
            )
        axis = (row["prompt_id"], row["model_name"])
        observed.append(axis)
        arguments = {
            **row,
            "selector_identity": request["selector_identity"],
            "selected_at_utc": request["selected_at_utc"],
        }
        if "protocol_path" in request:
            arguments["protocol_path"] = request["protocol_path"]
        payload = build_selection_record(**arguments, root=root)
        path = selection_output_path(root, payload["prompt_id"], payload["model_name"])
        payloads[path] = payload
    if tuple(observed) != AXES:
        raise ValueError(
            "Selection-cohort axes must appear once each in canonical prompt-major/model order."
        )
    return payloads


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    if args.command == "seal-review":
        draft = _load_object(args.draft_json, "candidate-review draft")
        payload = build_candidate_source_review(**draft)
        output, sidecar = write_candidate_source_review_immutable(
            args.output_json, payload, root=root
        )
        result = {
            "status": "written",
            "review_json": str(output),
            "sidecar": str(sidecar),
            "document_sha256": payload["document_sha256"],
        }
    elif args.command == "select":
        raise RuntimeError(
            "Single-axis canonical publication is disabled because it can expose a partial "
            "scientific state. Use publish-cohort with all 36 axes."
        )
    elif args.command == "validate":
        result = validate_selection_record(
            args.selection_json,
            root=root,
            final_manifest_paths=args.final_manifest,
        )
    elif args.command == "publish-cohort":
        request = _load_object(args.request_json, "selection-cohort request")
        payloads = _build_cohort_payloads(request, root=root)
        written = publish_selection_cohort(payloads, root=root)
        reopened = read_selection_cohort(root=root)
        result = {
            "status": "written",
            "selection_count": len(written),
            "selection_paths": [str(path) for path in written],
            "commit_path": reopened["commit_path"],
            "commit_sha256": reopened["commit_sha256"],
            "commit_file_sha256": reopened["commit_file_sha256"],
            "candidate_bindings": reopened["candidate_bindings"],
        }
    else:
        result = read_selection_cohort(root=root)
        result = {
            key: value for key, value in result.items() if key not in {"selections", "topology"}
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
