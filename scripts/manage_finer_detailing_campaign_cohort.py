#!/usr/bin/env python3
"""Build, publish, validate, or read one exact production campaign cohort.

This command owns manifest-cohort state only.  It never submits, releases, or
monitors a Slurm job.  ``build`` is read-only and keeps all manifests in memory;
``publish`` is the sole mutating mode and enters the existing atomic commit-last
publisher.  ``validate`` and ``read`` both strictly authenticate an already
committed cohort, with ``read`` returning its complete member bindings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from hierasafe_flow.benchmarks.finer_detailing_campaign_launch import (
    COHORT_KINDS,
    SEED_LADDER,
    SELECTED_SEED_FINAL,
    ValidatedCampaignCohort,
    read_campaign_cohort,
)
from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.evaluation import finer_detailing_campaign as campaign
from hierasafe_flow.evaluation import finer_detailing_selection_cohort as selection_cohort


ACTIONS = ("build", "publish", "validate", "read")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(project_root()))
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ACTIONS:
        command = commands.add_parser(action)
        command.add_argument("--cohort-kind", required=True, choices=COHORT_KINDS)
        if action in {"build", "publish"}:
            command.add_argument(
                "--flux1-native-equivalence-acceptance-receipt",
                required=True,
                help=(
                    "Absolute canonical acceptance_receipt.json for the admitted "
                    "FLUX-v3 native-equivalence execution DAG."
                ),
            )
    return parser


def _expected_paths(kind: str, root: Path) -> tuple[Path, ...]:
    if kind == SEED_LADDER:
        return campaign.canonical_ladder_manifest_paths(root)
    if kind == SELECTED_SEED_FINAL:
        return campaign.canonical_final_manifest_paths(root)
    raise ValueError(f"Unknown campaign cohort kind {kind!r}; expected {COHORT_KINDS}.")


def _selection_snapshot(root: Path) -> tuple[dict[str, Any], tuple[Path, ...]]:
    snapshot = selection_cohort.read_selection_cohort(root=root)
    paths = tuple(Path(value) for value in snapshot["selection_paths"])
    expected = selection_cohort.canonical_selection_paths(root)
    if (
        paths != expected
        or paths != campaign.canonical_selection_paths(root)
        or snapshot.get("selection_count") != campaign.EXPECTED_SELECTION_RECORDS
    ):
        raise ValueError("Final campaign requires the exact committed 36-selection cohort.")
    return snapshot, paths


def _build_exact_cohort(
    kind: str,
    root: Path,
    *,
    flux1_native_equivalence_acceptance_receipt_path: str | Path,
) -> tuple[dict[Path, dict[str, Any]], dict[str, Any] | None]:
    if kind == SEED_LADDER:
        manifests = campaign.build_production_seed_ladder(
            root,
            flux1_native_equivalence_acceptance_receipt_path=(
                flux1_native_equivalence_acceptance_receipt_path
            ),
        )
        selection = None
    elif kind == SELECTED_SEED_FINAL:
        selection, selection_paths = _selection_snapshot(root)
        manifests = campaign.build_production_final_campaign(
            root=root,
            flux1_native_equivalence_acceptance_receipt_path=(
                flux1_native_equivalence_acceptance_receipt_path
            ),
        )
        reopened, reopened_paths = _selection_snapshot(root)
        if (
            reopened["commit_sha256"] != selection["commit_sha256"]
            or reopened["commit_file_sha256"] != selection["commit_file_sha256"]
            or reopened_paths != selection_paths
        ):
            raise ValueError("Selection cohort changed while final manifests were built.")
    else:  # pragma: no cover - argparse and _expected_paths both guard this
        raise ValueError(f"Unknown campaign cohort kind {kind!r}.")

    expected = _expected_paths(kind, root)
    if tuple(manifests) != expected:
        raise RuntimeError("Production builder returned a noncanonical campaign path set/order.")
    return manifests, selection


def _manifest_summary(
    kind: str,
    manifests: Mapping[Path, Mapping[str, Any]],
    *,
    selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    logical_rows = sum(int(manifest["num_jobs"]) for manifest in manifests.values())
    media_rows = sum(int(manifest["expected_media_jobs"]) for manifest in manifests.values())
    unsupported_rows = sum(
        int(manifest["expected_not_supported_jobs"]) for manifest in manifests.values()
    )
    expected_counts = (
        {
            "manifest_count": campaign.EXPECTED_LADDER_MANIFESTS,
            "logical_rows": campaign.EXPECTED_LADDER_MEDIA,
            "media_rows": campaign.EXPECTED_LADDER_MEDIA,
            "unsupported_rows": 0,
            "exact_one_media_rows": 0,
        }
        if kind == SEED_LADDER
        else {
            "manifest_count": campaign.EXPECTED_FINAL_MANIFESTS,
            "logical_rows": campaign.EXPECTED_FINAL_LOGICAL,
            "media_rows": campaign.EXPECTED_FINAL_MEDIA,
            "unsupported_rows": campaign.EXPECTED_FINAL_UNSUPPORTED,
            "exact_one_media_rows": campaign.EXPECTED_FINAL_EXACT_ONE_MEDIA,
        }
    )
    observed_counts = {
        "manifest_count": len(manifests),
        "logical_rows": logical_rows,
        "media_rows": media_rows,
        "unsupported_rows": unsupported_rows,
        "exact_one_media_rows": expected_counts["exact_one_media_rows"],
    }
    if observed_counts != expected_counts:
        raise RuntimeError(
            "Built campaign cohort count drift: "
            f"expected={expected_counts}, observed={observed_counts}."
        )
    payload: dict[str, Any] = {
        "cohort_kind": kind,
        "counts": observed_counts,
        "manifest_bindings": [
            {
                "path": str(path),
                "manifest_sha256": manifest["manifest_sha256"],
            }
            for path, manifest in manifests.items()
        ],
    }
    if selection is not None:
        payload["selection_cohort"] = {
            "commit_path": selection["commit_path"],
            "commit_sha256": selection["commit_sha256"],
            "commit_file_sha256": selection["commit_file_sha256"],
            "selection_count": selection["selection_count"],
        }
    return payload


def _authenticated_summary(
    cohort: ValidatedCampaignCohort, *, include_members: bool
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cohort_kind": cohort.kind,
        "cohort_root": str(cohort.root),
        "cohort_commit_path": str(cohort.commit_path),
        "cohort_commit_sha256": cohort.digest,
        "counts": dict(cohort.commit["counts"]),
    }
    if cohort.selection_cohort is not None:
        payload["selection_cohort"] = dict(cohort.selection_cohort)
    if include_members:
        payload["manifest_bindings"] = [
            {
                "path": str(path),
                "manifest_sha256": manifest["manifest_sha256"],
            }
            for path, manifest in zip(cohort.manifest_paths, cohort.manifests, strict=True)
        ]
    return payload


def _publish_exact_cohort(
    kind: str,
    root: Path,
    *,
    flux1_native_equivalence_acceptance_receipt_path: str | Path,
) -> ValidatedCampaignCohort:
    manifests, selection = _build_exact_cohort(
        kind,
        root,
        flux1_native_equivalence_acceptance_receipt_path=(
            flux1_native_equivalence_acceptance_receipt_path
        ),
    )
    if kind == SEED_LADDER:
        published = campaign.publish_production_seed_ladder(manifests, root=root)
    else:
        if selection is None:  # pragma: no cover - guarded by _build_exact_cohort
            raise RuntimeError("Final publication lost its selection-cohort binding.")
        published = campaign.publish_production_final_campaign(manifests, root=root)
    cohort = read_campaign_cohort(kind, root=root)
    if cohort.manifest_paths != published:
        raise RuntimeError("Published campaign paths differ from the authenticated cohort.")
    return cohort


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve(strict=True)
    if args.action == "build":
        manifests, selection = _build_exact_cohort(
            args.cohort_kind,
            root,
            flux1_native_equivalence_acceptance_receipt_path=(
                args.flux1_native_equivalence_acceptance_receipt
            ),
        )
        result = {
            "status": "built_and_validated_in_memory_without_publication",
            **_manifest_summary(args.cohort_kind, manifests, selection=selection),
            "publication_performed": False,
            "submission_performed": False,
        }
    elif args.action == "publish":
        cohort = _publish_exact_cohort(
            args.cohort_kind,
            root,
            flux1_native_equivalence_acceptance_receipt_path=(
                args.flux1_native_equivalence_acceptance_receipt
            ),
        )
        result = {
            "status": "atomically_published_and_authenticated",
            **_authenticated_summary(cohort, include_members=True),
            "publication_performed": True,
            "submission_performed": False,
        }
    elif args.action == "validate":
        cohort = read_campaign_cohort(args.cohort_kind, root=root)
        result = {
            "status": "committed_campaign_cohort_valid",
            **_authenticated_summary(cohort, include_members=False),
            "publication_performed": False,
            "submission_performed": False,
        }
    else:
        cohort = read_campaign_cohort(args.cohort_kind, root=root)
        result = {
            "status": "committed_campaign_cohort_authenticated_and_read",
            **_authenticated_summary(cohort, include_members=True),
            "publication_performed": False,
            "submission_performed": False,
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
