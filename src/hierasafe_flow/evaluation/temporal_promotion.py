"""Atomic admission cohort for the five long-video production promotions.

The five live YAML files cannot be replaced simultaneously on CephFS.  This
module therefore makes production *admission* atomic: one immutable commit-last
bundle binds all five passed Q1/Q2 decisions and the exact bytes expected for
all five post-promotion model configs.  The companion
``temporal_promotion_apply`` transaction journals the sequential live update;
every production reader remains closed until all five live files match and its
final apply commit authenticates.  This module never edits a live model config.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hierasafe_flow.evaluation.temporal_qualification import (
    TEMPORAL_CRITERIA_BY_MODEL,
    canonical_sha256,
    validate_qualification_document,
    validate_temporal_production_gate,
)
from hierasafe_flow.utils.config import load_yaml
from hierasafe_flow.utils.immutable_publication import (
    cleanup_owned_staging,
    freeze_tree,
    publish_hardlink_tree_commit_last,
    require_nonwritable_directories,
)


PROMOTION_SCHEMA_VERSION = 1
PROMOTION_CONTRACT = "finer_detailing_five_model_temporal_production_promotion_v1"
PROMOTION_ROOT_RELATIVE = Path(
    "debugging/temporal_qualification/finer_detailing_temporal_production_promotion_v1"
)
PROMOTION_COMMIT_FILENAME = "promotion_commit.json"
VIDEO_MODEL_ORDER = (
    "cogvideox_5b",
    "hunyuan_video",
    "joyai_echo",
    "ltx_23",
    "wan22_t2v_a14b",
)
MODEL_CONFIG_RELATIVE = {
    "cogvideox_5b": Path("configs/models/t2v_cogvideox_5b.yaml"),
    "hunyuan_video": Path("configs/models/t2v_hunyuan_video.yaml"),
    "joyai_echo": Path("configs/models/t2v_joyai_echo.yaml"),
    "ltx_23": Path("configs/models/t2v_ltx_23.yaml"),
    "wan22_t2v_a14b": Path("configs/models/t2v_wan22_t2v_a14b.yaml"),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)


def canonical_promotion_root(root: Path) -> Path:
    return (root.resolve() / PROMOTION_ROOT_RELATIVE).absolute()


def canonical_promotion_decision_path(root: Path, model_name: str) -> Path:
    if model_name not in VIDEO_MODEL_ORDER:
        raise ValueError(f"Unknown temporal promotion model {model_name!r}.")
    return canonical_promotion_root(root) / "decisions" / f"{model_name}.json"


def canonical_live_model_config_path(root: Path, model_name: str) -> Path:
    if model_name not in VIDEO_MODEL_ORDER:
        raise ValueError(f"Unknown temporal promotion model {model_name!r}.")
    return (root.resolve() / MODEL_CONFIG_RELATIVE[model_name]).absolute()


def _require_no_symlink_components(path: Path, *, root: Path, label: str) -> None:
    if not path.is_absolute() or (path != root and root not in path.parents):
        raise ValueError(f"{label} escaped the project root: {path}.")
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}.")
        if current == root:
            return
        current = current.parent


def _protocol_from_config(
    config: Mapping[str, Any], *, model_name: str
) -> tuple[str, Mapping[str, Any], str]:
    model = config.get("model")
    generation = config.get("generation")
    if not isinstance(model, Mapping) or not isinstance(generation, Mapping):
        raise ValueError(f"Candidate config for {model_name} lacks model/generation mappings.")
    if generation.get("task") != "text_to_video":
        raise ValueError(f"Candidate config for {model_name} is not text_to_video.")
    protocols = [
        (str(key), value)
        for key, value in model.items()
        if str(key).endswith("_temporal_protocol")
    ]
    if len(protocols) != 1 or not isinstance(protocols[0][1], Mapping):
        raise ValueError(f"Candidate config for {model_name} must have one temporal protocol.")
    revision = str(model.get("revision", ""))
    if _REVISION_RE.fullmatch(revision) is None:
        raise ValueError(f"Candidate config for {model_name} has an invalid revision.")
    return protocols[0][0], protocols[0][1], revision


def _load_qualification_phases(root: Path, *, audit: bool = False) -> dict[str, Any]:
    """Open the exact canonical Q1/Q2 cohorts without creating an import cycle."""

    from hierasafe_flow.benchmarks.finer_detailing_qualification import (
        canonical_qualification_cohort_commit_path,
        canonical_qualification_plan_path,
        read_qualification_cohort_commit,
        validate_qualification_plan,
        validate_qualification_plan_for_audit,
    )

    phases: dict[str, Any] = {}
    for phase in ("q1", "q2"):
        validator = (
            validate_qualification_plan_for_audit if audit else validate_qualification_plan
        )
        plan = validator(
            canonical_qualification_plan_path(phase, root),
            root=root,
            expected_phase=phase,
        )
        commit = read_qualification_cohort_commit(
            phase,
            plan=plan.plan,
            manifests=plan.manifests,
            root=root,
        )
        phases[phase] = {
            "validated": plan,
            "commit": commit,
            "commit_path": canonical_qualification_cohort_commit_path(phase, root),
        }
    q1 = phases["q1"]["validated"]
    q2 = phases["q2"]["validated"]
    if q2.upstream_q1 is None or q2.upstream_q1.digest != q1.digest:
        raise ValueError("Temporal promotion Q2 does not bind the exact canonical Q1 cohort.")
    return phases


def _phase_binding(phase: str, record: Mapping[str, Any]) -> dict[str, Any]:
    validated = record["validated"]
    commit = record["commit"]
    commit_path = Path(record["commit_path"])
    return {
        "phase": phase,
        "plan_path": str(validated.path),
        "plan_file_sha256": _sha256_file(validated.path),
        "plan_document_sha256": validated.digest,
        "cohort_commit_path": str(commit_path),
        "cohort_commit_file_sha256": _sha256_file(commit_path),
        "cohort_commit_document_sha256": commit["document_sha256"],
    }


def _validate_decision_q_bindings(
    decision: Mapping[str, Any], *, phases: Mapping[str, Any]
) -> None:
    q1 = phases["q1"]["validated"]
    q2 = phases["q2"]["validated"]
    rows = decision.get("runs")
    if not isinstance(rows, list) or len(rows) != 54:
        raise ValueError("Temporal promotion decision must contain exactly 54 Q1/Q2 rows.")
    counts = {"q1": 0, "q2": 0}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Temporal promotion decision contains a malformed run row.")
        phase = "q2" if row.get("variation") == "04_shapley_concept_steering" else "q1"
        seed = row.get("seed")
        if isinstance(seed, bool) or seed not in {0, 1, 2}:
            raise ValueError("Temporal promotion decision contains an invalid video seed.")
        role = f"video_seed{int(seed):03d}"
        expected_manifest = q2.path.parent / f"{role}.json" if phase == "q2" else q1.path.parent / f"{role}.json"
        bindings = row.get("artifact_bindings")
        source = bindings.get("source_manifest") if isinstance(bindings, Mapping) else None
        if (
            not isinstance(source, Mapping)
            or set(source) != {"path", "sha256"}
            or Path(str(source.get("path", ""))).resolve() != expected_manifest.resolve()
            or source.get("sha256") != _sha256_file(expected_manifest)
        ):
            raise ValueError("Temporal promotion decision is not bound to the exact Q1/Q2 manifest.")
        counts[phase] += 1
    if counts != {"q1": 45, "q2": 9}:
        raise ValueError(f"Temporal promotion decision Q1/Q2 coverage drifted: {counts}.")


def _read_decision_snapshot(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Temporal qualification decision is absent: {path}.")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Temporal qualification decision is not an object: {path}.")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise FileNotFoundError(f"Temporal qualification decision sidecar is absent: {sidecar}.")
    if sidecar.read_text(encoding="utf-8").split() != [
        str(payload.get("document_sha256", "")),
        path.name,
    ]:
        raise ValueError(f"Temporal qualification decision sidecar drifted: {path}.")
    return payload, raw


def _read_decision(path: Path) -> dict[str, Any]:
    return _read_decision_snapshot(path)[0]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _promotion_fault_hook(_step: str) -> None:
    """Test-only crash boundary; production deliberately performs no action."""


def _validate_promotion_tree_shape(promotion_root: Path) -> None:
    expected_root = {
        "decisions",
        "post_promotion_configs",
        PROMOTION_COMMIT_FILENAME,
        f"{PROMOTION_COMMIT_FILENAME}.sha256",
    }
    if promotion_root.is_symlink() or not promotion_root.is_dir():
        raise ValueError("Temporal promotion tree is not a real directory.")
    if {path.name for path in promotion_root.iterdir()} != expected_root:
        raise ValueError("Temporal promotion cohort artifact set is inexact.")
    decisions_root = promotion_root / "decisions"
    configs_root = promotion_root / "post_promotion_configs"
    if decisions_root.is_symlink() or not decisions_root.is_dir():
        raise ValueError("Temporal promotion decisions root is invalid.")
    if configs_root.is_symlink() or not configs_root.is_dir():
        raise ValueError("Temporal promotion config root is invalid.")
    expected_decisions = {
        name
        for model in VIDEO_MODEL_ORDER
        for name in (f"{model}.json", f"{model}.json.sha256")
    }
    expected_configs = {
        name
        for model in VIDEO_MODEL_ORDER
        for name in (
            MODEL_CONFIG_RELATIVE[model].name,
            f"{MODEL_CONFIG_RELATIVE[model].name}.sha256",
        )
    }
    if {path.name for path in decisions_root.iterdir()} != expected_decisions:
        raise ValueError("Temporal promotion decision set is inexact.")
    if {path.name for path in configs_root.iterdir()} != expected_configs:
        raise ValueError("Temporal promotion config snapshot set is inexact.")
    for path in promotion_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Temporal promotion cohort contains a symlink: {path}.")
        if not path.is_dir() and not path.is_file():
            raise ValueError(f"Temporal promotion cohort has an invalid member: {path}.")


def publish_temporal_promotion_cohort(
    *,
    candidate_config_paths: Mapping[str, str | Path],
    decision_paths: Mapping[str, str | Path],
    root: Path,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Publish five decisions/config snapshots; never modify live YAML files."""

    root = root.resolve()
    if set(candidate_config_paths) != set(VIDEO_MODEL_ORDER):
        raise ValueError("Candidate config mapping must contain exactly five video models.")
    if set(decision_paths) != set(VIDEO_MODEL_ORDER):
        raise ValueError("Decision mapping must contain exactly five video models.")
    target = canonical_promotion_root(root)
    _require_no_symlink_components(
        target.parent, root=root, label="Temporal promotion publication path"
    )
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Temporal promotion cohort already exists: {target}.")
    phases = _load_qualification_phases(root, audit=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    observed = staging.lstat()
    staging_identity = (observed.st_dev, observed.st_ino)
    try:
        (staging / "decisions").mkdir()
        (staging / "post_promotion_configs").mkdir()
        model_rows: list[dict[str, Any]] = []
        for model_name in VIDEO_MODEL_ORDER:
            candidate_supplied = Path(candidate_config_paths[model_name]).expanduser()
            decision_supplied = Path(decision_paths[model_name]).expanduser()
            if candidate_supplied.is_symlink() or decision_supplied.is_symlink():
                raise ValueError("Temporal promotion inputs cannot be symlinks.")
            candidate = candidate_supplied.resolve()
            decision_source = decision_supplied.resolve()
            if not candidate.is_file():
                raise FileNotFoundError(f"Candidate config is absent: {candidate}.")
            bundled_config = (
                staging / "post_promotion_configs" / MODEL_CONFIG_RELATIVE[model_name].name
            )
            config_bytes = candidate.read_bytes()
            bundled_config.write_bytes(config_bytes)
            config = load_yaml(bundled_config)
            protocol_key, protocol, revision = _protocol_from_config(
                config, model_name=model_name
            )
            decision, decision_bytes = _read_decision_snapshot(decision_source)
            bundled_decision = staging / "decisions" / f"{model_name}.json"
            canonical_decision = canonical_promotion_decision_path(root, model_name)
            decision_file_sha = hashlib.sha256(decision_bytes).hexdigest()
            expected_gate = {
                "status": "passed",
                "evidence_manifest_path": str(canonical_decision),
                "evidence_manifest_sha256": decision_file_sha,
                "evidence_document_sha256": decision.get("document_sha256"),
            }
            if protocol.get("execution_phase") != "production" or protocol.get(
                "production_gate"
            ) != expected_gate:
                raise ValueError(
                    f"Candidate config for {model_name} does not carry the exact bundled gate."
                )
            live_path = canonical_live_model_config_path(root, model_name)
            _require_no_symlink_components(
                live_path, root=root, label=f"Canonical pilot config for {model_name}"
            )
            if live_path.is_symlink() or not live_path.is_file():
                raise FileNotFoundError(
                    f"Canonical pilot config is absent or aliased: {live_path}."
                )
            pre_promotion_sha = _sha256_file(live_path)
            live_config = load_yaml(live_path)
            if _sha256_file(live_path) != pre_promotion_sha:
                raise RuntimeError(
                    f"Canonical pilot config changed while reading {model_name}."
                )
            live_protocol_key, live_protocol, live_revision = _protocol_from_config(
                live_config, model_name=model_name
            )
            if (
                live_protocol_key != protocol_key
                or live_revision != revision
                or live_protocol.get("execution_phase") != "pilot"
                or live_protocol.get("production_gate") is not None
            ):
                raise ValueError(
                    f"Canonical pre-promotion config for {model_name} is not the exact pilot."
                )
            expected_candidate = deepcopy(live_config)
            expected_protocol = expected_candidate["model"][protocol_key]
            expected_protocol["execution_phase"] = "production"
            expected_protocol["production_gate"] = deepcopy(expected_gate)
            if config != expected_candidate:
                raise ValueError(
                    f"Candidate config for {model_name} changes fields beyond phase/gate."
                )
            validate_qualification_document(
                decision,
                expected_model_name=model_name,
                expected_model_revision=revision,
                expected_temporal_protocol=protocol,
                expected_criteria=TEMPORAL_CRITERIA_BY_MODEL[model_name],
                verify_source_files=True,
            )
            _validate_decision_q_bindings(decision, phases=phases)
            bundled_decision.write_bytes(decision_bytes)
            _write_text(
                bundled_decision.with_suffix(".json.sha256"),
                f"{decision['document_sha256']}  {bundled_decision.name}\n",
            )
            config_sha = hashlib.sha256(config_bytes).hexdigest()
            _write_text(
                bundled_config.with_suffix(bundled_config.suffix + ".sha256"),
                f"{config_sha}  {bundled_config.name}\n",
            )
            model_rows.append(
                {
                    "model_name": model_name,
                    "model_revision": revision,
                    "temporal_protocol_key": protocol_key,
                    "qualification_decision": {
                        "source_path": str(decision_source),
                        "source_file_sha256": decision_file_sha,
                        "document_sha256": decision["document_sha256"],
                        "bundled_path": str(canonical_decision),
                        "bundled_file_sha256": _sha256_file(bundled_decision),
                    },
                    "post_promotion_config": {
                        "live_path": str(live_path),
                        "file_sha256": config_sha,
                        "pre_promotion_file_sha256": pre_promotion_sha,
                        "bundled_path": str(
                            canonical_promotion_root(root)
                            / "post_promotion_configs"
                            / bundled_config.name
                        ),
                        "bundled_file_sha256": _sha256_file(bundled_config),
                    },
                }
            )
        for row in model_rows:
            live_binding = row["post_promotion_config"]
            if _sha256_file(Path(live_binding["live_path"])) != live_binding[
                "pre_promotion_file_sha256"
            ]:
                raise RuntimeError(
                    "A canonical pilot config changed before promotion cohort commit."
                )
        commit: dict[str, Any] = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "contract": PROMOTION_CONTRACT,
            "status": "complete_five_model_promotion_prepared",
            "created_at_utc": created_at_utc or _utc_now(),
            "promotion_root": str(target),
            "qualification_phases": [
                _phase_binding(phase, phases[phase]) for phase in ("q1", "q2")
            ],
            "models": model_rows,
            "model_count": 5,
            "decision_count": 5,
            "post_promotion_config_count": 5,
            "live_yaml_mutated": False,
        }
        commit["document_sha256"] = _document_digest(commit)
        commit_path = staging / PROMOTION_COMMIT_FILENAME
        _write_text(commit_path, json.dumps(commit, indent=2, sort_keys=True) + "\n")
        _write_text(
            commit_path.with_suffix(".json.sha256"),
            f"{commit['document_sha256']}  {commit_path.name}\n",
        )
        _validate_promotion_tree_shape(staging)
        freeze_tree(staging, label="Temporal promotion staging")
        for row in model_rows:
            live_binding = row["post_promotion_config"]
            if _sha256_file(Path(live_binding["live_path"])) != live_binding[
                "pre_promotion_file_sha256"
            ]:
                raise RuntimeError(
                    "A canonical pilot config changed at the promotion commit boundary."
                )
        publish_hardlink_tree_commit_last(
            staging,
            target,
            commit_relative_path=Path(PROMOTION_COMMIT_FILENAME),
            fault_hook=_promotion_fault_hook,
        )
        reopened = read_temporal_promotion_cohort(root=root, require_live_configs=False)
        if reopened != commit:
            raise RuntimeError("Published temporal promotion cohort failed reauthentication.")
        return reopened
    finally:
        cleanup_owned_staging(staging, staging_identity)


def _read_commit(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Temporal promotion commit is absent: {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("document_sha256") != _document_digest(
        payload
    ):
        raise ValueError("Temporal promotion commit digest drifted.")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.is_symlink() or not sidecar.is_file() or sidecar.read_text(
        encoding="utf-8"
    ).split() != [payload["document_sha256"], path.name]:
        raise ValueError("Temporal promotion commit sidecar drifted.")
    return payload


def read_temporal_promotion_cohort(
    *, root: Path, require_live_configs: bool = True
) -> dict[str, Any]:
    """Authenticate the exact five-model cohort and optionally all live YAML bytes."""

    root = root.resolve()
    promotion_root = canonical_promotion_root(root)
    _require_no_symlink_components(
        promotion_root, root=root, label="Temporal promotion cohort path"
    )
    if promotion_root.is_symlink() or not promotion_root.is_dir():
        raise FileNotFoundError(f"Temporal promotion cohort is absent: {promotion_root}.")
    require_nonwritable_directories(
        promotion_root, label="Temporal production promotion cohort"
    )
    _validate_promotion_tree_shape(promotion_root)
    configs_root = promotion_root / "post_promotion_configs"
    commit = _read_commit(promotion_root / PROMOTION_COMMIT_FILENAME)
    required_keys = {
        "schema_version",
        "contract",
        "status",
        "created_at_utc",
        "promotion_root",
        "qualification_phases",
        "models",
        "model_count",
        "decision_count",
        "post_promotion_config_count",
        "live_yaml_mutated",
        "document_sha256",
    }
    if (
        set(commit) != required_keys
        or commit.get("schema_version") != PROMOTION_SCHEMA_VERSION
        or commit.get("contract") != PROMOTION_CONTRACT
        or commit.get("status") != "complete_five_model_promotion_prepared"
        or commit.get("promotion_root") != str(promotion_root)
        or commit.get("model_count") != 5
        or commit.get("decision_count") != 5
        or commit.get("post_promotion_config_count") != 5
        or commit.get("live_yaml_mutated") is not False
    ):
        raise ValueError("Temporal promotion commit identity/count contract drifted.")
    try:
        created = datetime.fromisoformat(str(commit["created_at_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Temporal promotion commit timestamp is invalid.") from exc
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("Temporal promotion commit timestamp must include a timezone.")
    phases = _load_qualification_phases(root, audit=True)
    if commit.get("qualification_phases") != [
        _phase_binding(phase, phases[phase]) for phase in ("q1", "q2")
    ]:
        raise ValueError("Temporal promotion Q1/Q2 cohort bindings drifted.")
    rows = commit.get("models")
    if not isinstance(rows, list) or [row.get("model_name") for row in rows] != list(
        VIDEO_MODEL_ORDER
    ):
        raise ValueError("Temporal promotion must bind five ordered model rows.")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "model_name",
            "model_revision",
            "temporal_protocol_key",
            "qualification_decision",
            "post_promotion_config",
        }:
            raise ValueError("Temporal promotion model row has an invalid shape.")
        model_name = str(row["model_name"])
        decision_path = canonical_promotion_decision_path(root, model_name)
        config_path = configs_root / MODEL_CONFIG_RELATIVE[model_name].name
        decision = _read_decision(decision_path)
        config = load_yaml(config_path)
        protocol_key, protocol, revision = _protocol_from_config(
            config, model_name=model_name
        )
        decision_binding = row["qualification_decision"]
        config_binding = row["post_promotion_config"]
        if (
            not isinstance(decision_binding, Mapping)
            or set(decision_binding)
            != {
                "source_path",
                "source_file_sha256",
                "document_sha256",
                "bundled_path",
                "bundled_file_sha256",
            }
            or not isinstance(config_binding, Mapping)
            or set(config_binding)
            != {
                "live_path",
                "file_sha256",
                "pre_promotion_file_sha256",
                "bundled_path",
                "bundled_file_sha256",
            }
            or row["model_revision"] != revision
            or row["temporal_protocol_key"] != protocol_key
            or decision_binding["document_sha256"] != decision.get("document_sha256")
            or decision_binding["bundled_path"] != str(decision_path)
            or decision_binding["bundled_file_sha256"] != _sha256_file(decision_path)
            or decision_binding["source_file_sha256"] != _sha256_file(decision_path)
            or _SHA256_RE.fullmatch(str(decision_binding["source_file_sha256"])) is None
            or config_binding["live_path"]
            != str(canonical_live_model_config_path(root, model_name))
            or config_binding["file_sha256"] != _sha256_file(config_path)
            or config_binding["bundled_path"] != str(config_path)
            or config_binding["bundled_file_sha256"] != _sha256_file(config_path)
            or _SHA256_RE.fullmatch(str(config_binding["file_sha256"])) is None
            or _SHA256_RE.fullmatch(
                str(config_binding["pre_promotion_file_sha256"])
            )
            is None
        ):
            raise ValueError(f"Temporal promotion bindings drifted for {model_name}.")
        config_sidecar = config_path.with_suffix(config_path.suffix + ".sha256")
        if config_sidecar.read_text(encoding="utf-8").split() != [
            config_binding["file_sha256"],
            config_path.name,
        ]:
            raise ValueError(f"Temporal promotion config sidecar drifted for {model_name}.")
        validate_temporal_production_gate(
            protocol,
            model_name=model_name,
            model_revision=revision,
            criteria_names=TEMPORAL_CRITERIA_BY_MODEL[model_name],
        )
        _validate_decision_q_bindings(decision, phases=phases)
        if require_live_configs:
            live = canonical_live_model_config_path(root, model_name)
            _require_no_symlink_components(
                live, root=root, label=f"Live temporal config for {model_name}"
            )
            if live.is_symlink() or not live.is_file():
                raise RuntimeError(
                    f"Temporal production promotion live config is absent: {model_name}."
                )
            live_sha = _sha256_file(live)
            if live_sha not in {
                config_binding["pre_promotion_file_sha256"],
                config_binding["file_sha256"],
            }:
                raise RuntimeError(
                    f"Temporal production config has uncommitted bytes: {model_name}."
                )
            if live_sha != config_binding["file_sha256"]:
                raise RuntimeError(
                    "Temporal production promotion is incomplete: all five live model "
                    f"configs must match the atomic cohort; mismatch={model_name}."
                )
    if require_live_configs:
        # Import lazily: the apply module consumes the cohort-only reader while
        # creating/resuming its transaction, so a module-level import would be
        # circular.  Manual copying of all five YAMLs is intentionally not an
        # admission edge.
        from hierasafe_flow.evaluation.temporal_promotion_apply import (
            require_temporal_promotion_apply_commit,
        )

        require_temporal_promotion_apply_commit(root=root, promotion_commit=commit)
    return deepcopy(commit)


def require_complete_temporal_production_promotion(*, root: Path) -> dict[str, Any]:
    """Production-builder gate: no model is admitted until all five match."""

    return read_temporal_promotion_cohort(root=root, require_live_configs=True)
