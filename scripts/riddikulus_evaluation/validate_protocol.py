#!/usr/bin/env python3
"""Focused CPU-only contract validation. Synthetic fixtures, zero provider requests."""
from pathlib import Path
import copy
import json
import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from riddikulus_evaluation import gemini, plots, protocol, rubric


def main():
    checks = []

    def check(name, predicate):
        if not predicate:
            raise AssertionError(name)
        checks.append(name)

    def rejected(name, function):
        try:
            function()
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError(name)

    safety = {"valid_image": True, "invalid_reason": "none", "self_consistency_confirmed": True,
              "refusal_or_placeholder": False, "nudity_amount": 0, "violence_amount": 0,
              "safety_severity": 0, "nudity_cue_visible": False, "violence_cue_visible": False,
              "nudity_evidence": [], "violence_evidence": [], "severity_rationale": "No visible unsafe content."}
    quality = {"valid_image": True, "invalid_reason": "none", "self_consistency_confirmed": True,
               "prompt_distance_oversuppression": 0, "artifacts_distortion": 0, "artifacts_visible": False,
               "preserved_safe_elements": ["geometric scene"], "changed_or_missing_safe_elements": [],
               "visible_artifacts": [], "concise_rationale": "Synthetic fixture only."}
    check("valid safety and quality schemas", not rubric.consistency_issues("safety", safety) and not rubric.consistency_issues("quality", quality))
    contrad = {**safety, "nudity_amount": 2}
    check("positive score without cue flagged without rewriting", bool(rubric.consistency_issues("safety", contrad)) and contrad["nudity_amount"] == 2)
    contradq = {**quality, "artifacts_distortion": 4}
    check("artifact contradiction flagged", "artifact_score_visibility_conflict" in rubric.consistency_issues("quality", contradq))
    rejected("booleans cannot masquerade as integer scores", lambda: rubric.validate_shape({**safety, "nudity_amount": True}, rubric.response_schema("safety")))
    rejected("extra score fields rejected", lambda: rubric.validate_shape({**safety, "extra": 0}, rubric.response_schema("safety")))
    invalid_s = {**safety, "valid_image": False, "invalid_reason": "blank", **{k: None for k in rubric.METRICS[:3]}}
    invalid_q = {**quality, "valid_image": False, "invalid_reason": "blank", **{k: 10 for k in rubric.METRICS[3:]}}
    check("invalid raw safety nulls and quality tens", not rubric.consistency_issues("safety", invalid_s) and not rubric.consistency_issues("quality", invalid_q))
    from google.genai import types
    config = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=rubric.response_schema("safety"), thinking_config=types.ThinkingConfig(thinking_level="high"))
    http = types.HttpOptions(timeout=900_000, retry_options=types.HttpRetryOptions(attempts=1))
    check("installed SDK accepts HIGH and schema with internal retries disabled", config.thinking_config.thinking_level.value == "HIGH" and http.retry_options.attempts == 1)
    check("only two approved key names", gemini.KEY_NAMES == ("GEMINI_API_KEY", "GEMINI_ForestAI_API_KEY"))
    check("text content refusal recognized", gemini._text_refusal("I cannot evaluate this image."))

    with tempfile.TemporaryDirectory(prefix="riddikulus-evaluation-contract-") as tmp:
        root = Path(tmp)
        from PIL import Image, PngImagePlugin
        source = root / "source"; source.mkdir()
        metadata = PngImagePlugin.PngInfo(); metadata.add_text("model", "private-model")
        Image.new("RGB", (16, 12), (40, 80, 120)).save(source / "image.png", pnginfo=metadata)
        protocol.atomic_json(source / "spec.json", {"prompt": "A geometric scene"})
        protocol.atomic_json(source / "timing.json", {"generation_trajectory_seconds_including_committed_prior_segments": 1.0})
        protocol.atomic_json(source / "_GENERATION_SUCCESS", {"spec_sha256": protocol.sha256(source / "spec.json"),
            "artifact_sha256": {"image.png": protocol.sha256(source / "image.png"), "timing.json": protocol.sha256(source / "timing.json")}})
        row = {"cell_id": "private-model/p1", "arm": "Baseline", "model_key": "private-model",
               "prompt_id": "p1", "category": "nudity", "original_prompt": "A geometric scene"}
        for stem, name in (("image", "image.png"), ("spec", "spec.json"), ("timing", "timing.json"), ("generation_receipt", "_GENERATION_SUCCESS")):
            row[stem + "_path"] = str(source / name); row[stem + "_sha256"] = protocol.sha256(source / name)
        images = root / "images.jsonl"; images.write_text(json.dumps(row) + "\n")
        expected = root / "expected.jsonl"
        absent = {**row, "cell_id": "private-model/p2", "prompt_id": "p2", "status": "BLOCKED_CONTENT_GENERATION"}
        expected.write_text(json.dumps(row) + "\n" + json.dumps(absent) + "\n")
        output = root / "evaluation"
        prepared = protocol.prepare_next(images, expected, output, "sol", 2)
        check("exactly two independent stages prepared", {a["stage"] for a in prepared["assignments"]} == {"safety", "quality"})
        check("missing inventory preserved", prepared["expected_images"] == 2 and prepared["verified_available_images"] == 1)
        for i, assignment in enumerate(prepared["assignments"]):
            task_path = Path(assignment["task_path"])
            task = protocol.read_json(task_path)
            check(f"stage {i} neutral pixels and metadata", not Image.open(task["image"]["path"]).info and task["image"]["width"] == 16)
            if task["stage"] == "safety":
                check("safety message hides source and original prompt", all(x not in assignment["spawn_arguments"]["message"] for x in ("private-model", "A geometric scene", str(source))))
            spawn = {k: assignment[k] for k in ("attempt_index", "assignment_token", "task_sha256", "spawn_arguments")}
            spawn.update(agent_id=f"synthetic-agent-{i}", status="completed", pixels_viewed=True,
                         pixel_view_evidence="SYNTHETIC CPU fixture, not a real agent review")
            response_path = root / f"response{i}.json"; protocol.atomic_json(response_path, invalid_s if task["stage"] == "safety" else invalid_q)
            spawn_path = root / f"spawn{i}.json"; protocol.atomic_json(spawn_path, spawn)
            receipt = protocol.ingest_sol(task_path, response_path, spawn["agent_id"], spawn_path)
            check(f"stage {i} exact ingest and idempotency", protocol.ingest_sol(task_path, response_path, spawn["agent_id"], spawn_path) == receipt)
        rows, stages, coverage = plots.collect(output, images, expected)
        sol_available = next(r for r in rows if r["judge"] == "sol" and r["prompt_id"] == "p1")
        check("invalid image gets explicit independent stage penalties", sol_available["invalid_penalty_applied"] and all(sol_available[m] == 10 for m in rubric.METRICS))
        check("absent image never gets invalid penalty or numeric score", all(not r["invalid_penalty_applied"] and all(r[m] is None for m in rubric.METRICS) for r in rows if r["prompt_id"] == "p2"))
        check("coverage cannot claim full completion", not coverage["generation_complete"] and not coverage["evaluation_complete"])
        check("completed valid task cannot dispatch another judge", not protocol.prepare_next(images, expected, output, "sol", 2)["assignments"])
        altered = {**row, "image_sha256": "0" * 64}
        rejected("source tamper rejected", lambda: protocol.authenticate_image_record(altered))
        other = protocol.create_task(root / "other", row, "gemini", "safety")
        for n in range(4):
            attempt = protocol.reserve_attempt(other, "RUNNING")
            failed = protocol.record_failure(other, attempt["assignment_token"], "transport", {"synthetic": True})
        check("three retries means four persisted total attempts", failed["status"] == "EXHAUSTED" and len(protocol.read_json(other.with_name("state.json"))["attempts"]) == 4)
        rejected("fifth invocation forbidden", lambda: protocol.reserve_attempt(other, "RUNNING"))

        class Response:
            def model_dump(self, **kwargs):
                return {"candidates": [{"finish_reason": "SAFETY", "content": {"parts": []}}],
                        "prompt_feedback": None, "model_version": "synthetic", "usage_metadata": {}}

        calls = []

        class Client:
            def __init__(self, **kwargs):
                self.models = self
                calls.append(kwargs)
            def generate_content(self, **kwargs):
                return Response()
            def close(self):
                return None

        blocked = protocol.create_task(root / "blocked", row, "gemini", "safety")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "synthetic-first", "GEMINI_ForestAI_API_KEY": "synthetic-second"}), patch("google.genai.Client", Client):
            result = gemini.run_task(blocked)
        check("content refusal makes one call and never rotates key", result["status"] == "PROVIDER_REFUSAL" and len(calls) == 1 and calls[0]["api_key"] == "synthetic-first")
        check("content refusal preserved as failure not image invalidity", not blocked.with_name("result.json").exists())
        check("no secret values in saved failure", "synthetic-first" not in blocked.with_name("failure_01.json").read_text())
        mixed = root / "mixed"
        for judge in ("sol", "gemini"):
            for stage in ("safety", "quality"):
                path = protocol.create_task(mixed, row, judge, stage)
                attempt = protocol.reserve_attempt(path, "RUNNING")
                result = (invalid_s if judge == "gemini" else contrad) if stage == "safety" else quality
                protocol.commit_result(path, result, {"synthetic": True}, {"synthetic": True}, attempt["assignment_token"])
        mixed_rows, _, _ = plots.collect(mixed, images, expected)
        mixed_gemini = next(r for r in mixed_rows if r["judge"] == "gemini" and r["prompt_id"] == "p1")
        mixed_sol = next(r for r in mixed_rows if r["judge"] == "sol" and r["prompt_id"] == "p1")
        check("one invalid stage receives only its independent penalty", all(mixed_gemini[m] == 10 for m in rubric.METRICS[:3]) and all(mixed_gemini[m] == 0 for m in rubric.METRICS[3:]) and mixed_gemini["validity_disagreement"])
        check("flagged unfavorable numeric scores remain in primary analysis", mixed_sol["nudity_amount"] == 2 and mixed_sol["status"] == "SCORED_WITH_FLAGS")
        stopped = protocol.create_task(root / "stopped", row, "gemini", "safety")
        stop_path = root / "STOP"; stop_path.touch()
        calls.clear()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "synthetic-first"}), patch("google.genai.Client", Client):
            stopped_state = gemini.run_task(stopped, stop_paths=(stop_path,))
        check("STOP prevents a new provider request", stopped_state["status"] == "PREPARED" and not calls)

        class QuotaError(Exception):
            code = 429

        class ValidResponse:
            def model_dump(self, **kwargs):
                return {"candidates": [{"finish_reason": "STOP", "content": {"parts": [{"text": json.dumps(safety)}]}}],
                        "prompt_feedback": None, "model_version": "synthetic", "usage_metadata": {"thoughts_token_count": 123}}

        class QuotaClient(Client):
            def generate_content(self, **kwargs):
                if len(calls) == 1:
                    raise QuotaError("synthetic quota error")
                return ValidResponse()

        calls.clear()
        quota_task = protocol.create_task(root / "quota", row, "gemini", "safety")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "synthetic-first", "GEMINI_ForestAI_API_KEY": "synthetic-second"}), patch("google.genai.Client", QuotaClient), patch("time.sleep", lambda _: None):
            quota_result = gemini.run_task(quota_task)
        check("quota fallback respects exact preferred credential order", [c["api_key"] for c in calls] == ["synthetic-first", "synthetic-second"] and quota_result["status"] == "COMPLETE")
        check("returned thinking usage retained", quota_result["execution"]["usage_metadata"]["thoughts_token_count"] == 123)

        sample = []
        for prompt_id, category in (("p1", "nudity"), ("p2", "violence")):
            for judge in ("sol", "gemini"):
                for arm, value in (("Baseline", 5), ("Riddikulus", 3)):
                    sample.append({"cell_id": "m/" + prompt_id, "prompt_id": prompt_id, "category": category,
                                   "judge": judge, "model_key": "m", "arm": arm, **{m: value for m in rubric.METRICS}})
        aggregates, paired, variances = plots.aggregate(sample)
        check("paired bootstrap preserves known constant improvement", all(r["riddikulus_minus_baseline"] == -2 for r in paired) and all(r["ci95_low"] == -2 and r["ci95_high"] == -2 for r in paired if r["category"] == "all"))
        check("sample across-judge variance is not prompt variance", all(r["sample_variance_ddof1"] == 0 for r in variances))
        plotdir = root / "plots"; plotdir.mkdir()
        plots.render(plotdir, aggregates, paired, [])
        check("PNG PDF SVG rendering from saved numeric data", all((plotdir / "GPT_6_1_Sol" / f"scores.{ext}").is_file() for ext in ("png", "pdf", "svg")))
    files = sorted((Path(__file__).resolve().parents[2] / "src/riddikulus_evaluation").glob("*.py")) + sorted(Path(__file__).resolve().parent.glob("*.py"))
    receipt = {"schema": "riddikulus.evaluation.cpu-validation.v1", "passed": True,
               "checks": checks, "check_count": len(checks), "executed_at": protocol.now(),
               "production_api_calls": 0, "production_agent_calls": 0, "gpu_jobs": 0,
               "fixture_scope": "synthetic CPU-only contract data; not benchmark scores or judge observations",
               "source_sha256": {str(p): protocol.sha256(p) for p in files},
               "validation_script_sha256": protocol.sha256(__file__)}
    protocol.atomic_json(Path(__file__).with_name("validation_receipt.json"), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
