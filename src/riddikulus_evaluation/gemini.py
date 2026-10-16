"""One exact Gemini judge, HIGH thinking, two ordered credential names, bounded retries.

Importing this module never reads a credential or calls a service. Calling run_task is
an explicit production judgment. A provider refusal is terminal and never rotates keys.
"""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import time

from . import protocol, rubric

KEY_NAMES = ("GEMINI_API_KEY", "GEMINI_ForestAI_API_KEY")
MODEL = "gemini-3.8-flash"
THINKING_LEVEL = "high"
THINKING_SOURCES = [
    "https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash",
    "https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.8",
    "https://ai.google.dev/gemini-api/docs/thinking",
]
REFUSAL_REASONS = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT"}


def configuration() -> dict:
    return {"model": MODEL, "thinking_level": THINKING_LEVEL,
            "thinking_interpretation": "highest documented supported level; no MAX enum",
            "thinking_sources": THINKING_SOURCES, "documentation_checked": "2026-10-16",
            "credential_priority": list(KEY_NAMES), "content_refusal_key_rotation": False,
            "max_technical_retries": protocol.MAX_TECHNICAL_RETRIES,
            "max_total_attempts_per_stage": 1 + protocol.MAX_TECHNICAL_RETRIES,
            "sdk_internal_retries": 0, "request_timeout_seconds": 900,
            "sampling_parameters": "provider defaults; no temperature/top_p/top_k/candidate_count override",
            "max_output_tokens": None, "output_budget": "provider default, not an artificial low thinking cap",
            "safety_settings": "provider defaults; a content block is recorded without fallback",
            "google_genai_version": importlib.metadata.version("google-genai")}


def _redact(value, secrets: list[str]):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {k: _redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, secrets) for v in value]
    return value


def _error_class(exc: Exception) -> tuple[str, bool]:
    """Return failure class and whether credential fallback is appropriate."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    message = str(exc).lower()
    if any(term in message for term in ("content policy", "prohibited content", "safety block", "blocked content")):
        return "provider_refusal", False
    if code in (401, 403):
        return "authentication", True
    if code == 429:
        return "rate_limit", True
    if isinstance(code, int) and 500 <= code < 600:
        return "server", False
    if isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in {
        "ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError", "ServerDisconnectedError"
    }:
        return "transport", False
    return "invalid_input", False


def _text_refusal(text: str) -> bool:
    prefix = text.strip().lower()
    return prefix.startswith(("i cannot", "i can't", "i’m unable", "i'm unable", "i am unable",
                              "sorry, i cannot", "sorry, i can't", "i'm sorry", "i’m sorry"))


def _name(value) -> str:
    return str(value).split(".")[-1] if value is not None else ""


def response_content(response) -> tuple[str, dict, str | None]:
    raw = response.model_dump(mode="json", exclude_none=False)
    feedback = raw.get("prompt_feedback") or {}
    reason = _name(feedback.get("block_reason"))
    if reason and reason not in {"BLOCK_REASON_UNSPECIFIED", "0", "None"}:
        return "", raw, reason
    candidates = raw.get("candidates") or []
    for candidate in candidates:
        if _name(candidate.get("finish_reason")) in REFUSAL_REASONS:
            return "", raw, _name(candidate["finish_reason"])
    if len(candidates) != 1:
        raise ValueError("Expected exactly one Gemini candidate")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought", False))
    return text, raw, None


def run_task(task_path: Path, *, stop_paths: tuple[Path, ...] = (), stop_check=None) -> dict:
    from google import genai
    from google.genai import types

    task_path = task_path.resolve()
    task, _ = protocol.authenticate_task(task_path)
    if task["judge"] != "gemini":
        raise ValueError("Gemini runner received another judge's task")
    state = protocol.read_json(task_path.with_name("state.json"))
    if state["status"] in {"COMPLETE", "COMPLETE_WITH_ISSUES"}:
        return protocol.verified_result(task_path)[2]
    if state["status"] in protocol.TERMINAL:
        return state
    credentials = [(name, os.environ.get(name)) for name in KEY_NAMES if os.environ.get(name)]
    if not credentials:
        raise RuntimeError("Neither approved Gemini credential environment name is configured")
    secrets = [value for _, value in credentials]
    # On a resumed technical retry retain the last recorded authorized key choice.
    key_index = 0
    if state["attempts"]:
        last = task_path.parent / f"failure_{state['attempts'][-1]['index']:02d}.json"
        if last.exists():
            detail = protocol.read_json(last)["detail"]
            preferred = detail.get("next_credential_env")
            key_index = next((i for i, (name, _) in enumerate(credentials) if name == preferred), 0)
    while True:
        if any(path.exists() for path in stop_paths) or (stop_check is not None and stop_check()):
            return protocol.read_json(task_path.with_name("state.json"))
        reservation = protocol.reserve_attempt(task_path, "RUNNING")
        key_name, key_value = credentials[key_index]
        request_config = configuration()
        execution = {"transport": "google_genai", "requested_model": MODEL,
                     "credential_env_name": key_name, "configuration": request_config,
                     "started_at": protocol.now()}
        protocol.atomic_json(task_path.parent / f"request_{reservation['index']:02d}.json",
                             {"task_sha256": protocol.sha256(task_path), "reservation": reservation,
                              "execution": execution}, immutable=True)
        started = time.monotonic()
        raw = None
        failure_class = None
        next_key = key_index
        client = genai.Client(api_key=key_value, http_options=types.HttpOptions(
            timeout=900_000, retry_options=types.HttpRetryOptions(attempts=1)))
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[task["prompt"], types.Part.from_bytes(
                    data=Path(task["image"]["path"]).read_bytes(), mime_type="image/png")],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", response_json_schema=task["response_schema"],
                    thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL)),
            )
            raw = response.model_dump(mode="json", exclude_none=False)
            text, raw, refusal_reason = response_content(response)
            execution.update({"elapsed_seconds": time.monotonic() - started,
                              "completed_at": protocol.now(), "returned_model_version": raw.get("model_version"),
                              "response_id": raw.get("response_id"), "usage_metadata": raw.get("usage_metadata"),
                              "finish_reasons": [c.get("finish_reason") for c in raw.get("candidates") or []]})
            if refusal_reason:
                return protocol.record_failure(task_path, reservation["assignment_token"], "provider_refusal",
                    _redact({"reason": refusal_reason, "raw_response": raw, "execution": execution}, secrets))
            if _text_refusal(text):
                return protocol.record_failure(task_path, reservation["assignment_token"], "provider_refusal",
                    _redact({"reason": "explicit_text_refusal", "text": text, "raw_response": raw,
                             "execution": execution}, secrets))
            # Structural parsing errors may retry; valid but contradictory scores may not.
            try:
                parsed = json.loads(text)
                rubric.validate_shape(parsed, task["response_schema"])
            except (ValueError, TypeError) as exc:
                failure_class = "schema"
                detail = {"error_type": type(exc).__name__, "message": str(exc), "text": text,
                          "raw_response": raw, "execution": execution}
            else:
                return protocol.commit_result(task_path, parsed, _redact({"text": text, "response": raw}, secrets),
                                              execution, reservation["assignment_token"])
        except Exception as exc:
            failure_class, rotate = _error_class(exc)
            if rotate and key_index + 1 < len(credentials):
                next_key = key_index + 1
            detail = {"error_type": type(exc).__name__, "message": str(exc),
                      "elapsed_seconds": time.monotonic() - started, "raw_response": raw,
                      "execution": execution, "credential_env_name": key_name,
                      "next_credential_env": credentials[next_key][0]}
        finally:
            client.close()
        failure = protocol.record_failure(task_path, reservation["assignment_token"], failure_class,
                                          _redact(detail, secrets))
        if failure["status"] != "TECHNICAL_FAILURE":
            return failure
        key_index = next_key
        time.sleep(min(60, 5 * 2 ** (reservation["index"] - 1)))
