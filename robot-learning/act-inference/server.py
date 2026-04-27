"""Minimal FastAPI inference server for ACT policies.

Loads the policy lazily on the first /predict call to keep startup fast and
keep /healthz responding while big model downloads happen in the background.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

POLICY_REPO_ID = os.environ.get("POLICY_REPO_ID", "ETHRC/act-towelspring26")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/checkpoints/act")

app = FastAPI(title="ACT Inference", version="0.1.0")

_policy: Any | None = None
_policy_lock = threading.Lock()
_policy_error: str | None = None


def _load_policy() -> Any:
    """Lazy import + load so /healthz works before lerobot is initialised."""
    global _policy, _policy_error
    if _policy is not None:
        return _policy
    with _policy_lock:
        if _policy is not None:
            return _policy
        try:
            # TODO: replace with the real ACT policy load once the team picks
            # the canonical loader (lerobot.policies.factory or a custom path).
            # Keeping the import lazy means startup doesn't block on a 1+GB
            # checkpoint download, which is friendly to the readiness probe.
            from lerobot.common.policies.factory import make_policy_from_pretrained  # type: ignore

            _policy = make_policy_from_pretrained(POLICY_REPO_ID, device="cuda")
        except Exception as exc:  # noqa: BLE001 — we want the message in the response
            _policy_error = f"{type(exc).__name__}: {exc}"
            raise
        return _policy


class PredictRequest(BaseModel):
    observation: dict[str, Any]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    if _policy is None:
        return {"status": "policy-not-loaded", "policy": POLICY_REPO_ID}
    return {"status": "ready", "policy": POLICY_REPO_ID}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    try:
        policy = _load_policy()
    except Exception:
        raise HTTPException(status_code=503, detail=_policy_error or "policy load failed")

    # TODO: wire up real inference. The policy interface depends on the lerobot
    # version pinned in pyproject.toml — this stub keeps the contract stable.
    action = policy.select_action(req.observation)  # type: ignore[attr-defined]
    return {"action": action.tolist() if hasattr(action, "tolist") else action}
