"""Vision embedder FastAPI server.

Lazily loads a HuggingFace vision encoder on the first /embed call. Until
then, /healthz responds immediately so the trainer's wait-for-embedder
initContainer can release.

The model load is intentionally lazy: a 1+GB DINOv2 download blocks for a
while on first start, and we don't want the readiness probe to time out
during that window.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL", "facebook/dinov2-base")

app = FastAPI(title="ACT Conditioning Embedder", version="0.1.0")

_model: Any | None = None
_processor: Any | None = None
_load_lock = threading.Lock()
_load_error: str | None = None


def _ensure_loaded() -> tuple[Any, Any]:
    global _model, _processor, _load_error
    if _model is not None and _processor is not None:
        return _model, _processor
    with _load_lock:
        if _model is not None and _processor is not None:
            return _model, _processor
        try:
            # Imported lazily so /healthz works before transformers is installed
            # — keeps the initContainer probe in the trainer Job from racing
            # the first `uv sync` here.
            import torch
            from transformers import AutoImageProcessor, AutoModel  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
            _processor = AutoImageProcessor.from_pretrained(EMBEDDER_MODEL)
            _model = AutoModel.from_pretrained(EMBEDDER_MODEL).to(device).eval()
        except Exception as exc:  # noqa: BLE001
            _load_error = f"{type(exc).__name__}: {exc}"
            raise
        return _model, _processor  # type: ignore[return-value]


class EmbedRequest(BaseModel):
    # images: list of HxWxC float arrays in [0, 1]. Kept loose so trainers
    # can experiment without the embedder needing to redeploy.
    images: list[list[list[list[float]]]]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    if _model is None:
        return {"status": "model-not-loaded", "model": EMBEDDER_MODEL}
    return {"status": "ready", "model": EMBEDDER_MODEL}


@app.post("/embed")
def embed(req: EmbedRequest) -> dict[str, Any]:
    try:
        model, processor = _ensure_loaded()
    except Exception:
        raise HTTPException(status_code=503, detail=_load_error or "model load failed")

    import numpy as np  # local import keeps cold start fast
    import torch

    arr = np.asarray(req.images, dtype=np.float32)  # [B, H, W, C]
    inputs = processor(images=list(arr), return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token embedding (first token) as the conditioning vector.
    cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return {"embeddings": cls.tolist(), "model": EMBEDDER_MODEL}
