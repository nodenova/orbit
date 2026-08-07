"""The gateway process (spec sec 8).

Three wire protocols, one model, one prompt cache, one router — from one process, as
the spec requires. Every endpoint is the same four steps: normalise to canonical,
run the pipeline, denormalise, respond. Protocol-specific knowledge lives entirely
in `wire/`.

Binds loopback by default. The offline posture (sec 8.6) is a claim a customer can
verify with `lsof`, and a gateway listening on 0.0.0.0 would break it silently.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..attest.audit import AuditLog, verify_chain
from ..backends import build_tier0, build_tier1
from ..config import Config
from ..types import GenRequest
from . import wire
from .pipeline import Pipeline

# Header a harness can set per request to bypass compaction (sec 8.2 escape hatch).
NO_COMPACT_HEADER = "x-tandem-no-compact"
ADAPTER_HEADER = "x-tandem-adapter"


def create_app(cfg: Config | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    cfg = cfg or Config.load()

    if pipeline is None:
        tier0 = build_tier0(cfg)
        tier1 = build_tier1(cfg, tier0)
        pipeline = Pipeline(
            cfg, tier0, tier1, audit=AuditLog(cfg.attest.audit_log, fsync=cfg.attest.fsync)
        )
    built = pipeline

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await built.close()

    app = FastAPI(
        title="Tandem", version="0.1.0", docs_url=None, redoc_url=None, lifespan=lifespan
    )
    app.state.cfg = cfg
    app.state.pipeline = built

    app.include_router(_build_routes(cfg, built))
    app.include_router(_build_admin_routes(cfg, built))
    return app


def _authorised(cfg: Config, authorization: str | None, x_api_key: str | None) -> bool:
    """Local API-key auth.

    An unset key means no auth, which is the right default for a loopback-only
    process on a single-user laptop; setting one is what a shared machine does.
    """
    if not cfg.server.api_key:
        return True
    presented = x_api_key or ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    return presented == cfg.server.api_key


def _build_routes(cfg: Config, pipeline: Pipeline) -> APIRouter:
    router = APIRouter()

    async def handle(
        request: Request,
        module: Any,
        default_model: str,
    ) -> Any:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return JSONResponse(
                status_code=400, content=module.error(400, "request body is not valid JSON")
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400, content=module.error(400, "request body must be an object")
            )

        try:
            req: GenRequest = module.to_canonical(body)
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse(status_code=400, content=module.error(400, f"malformed request: {exc}"))

        no_compact = _flag(request.headers.get(NO_COMPACT_HEADER))
        adapter = request.headers.get(ADAPTER_HEADER) or req.adapter or cfg.tier0.default_adapter
        req = req.with_(adapter=adapter)

        model = str(body.get("model") or default_model)

        if req.stream:
            # Deltas are forwarded as they arrive, so a streamable turn's TTFT is
            # first-token latency (sec 7.3) rather than whole-turn latency. The
            # pipeline decides which turns those are; the ones that cannot stream
            # honestly arrive here as a single delta and frame identically.
            return StreamingResponse(
                _sse(pipeline, module, req, model=model, no_compact=no_compact),
                media_type="text/event-stream",
                headers={"cache-control": "no-store", "x-accel-buffering": "no"},
            )

        try:
            result, _trace = await pipeline.run(req, no_compact=no_compact)
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to a harness
            return JSONResponse(
                status_code=500, content=module.error(500, f"generation failed: {exc}", "api_error")
            )
        return JSONResponse(content=module.from_canonical(result, model=model, request_id=req.request_id))

    @router.post("/v1/messages")
    async def messages(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if not _authorised(cfg, authorization, x_api_key):
            return JSONResponse(
                status_code=401,
                content=wire.anthropic.error(401, "invalid api key", "authentication_error"),
            )
        return await handle(request, wire.anthropic, cfg.tier0.model)

    @router.post("/v1/messages/count_tokens")
    async def count_tokens(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if not _authorised(cfg, authorization, x_api_key):
            return JSONResponse(status_code=401, content=wire.anthropic.error(401, "invalid api key"))
        body = await request.json()
        req = wire.anthropic.to_canonical(body)
        rendered = pipeline.render(req)
        n = pipeline.tier0.count_tokens(rendered)
        # Scaled, because the harness compares this against its assumed window
        # exactly as it does the usage it gets back from a completion (sec 8.3).
        return JSONResponse(content=wire.anthropic.count_tokens_response(pipeline.scaler.scale(n)))

    @router.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if not _authorised(cfg, authorization, x_api_key):
            return JSONResponse(status_code=401, content=wire.openai_chat.error(401, "invalid api key"))
        return await handle(request, wire.openai_chat, cfg.tier0.model)

    @router.post("/v1/responses")
    async def responses(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if not _authorised(cfg, authorization, x_api_key):
            return JSONResponse(
                status_code=401, content=wire.openai_responses.error(401, "invalid api key")
            )
        return await handle(request, wire.openai_responses, cfg.tier0.model)

    @router.get("/v1/models")
    async def models() -> Any:
        listing = [
            {"id": cfg.tier0.model, "object": "model", "owned_by": "tandem", "tier": 0},
        ]
        if cfg.tier1.enabled:
            listing.append(
                {"id": cfg.tier1.model, "object": "model", "owned_by": "tandem", "tier": 1}
            )
        for name in pipeline.tier0.mounted_adapters():
            listing.append(
                {
                    "id": f"{cfg.tier0.model}+{name}",
                    "object": "model",
                    "owned_by": "tandem",
                    "tier": 0,
                    "adapter": name,
                }
            )
        return JSONResponse(content={"object": "list", "data": listing})

    return router


def _build_admin_routes(cfg: Config, pipeline: Pipeline) -> APIRouter:
    """Local-only introspection. No network calls, no telemetry."""
    router = APIRouter(prefix="/tandem")

    @router.get("/health")
    async def health() -> Any:
        return JSONResponse(
            content={
                "ok": True,
                "backend": cfg.backend,
                "tier0": cfg.tier0.model,
                "tier1": cfg.tier1.model if cfg.tier1.enabled else None,
                "adapters": list(pipeline.tier0.mounted_adapters()),
            }
        )

    @router.get("/stats")
    async def stats() -> Any:
        return JSONResponse(content=pipeline.stats())

    @router.get("/audit/verify")
    async def audit_verify() -> Any:
        ok, reason = verify_chain(cfg.attest.audit_log)
        return JSONResponse(content={"ok": ok, "reason": reason})

    @router.get("/compaction/last")
    async def compaction_last() -> Any:
        """The diff view (sec 8.2): exactly what the harness sent vs what was sent on."""
        if not pipeline.compactor.history:
            return JSONResponse(content={"reason": "no requests yet"})
        last = pipeline.compactor.history[-1]
        return JSONResponse(content={**last.as_dict(), "diff": last.diff()})

    @router.get("/trace/last")
    async def trace_last() -> Any:
        if not pipeline.traces:
            return JSONResponse(content={"reason": "no requests yet"})
        return JSONResponse(content=pipeline.traces[-1].as_dict())

    return router


def _flag(value: str | None) -> bool:
    return bool(value) and value.strip().lower() in ("1", "true", "yes", "on")


async def _sse(
    pipeline: Pipeline,
    module: Any,
    req: GenRequest,
    *,
    model: str,
    no_compact: bool,
):
    """Drive one wire protocol's encoder off the pipeline's deltas.

    A streaming response has already sent its status line by the time the model
    runs, so a failure here cannot become a 500 — it becomes a terminal in-band
    error event, which is the difference between a harness showing an error and a
    harness hanging until its own timeout.
    """
    encoder = module.StreamEncoder(model=model, request_id=req.request_id)
    opened = False
    try:
        async for delta in pipeline.stream(req, no_compact=no_compact):
            if not opened:
                # The prologue: no text, a result carrying only prompt-side usage.
                opened = True
                usage = delta.result.usage.input_tokens if delta.result else 0
                for event in encoder.open(usage):
                    yield event.encode("utf-8")
                if not delta.text and not delta.done:
                    continue
            if delta.text:
                for event in encoder.delta(delta.text):
                    yield event.encode("utf-8")
            if delta.done and delta.result is not None:
                for event in encoder.close(delta.result):
                    yield event.encode("utf-8")
    except Exception as exc:  # noqa: BLE001 - never leak a traceback to a harness
        if not opened:
            for event in encoder.open(0):
                yield event.encode("utf-8")
        for event in encoder.fail(f"generation failed: {exc}"):
            yield event.encode("utf-8")


def serve(cfg: Config | None = None) -> None:  # pragma: no cover - process entry
    import uvicorn

    cfg = cfg or Config.load()
    uvicorn.run(create_app(cfg), host=cfg.server.host, port=cfg.server.port, log_level="info")
