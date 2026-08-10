"""The gateway process (spec sec 8).

Three wire protocols, one model, one prompt cache, one router — from one process, as
the spec requires. Every endpoint is the same four steps: normalise to canonical,
run the pipeline, denormalise, respond. Protocol-specific knowledge lives entirely
in `wire/`.

Binds loopback by default. The offline posture (sec 8.6) is a claim a customer can
verify with `lsof`, and a gateway listening on 0.0.0.0 would break it silently.
"""

from __future__ import annotations

import hmac
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from orbit.attest.audit import AuditLog, verify_chain
from orbit.backends import build_tier0, build_tier1
from orbit.config import Config
from orbit.gateway import wire
from orbit.gateway.pipeline import Pipeline
from orbit.types import GenRequest

# Header a harness can set per request to bypass compaction (sec 8.2 escape hatch).
NO_COMPACT_HEADER = "x-orbit-no-compact"
ADAPTER_HEADER = "x-orbit-adapter"

# Loopback names only, matching the default bind. Overridden by
# `[server] allowed_hosts` for a deployment that fronts the gateway with a proxy.
_DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "[::1]", "::1")


def create_app(cfg: Config | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    cfg = cfg or Config.load()

    if pipeline is None:
        tier0 = build_tier0(cfg)
        tier1 = build_tier1(cfg, tier0)
        pipeline = Pipeline(
            cfg,
            tier0,
            tier1,
            audit=AuditLog(cfg.attest.audit_log, fsync=cfg.attest.fsync),
        )
    built = pipeline

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await built.close()

    app = FastAPI(
        title="Orbit",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    # Host allow-list, not CORS. A browser will not send a cross-origin POST here
    # without a preflight, and we answer no preflight — but DNS rebinding does not
    # need one: the attacker's page keeps its own origin and re-points its own
    # hostname at 127.0.0.1, so the request is same-origin by the browser's rules
    # and reaches a loopback bind. The Host header still carries the attacker's
    # name, and a page cannot forge it, so pinning Host to loopback names is what
    # turns "binds loopback by default" into a boundary. This matters most in the
    # default configuration, where `api_key` is unset and there is nothing else in
    # the way of an unauthenticated coding agent.
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(
            getattr(cfg.server, "allowed_hosts", _DEFAULT_ALLOWED_HOSTS)
        ),
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

    Compared with `compare_digest` rather than `==`: the key is a secret presented
    by an unauthenticated caller, and `==` on `str` short-circuits at the first
    differing byte. The timing signal is small over loopback and not small over a
    LAN, which is exactly the deployment that sets a key in the first place.
    """
    if not cfg.server.api_key:
        return True
    presented = x_api_key or ""
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    return hmac.compare_digest(presented, cfg.server.api_key)


def _unauthorised(module: Any = None) -> JSONResponse:
    """The 401 body, in the caller's protocol shape where there is one."""
    if module is not None:
        return JSONResponse(
            status_code=401,
            content=module.error(401, "invalid api key", "authentication_error"),
        )
    return JSONResponse(
        status_code=401, content={"error": {"message": "invalid api key"}}
    )


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
                status_code=400,
                content=module.error(400, "request body is not valid JSON"),
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content=module.error(400, "request body must be an object"),
            )

        try:
            req: GenRequest = module.to_canonical(body)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            # AttributeError belongs here with the rest: a body can be valid JSON,
            # be an object, and still be the wrong *shape* two levels down —
            # `{"messages": ["hello"]}` reaches `.get` on a str. That is a client
            # error and has to arrive as this protocol's 400, not as a bare 500
            # text/plain that no harness knows how to read.
            return JSONResponse(
                status_code=400, content=module.error(400, f"malformed request: {exc}")
            )

        no_compact = _flag(request.headers.get(NO_COMPACT_HEADER))
        adapter = (
            request.headers.get(ADAPTER_HEADER)
            or req.adapter
            or cfg.tier0.default_adapter
        )
        unknown = _unknown_adapter(pipeline, adapter)
        if unknown is not None:
            return JSONResponse(status_code=400, content=module.error(400, unknown))
        req = req.with_(adapter=adapter)

        # Minted here rather than inside the pipeline. `_begin` mints onto its own
        # copy of the request, which never comes back out, so the id the harness
        # sees in the response used to share no field with the id in the audit log
        # — leaving an auditor to correlate a customer's response against sec 9.2
        # by timestamp alone. One id, minted before either consumer runs.
        req = req.with_(request_id=req.request_id or uuid.uuid4().hex)

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
                status_code=500,
                content=module.error(500, f"generation failed: {exc}", "api_error"),
            )
        return JSONResponse(
            content=module.from_canonical(
                result, model=model, request_id=req.request_id
            )
        )

    @router.post("/v1/messages")
    async def messages(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if not _authorised(cfg, authorization, x_api_key):
            return _unauthorised(wire.anthropic)
        return await handle(request, wire.anthropic, cfg.tier0.model)

    @router.post("/v1/messages/count_tokens")
    async def count_tokens(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if not _authorised(cfg, authorization, x_api_key):
            return _unauthorised(wire.anthropic)
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return JSONResponse(
                status_code=400,
                content=wire.anthropic.error(400, "request body is not valid JSON"),
            )
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content=wire.anthropic.error(400, "request body must be an object"),
            )
        try:
            req = wire.anthropic.to_canonical(body)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            return JSONResponse(
                status_code=400,
                content=wire.anthropic.error(400, f"malformed request: {exc}"),
            )

        # Counted against the prompt that would actually be sent, which means
        # compaction runs first — exactly as it does on the completion path.
        # Counting the raw harness prompt over-reports by the full compaction
        # multiplier (~33x on the CC_SYSTEM fixture), and Claude Code drives both
        # its context meter and its auto-compact threshold off this number: an
        # inflated count makes the harness compact its own history far too
        # aggressively, discarding context the model could have had. The escape
        # hatch has to be honoured here too, or the count stops describing the
        # request the harness is about to send.
        no_compact = _flag(request.headers.get(NO_COMPACT_HEADER))
        n = pipeline.count_prompt_tokens(req, no_compact=no_compact)
        # Scaled, because the harness compares this against its assumed window
        # exactly as it does the usage it gets back from a completion (sec 8.3).
        return JSONResponse(
            content=wire.anthropic.count_tokens_response(pipeline.scaler.scale(n))
        )

    @router.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if not _authorised(cfg, authorization, x_api_key):
            return _unauthorised(wire.openai_chat)
        return await handle(request, wire.openai_chat, cfg.tier0.model)

    @router.post("/v1/responses")
    async def responses(
        request: Request,
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if not _authorised(cfg, authorization, x_api_key):
            return _unauthorised(wire.openai_responses)
        return await handle(request, wire.openai_responses, cfg.tier0.model)

    @router.get("/v1/models")
    async def models(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        # Authenticated like the completion routes: the listing names the served
        # model and every mounted adapter, which is the shape of the deployment.
        if not _authorised(cfg, authorization, x_api_key):
            return _unauthorised(wire.openai_chat)
        listing = [
            {"id": cfg.tier0.model, "object": "model", "owned_by": "orbit", "tier": 0},
        ]
        if cfg.tier1.enabled:
            listing.append(
                {
                    "id": cfg.tier1.model,
                    "object": "model",
                    "owned_by": "orbit",
                    "tier": 1,
                }
            )
        for name in pipeline.tier0.mounted_adapters():
            listing.append(
                {
                    "id": f"{cfg.tier0.model}+{name}",
                    "object": "model",
                    "owned_by": "orbit",
                    "tier": 0,
                    "adapter": name,
                }
            )
        return JSONResponse(content={"object": "list", "data": listing})

    return router


def _build_admin_routes(cfg: Config, pipeline: Pipeline) -> APIRouter:
    """Local-only introspection. No network calls, no telemetry.

    "Local-only" is a statement about the bind address, not an access control, so
    every route here takes the same API key as `/v1/messages`. They used to take
    none — and they are the *more* sensitive half of the surface, not the less:
    `/compaction/last` returns the previous request's full raw system prompt,
    which for a coding agent routinely embeds repository context, file paths and
    project instructions, and `/health` enumerates the mounted adapter names.
    A deployment that sets a key has decided its callers are not all trusted;
    that decision has to cover the routes that disclose the most.
    """
    router = APIRouter(prefix="/orbit")

    @router.get("/health")
    async def health(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if not _authorised(cfg, authorization, x_api_key):
            return _unauthorised()
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
    async def stats(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if not _authorised(cfg, authorization, x_api_key):
            return _unauthorised()
        return JSONResponse(content=pipeline.stats())

    @router.get("/audit/verify")
    async def audit_verify(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if not _authorised(cfg, authorization, x_api_key):
            return _unauthorised()
        ok, reason = verify_chain(cfg.attest.audit_log)
        return JSONResponse(content={"ok": ok, "reason": reason})

    @router.get("/compaction/last")
    async def compaction_last(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        """The diff view (sec 8.2): exactly what the harness sent vs what was sent on."""
        if not _authorised(cfg, authorization, x_api_key):
            return _unauthorised()
        if not pipeline.compactor.history:
            return JSONResponse(content={"reason": "no requests yet"})
        last = pipeline.compactor.history[-1]
        return JSONResponse(content={**last.as_dict(), "diff": last.diff()})

    @router.get("/trace/last")
    async def trace_last(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Any:
        if not _authorised(cfg, authorization, x_api_key):
            return _unauthorised()
        if not pipeline.traces:
            return JSONResponse(content={"reason": "no requests yet"})
        return JSONResponse(content=pipeline.traces[-1].as_dict())

    return router


def _unknown_adapter(pipeline: Pipeline, adapter: str | None) -> str | None:
    """An error message if `adapter` is named but not mounted, else None.

    Refused rather than ignored. Serving the base model for an adapter the caller
    asked for would be a wrong answer that looks like a right one, and the receipt
    makes it worse than silent: `_build_receipt` stamps `adapter_name` from the
    request and `adapter_blake3` from the backend, so an unmounted name produces
    an attestation for an adapter that never touched the generation. That is the
    exact failure mode sec 4.2's isolation gate exists to rule out, arriving
    through the front door instead. A typo in `default_adapter` does it to every
    request in the deployment, which is why this checks the resolved name rather
    than only the header.
    """
    if not adapter:
        return None
    mounted = pipeline.tier0.mounted_adapters()
    if adapter in mounted:
        return None
    known = ", ".join(mounted) if mounted else "none"
    return f"unknown adapter {adapter!r}; mounted adapters: {known}"


def _flag(value: str | None) -> bool:
    return value is not None and value.strip().lower() in ("1", "true", "yes", "on")


async def _sse(
    pipeline: Pipeline,
    module: Any,
    req: GenRequest,
    *,
    model: str,
    no_compact: bool,
) -> AsyncIterator[str]:
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
    uvicorn.run(
        create_app(cfg), host=cfg.server.host, port=cfg.server.port, log_level="info"
    )
