"""ollama client for the A1 arm: one turn in, one `Turn` out, plus the host assertions.

`/api/chat` is the primary endpoint rather than `/v1/chat/completions` for four
reasons, three of them things the OpenAI-compatible path cannot express at all:
per-request `options.num_ctx` and `keep_alive`, the `think` switch, and
`prompt_eval_duration`/`eval_duration` straight off the runner. The fourth is the
decisive one — `/v1` re-serialises tool-call `arguments` back into a JSON *string*,
which reintroduces the escaping step this model's native XML tool-call format exists
to avoid. `--openai-compat` is therefore a portability check and not a second way to
run the experiment.

Two rules in here are load-bearing and both fail silently if broken:

  * **Reasoning never enters `content`.** ollama returns it in its own `thinking`
    field and drops that field on the way back in, so reasoning kept as a field costs
    nothing. Inlined into `content` it is re-rendered into the prompt every turn for
    the rest of the episode: measured at 42 prompt tokens as a field against 703
    inlined, on the same three-message array.
  * **Prefix reuse shows up in the duration, not the count.** ollama reports
    `prompt_eval_count` as the full prompt length every turn whether it evaluated it
    or not. Reuse is a flat `prompt_eval_duration` against a rising count, equivalently
    an `implied_prefill_rate` that climbs past the single-pass ceiling below. A harness
    asserting the count collapses fails on a healthy machine.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "a1-eval"
DEFAULT_KEEP_ALIVE = "30m"
# Explicit, never inherited: the published tag carries no window and would serve
# ollama's small default. This is not the model's ceiling — the trained window is
# 262,144, and 131,072 costs ~2 GiB more resident than 65,536 on this host.
DEFAULT_NUM_CTX = 65536

# A per-turn decode cap, because without one a turn can eat the whole window. Measured:
# greedy with no cap, asked one trivial question with thinking on, decoded 13,587 tokens
# at ~56 tok/s and was still going when it was killed — it would have run to num_ctx,
# roughly 20 minutes for a one-sentence answer. This is the runaway the card's
# `presence_penalty 1.1` exists to suppress, and it does not show up in a 700-token
# probe. 2,048 leaves room for reasoning plus a file write and bounds a turn at ~36 s.
DEFAULT_MAX_OUTPUT_TOKENS = 2048

# The single-pass prefill rate measured on this host at prompts under 16 k. A turn
# whose implied rate sits near it evaluated the whole prompt; one far above it reused
# the prefix, which is the only way to tell the two apart from a response body.
SINGLE_PASS_PREFILL_TOK_S = 1200.0


class TransportError(RuntimeError):
    """The endpoint refused, or the host is in a state that would make a run a lie."""


class TruncatedToolCall(Exception):
    """`num_predict` cut a tool call mid-JSON and ollama could not parse the remains.

    The same event as `done_reason == "length"` on a turn that emitted no call, and
    it must be handled the same way — the model spent its budget and needs another
    turn, not an aborted episode. It arrives differently because ollama parses tool
    arguments server-side and reports the failure as a stream error, so it reached
    the loop as a fatal `TransportError` and killed the run outright: measured on
    `mid-08` at a 2,048-token cap, where the episode died on turn 4 of a 30-turn
    budget. At 4,096 the same task completed. A cap that ends a run rather than a
    turn silently converts a configuration choice into a model result.
    """


class ContextOverflow(Exception):
    """The prompt did not fit. ollama types this rather than truncating silently.

    Silent truncation would be the dangerous outcome: it drops the *oldest* tokens,
    which here is the pinned task prompt, and losing that is what makes the model's
    own template raise `No user query found in messages` deep into an episode.
    """

    def __init__(self, n_prompt_tokens: int, n_ctx: int) -> None:
        super().__init__(f"prompt {n_prompt_tokens} tokens exceeds num_ctx {n_ctx}")
        self.n_prompt_tokens = n_prompt_tokens
        self.n_ctx = n_ctx


@dataclass(frozen=True, slots=True)
class Sampling:
    """A named, fully pinned sampling set. Both are deterministic with `seed` set."""

    mode: str
    temperature: float
    top_p: float
    seed: int = 0
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repeat_penalty: float | None = None

    def options(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }
        for key, value in (
            ("top_k", self.top_k),
            ("min_p", self.min_p),
            ("presence_penalty", self.presence_penalty),
            ("repeat_penalty", self.repeat_penalty),
        ):
            if value is not None:
                out[key] = value
        return out


GREEDY = Sampling(mode="greedy", temperature=0.0, top_p=1.0, seed=0)
# The published card's set. It is *not* what buys determinism — with `seed` pinned,
# both of these reproduce byte-identical 700-token generations — so greedy is chosen
# for having one less variable, not for repeatability. Greedy is off the distribution
# the model was tuned on, which is why it is a declared delta rather than a default
# nobody examined. `repeat_penalty` is ollama's spelling of `repetition_penalty`.
CARD = Sampling(
    mode="card",
    temperature=0.85,
    top_p=0.95,
    seed=0,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.1,
    repeat_penalty=1.0,
)
SAMPLINGS: dict[str, Sampling] = {GREEDY.mode: GREEDY, CARD.mode: CARD}


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    salvaged: bool = False


@dataclass(slots=True)
class Turn:
    """One assistant turn, with the wire facts needed to read it months later."""

    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_eval_count: int = 0
    eval_count: int = 0
    prompt_eval_ms: float = 0.0
    eval_ms: float = 0.0
    wall_s: float = 0.0
    ttft_s: float = 0.0
    done_reason: str = ""
    salvaged: int = 0
    trailing_prose_dropped: int = 0

    @property
    def implied_prefill_rate(self) -> float:
        """`prompt_eval_count / prompt_eval_duration` — the prefix-reuse read.

        Near the single-pass ceiling means the whole prompt was evaluated; far above
        it means only the delta was, which is the reuse this harness is built for.
        """
        if not self.prompt_eval_ms:
            return 0.0
        return self.prompt_eval_count / (self.prompt_eval_ms / 1000)

    def stats(self) -> dict[str, Any]:
        return {
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "prompt_eval_ms": round(self.prompt_eval_ms, 1),
            "eval_ms": round(self.eval_ms, 1),
            "implied_prefill_tok_s": round(self.implied_prefill_rate, 1),
            "wall_s": round(self.wall_s, 2),
            "ttft_s": round(self.ttft_s, 2),
            "done_reason": self.done_reason,
            "tool_calls": [call.name for call in self.tool_calls],
            "salvaged": self.salvaged,
        }


@dataclass(slots=True)
class HostState:
    """What the host was at preflight, in the terms that make a run reproducible."""

    ollama_version: str = ""
    model: str = ""
    model_digest: str = ""
    model_digest_source: str = ""
    trained_context: int = 0
    env: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_FUNCTION_BLOCK = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)


def _unwrap(value: str) -> str:
    """Undo the newline the template puts around a parameter value, and only that.

    The format is `<parameter=name>` newline, the value verbatim, newline
    `</parameter>`, and the value is raw text with no escaping — which is the whole
    reason this format is worth building on. `.strip()` here would eat trailing blank
    lines out of a file the model is writing, so exactly one newline comes off each end.
    """
    value = value.removeprefix("\n")
    value = value.removesuffix("\n")
    return value


def parse_native_tool_calls(text: str) -> tuple[list[ToolCall], int]:
    """Salvage `<tool_call>` blocks the server did not parse, and count dropped prose.

    The template permits reasoning *before* a call and forbids it after, so text
    following the last `</tool_call>` is discarded with a counter rather than fed back
    as an error the model then has to cope with.
    """
    calls: list[ToolCall] = []
    for block in _TOOL_CALL_BLOCK.findall(text):
        function = _FUNCTION_BLOCK.search(block)
        if function is None:
            continue
        arguments = {
            key: _unwrap(value) for key, value in _PARAMETER.findall(function.group(2))
        }
        calls.append(
            ToolCall(name=function.group(1), arguments=arguments, salvaged=True)
        )
    dropped = len(text.rsplit("</tool_call>", 1)[-1].strip()) if calls else 0
    return calls, dropped


def _post(url: str, body: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def _get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload: dict[str, Any] = json.loads(response.read())
        return payload


def _post_json(url: str, body: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    try:
        with _post(url, body, timeout) as response:
            payload: dict[str, Any] = json.loads(response.read())
            return payload
    except urllib.error.HTTPError as exc:
        _raise_for_body(exc)


def _raise_for_chunk(message: str) -> NoReturn:
    """Classify a mid-stream ollama error. Only one of them is recoverable."""
    if "invalid tool call arguments" in message:
        raise TruncatedToolCall(message[:400])
    raise TransportError(message[:400])


def _raise_for_body(exc: urllib.error.HTTPError) -> NoReturn:
    """Turn ollama's error body into the typed failure the loop can act on."""
    raw = exc.read().decode(errors="replace")
    try:
        parsed = json.loads(raw)
    except ValueError:
        raise TransportError(f"HTTP {exc.code}: {raw[:400]}") from exc
    node = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(node, dict):
        if node.get("type") == "exceed_context_size_error":
            raise ContextOverflow(
                n_prompt_tokens=int(node.get("n_prompt_tokens") or 0),
                n_ctx=int(node.get("n_ctx") or 0),
            ) from exc
        raise TransportError(f"HTTP {exc.code}: {node.get('message') or node}") from exc
    raise TransportError(f"HTTP {exc.code}: {str(node or raw)[:400]}") from exc


def _model_digest(model: str) -> tuple[str, str]:
    """The GGUF blob's sha256, which is also the source repository's LFS oid.

    Read out of the on-disk manifest because no endpoint returns it: `/api/tags` gives
    the *manifest* digest, which changes when a tag's parameters change and therefore
    cannot identify the weights. Recording nothing is worse than recording which
    source the value came from, so both outcomes are reported rather than one guessed.
    """
    name, _, tag = model.partition(":")
    root = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")
    manifest = (
        root / "manifests" / "registry.ollama.ai" / "library" / name / (tag or "latest")
    )
    if manifest.is_file():
        try:
            layers = json.loads(manifest.read_text()).get("layers") or []
        except (OSError, ValueError):
            layers = []
        for layer in layers:
            if str(layer.get("mediaType") or "").endswith(".model"):
                digest = str(layer.get("digest") or "").removeprefix("sha256:")
                return digest, "gguf-blob"
    return "", "unavailable"


class Transport:
    """One resident model, one endpoint, and the wire facts the artifact records."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        num_ctx: int = DEFAULT_NUM_CTX,
        think: bool = True,
        sampling: Sampling = GREEDY,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        openai_compat: bool = False,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.think = think
        self.sampling = sampling
        self.max_output_tokens = max_output_tokens
        self.keep_alive = keep_alive
        self.openai_compat = openai_compat
        self.keep_alive_verified = False

    @property
    def endpoint(self) -> str:
        return "/v1/chat/completions" if self.openai_compat else "/api/chat"

    def resident(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = (
            _get_json(f"{self.host}/api/ps").get("models") or []
        )
        return models

    def preflight(self) -> HostState:
        """Abort rather than run against a host that would make the numbers a lie.

        Three things have to be true and none of them is recoverable from a result file
        afterwards: nothing else large is resident, the tag this names is the weights
        that answer, and `keep_alive` actually took. With `OLLAMA_KEEP_ALIVE=0` in the
        environment a dropped `keep_alive` field unloads the model after every single
        request, and the only symptom is a wall clock that looks like a slower model.
        """
        state = HostState(model=self.model)
        try:
            version = _get_json(f"{self.host}/api/version").get("version")
        except (urllib.error.URLError, OSError) as exc:
            raise TransportError(f"no ollama at {self.host}: {exc}") from exc
        state.ollama_version = str(version or "")

        for entry in self.resident():
            name = str(entry.get("name") or entry.get("model") or "?")
            if name.split(":")[0] != self.model.split(":")[0]:
                size_gib = float(entry.get("size") or 0) / 2**30
                raise TransportError(
                    f"{name} is already resident at {size_gib:.2f} GiB — one model at a "
                    "time on this host; unload it before running this arm"
                )

        state.env = {
            "OLLAMA_KEEP_ALIVE": os.environ.get("OLLAMA_KEEP_ALIVE", "unset"),
            "OLLAMA_NUM_PARALLEL": os.environ.get("OLLAMA_NUM_PARALLEL", "unset"),
        }
        # Both are read by the server at startup, not by this client, so neither can be
        # pinned from here: the harness records what the login environment holds and
        # says what it costs. The per-request `keep_alive` overrides the first, verified
        # on the first response rather than assumed. The second admits a second slot
        # that would evict this episode's prefix, which the residency check above makes
        # unlikely rather than impossible.
        if state.env["OLLAMA_NUM_PARALLEL"] not in ("1", "unset"):
            state.warnings.append(
                f"OLLAMA_NUM_PARALLEL={state.env['OLLAMA_NUM_PARALLEL']} in this "
                "environment and only a server restart changes it; a concurrent client "
                "could evict this episode's prefix"
            )

        try:
            shown = _post_json(f"{self.host}/api/show", {"model": self.model})
        except (TransportError, urllib.error.URLError, OSError) as exc:
            state.warnings.append(f"/api/show failed: {exc}")
            return state
        for key, value in (shown.get("model_info") or {}).items():
            if key.endswith(".context_length") and isinstance(value, int):
                state.trained_context = value
                break
        state.model_digest, state.model_digest_source = _model_digest(self.model)
        return state

    def verify_keep_alive(self) -> bool:
        """True when the resident copy is held into the future rather than unloaded."""
        for entry in self.resident():
            raw = str(entry.get("expires_at") or "")
            try:
                expires = datetime.fromisoformat(raw)
            except ValueError:
                continue
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=UTC)
            self.keep_alive_verified = expires > datetime.now(UTC)
            return self.keep_alive_verified
        return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        timeout: float = 900.0,
        think: bool | None = None,
    ) -> Turn:
        """`think` overrides the episode default for one call, and only downward in cost.

        The loop uses it for a forced final answer: a turn cut off mid-reasoning has
        already proved the model will spend the whole budget thinking, and with thinking
        off this model answers directly — measured at 31 tokens against a 2,048-token cap.
        """
        if self.openai_compat:
            return self._chat_openai(messages, tools, timeout=timeout)
        return self._chat_native(messages, tools, timeout=timeout, think=think)

    def _chat_native(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        timeout: float,
        think: bool | None = None,
    ) -> Turn:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": self.think if think is None else think,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.max_output_tokens,
                **self.sampling.options(),
            },
        }
        if tools:
            body["tools"] = tools

        turn = Turn()
        content: list[str] = []
        thinking: list[str] = []
        started = time.perf_counter()
        try:
            with _post(f"{self.host}/api/chat", body, timeout) as response:
                for raw in response:
                    line = raw.decode(errors="replace").strip()
                    if not line:
                        continue
                    chunk: dict[str, Any] = json.loads(line)
                    if chunk.get("error"):
                        _raise_for_chunk(str(chunk["error"]))
                    message = chunk.get("message") or {}
                    piece = str(message.get("content") or "")
                    reasoning = str(message.get("thinking") or "")
                    if (piece or reasoning) and not turn.ttft_s:
                        turn.ttft_s = time.perf_counter() - started
                    content.append(piece)
                    thinking.append(reasoning)
                    for call in message.get("tool_calls") or []:
                        function = call.get("function") or {}
                        arguments = function.get("arguments")
                        turn.tool_calls.append(
                            ToolCall(
                                name=str(function.get("name") or ""),
                                arguments=dict(arguments)
                                if isinstance(arguments, dict)
                                else {},
                            )
                        )
                    if chunk.get("done"):
                        turn.done_reason = str(chunk.get("done_reason") or "")
                        turn.prompt_eval_count = int(
                            chunk.get("prompt_eval_count") or 0
                        )
                        turn.eval_count = int(chunk.get("eval_count") or 0)
                        turn.prompt_eval_ms = (
                            float(chunk.get("prompt_eval_duration") or 0) / 1e6
                        )
                        turn.eval_ms = float(chunk.get("eval_duration") or 0) / 1e6
        except urllib.error.HTTPError as exc:
            _raise_for_body(exc)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc

        turn.wall_s = time.perf_counter() - started
        turn.content = "".join(content)
        turn.thinking = "".join(thinking)
        self._salvage(turn)
        return turn

    def _chat_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        timeout: float,
    ) -> Turn:
        """The portability check. Unstreamed, because no reason to stream survives here.

        `num_ctx`, `keep_alive` and `think` cannot be expressed on this path at all,
        there are no durations to read, and `arguments` comes back as a JSON string
        rather than an object. It answers "does the model work behind an OpenAI-shaped
        server" and nothing about the experiment's central claim.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "seed": self.sampling.seed,
            "max_tokens": self.max_output_tokens,
        }
        if tools:
            body["tools"] = tools

        started = time.perf_counter()
        try:
            payload = _post_json(f"{self.host}/v1/chat/completions", body, timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc

        turn = Turn()
        turn.wall_s = turn.ttft_s = time.perf_counter() - started
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        turn.content = str(message.get("content") or "")
        turn.thinking = str(message.get("reasoning") or "")
        turn.done_reason = str(choice.get("finish_reason") or "")
        usage = payload.get("usage") or {}
        turn.prompt_eval_count = int(usage.get("prompt_tokens") or 0)
        turn.eval_count = int(usage.get("completion_tokens") or 0)
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            turn.tool_calls.append(
                ToolCall(
                    name=str(function.get("name") or ""),
                    arguments=_decode_arguments(function.get("arguments")),
                )
            )
        self._salvage(turn)
        return turn

    def _salvage(self, turn: Turn) -> None:
        if turn.tool_calls or "<tool_call>" not in turn.content:
            return
        calls, dropped = parse_native_tool_calls(turn.content)
        if not calls:
            return
        turn.tool_calls.extend(calls)
        turn.salvaged = len(calls)
        turn.trailing_prose_dropped = dropped


def _decode_arguments(raw: Any) -> dict[str, Any]:
    """`/v1` hands back `arguments` as a JSON string; `/api/chat` hands back an object."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except ValueError:
            return {}
        if isinstance(decoded, dict):
            return dict(decoded)
    return {}
