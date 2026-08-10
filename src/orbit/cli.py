"""`orbit` — the command line.

Subcommands map onto the spec's milestones, so a user working through M0-M6 can run
each gate by name:

    orbit serve                      sec 8   gateway, three wire protocols
    orbit doctor                     sec 8.6 offline posture + runtime status
    orbit extract a0|a1|a2           sec 6   build an adapter corpus
    orbit profile                    sec 6.4 routing profile
    orbit train sft|dpo              sec 6   train an adapter
    orbit eval merge                 sec 10.1 the merge eval, four bars
    orbit gate toolcall|isolation    sec 10.2, 4.2
    orbit bench latency|tier1        sec 10.4, M0 Gate A/B
    orbit audit verify               sec 9.2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orbit.config import Config

if TYPE_CHECKING:
    from orbit.backends.base import Backend
    from orbit.eval.merge_eval import Arm
    from orbit.types import GenRequest


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


# --- serve ------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    from orbit.gateway.app import serve

    cfg = Config.load(args.config)
    if args.port:
        cfg.server.port = args.port
    if args.backend:
        cfg.backend = args.backend
    if args.no_compact:
        cfg.compaction.enabled = False
    print(
        f"orbit serving on http://{cfg.server.host}:{cfg.server.port}", file=sys.stderr
    )
    print(f"  backend={cfg.backend} tier0={cfg.tier0.model}", file=sys.stderr)
    print(
        f"  tier1={'on ' + cfg.tier1.model if cfg.tier1.enabled else 'off'}",
        file=sys.stderr,
    )
    print("  /v1/messages  /v1/chat/completions  /v1/responses", file=sys.stderr)
    serve(cfg)
    return 0


# --- doctor -----------------------------------------------------------------


def _rung_note(cfg: Config) -> str:
    """One line on what the configured rung of the sec 5.5 ladder actually is.

    Each rung answers a different question, and the differences are the kind a user
    finds out about at the wrong moment: rung 3 is weaker than it looks, rung 2 is
    slower than it looks, and rung 4 is not local at all.
    """
    from orbit.backends import REMOTE_RUNG, RESIDENT_SWAP_RUNG, SECOND_OPINION_RUNG

    return {
        SECOND_OPINION_RUNG: (
            "rung 3: the resident model with its adapter unmounted — weaker than the "
            "streamed verifier, but it still catches adapter overfit (sec 5.5)"
        ),
        RESIDENT_SWAP_RUNG: (
            "rung 2: the 80B swapped into residency. Tier 0 is evicted while it "
            "serves, so tier-0 requests wait out the swap (~10 s each way, sec 5.5)"
        ),
        REMOTE_RUNG: (
            "rung 4: a remote API. Verdicts and the code they judge leave this "
            "machine, and the sec 8.6 offline claim does not hold while it is on"
        ),
    }.get(cfg.tier1.rung, "")


def _tier1_model(cfg: Config) -> str:
    """Which weights the configured rung would actually use."""
    from orbit.backends import REMOTE_RUNG, SECOND_OPINION_RUNG

    if cfg.tier1.rung == SECOND_OPINION_RUNG:
        # Rung 3 is tier 0 itself, with the adapter off.
        return cfg.tier0.model
    if cfg.tier1.rung == REMOTE_RUNG:
        return cfg.tier1.remote_model or cfg.tier1.model
    return cfg.tier1.model


def cmd_doctor(args: argparse.Namespace) -> int:
    from orbit.backends import REMOTE_RUNG, build_tier0, build_tier1
    from orbit.backends.base import BackendUnavailable
    from orbit.eval.latency import Environment
    from orbit.gateway.toolcall.constrain import Constrainer
    from orbit.offline import verify

    cfg = Config.load(args.config)
    out: dict[str, Any] = {
        "environment": Environment.detect().as_dict(),
        "backend": cfg.backend,
        "constrained_decoding": Constrainer(enabled=cfg.toolcall.constrain).status(),
        "offline": verify(
            strict_deps=args.strict_deps,
            # A configured rung 4 breaks the posture whether or not lsof caught a
            # call: the claim is about what this process will do, not what it has
            # done so far.
            remote_tier1=cfg.tier1.enabled and cfg.tier1.rung == REMOTE_RUNG,
        ).as_dict(),
    }
    tier0 = None
    try:
        tier0 = build_tier0(cfg)
        out["tier0"] = {
            "ok": True,
            "model": cfg.tier0.model,
            "container_hash": tier0.container_hash(),
            "adapters": list(tier0.mounted_adapters()),
            "supports_kv_state": tier0.supports_state(),
        }
    except (BackendUnavailable, ValueError) as exc:
        out["tier0"] = {"ok": False, "reason": str(exc)}

    try:
        # Rungs 2 and 3 both involve the resident model, so they need that backend.
        # When tier 0 itself failed to build there is nothing to serve them from, and
        # the honest answer is to say so rather than report tier 1 as merely disabled.
        tier1 = build_tier1(cfg, tier0)
        out["tier1"] = (
            {
                "ok": True,
                "rung": cfg.tier1.rung,
                "model": _tier1_model(cfg),
                # None on rung 4 by construction: you cannot attest to a model you
                # do not hold (sec 9.1).
                "container_hash": tier1.container_hash(),
                "note": _rung_note(cfg),
            }
            if tier1
            else {"ok": False, "reason": "tier 1 disabled in config"}
        )
    except (BackendUnavailable, ValueError) as exc:
        out["tier1"] = {"ok": False, "reason": str(exc)}

    from orbit.adapters.train import trainer_available

    ok, detail = trainer_available()
    out["trainer"] = {"ok": ok, "detail": detail}
    _print(out)
    return 0 if out["tier0"].get("ok") else 1


def cmd_offline_env(args: argparse.Namespace) -> int:
    from orbit.offline import env_exports

    print(env_exports())
    return 0


# --- extract ----------------------------------------------------------------


def cmd_extract(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "a0":
        from orbit.adapters import extract_a0

        traces = extract_a0.generate(n=args.n, seed=args.seed)
        path = extract_a0.write_jsonl(traces, out_dir / "train.jsonl")
        _print({"corpus": str(path), **extract_a0.report(traces)})
        return 0

    if args.kind == "a1":
        from orbit.adapters import extract_a1
        from orbit.adapters.filters import ExtractionFilters

        filters = ExtractionFilters()
        if args.merge_policy:
            filters.merge_policy = args.merge_policy
        train, held, report = extract_a1.extract(
            args.repo,
            filters=filters,
            limit=args.limit,
            since=args.since,
            holdout=args.holdout,
        )
        extract_a1.write_jsonl(train, out_dir / "train.jsonl")
        extract_a1.write_manifest(train, out_dir / "manifest.jsonl")
        if held:
            extract_a1.write_jsonl(held, out_dir / "holdout.jsonl")
            extract_a1.write_manifest(held, out_dir / "holdout_manifest.jsonl")
        payload = report.as_dict()
        payload["corpus"] = str(out_dir / "train.jsonl")
        payload["holdout"] = len(held)
        _print(payload)
        return 0 if not report.thin else 2

    from orbit.adapters import extract_a2
    from orbit.adapters.filters import ExtractionFilters

    # Distinct names from the a1 branch above: A2 yields PreferencePair/A2Report
    # where A1 yields Pair/ExtractionReport, and reusing the names binds this
    # function's locals to whichever branch is read first.
    a2_train, a2_held, a2_report = extract_a2.extract(
        args.repo,
        filters=ExtractionFilters(),
        limit=args.limit,
        reviews_path=args.reviews,
        holdout=args.holdout,
    )
    extract_a2.write_jsonl(a2_train, out_dir / "train.jsonl")
    if a2_held:
        extract_a2.write_jsonl(a2_held, out_dir / "holdout.jsonl")
    payload = a2_report.as_dict()
    payload["corpus"] = str(out_dir / "train.jsonl")
    _print(payload)
    return 0 if not a2_report.thin else 2


# --- profile ----------------------------------------------------------------


def cmd_profile(args: argparse.Namespace) -> int:
    """Build a routing profile from a recorded activation dump (sec 6.4).

    The forward pass that produces the counts is backend work and needs the real
    model; this builds the sidecar from its output, checks it, and writes it.
    """
    from orbit.adapters import profile as prof

    raw = json.loads(Path(args.counts).read_text(encoding="utf-8"))
    built = prof.build(
        raw["counts"],
        raw.get("mass"),
        model_name=args.model,
        model_hash=raw.get("model_hash", ""),
        corpus_hash=raw.get("corpus_hash", ""),
        n_tokens=raw.get("n_tokens", 0),
    )
    path = built.write(args.out)
    _print({"profile": str(path), "sanity": prof.sanity(built)})
    return 0


# --- train ------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    from orbit.adapters.train import DPOConfig, SFTConfig, train_dpo, train_sft
    from orbit.attest.provenance import SourceKind

    cfg = Config.load(args.config)
    corpus = Path(args.corpus)
    output = Path(args.out)

    # Counted once, before either branch. The generator form left the corpus handle
    # open until the GC got to it, which on a dry run is the whole process lifetime.
    with corpus.open(encoding="utf-8") as fh:
        n_pairs = sum(1 for _ in fh)

    if args.method == "sft":
        result = train_sft(
            model=args.model or cfg.tier0.model,
            corpus=corpus,
            output=output,
            adapter_name=args.name,
            source_kind=SourceKind(args.source_kind),
            source_repo=args.repo or "",
            n_pairs=n_pairs,
            cfg=SFTConfig(neftune_alpha=args.neftune),
            dry_run=args.dry_run,
        )
    else:
        result = train_dpo(
            model=args.model or cfg.tier0.model,
            corpus=corpus,
            output=output,
            adapter_name=args.name,
            parent_adapter=Path(args.mount_adapter) if args.mount_adapter else None,
            source_repo=args.repo or "",
            n_pairs=n_pairs,
            cfg=DPOConfig(),
            dry_run=args.dry_run,
        )
    _print(result.as_dict())
    return 0 if result.ok else 1


# --- eval -------------------------------------------------------------------


def _review_proxy(args: argparse.Namespace, verifier: Any) -> Any:
    """Build the review-comment proxy named on the command line, or None."""
    from orbit.eval.merge_eval import scored_review_proxy, tier1_review_proxy

    choice = args.review_proxy or "none"
    if choice == "none":
        return None
    if choice == "tier1":
        if not verifier.available:
            raise SystemExit("--review-proxy tier1 needs tier 1 enabled in the config")
        return tier1_review_proxy(verifier)
    scores = json.loads(Path(args.review_scores).read_text(encoding="utf-8"))
    return scored_review_proxy({str(k): float(v) for k, v in scores.items()})


def cmd_eval_merge(args: argparse.Namespace) -> int:
    from orbit.adapters.extract_a1 import extract
    from orbit.backends import build_tier0, build_tier1
    from orbit.eval.merge_eval import Arm, cases_from_holdout, run
    from orbit.eval.worktree import from_config as build_runner
    from orbit.gateway.pipeline import Pipeline

    cfg = Config.load(args.config)
    if args.review_proxy == "file" and not args.review_scores:
        raise SystemExit("--review-proxy file needs --review-scores PATH")

    _train, held, _report = extract(args.repo, limit=args.limit, holdout=args.holdout)
    cases = cases_from_holdout(held)
    if not cases:
        _print({"error": "no held-out cases; increase --holdout or --limit"})
        return 1

    tier0 = build_tier0(cfg)
    tier1 = build_tier1(cfg, tier0)
    pipeline = Pipeline(cfg, tier0, tier1)

    # The worktree runner is what makes `test_pass` and `convention conformance`
    # measurable at all; without a `[eval]` block in orbit.toml it is None and
    # `compare_arms` says so rather than passing the M3 gate on two metrics.
    runner = None if args.no_worktree else build_runner(cfg.eval, repo=args.repo)
    proxy = _review_proxy(args, pipeline.verifier)

    async def gen(
        adapter: str | None, cascade: bool
    ) -> Callable[[GenRequest], Awaitable[str]]:
        async def _gen(req: GenRequest) -> str:
            req = req.with_(adapter=adapter)
            if cascade:
                result, _ = await pipeline.run(req)
                return result.text
            return (await tier0.generate(req)).text

        return _gen

    async def build_arms() -> list[Arm]:
        arms = [Arm(name="tier0 base", generate=await gen(None, False))]
        if args.a1:
            arms.append(
                Arm(
                    name="tier0 + A1",
                    generate=await gen(args.a1, False),
                    adapter=args.a1,
                )
            )
        if args.a2:
            arms.append(
                Arm(
                    name="tier0 + A1 + A2",
                    generate=await gen(args.a2, False),
                    adapter=args.a2,
                )
            )
        if cfg.tier1.enabled:
            arms.append(
                Arm(
                    name="cascade + tier1 rerank",
                    generate=await gen(args.a1, True),
                    adapter=args.a1,
                )
            )
        return arms

    async def main() -> int:
        # One run of the repo's own checks before spending hours on the eval. A
        # suite that already fails at the base makes `test_pass_rate` a statement
        # about the repository rather than about any arm.
        base_health = None
        if runner is not None and runner.measures_tests:
            outcome = await runner.verify_base()
            base_health = outcome.as_dict()
            if outcome.tests_passed is False:
                print(
                    f"warning: `{' '.join(runner.test_command)}` already fails at "
                    f"{cfg.eval.base_rev} — test_pass_rate will not discriminate "
                    "between arms",
                    file=sys.stderr,
                )

        arms = await build_arms()
        report = await run(
            cases, arms, repo=Path(args.repo), runner=runner, review_proxy=proxy
        )
        report.base_health = base_health
        if args.out:
            report.write(args.out)
        print(report.table(), file=sys.stderr)
        _print(
            {
                "measured": report.measured,
                "base_health": base_health,
                "arms": [a.as_dict() for a in report.arms],
                "comparisons": report.comparisons,
            }
        )
        # M3 gate: the first non-base arm must beat base on >=3 of 5.
        return 0 if (report.comparisons and report.comparisons[0]["pass"]) else 1

    return asyncio.run(main())


def cmd_eval_regression(args: argparse.Namespace) -> int:
    """Sec 10.3. Exit 1 on a regression; exit 0 on a clean run or a first baseline."""
    from orbit.attest.receipt import engine_commit
    from orbit.backends import build_tier0
    from orbit.eval.regression import Baseline, check_comparable, compare, run
    from orbit.eval.regression_items import SUITE, by_category

    cfg = Config.load(args.config)
    tier0 = build_tier0(cfg)
    items = by_category(args.category) if args.category else SUITE

    results = asyncio.run(run(tier0, items, adapter=args.adapter))

    baseline_path = Path(args.baseline)
    baseline = None if args.rebaseline else Baseline.load(baseline_path)
    warning = check_comparable(baseline, tier0.container_hash(), args.adapter)
    if warning:
        # A baseline from a different model would report a deliberate change as a
        # regression, which teaches everyone to ignore the suite.
        baseline = None

    report = compare(results, baseline)
    if warning:
        report.comparable = False
        report.note = warning

    if baseline is None:
        Baseline.from_results(
            results,
            container_hash=tier0.container_hash(),
            adapter=args.adapter,
            engine_commit=engine_commit(),
        ).write(baseline_path)
        report.baseline_written = True
        if not report.note:
            report.note = (
                f"No baseline: recorded {len(results)} items at {baseline_path}. "
                "Re-run after a change to see what moved."
            )

    payload = report.as_dict()
    payload["items_run"] = len(results)
    payload["baseline_path"] = str(baseline_path)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    _print(payload)
    print(report.summary(), file=sys.stderr)
    return 0 if report.clean else 1


def cmd_gate(args: argparse.Namespace) -> int:
    from orbit.backends import build_tier0, build_tier1
    from orbit.eval.gates import adapter_isolation_gate, toolcall_gate
    from orbit.gateway.pipeline import Pipeline

    cfg = Config.load(args.config)

    if args.which == "toolcall":
        tier0 = build_tier0(cfg)
        pipeline = Pipeline(cfg, tier0, build_tier1(cfg, tier0))

        async def run_turn(req: GenRequest) -> Any:
            prepared = pipeline._prepare_sampling(req)
            result, _ = await pipeline.cascade.produce(prepared)
            _result, info = await pipeline._settle_tool_calls(prepared, result)
            return info

        result = asyncio.run(toolcall_gate(run_turn, runs=args.runs))
        _print(result.as_dict())
        return 0 if result.passed else 1

    from orbit.backends.mock import MockBackend

    def factory(names: Sequence[str]) -> Backend:
        if cfg.backend == "mock":
            return MockBackend(adapters=tuple(names))
        from orbit.backends.mlx_tier0 import MLXTier0Backend

        backend = MLXTier0Backend(
            cfg.tier0.container_path or cfg.tier0.model, adapter_dir=None
        )
        for name in names:
            backend.mount(name, Path(cfg.tier0.adapter_dir) / name)
        return backend

    adapters = args.adapters or list(build_tier0(cfg).mounted_adapters())
    result = asyncio.run(adapter_isolation_gate(factory, adapters))
    _print(result.as_dict())
    return 0 if result.passed else 1


def cmd_bench(args: argparse.Namespace) -> int:
    from orbit.backends import STREAMED_RUNG, build_tier0, build_tier1
    from orbit.eval.latency import Environment, LatencyReport, m0_gate_a, measure

    cfg = Config.load(args.config)

    if args.which == "tier1":
        if not cfg.tier1.enabled:
            _print({"error": "tier 1 is disabled; enable it in config to run Gate B"})
            return 1

        # Rung 1 reaches the engine over a socket and never reads tier 0, so Gate B is
        # a one-model measurement — but `build_tier0` calls `mlx_lm.load()` eagerly,
        # and building one here cost 23.0 GiB on a host measured with 25.9 GB of
        # headroom. That is the difference between Gate B running and Gate B being
        # the second model on a 36 GB box. The rungs that *do* serve from tier 0
        # (3 and 2) carry no prefill instrument, so for them this command can only
        # print the error below — which it now does without loading anything.
        tier1 = build_tier1(cfg) if cfg.tier1.rung == STREAMED_RUNG else None

        # Gate B instruments *streamed* prefill, which only rung 1 has. Rung 3 serves
        # the verifier from tier 0's own weights and rung 1 on the mock backend is a
        # MockBackend; neither carries the instrument. Say which rung is configured
        # rather than dying with an AttributeError on the host whose orbit.toml
        # deliberately runs rung 3.
        if tier1 is None or not hasattr(tier1, "gate_b_report"):
            _print(
                {
                    "error": "Gate B measures streamed prefill and needs tier1.rung "
                    f"= 'streamed' on a real engine; this config is rung "
                    f"'{cfg.tier1.rung}' on backend '{cfg.backend}'",
                    "rung": cfg.tier1.rung,
                }
            )
            return 1

        # Deliberately duck-typed, and the hasattr guard above is what makes it
        # safe. The prefill instrument lives only on the rung-1 streamed backend, so
        # declaring it on `Backend` would put a method on every implementation that
        # cannot honour it — which is the same mistake sec 5.1 avoids by giving
        # tier 1 no `generate`.
        instrumented: Any = tier1

        async def gate_b() -> int:
            for frontier in (4_000, 8_000, 16_000):
                await instrumented.measure_prefill(frontier)
            report = instrumented.gate_b_report(
                threshold_tok_per_s=cfg.gates.gate_b_prefill_tok_per_s
            )
            _print(report)
            return 0 if report["pass"] else 1

        return asyncio.run(gate_b())

    tier0 = build_tier0(cfg)

    async def main() -> int:
        cold = await measure(tier0, adapter=args.adapter, cold=True)
        warm = await measure(tier0, adapter=args.adapter, cold=False)
        report = LatencyReport(
            environment=Environment.detect().as_dict(),
            container_hash=tier0.container_hash(),
            adapter_hash=tier0.adapter_hash(args.adapter),
            command=" ".join(sys.argv),
            samples=[*cold, *warm],
            contract_ttft_s=cfg.gates.contract_chat_ttft_s,
            contract_tok_per_s=cfg.gates.contract_chat_tok_per_s,
        )
        if args.out:
            report.write(args.out)
        print(report.table(), file=sys.stderr)
        _print(
            {
                "gate_a": m0_gate_a(
                    report.samples,
                    args.toolcall_failure_rate,
                    ttft_s=cfg.gates.gate_a_ttft_s,
                    decode_tok_per_s=cfg.gates.gate_a_decode_tok_per_s,
                    toolcall_failure_budget=cfg.gates.gate_a_toolcall_failure_rate,
                )
            }
        )
        return 0

    return asyncio.run(main())


def cmd_audit(args: argparse.Namespace) -> int:
    from orbit.attest.audit import verify_chain

    cfg = Config.load(args.config)
    ok, reason = verify_chain(args.log or cfg.attest.audit_log)
    _print({"ok": ok, "reason": reason})
    return 0 if ok else 1


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orbit", description=__doc__.splitlines()[0])
    p.add_argument("--config", default=None, help="path to orbit.toml")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the gateway (sec 8)")
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--backend", choices=("mock", "mlx"), default=None)
    s.add_argument(
        "--no-compact", action="store_true", help="disable harness compaction"
    )
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("doctor", help="runtime status and offline posture (sec 8.6)")
    s.add_argument(
        "--strict-deps", action="store_true", help="also audit the import graph"
    )
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser(
        "offline-env", help="print the harness environment to export (sec 8.6)"
    )
    s.set_defaults(func=cmd_offline_env)

    s = sub.add_parser("extract", help="build an adapter corpus (sec 6)")
    s.add_argument("kind", choices=("a0", "a1", "a2"))
    s.add_argument("--repo", default=".", help="repository to extract from (a1/a2)")
    s.add_argument("--out", default="corpus", help="output directory")
    s.add_argument("--limit", type=int, default=None, help="max commits to walk")
    s.add_argument("--since", default="", help="git --since expression")
    s.add_argument(
        "--holdout", type=int, default=0, help="reserve K most recent for eval"
    )
    s.add_argument("--n", type=int, default=2000, help="traces to generate (a0)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--reviews", default=None, help="forge review export (a2)")
    s.add_argument(
        "--merge-policy", choices=("auto", "skip", "first_parent"), default=None
    )
    s.set_defaults(func=cmd_extract)

    s = sub.add_parser("profile", help="build a routing profile sidecar (sec 6.4)")
    s.add_argument("--counts", required=True, help="JSON activation-count dump")
    s.add_argument(
        "--model", default="", help="model name, decides count- vs mass-ranking"
    )
    s.add_argument("--out", default="profiles/routing_profile.json")
    s.set_defaults(func=cmd_profile)

    s = sub.add_parser("train", help="train an adapter (sec 6)")
    s.add_argument("method", choices=("sft", "dpo"))
    s.add_argument("--corpus", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--model", default=None)
    s.add_argument("--repo", default=None)
    s.add_argument(
        "--source-kind",
        choices=("customer_repo", "permissive_corpus", "synthetic_harness"),
        default="customer_repo",
    )
    s.add_argument("--mount-adapter", default=None, help="A1 adapter to start DPO from")
    s.add_argument("--neftune", type=float, default=0.0)
    s.add_argument(
        "--dry-run", action="store_true", help="print the command without running it"
    )
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("eval", help="run an evaluation")
    esub = s.add_subparsers(dest="eval_kind", required=True)
    e = esub.add_parser("merge", help="repo-held-out merge eval, four bars (sec 10.1)")
    e.add_argument("--repo", default=".")
    e.add_argument("--holdout", type=int, default=25)
    e.add_argument("--limit", type=int, default=500)
    e.add_argument("--a1", default=None, help="A1 adapter name")
    e.add_argument("--a2", default=None, help="A2 adapter name")
    e.add_argument("--out", default=None)
    e.add_argument(
        "--review-proxy",
        choices=("none", "tier1", "file"),
        default="none",
        help="review-comment metric: tier-1 judgement, or scores from --review-scores",
    )
    e.add_argument(
        "--review-scores", default=None, help="JSON {commit_sha: probability}"
    )
    e.add_argument(
        "--no-worktree",
        action="store_true",
        help="skip the test and lint metrics even when orbit.toml [eval] configures them",
    )
    e.set_defaults(func=cmd_eval_merge)

    e = esub.add_parser(
        "regression",
        help="regression detector, not a benchmark (sec 10.3)",
        description=(
            "Runs a fixed curated set and reports what changed against a recorded "
            "baseline. There is no score: the output is a per-item diff, because a "
            "single number tells you nothing after a kernel change and invites "
            "publication the spec explicitly rules out (sec 10.3)."
        ),
    )
    e.add_argument("--adapter", default=None)
    e.add_argument(
        "--baseline",
        # Committed, and deliberately not under `var/`: everything there is
        # reproducible output, while this is the *reference* the output is read
        # against, and re-recording it after a change compares the change with
        # itself (T29). A baseline from another container is detected and ignored
        # rather than believed — `check_comparable`, and the caller above.
        default="baselines/regression-baseline.json",
        help="recorded reference; written on first run",
    )
    e.add_argument(
        "--rebaseline",
        action="store_true",
        help="overwrite the baseline with this run (after a deliberate model change)",
    )
    e.add_argument(
        "--category",
        choices=("reasoning", "maths", "code_localisation"),
        default=None,
        help="restrict to one category; the full set is the comparable one",
    )
    e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_eval_regression)

    s = sub.add_parser("gate", help="run a blocking gate")
    s.add_argument("which", choices=("toolcall", "isolation"))
    s.add_argument("--runs", type=int, default=100)
    s.add_argument("--adapters", nargs="*", default=None)
    s.set_defaults(func=cmd_gate)

    s = sub.add_parser("bench", help="latency suite (sec 10.4) / M0 gates")
    s.add_argument("which", choices=("latency", "tier1"))
    s.add_argument("--adapter", default=None)
    s.add_argument("--out", default=None)
    s.add_argument("--toolcall-failure-rate", type=float, default=0.0)
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("audit", help="audit log operations (sec 9.2)")
    s.add_argument("action", choices=("verify",))
    s.add_argument("--log", default=None)
    s.set_defaults(func=cmd_audit)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
