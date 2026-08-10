---
name: real-weights
description: Pre-flight, pilot ladder and memory accounting for running Orbit against real MLX weights on the 36 GB M4 Max. Use before any command that loads a model — doctor, serve, bench, eval, train, either gate, or any mlx_lm call — and before downloading, converting or timing a model. Also use when a run is slow, wedged, OOM-killed, or reports throughput that looks wrong.
---

# Running against real weights

The baseline host is an **M4 Max, 36 GB, 1 TB**. Metal's ceiling is
`max_recommended_working_set_size` = **28.08 GiB**, not 36 GB. Tier 0 alone is
**23.0 GiB**. There is room for one model and nothing else.

Overcommitting unified memory here does not raise — it wedges the machine or reboots it.
Everything below exists to keep a measurement from becoming a reboot.

## 1. Pre-flight, every time

```bash
curl -s localhost:11434/api/ps                 # expect {"models":[]}
lsof -nP -iTCP:8081 -sTCP:LISTEN               # expect nothing at rung 3
lsof -nP -iTCP:8080 -sTCP:LISTEN               # a Orbit gateway already holding weights
memory_pressure | tail -5                      # or the vm_stat reading in §3
```

- `ollama serve` is resident from login. It holds nothing until asked, then loads
  17–23 GB for whoever asks. It is a **neighbour, not a dependency** — do not start it,
  and do not assume Orbit's failure is its fault.
- A stale `mlx-optiq` on 8081 survives Orbit restarts, because Orbit never spawned it.
  At rung 3 nothing should be there; if something is, find out whose it is before you
  add 23 GiB next to it.

Abort the run rather than "seeing what happens" if either check comes back non-empty.

## 2. The pilot ladder

**Never open with the full model, the full tier or the full run count.** Each rung is
authorised by the measurement from the rung below it:

| Rung | What | What it answers |
|---|---|---|
| 0 | `backend = "mock"` config, or `pytest` | Is the wiring right? Costs nothing. |
| 1 | **one** generation, warmed up | Does it load, what is the real footprint, what headroom is left? |
| 2 | **eight** runs | Is throughput stable, does memory grow per request? |
| 3 | the full run (100-run gate, an eval sweep, a training pass) | The number you actually wanted. |

Rules that make the ladder worth having:

- **Record the footprint at rung 1 and compare it to 28.08 GiB before rung 2.** If the
  measured cost does not leave room for the next rung, stop and report the measurement.
  Do not try it anyway.
- **Two models is a different rung, not a bigger one.** Anything that holds tier 0 and a
  verifier at once (rung 2 swap, `gate isolation` with adapters present) is 2 × 23.0 GiB
  against 28.08 and must be reasoned about before it is run, not after.
- **The 100-run tool-call gate is ~2 min when it works and ~20 min when it does not.**
  Find that out on run one, not on run ninety.
- One call before eight, eight before a hundred — including when a previous session
  already did it. A config change invalidates the ladder.

## 3. Reading memory correctly

```bash
sysctl hw.memsize
vm_stat | head -8                              # pages are 16384 B on Apple Silicon
sysctl vm.swapusage
```

- **Headroom is `total − active`, never `Pages free`.** macOS reports most of a healthy
  machine as reclaimable; `Pages free` on an idle 36 GB box is routinely under 1 GB and
  means nothing. Tier 0 needs ~27 GB of headroom by the `total − active` measure.
- **Thrashing is `Pageouts` climbing, not `vm.swapusage used` being non-zero.** A guard
  written against `used` aborts runs that were never in trouble.
- Watch `Pageouts` across the run, not once at the start. A run that pages is a run whose
  timings are fiction, whatever it finally reports.

## 4. What loads 23.0 GiB

`MLXTier0Backend.__init__` calls `mlx_lm.load()` eagerly, so weights land at construction
— not at first request:

| Loads weights | Never loads weights |
|---|---|
| `doctor`, `serve`, `bench`, `eval`, `train`, `gate toolcall`, `gate isolation` | `pytest`, `extract`, `profile`, `audit verify` |

For a cheap question, point a `backend = "mock"` config at it:

```bash
orbit --config orbit.mock.toml doctor        # --config is GLOBAL: before the subcommand
```

`orbit --config … doctor` with the mlx backend is a 23 GiB answer to a question about
config parsing. `--config` after the subcommand is a different, silent mistake.

`orbit gate isolation` builds tier 0 **once per adapter plus two**, with two live across
one `asyncio.gather` (`eval/gates.py:178-180`). Mounting a subset is the point of the
test, so it cannot reuse one instance. It is harmless only while `adapters/` does not
exist and the gate short-circuits — the moment a trained adapter lands, this is the
likeliest way to wedge the machine.

## 5. Timing anything

- **Warm up first.** The first generation after a load pays ~9 s of Metal kernel
  compilation. Timed cold, tier 0 reads ~4 tok/s where the truth is ~27.
- Report prefill and decode separately; a single tok/s figure over a short generation is
  mostly prefill.
- Reference points on this host: tier 0 ~65 tok/s unconstrained, ~27 tok/s with a
  constrained decode. The 2.4× is the sync `tokens.tolist()` forces and is not removable
  from a Python-side constrained decode. A number far outside these is a measurement bug
  before it is a hardware finding.
- `docs/platform.md` holds the recorded figures. Compare against it rather than
  re-deriving, and update it when a figure genuinely moves.

## 6. Models, caches and downloads

- **Look in both local caches before downloading**: `hf cache list` and `ollama list`
  overlap.
- **Neither format can serve the other.** Tier 0 needs a snapshot directory of OptiQ
  safetensors with a `config.json`; ollama stores opaque GGUF blobs and has none.
- **Check the model can run here before spending an hour fetching it.** `orbit.toml`
  documents tier-1 rung 1 as arithmetically dead on this host — 122B does not fit, at any
  quantisation, alongside anything.
- `HF_HUB_OFFLINE=1` for anything touching a local snapshot, so a cache miss fails loudly
  instead of silently re-downloading.
- Corporate TLS interception means Python cannot reach Hugging Face at all until
  `SSL_CERT_FILE` points at a Keychain export — `docs/platform.md` §5 has the procedure.

## 7. When it goes wrong

- **Slow, with `Pageouts` climbing** — it is paging. Kill it; the numbers are worthless.
  Something else is resident, or the run is a rung too high.
- **Killed with no traceback** — the kernel took it. Re-check §1, then drop a rung.
- **Throughput an order of magnitude low** — you did not warm up (§5), or another process
  holds weights (§1).
- **`BackendUnavailable`** — `mlx`/`mlx-lm` missing, or the model path is a GGUF blob
  rather than a snapshot directory (§6).

Report what you measured, including the rung you stopped at and why. A stopped ladder
with a recorded reason is a result; an unattended full run that reboots the machine is
not.
