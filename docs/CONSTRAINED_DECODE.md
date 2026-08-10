# Constrained decoding — where the 2.4× goes, and what to do about it

| | |
|---|---|
| **Purpose** | Scope the cost of constrained decoding (65.6 → 27.1 tok/s) and decide what, if anything, needs to be written in a language other than Python. |
| **Answers** | "Where does the 21 ms/token actually go?" · "Is the sync removable?" · "Do we need a native grammar engine?" |
| **Does not answer** | Whether constrained decoding is worth its cost — it is, and `BASELINE.md` §2.3 is why (100/100 first-attempt tool calls against 0/100). What the host is doing to every throughput number — `HANDOFF.md` §5, T18. |
| **Status** | **F1 and F2 have landed and are now measured** — 2.55× → 2.38×, worth 11–15% (§8.1). **F3 is rejected on measurement** (§6); F4 is not attempted and is worth ~1.11×, not ~1.5× (§8.3). The ladder is complete through rung 3, which passes at 1.00. **T26 is decided** — the 425 ms state is cacheable in principle and not by us (§8.5). **T28 is closed** — §3.2 and §4 are rebuilt on both fixtures and rung 0 now predicts rung 1 to within 5%. |
| **The finding** | **~97% of what constrained decoding costs is one parser state: the gap between a backslash and its escape character, at ~425 ms per occurrence** (§8.4). It is reproducible with no weights, and the rung-0 fixture never entered it, which is why every projection in §3–§4 is 5× low. **It is not one state but one per enclosing object stack** (§8.5), which is what decides whether a cache may collapse it. |
| **Provenance** | Measured 2026-08-09 on the baseline M4 Max, `tools/constrained_decode_bench.py` (no weights) and `tools/constrained_decode_realweights.py` (rung 1–2, six variants). The host passed `PROCESSES.md` §3.1 before *and* after the run. Reproduction: §10. |

This file is committed on purpose. It lived in `/specs/`, which is gitignored, while
`BASELINE.md` pointed at it for "full scope, measurements and design" — the same mistake
`specs/NEXT_STEPS.md` made before it became `docs/HANDOFF.md`. A design a future session
needs cannot live in a directory that does not survive a clone.

---

## 1. What this overturns

`build_logits_processor`'s docstring (`backends/mlx_tier0.py`) states:

> The cost that does matter is not in this function. […] only ~6 ms of that 21 ms/token
> gap is above — the rest is the sync `tokens.tolist()` forces, which serialises the
> decode loop MLX otherwise pipelines. **That sync is not removable**: the mask depends
> on the token just sampled, so any Python-side constrained decode pays it.

Two of those three claims do not survive measurement.

| Claim | Verdict | Evidence |
|---|---|---|
| Only ~6 ms of the gap is in this function | **Wrong** — that profile is the median of a bimodal distribution. Real per-token host cost averages 8.0 ms and peaks at 29 ms. | §3.2 |
| The rest is the sync `tokens.tolist()` forces | **Wrong attribution, and now confirmed against real weights.** A null processor that does only the sync costs **0.62 ms/token** of the 22.81 (§8.2). What costs is the filter-and-mask work itself — the sync is where it lands, not why it costs. | §3.1, **§8.2** |
| That sync is not removable | **True but irrelevant.** The sync is unavoidable; its *cost* is not. Dispatching the forward pass before doing host work hides it entirely. | §3.1, §5 F4 |

The docstring also rejects a mask cache on a premise that no longer holds:

> three consecutive steps inside a JSON string share one parser state, so LMFE hands
> back **the same list object** […] Caching on `id()` (pinning the list […])

Measured over 42 consecutive token positions: **0 identical objects.** `lm-format-enforcer`
≥ 0.11 builds a fresh list on every call, so an `id()`-keyed cache *cannot hit* — which is
exactly consistent with that experiment measuring "27.1 against 27.6, i.e. nothing". The
cache was not useless; **it was incapable of hitting.** Content is identical 40% of the
time (§3.4).

The docstring's closing hedge was right, and is what this document acts on:

> Re-measure before adding it back; **a call with a long string argument is where it
> would finally pay.**

That is the workload measured here.

---

## 2. Mechanism

### 2.1 The loop mlx_lm actually runs

`mlx_lm/generate.py`, verbatim in structure:

```python
y, logprobs = _step(prompt)
mx.async_eval(y, logprobs)
while True:
    next_y, next_logprobs = _step(y)  # builds step n+1 while y is STILL LAZY
    mx.async_eval(next_y, next_logprobs)
    yield y.item(), logprobs  # syncs step n — already running on GPU
    y, logprobs = next_y, next_logprobs
```

`_step(y)` is handed `y` unevaluated. MLX builds step *n+1*'s graph without needing token
*n* on the host, so Python's graph construction overlaps the GPU's execution of step *n*.
That one-step lookahead is what makes unconstrained decode GPU-bound.

A logits processor runs *inside* `_step`. Calling `tokens.tolist()` there forces `mx.eval`
on `y` in the middle of building step *n+1* — the lookahead collapses, and every
millisecond of host work after it is a millisecond the GPU spends idle.

### 2.2 Why the sync is not the cost

The sync must happen: the mask for step *n+1* depends on the token sampled at step *n*.
But a sync only costs wall-clock if the host then does something slow while holding the
GPU. On a synthetic decode loop of the right cost (§3.1), making the loop sync with no
other work costs **nothing** (13.24 against 13.12 ms/token). Adding 6 ms of host work
behind that sync costs **+7.1 ms**.

The correct statement: *the cost is host work serialised against an idle GPU.* The fix is
not to remove the sync but to move the work off the critical path.

---

## 3. Measurements

All rung 0 — no weights, so none of it is affected by T18's host degradation except where
noted in §8.

### 3.1 The mechanism, isolated — synthetic model

48-layer synthetic decoder, vocab 248,077, sized to 13.1 ms/token (~278 GB/s effective, a
healthy host). `--host-ms 6` simulates LMFE.

| Variant | ms/token | tok/s | vs A |
|---|---|---|---|
| A unconstrained | 13.12 | 76.2 | 1.00× |
| B `tokens.tolist()` only, no mask | 13.24 | 75.6 | 1.01× |
| C mask only, no `tolist()` | 13.02 | 76.8 | 0.99× |
| D `tolist()` + mask | 13.23 | 75.6 | 1.01× |
| **E** `tolist()` + mask + **6 ms host spin** | **20.23** | 49.4 | **1.54×** |
| F restructured loop, no host work | 13.06 | 76.6 | 1.00× |
| **G restructured loop, 6 ms spin + mask** | **13.01** | **76.8** | **0.99×** |

- **B ≈ A**: the sync is free.
- **E = A + 7.1 ms** for a 6 ms spin: serialised host work costs ~1:1, plus a small penalty
  for the collapsed lookahead.
- **G ≈ A**: the *same* 6 ms of host work, restructured, costs nothing. The forward pass is
  ~13 ms of GPU time and 6 ms of CPU work fits inside it.

Sweeping the allowed-set size (1,000 / 50,000) leaves C ≈ A: the GPU-side scatter is
size-independent. The cost is on the host.

### 3.2 What the real filter costs — real tokenizer + LMFE

*Re-measured against both fixtures on 2026-08-10 (T28), host at 316/346 GB/s.
`tools/constrained_decode_bench.py filter`; `--per-token` for the raw dump.*

`TARGET_CALL` is the 43-token `edit_file` call this section was originally written
against. `ESCAPED_CALL` is the same call as the model actually emitted it, and the only
difference is two newlines whose backslash the tokenizer splits. **Everything below is a
consequence of that one difference.**

| Parser state | `|allowed|` | \#tokens | LMFE ms (median) | pre-F1 | **landed** | share |
|---|---|---|---|---|---|---|
| **`TARGET_CALL` — 43 tokens** | | | | 305.9 | **127.6 ms** | |
| JSON skeleton | 3 – 269 | 20 | 0.39 | 22.8 | 22.8 | 18% |
| string-open transition | 649 – 658 | 3 | 24.14 | 74.4 | 74.3 | 58% |
| string interior | 246,908 | 20 | 0.95 | 208.7 | 30.5 | 24% |
| **`ESCAPED_CALL` — 46 tokens** | | | | 1,166.4 | **966.9 ms** | |
| JSON skeleton | 3 – 269 | 19 | 0.28 | 16.0 | 16.0 | 2% |
| string-open transition | 649 – 658 | 3 | 24.08 | 74.5 | 74.4 | 8% |
| **post-backslash escape** | **5,795 – 5,798** | **2** | **418.75** | 839.7 | **838.8** | **87%** |
| string interior | 246,908 | 22 | 0.97 | 236.1 | 37.7 | 4% |

**landed** is the path in the tree — LMFE, the F2 content key, and a mask built only on a
miss. The two fixtures walk the same states in nearly the same numbers; the escaped one
additionally enters a state the other never does, and **two tokens out of 46 are 87% of
the call.** F1 is what collapses the string-interior column, 208.7 → 30.5.

| | `TARGET_CALL` | `ESCAPED_CALL` | hardware, §8 |
|---|---|---|---|
| LMFE alone | 2.38 ms/token | **20.47** | ~20.4 (§8.4) |
| F1 removes | 3.15 | 3.38 | **2.99** (§8.1) |
| F2 removes | 1.05 | 1.00 | 0.08 – 0.54 (§8.1) |
| F2 hit rate | 40% | 37% | 35% |
| pre-F1 total | 7.11 ms/token | 25.36 | — |
| **landed total** | **2.97 ms/token** | **21.02** | **22.15** (§8.2, F→D) |

**This is the section T28 said was 5× low, and re-measuring closes the gap: 21.02 against
hardware's 22.15 — 95%, and LMFE alone lands within 0.5%.** Rung 0 predicts rung 1 once
the fixture contains what the workload contains. F1's value is fixture-independent and
matches hardware on both. **The one component rung 0 still over-reads is F2**, at ~1.0
ms/token against 0.08–0.54, because this harness forces an `mx.eval(mask)` per token that
the production path does not pay — so a skipped mask is worth more here than in the tree.

Run-to-run spread across three runs is ~0.5% on the totals and ~2% on the escape median;
the numbers above are one run, not a mean of three, so they stay internally consistent.

### 3.3 The three costs are each removable

| Finding | Measured |
|---|---|
| LMFE returns `list[int]` — every element is already `int` | `type(a) == list`, `type(a[0]) == int`, all-int over 5,000 samples |
| `[int(t) for t in allowed if 0 <= int(t) < vocab]` | **7.39 ms** at \|allowed\|=246,908 — an int→int conversion |
| Model logit width vs tokenizer vocabulary | **248,320 ≥ 248,077** → the bounds filter is *provably* unnecessary on this container |
| `mx.array(246,908 ints)` host→device | 1.48 ms |
| mask: `mx.full` + `mx.array` + scatter | 2.00 ms |
| mask: `mx.full` + scatter, id array prebuilt | 1.01 ms |
| The 26 ms LMFE calls, same prefix re-queried | **0.00 ms** — one-time parser-state builds, not per-token costs |

The 26 ms calls are a **per-request** cost, not a per-token one: `token_filter` builds a
fresh `TokenEnforcer` per request by design (`constrain.py`, to bound memory), so every
request re-pays ~78 ms of identical state construction for the same schema.

### 3.4 A sound cache key

`id()` is impossible. Over 42 consecutive token positions:

| | hits |
|---|---|
| identical **object** | **0 / 42 (0%)** |
| identical **content** | 17 / 42 (40%) |

A wrong mask is a silently wrong constraint, so a probabilistic key is out of the question.
Full content comparison is sound and cheap:

| | ms |
|---|---|
| `len(a) == len(prev)` | 0.0001 |
| `a == prev` (hit) | **0.12** |
| `a == prev` (miss, differing at the last element — the worst case) | **0.13** |

**Per string-interior token:**

| | ms | |
|---|---|---|
| current | **8.36** | |
| fixed, cache miss | **2.15** | 3.9× |
| fixed, cache hit | **0.12** | **70×** |

**The comparison must be measured against two real LMFE outputs, and an earlier run of
this got it wrong.** Building the second list as `list(allowed)` — or independently as
`list(range(n))` — measures the wrong thing in *opposite* directions: a shallow copy shares
every element object so `==` short-circuits on identity, and two independently built lists
share none so it compares 246,908 integers by value (0.87 ms). LMFE's consecutive outputs
share element objects, which is why the honest figure is 0.12 ms. `cmd_key` pulls two
genuinely consecutive filter results rather than copying one.

---

## 4. Cost model

**Measured, not derived — rebuilt 2026-08-10 against both fixtures (T28).** The section
that stood here was derived from §3's components against `TARGET_CALL` and projected
2.7 ms/token where hardware measures 22.15. It was not wrong about its components; it was
wrong about which states a real call visits. This one is `filter`'s output.

| Token class | \#tokens | pre-F1 | **landed** | with a sound key (§8.5) |
|---|---|---|---|---|
| JSON skeleton | 19 | 16.0 | **16.0** | 16.0 |
| string-open transition | 3 | 74.5 | **74.4** | 74.4 |
| **post-backslash escape** | **2** | **839.7** | **838.8** | **838.8** — 2 states, nothing to collapse |
| string interior | 22 | 236.1 | **37.7** | 37.7 |
| **total host work** | 46 | **1,166.4 ms** | **966.9 ms** | 966.9 ms |
| **per token** | | **25.36 ms** | **21.02 ms** | 21.02 ms |

**F1 is worth 3.38 ms/token here and F2 1.00**, against 3.15 and 1.05 on `TARGET_CALL` —
both fixture-independent, both matching hardware (§8.1). What is *not*
fixture-independent is the total, because 87% of it is two tokens.

**The same call with no escape costs 2.97 ms/token** (§3.2). The cost of a tool call is
therefore not a per-token property at all:

> **host ms ≈ 2.9 ms × tokens + 419 ms × split escapes.**

Both coefficients are measured — 128.1 ms over the 44 non-escape tokens, 838.8 over the
two — the second dominates anything a coding agent emits, and `escape` shows it is dead
linear. A twenty-line edit whose lines end in `)` is ~19 splits and ~200 tokens: **~8.0 s
of ~8.6 s is the second term**, which is where `BASELINE.md` §2.3's ~8.5 s comes from.

**Where each removal now stands.** F1 and F2 are landed and their value is confirmed on
both fixtures. F3 is rejected (§6.1), re-tested against `ESCAPED_CALL` at 9.5× the stake.
The escape term is collapsible in principle and not by us (§8.5) — and the last column
above shows why it collapses *nothing* on this particular fixture: its two escapes sit in
different properties, so they are two states, not one. It is a twenty-line edit, all
escapes inside one string, where a sound key would take ~8.0 s to ~0.42 s.

---

## 5. The changes

Ordered by measured value per unit of risk. **None is native code.** F1 and F2 have
landed; F3 was measured and rejected (§6); F4 is the only one still open.

### F1 — delete the redundant `int()` conversion · **measured 2.99 ms/token** · **landed**

```python
ids = [int(t) for t in allowed if 0 <= int(t) < vocab]  # 7.39 ms, converts int -> int
```

LMFE already returns `list[int]`. The bounds guard is load-bearing in general — the
docstring is right that `len(tokenizer)` and the logit width differ — but it is a **fixed
property of (tokenizer, model)**, knowable once at backend construction, not per token. On
this container the width is 248,320 against a 248,077 vocabulary, so no id can be out of
range and the filter is provably dead.

**As landed:** `build_logits_processor` takes an `id_bound` and skips the filter when
`logits.shape[-1] >= id_bound`. The guard did not disappear; it stopped re-deciding a fixed
property of (tokenizer, model) once per token. Omitting the bound keeps the old per-token
behaviour, so the fast path is opt-in.

**One correction to the design as written.** It said `safe_width = logit_width >=
len(tokenizer)`. Length alone is *not* the bound: `Constrainer.vocabulary()` enumerates
`range(len(inner))` but passes the stop ids in separately, so a stop id outside the
enumerated range is possible and the length would not cover it. The bound is
`constrain.logit_width_bound` — `max(len(inner), max(stop_ids) + 1)` — computed where both
are already known.

**Risk:** none identified. The failure mode if the bound were computed wrongly is a scatter
outside the row, which is the exact hazard the guard names — so it is derived from the
model's own logits, not from config.

### F2 — single-slot mask cache keyed on content · **measured 0.08–0.54 ms/token** · **landed**

Replace the impossible `id()` key with `!=` on the list. Sound, **0.12 ms**, hits 40% of
consecutive positions and ~85% within a string run.

**As landed:** one slot holding `(ids, width, dtype, mask)`. Single-slot is deliberate — it
bounds memory absolutely, the access pattern is a run rather than a working set, and the
processor is per-request so the slot dies with the turn. Width and dtype are part of the
key because a mask is only valid for the row it was built for.

**The key is `prev is not ids and prev == ids`, and the identity half is not redundant.**
If LMFE ever returned the same list mutated in place, a content key alone would compare the
new contents against themselves, report a hit, and reuse a mask built from the old ones — a
silently wrong constraint, which is the one failure §7.1 cannot tolerate. Today it costs
nothing (0 of 42 shared an identity) and it fails toward a rebuild.
`tests/test_constrain_mlx.py::test_the_cache_does_not_hit_on_a_list_mutated_in_place` pins
it.

**This supersedes the rejected-cache comment rather than contradicting it.** That comment
records a real experiment whose null result is now explained: the key could not hit. The
comment was rewritten to say so, not deleted (`CLAUDE.md`).

### F3 — reuse parser state across requests · ~78 ms/request · **rejected, §6**

Proposed as "share the schema-derived `JsonSchemaParser`, not the token-keyed state". The
part it named is not where the cost is: constructing that parser takes **0.035 ms**.
Measurement and the full reasoning are in §6.

### F4 — restructure the decode loop so host work overlaps the forward pass · **~1.11×, §8.3**

```python
while True:
    logits = model_call(y)
    mx.async_eval(logits)  # GPU starts the forward pass NOW
    tok = y.item()  # sync — cheap, y was dispatched last step
    mask = build_mask(filter(history + [tok]))  # host work, GPU is busy
    y = sample(logits + mask)
    mx.async_eval(y)
    yield tok
```

Measured as variant G: 13.01 ms/token against 13.12 unconstrained, with a full 6 ms of host
work in the loop. **Zero overhead.**

**Cost:** Orbit must own its decode loop rather than passing `logits_processors` to
`mlx_lm.stream_generate`. That is a real increase in surface area — prompt-cache
integration, `max_tokens`, stop-token handling and `mx.clear_cache()` cadence all currently
come free from `generate_step`. It is the largest change here and the last one to make.

**F1+F2 have landed and are now measured, and every earlier estimate of F4's value was
wrong — including the one that replaced them.** The first read "if F1+F2+F3 bring host work
to ~0.9 ms/token against a ~15 ms forward pass, F4 buys at most ~9%"; the second raised the
floor to **2.7 ms/token** when §6.1 rejected F3; the third took rung 1's **22.15 ms/token**
residual and read off **~44 tok/s, a ~1.5× win**. That third one divides by the wrong
thing: 22.15 is an average over a distribution that is ~2.9 ms on 96% of tokens and ~419 ms
on the rest, and overlap is a per-token `max()`. **Measured against the distribution, F4 is
~1.11× on a call with escapes and ~1.25× on one without** — §8.3. Worth having, not worth
owning the decode loop for on its own.

**It is still not the first thing to do.** F4 hides host work; §8.2's unexplained 5× is
that host work, and removing it needs no ownership of the decode loop. Own the loop only if
§9 item 3 closes and a serialised residual survives it.

---

## 6. What is not worth doing

| Option | Verdict |
|---|---|
| **Rust/C++ grammar engine bound into the sampling loop** | **Not justified.** It would attack the ~1 ms/token that survives F1–F3, and F4 hides that for free. The 2.4× is Python *waste* and loop *structure*, not a language boundary. Revisit only if F1–F4 land and a measured residual remains. |
| **XGrammar** | Hard-requires PyTorch, which would break an MLX-native runtime (sec 5.1). Already recorded in `pyproject.toml`; unchanged by these findings. |
| **Bitmask output from LMFE** | Requires torch — `constrain.py` already passes `False` for exactly this reason. |
| **Complement masking** (scatter 1,170 `-inf` into zeros instead of 246,908 zeros into `-inf`) | Measured 1.01 ms against 1.01 ms for a prebuilt-array scatter — **no win**, and computing the complement is an O(vocab) host-side set difference. Rejected on measurement. |
| **Caching on `id()`** | Impossible: 0/42. |
| **Caching on `len(allowed)` or a sampled fingerprint** | Sound-looking and wrong. Two different parser states can share a length, and the failure is a wrong mask — a silently wrong constraint. Rejected on principle, not on cost. |
| **Caching the 425 ms post-backslash set on "we are after a backslash"** | **Rejected on measurement, §8.5.** The set depends on the enclosing object, not the escape: in the last property that key permits a comma where the object must close. The 75% it would win is real, and the sound key is stack-wide and upstream's to write. |
| **F3 — reusing parser state across requests** | **Rejected on measurement.** The thing it proposed sharing costs 0.035 ms; the thing that costs 78 ms is not shareable. Below. |

### 6.1 Why F3 cannot work — measured 2026-08-09

> **Re-measured against `ESCAPED_CALL` after §8.4, and the verdict holds.** Everything
> below was originally measured on `TARGET_CALL`, which contains no escape sequence and
> never enters the 425 ms state. `reuse` now runs both fixtures:
>
> | Fixture | fresh enforcer per request | one enforcer shared | `allowed_token_cache` |
> |---|---|---|---|
> | `TARGET_CALL` | 108.8 / 100.1 ms | 98.2 → **101.1** ms | **0** |
> | `ESCAPED_CALL` | 933.3 / **932.5** ms | 930.6 → **970.6** ms | **0** |
>
> **F3 stays rejected, and now on the workload that matters.** The stake is 9.5× larger
> and sharing still recovers nothing — `prefix_states` doubles (46 → 92) and the cache
> that would help stays empty, exactly as the mechanism below predicts. Two further facts
> the escaped fixture makes visible: request 2 costs the same as request 1 to within
> 0.1%, so the 425 ms is re-paid deterministically per request; and **the two escapes
> inside a single request cost ~420 ms each**, so nothing collapses them within a turn
> either.

*`tools/constrained_decode_bench.py reuse`, tier-0 tokenizer, no weights.*

F3 read the profile as "three ~26 ms one-time state builds with a 0.00 ms repeat cost" and
concluded that a second request could skip them by sharing the schema-derived parser. Both
halves of that are wrong, and the second is wrong in a way the source settles:

| What F3 proposed sharing | Cost |
|---|---|
| `JsonSchemaParser(schema)` | **0.035 ms** |
| `TokenEnforcer(vocabulary, parser)` | **0.001 ms** |

The 78 ms is neither. It is three recursive `_collect_allowed_tokens` walks of the token
prefix tree at the string-open transitions, and **LMFE does not cache those for a JSON
schema**: `_compute_allowed_tokens` stores its result in `allowed_token_cache` only when
`parser.cache_key()` is not None, and `JsonSchemaParser` inherits the base implementation,
which returns None. The cache is observably empty — **0 entries after a full 43-token
call**. The "0.00 ms repeat cost" came from `prefix_states`, which is keyed on the full
token tuple and is the per-request dict `constrain.py` exists to bound.

Sharing the enforcer anyway — the thing F3 itself forbids — was measured to confirm there
was nothing to trade against: **98.3 → 97.2 ms**, inside noise, while `prefix_states` grew
43 → 86. Unbounded growth for no saving.

**What would recover it, and why it was not done.** Only a cache keyed on parser *state*,
which is exactly what `cache_key()` is and what upstream declines to implement for JSON
schemas. Writing one here would put a key we invented in front of a constraint whose
failure mode is a wrong mask — the same category as `len(allowed)` above, rejected on
principle two rows up. **The floor without F4 is 2.7 ms/token, not 0.9.**

> **That floor is now measured, and it is a floor only for a call with no escape in it —
> 2026-08-10, §3.2.** `TARGET_CALL` lands at **2.97 ms/token**, so the derivation was
> sound; `ESCAPED_CALL` lands at **21.02**. Read "2.7" as the cost of the states this
> paragraph is about, never as the cost of a call.

---

## 7. Correctness requirements

Any implementation must hold these. Each failure mode is silent.

1. **The mask must correspond to the parser state for the tokens actually sampled.** A
   cache key that can collide is a wrong constraint, not a slow one.
2. **The empty-allowed-set guard stays.** Masking everything makes the row `-inf`, which
   the downstream log-softmax turns into NaN and samples as garbage.
3. **The out-of-range-id guard stays**, as a per-backend fact rather than a per-token loop.
4. **`tests/fake_mlx.py` must apply the processors it is given.** It already does, and
   `CLAUDE.md` lists this as load-bearing: a fake that accepts and ignores them makes the
   whole gate green for free.
5. **`MockBackend` must not become easier to satisfy than the real backend.** It has failed
   that twice.
6. **Re-run `orbit gate toolcall --runs 100` after any change here** — blocking, and
   `CLAUDE.md` names this layer explicitly. It needs a healthy host (`HANDOFF.md` T18).

---

## 8. Validation — the ladder, complete

| Rung | What | Status |
|---|---|---|
| 0 | Synthetic loop, component microbenchmarks, real tokenizer + LMFE | **Complete, and re-run on both fixtures 2026-08-10** — §3, §4. On `ESCAPED_CALL` it reads 21.02 ms/token against rung 1's 22.15, so rung 0 is now a predictor rather than a 5×-low proxy |
| 1 | One generation per variant, real weights | **Complete, 2026-08-09** — §8.1 |
| 2 | Eight runs | **Complete** — reproduces rung 1 to within ~1% |
| 3 | `orbit gate toolcall --runs 100` | **Complete — passes 1.00, 100/100** |

The two earlier rung-1 attempts were void because the host was degraded (`HANDOFF.md` T18).
It is not degraded now: `mlxbench.py` read **321/347 GB/s and 11.92 TFLOP/s** before the
run and **320/343 GB/s and 11.97** after it, so `PROCESSES.md` §3.1 is satisfied at both
ends and these numbers may be believed.

### 8.1 What F1 and F2 are actually worth

*8 runs × 64 tokens per variant, one loaded model, `--max-tokens 64`, 406-token prompt.
`tok/s` is `mlx_lm`'s own `generation_tps`, so it excludes prefill; `ms/token` is its
reciprocal. Rung 1 (`--runs 1`) gave 90.15 / 26.72 / 29.74 / 30.08 — the same answer.*

| Variant | tok/s | ms/token | vs A |
|---|---|---|---|
| A unconstrained | **90.06** | 11.10 | 1.00× |
| E null processor | 89.75 | 11.14 | 1.00× |
| F null + `tokens.tolist()` | 85.01 | 11.76 | 1.04× |
| B current, pre-F1 | 27.04 | 36.98 | 2.55× |
| C = B + F1 | 29.42 | 33.99 | 2.38× |
| **D = C + F2** *(what is in the tree)* | **29.49** | **33.91** | **2.38×** |

**F1 and F2 work, and they are small: 2.55× → 2.38×, recovering 11–15%** of the penalty
across three runs. F1 is nearly all of it (2.99 ms/token measured against 3.57 predicted by
§3.2); F2 adds 0.08–0.54 ms/token against ~0.49 predicted, at a 35% content-hit rate
against the 40% §3.4 projected. **The rung-0 deltas predict the rung-1 deltas.** That is
the part of §3 that survives intact.

### 8.2 The sync really is free — and that is the bad news

E and F are new here, and they exist because A-vs-D left 22.8 ms/token unaccounted for.
They are null processors: E returns `logits` untouched and never reads `tokens`; F adds
only the `tokens.tolist()` every real processor must do.

| Step | Cost | What it prices |
|---|---|---|
| A → E | **+0.04 ms/token** | having a `logits_processors` entry at all |
| E → F | **+0.62 ms/token** | the sync `tokens.tolist()` forces |
| F → D | **+22.15 ms/token** | `token_filter` and mask construction |

**§2.2 and §1's second row are confirmed on real weights**, which they never were before —
the synthetic loop was a ~4 GB stand-in and the claim rested on it. The sync costs 0.62
ms/token, the plumbing costs nothing, and **essentially the whole penalty is inside the
filter-and-mask path.**

**And that path costs ~5× what §3.2 says it should.** Post-F1 host work there is 4.0
ms/token by rung-0 measurement (7.57 mean, less the 3.57 the `int()` conversion took);
rung 1 charges it **22.15**. **§8.4 explains the whole of it**, and the explanation has
nothing to do with having a model resident.

Three candidates were tested first and all three are dead, which is worth recording
because each is the kind of thing one would otherwise assume:

| Candidate | Verdict |
|---|---|
| The mask's GPU ops contend with the forward pass instead of running on an idle GPU | **No.** A stopwatch inside the processor (variant G) accounts for **21.50 of the 24.37 ms**; only 2.87 ms (12%) falls outside host work. MLX is lazy, so that stopwatch deliberately excludes the GPU work it queues. |
| `get_allowed_tokens` is handed prompt + generated, a 406-token prefix rung 0 never paid | **No.** `mlx_lm` seeds `tokens` from the last prompt token only, so the sequence is 49 long, not 455. Prepending a lead token to the rung-0 walk costs **1.0×**. |
| A saturated GPU down-clocks the CPU, so the same Python is slower beside a busy device | **No.** The rung-0 walk beside an unrelated GEMM loop saturating the GPU costs **2.55 ms against 2.39** — 7%, not 5×. |

### 8.3 What this does to F4

F4 hides host work behind the forward pass, so **its ceiling is the larger of the two, not
their sum**: 22.15 ms/token of host work against an 11.10 ms/token forward pass gives
~22.8 ms/token, or **~44 tok/s against today's 29.5** — a ~1.5× win.

That is far more than the "at most ~9%" §5 estimated, and the estimate was wrong because
it assumed the residual was the 2.7 ms/token §6.1 derived. It is 22.15. **But F4 is no
longer the biggest lever**: closing the 5× in §8.2 would beat it and costs no ownership of
the decode loop. Do §8.2 first.

> **Corrected 2026-08-10 — the ~1.5× above takes `max()` of an *average*, and §4 now shows
> the host cost is not remotely uniform.** 22.15 ms/token is ~2.9 ms on 96% of tokens and
> ~419 ms on the other 4%. The `max(forward, host)` has to be taken per token, and **419 ms
> of host work does not hide behind an 11 ms forward pass** — 11 ms of it hides.
>
> | 46 tokens, 2 split escapes | today | with F4 |
> |---|---|---|
> | forward + sync, 11.76 ms/token | 541 ms | 541 ms |
> | 44 non-escape tokens × 2.91 ms | 128 ms | **0 — hidden** |
> | 2 escapes × 419 ms | 838 ms | **814 ms — 11.76 hides, the rest does not** |
> | **total** | **1,507 ms → 30.5 tok/s** | **1,355 ms → 33.9 tok/s** |
>
> **F4 is worth ~1.11× on a call with escapes, and ~1.25× on one without** (43 tokens at
> 2.97 ms of host: 67.9 → 85.0 tok/s) — not ~1.5× on either. The "today" column reproduces
> the measured 29.5 tok/s to within 3%, which is what makes the F4 column worth reading at
> all. **F4 therefore stays below the escape term on any list ordered by measured value,
> and the escape term is the one lever that is not ours to pull** (§8.5).

---

### 8.4 It is one parser state, it costs 425 ms, and the fixture never entered it

*`tools/constrained_decode_bench.py escape`. **No weights** — this reproduces the
real-weights figure to within 2% in about a minute.*

The 22.15 ms/token is not spread across the decode at all. Of the ~1,000 ms of LMFE time
in a 49-token generation, **849 ms is two calls**:

| `|allowed|` | calls | median |
|---|---|---|
| 246,908 *(string interior)* | 22 | **1.19 ms** |
| **5,798 / 5,795** | **2** | **425 ms** |
| 658 / 649 | 3 | 26 ms |
| 267–269 | 8 | 1.25 ms |

The 246,908-token states cost 1.19 ms, exactly what §3.2 measured. Nothing is 5× slower.
**One state is 350× slower than its neighbours, and the rung-0 fixture never reaches it.**

**The state is the gap between a backslash and its escape character.** After `\` inside a
JSON string, only ~5,798 of 248,077 tokens can legally follow, and LMFE walks the token
prefix tree to find them — with no cache to keep the answer, because
`JsonSchemaParser.cache_key()` is None (§6.1).

**Whether that state is ever occupied is decided by the tokenizer, not the JSON**, and
this is the part that makes it invisible to a naive benchmark:

```
"compact(req)\n"   ->  '"' 'compact' '(req' ')\' 'n' '"'      the escape SPLITS -> 425 ms
"line0\nline1"     ->  '"' 'line' '0' '\n' 'line' '1' '"'     one token         -> free
```

A backslash that merges with the text before it (`)\`) leaves `n` as its own token and the
parser rests in the expensive state. A `\n` that tokenizes whole never rests there. Same
JSON, same schema, same escape count:

| body | newlines | splits | 43-token walk |
|---|---|---|---|
| `f(x)\n…` | 0 / 1 / 2 / 4 | 0 / 1 / 2 / 4 | 94 → 521 → 950 → **1807 ms** |
| `word\n…` | 0 / 1 / 2 / 4 | **0** | 93 → 95 → 98 → **105 ms** |

**~428 ms per split escape, dead linear, and zero for the same newlines that tokenize
whole.** `TARGET_CALL` — the rung-0 fixture — contains no escape at all, which is the
entire reason §3.2 read 2.5 ms/token where hardware read 20.4. The call the model actually
emitted had two newlines and **both split**.

> **The lesson is about the fixture, not about LMFE.** Every rung-0 conclusion in §3 and
> §6.1 was measured on one hand-written tool call chosen for being realistic-looking. It
> was realistic in shape and unrepresentative in the one dimension that dominates the cost.
> A benchmark fixture for a coding agent must contain the thing coding agents emit —
> multi-line code in string arguments — and this one did not.

**What it means for the product.** Cost scales with split escapes, not with tokens: a
tool call carrying twenty lines of code whose lines end in `)` costs ~8.5 s of host time
on its own. That is a latency question the 2.38× decode ratio does not express, and it
lands on exactly the workload Orbit exists to serve.

**Not decided here.** The state is the same one every time, so LMFE's own
`allowed_token_cache` would collapse every occurrence after the first — it is disabled
only because the JSON parser declines to provide a key. §6.1 rejected inventing a key
locally, on the principle that a key we invent sits in front of a constraint whose failure
is silent. That principle is unchanged; what has changed is the prize, from 1.1 ms to
425 ms per occurrence. Re-opened as §9 item 6, undecided.

> **Decided 2026-08-09 — §8.5.** "The state is the same one every time" is the sentence
> that was never checked, and it is false: the set is fixed by the enclosing object, not
> by the backslash.

### 8.5 It is not one state, and the key that would collapse it is not ours to write

*`tools/constrained_decode_bench.py statekey`. **No weights**, ~7 s.*

§8.4 left the deciding number open: the two occurrences returned **5,798** and **5,795**
allowed tokens, so "the same state every time" was an assumption with a 3-token
counterexample sitting in its own table. Comparing the sets themselves settles it.

| Compared | Same set? |
|---|---|
| the two occurrences within one request | **No** — differ by 3 |
| escape in a middle property vs in the last property | **No** — the same 3 |
| same property, 3-character body vs 45-character body | **Yes** — identical |
| four escapes inside one property | **Yes** — one distinct set |

The three tokens are `\",`, `/",` and `"",`. Each supplies the escape character, closes
the string and emits a comma **in one token**, and that is legal only while a required
property is still unwritten. **So the allowed set is fixed by the enclosing object stack
and not by the backslash**: 42 characters of string content before it change nothing,
and one unwritten property changes it.

**A key on "we are after a backslash" is therefore unsound, and it fails the documented
way** — in the last property it permits a comma where the object must close, which is a
wrong constraint rather than a slow one (§7 item 1). §6.1 rejected `len(allowed)` on that
principle; the principle now has a measured counterexample behind it.

**A sound key is stack-wide, and the stack is not all of the state.**
`JsonSchemaParser.get_allowed_characters` filters on `num_consecutive_whitespaces`, which
lives on the parser rather than in the stack; `ObjectParsingState.add_character` reads
`self.root.context.active_parser.last_parsed_string`, reaching through a `_Context` shared
by every clone; and `get_allowed_characters` **reassigns** `context.active_parser`. All
three run inside `_collect_allowed_tokens`, so a key must cover all three, per frame, and
return None whenever a frame cannot answer — the conservative composition
`UnionParser.cache_key` already uses.

**The prize is real, which is what makes this a decision rather than a dismissal:**

| Shape | Occurrences | Distinct sets | What a sound key collapses |
|---|---|---|---|
| `ESCAPED_CALL` — one escape in each of two properties | 2 | 2 | **nothing** |
| four `f(x)\n` lines in one property | 4 | **1** | **1,228 of 1,638 ms — 75%** |

A twenty-line edit is ~19 splits inside one property, so it is ~8.5 s against ~0.43 s.
The cache fires per *state*, not per request, and a coding agent's calls are long strings
in one property — the shape where it fires most.

**Decision: not built locally, and proposed upstream instead.** Three reasons, in order of
weight: the key sits in front of a constraint whose failure is silent; a correct one must
model three pieces of someone else's state including one reassigned mid-walk, so it breaks
on any LMFE release that moves them; and there is nothing to pick up — **0.11.3 is the
current release (24 Aug 2025) and `main` still defines only `shortcut_key()`**, so
"prefer upstream" here means contributing `JsonSchemaParser.cache_key()`, with `statekey`
as the reproducer and the 3-token difference as its acceptance test. Until that lands the
cost stands, and it is a latency characteristic of the workload rather than a bug: **F4
does not help it either**, since hiding 11 ms/token behind the forward pass leaves a
410 ms stall at 410 ms.

## 9. Open

| # | Item | Blocks |
|---|---|---|
| ~~1~~ | ~~**Rung 1 has never produced a valid number**~~ | **Closed 2026-08-09** — §8.1, on a host passing §3.1 at both ends. F1+F2 are worth 11–15%, and rung 3 passes at 1.00 |
| ~~3~~ | ~~The filter-and-mask path costs 22.15 ms/token where rung 0 prices it at 4.0~~ | **Closed by measurement — §8.4.** One parser state at ~425 ms per occurrence, reproducible without weights. Three plausible candidates were tested and killed first (§8.2) |
| ~~6~~ | ~~**Can the post-backslash allowed set be cached?**~~ | **Decided 2026-08-09 — §8.5, not locally.** The 5,798-vs-5,795 caveat was the answer: the set is fixed by the enclosing object stack, so a key on the backslash alone permits a comma where the object must close. A sound key is stack-wide, must also model `num_consecutive_whitespaces` and a `last_parsed_string` reached through a context reassigned mid-walk, and does not exist upstream to pick up. Proposed as an upstream contribution; nothing local goes in front of the mask |
| ~~7~~ | ~~Re-measure §3, §4 and §6.1 against a fixture that contains split escapes~~ | **Closed 2026-08-10.** `filter` now runs both fixtures and reports per state, pre-F1 and landed; §3.2 and §4 are rebuilt on it and §6.1 was already re-run. **Rung 0 now predicts rung 1: 21.02 ms/token against hardware's 22.15**, where the same harness on `TARGET_CALL` reads 2.97. It also corrected §8.3 — F4's ceiling was a `max()` over an average and is ~1.11×, not ~1.5× |
| ~~8~~ | ~~Does the cheap fix exist upstream — `JsonSchemaParser.cache_key()`~~ | **No — closed 2026-08-09.** 0.11.3 is the current release (24 Aug 2025) and `main` still defines only `shortcut_key()`, so the route is to contribute it, not to upgrade. The number that was missing is measured: the set is state-determined but the state is the whole stack — string content before the escape changes nothing, an unwritten property changes 3 tokens (§8.5) |
| 2 | Does F4's restructured loop reproduce variant G against real weights? | Whether F4 is worth owning the decode loop, and the answer got smaller twice. Ceiling re-derived against the measured *distribution* rather than its average: **~1.11× on a call with escapes, ~1.25× without** (§8.3), not the ~44 tok/s this row used to carry. Items 3, 6 and 8 are now closed and none of them left F4 anything — so this is the last open lever here, and it is worth ~10% for the cost of owning `stream_generate`'s loop |
| ~~3a~~ | ~~Splitting `JsonSchemaParser` reuse from `TokenEnforcer`'s token-keyed state~~ | **Closed by measurement, negative** — §6.1. The split is real and buys 0.035 ms |
| 4 | Is the 35% content-hit rate representative? Measured on one call shape. | F2's measured saving, which is the smaller half of a small win |
| ~~5~~ | ~~`build_logits_processor`'s docstring is wrong in two places~~ | **Closed.** Rewritten to describe what landed, both times keeping the rejected experiment and its explanation |

---

## 10. Reproduction

```bash
# host health — REQUIRED FIRST. Expect ~247 GB/s; 23 GB/s means stop (PROCESSES.md §3.1).
.venv-optiq/bin/python tools/mlxbench.py

# rung 0 — no weights
.venv/bin/python tools/constrained_decode_bench.py loop --layers 48   # §3.1 mechanism
HF_HUB_OFFLINE=1 .venv/bin/python tools/constrained_decode_bench.py filter      # §3.2, §4
# `filter` runs both fixtures; --per-token dumps every step instead of the state summary.
HF_HUB_OFFLINE=1 .venv/bin/python tools/constrained_decode_bench.py components  # §3.3
HF_HUB_OFFLINE=1 .venv/bin/python tools/constrained_decode_bench.py identity    # §3.4
HF_HUB_OFFLINE=1 .venv/bin/python tools/constrained_decode_bench.py key         # §3.4
HF_HUB_OFFLINE=1 .venv/bin/python tools/constrained_decode_bench.py reuse       # §6.1
HF_HUB_OFFLINE=1 .venv/bin/python tools/constrained_decode_bench.py escape      # §8.4
HF_HUB_OFFLINE=1 .venv/bin/python tools/constrained_decode_bench.py statekey    # §8.5
# `escape` reproduces the real-weights per-token cost to within 2% and needs no model.
# Prefer it to a 20.6 GiB load for anything about where the time goes. `statekey`
# compares the allowed sets themselves and is the acceptance test any cache key owes.

# rung 1 — real weights, ~20.6 GiB. Headroom >= 27 GB first; PROCESSES.md §3.
HF_HUB_OFFLINE=1 .venv/bin/python tools/constrained_decode_realweights.py \
    --runs 1 --max-tokens 64 --out var/constrained-decode.json

# rung 2 — §8.1 and §8.2 as tabulated. Six variants, ~9 min, one load.
HF_HUB_OFFLINE=1 .venv/bin/python tools/constrained_decode_realweights.py \
    --runs 8 --max-tokens 64 --out var/constrained-decode.json

# rung 3 — the sec 10.2 gate, blocking. ~2 min, loads tier 0.
HF_HUB_OFFLINE=1 .venv/bin/orbit gate toolcall --runs 100
```

**`var/` is gitignored, so `var/constrained-decode.json` does not survive a clone** — §8's
tables are the copy of record, which is why they carry every variant rather than a summary.

`loop` autotunes its layer count when `--layers` is omitted; pass `--layers 48` to
reproduce §3.1 exactly, since the autotuned value depends on host health.
