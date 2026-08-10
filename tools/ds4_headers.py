"""DeepSeek-V4-Flash's memory shape, off the safetensors headers, without downloading it.

A safetensors file begins with a u64 header length and then a JSON header giving every
tensor's dtype, shape and byte range. Eighteen HTTP Range requests therefore answer what
a 92.83 GB download would: what is streamed, what stays resident, what is dropped, and
what the per-token read actually costs.

Classification follows `optiq/runtime/moe_stream.py` rather than guessing: a tensor is a
streamed routed expert iff its key ends `.weight` and contains one of
``_EXPERT_SEGMENTS``, and `DeepseekV4Model.sanitize` drops every ``mtp.*`` key before the
model sees it. Note that this quant names routed experts ``.ffn.switch_mlp.`` -- the
first segment, not the ``.ffn.experts.`` one whose comment names DeepSeek.

Requires network. Takes the repo id as its one argument, because the two quants this
project has costed are both recorded and neither is the obvious default: no argument
reproduces `docs/platform.md` §4.6 (the OptiQ build, the one an engine would serve),
and passing `mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed` reproduces §4.4.
"""

import collections
import json
import re
import struct
import subprocess
import sys
from typing import Any

REPO = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "mlx-community/DeepSeek-V4-Flash-0731-OptiQ-2bit"
)
BASE = f"https://huggingface.co/{REPO}/resolve/main/"

EXPERT_SEGMENTS = (".switch_mlp.", ".mlp.experts.", ".ffn.experts.")
TOPK = 6


def get(path: str, a: int | None = None, b: int | None = None) -> bytes:
    cmd = ["curl", "-sSL", "--fail", "--max-time", "60"]
    if a is not None:
        cmd += ["-r", f"{a}-{b}"]
    out = subprocess.run([*cmd, BASE + path], capture_output=True, check=False)
    if out.returncode != 0:
        sys.exit(f"curl failed on {path}: {out.stderr.decode()[:300]}")
    return out.stdout


config = json.loads(get("config.json"))
layers = config["num_hidden_layers"]
print(f"{REPO}\n")
print(
    f"  layers {layers}  routed experts {config['n_routed_experts']}  "
    f"top-{config['num_experts_per_tok']}  shared {config['n_shared_experts']}"
)
print(
    f"  hash-routed layers {config['num_hash_layers']}  "
    f"first_k_dense_replace {config.get('first_k_dense_replace', 'absent -> all MoE')}"
)
quant = config.get("quantization", {})
scalars = {k: v for k, v in quant.items() if not isinstance(v, dict)}
overrides = {k: v for k, v in quant.items() if isinstance(v, dict)}
print(f"  quantization {scalars}, {len(overrides)} per-path overrides")
# `load_streaming` reads the quantisation mode from the top level only and has no
# per-path path, so a routed expert whose override *differs* there would be streamed at
# the wrong bit width. An override that merely restates the top level is harmless, and
# the OptiQ quant has 129 of exactly that kind — so compare values, and never report
# the presence of an override as the finding.
main_routed = {
    k: v for k, v in overrides.items() if "switch_mlp" in k and not k.startswith("mtp.")
}
divergent = {
    k: v
    for k, v in main_routed.items()
    if any(v[field] != scalars.get(field) for field in v)
}
print(
    f"  main-tower routed expert overrides: {len(main_routed)}"
    f" -> {len(divergent) or 'none'} diverging from the top level"
)
for k, v in sorted(divergent.items()):
    print(f"    {k} {v}")

weight_map = json.loads(get("model.safetensors.index.json"))["weight_map"]
shards = sorted(set(weight_map.values()))
print(f"\nreading {len(shards)} shard headers ({len(weight_map)} tensors)", flush=True)

tensors = {}
for s in shards:
    n = struct.unpack("<Q", get(s, 0, 7))[0]
    hdr = json.loads(get(s, 8, 8 + n - 1))
    hdr.pop("__metadata__", None)
    tensors.update(hdr)


def nbytes(m: dict[str, Any]) -> int:
    a, b = m["data_offsets"]
    return int(b) - int(a)


buckets: collections.Counter[str] = collections.Counter()
for k, m in tensors.items():
    mtp = k.startswith("mtp.")
    if any(s in k for s in EXPERT_SEGMENTS):
        buckets[("mtp" if mtp else "routed") + "." + k.rsplit(".", 1)[-1]] += nbytes(m)
    else:
        buckets["mtp.other" if mtp else "resident"] += nbytes(m)

routed = sum(v for k, v in buckets.items() if k.startswith("routed."))
mtp = sum(v for k, v in buckets.items() if k.startswith("mtp"))
meta = buckets["routed.scales"] + buckets["routed.biases"]

print(
    f"\n  routed .weight            {buckets['routed.weight'] / 1e9:7.2f} GB  streamed"
)
print(f"  routed .scales + .biases  {meta / 1e9:7.2f} GB  streamed iff over the budget")
print(f"  MTP / DSpark              {mtp / 1e9:7.2f} GB  dropped by sanitize()")
print(f"  resident                  {buckets['resident'] / 1e9:7.2f} GB")
print(f"  {'total':24s}  {sum(buckets.values()) / 1e9:7.2f} GB")

print("\nper-expert read pattern (one layer), as _ShardWeightReader issues it:")
per_expert = 0.0
for k, m in sorted(tensors.items()):
    if re.match(rf"model\.layers\.{layers // 2}\.ffn\.switch_mlp\.", k):
        stride = nbytes(m) / m["shape"][0]
        per_expert += stride
        print(
            f"  {k.split('.', 3)[-1]:34s} {m['dtype']:5s} {m['shape']!s:20s} "
            f"stride {stride / 1024:8.0f} KiB"
        )
print(f"\n  {per_expert / 1e6:.3f} MB per routed expert per layer")
print(f"  {per_expert * TOPK * layers / 1e9:.3f} GB per decoded token at top-{TOPK}")
print(f"  {routed / 1e9:.2f} GB per prefill chunk, if the chunk reaches every expert")
