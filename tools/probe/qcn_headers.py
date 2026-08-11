"""Qwen3-Coder-Next-4bit's memory shape off the safetensors headers, no download.

Same method as tools/probe/ds4_headers.py, retargeted: that script reads DeepSeek config
keys (n_routed_experts, num_hash_layers) this architecture does not have.
"""

import collections
import json
import struct
import subprocess
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-Coder-Next-4bit"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
EXPERT_SEGMENTS = (".switch_mlp.", ".mlp.experts.", ".ffn.experts.")


def get(path: str, a: int | None = None, b: int | None = None) -> bytes:
    cmd = ["curl", "-sSL", "--fail", "--max-time", "60"]
    if a is not None:
        cmd += ["-r", f"{a}-{b}"]
    out = subprocess.run([*cmd, BASE + path], capture_output=True, check=False)
    if out.returncode != 0:
        sys.exit(f"curl failed on {path}: {out.stderr.decode()[:300]}")
    return out.stdout


config = json.loads(get("config.json"))
quant = config.get("quantization", {})
topk = config["num_experts_per_tok"]
print(f"{REPO}\n")
print(
    f"  layers {config['num_hidden_layers']}  experts {config['num_experts']}  "
    f"top-{topk}  moe_intermediate {config['moe_intermediate_size']}  "
    f"hidden {config['hidden_size']}"
)
print(f"  arch {config['architectures']}  type {config['model_type']}")
print(f"  full_attention_interval {config.get('full_attention_interval')}")
print(f"  quantization {quant}\n")

index = json.loads(get("model.safetensors.index.json"))
shards = sorted(set(index["weight_map"].values()))
print(f"  {len(shards)} shards, {len(index['weight_map'])} tensors in the map")

buckets: dict[str, int] = collections.defaultdict(int)
counts: dict[str, int] = collections.defaultdict(int)
blocks: dict[str, collections.Counter[int]] = collections.defaultdict(
    collections.Counter
)
DT = {"F32": 4, "F16": 2, "BF16": 2, "U32": 4, "I32": 4, "U16": 2, "U8": 1, "I8": 1}

for shard in shards:
    raw = get(shard, 0, 7)
    (n,) = struct.unpack("<Q", raw)
    header = json.loads(get(shard, 8, 8 + n - 1))
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
        expert = any(s in key for s in EXPERT_SEGMENTS)
        if expert:
            kind = "routed expert " + key.rsplit(".", 1)[1]
            blocks[key.rsplit(".", 1)[1]][nbytes] += 1
        else:
            kind = "resident (non-expert)"
        buckets[kind] += nbytes
        counts[kind] += 1

total = sum(buckets.values())
print(f"\n  {'component':<28} {'GB':>8} {'tensors':>9}")
for kind, nbytes in sorted(buckets.items(), key=lambda kv: -kv[1]):
    print(f"  {kind:<28} {nbytes / 1e9:8.2f} {counts[kind]:9d}")
print(f"  {'TOTAL':<28} {total / 1e9:8.2f} {sum(counts.values()):9d}")

streamed = sum(v for k, v in buckets.items() if k.startswith("routed expert"))
resident = total - streamed
print(f"\n  streamed {streamed / 1e9:.2f} GB   resident {resident / 1e9:.2f} GB")

print("\n  per-read block sizes (what the SSD actually sees):")
per_expert_layer = 0
for suffix, hist in sorted(blocks.items()):
    for nbytes, k in sorted(hist.items()):
        print(f"    .{suffix:<8} {nbytes / 1024:9.1f} KiB  x{k}")
        per_expert_layer += nbytes / config["num_experts"] * k

layers = config["num_hidden_layers"]
tok = per_expert_layer * topk
print(f"\n  per routed expert per layer: {per_expert_layer / 1e6:.3f} MB")
print(f"  per decoded token at top-{topk} over {layers} layers: {tok / 1e6:.0f} MB")
print(f"  full expert sweep per prefill chunk: {streamed / 1e9:.2f} GB")
