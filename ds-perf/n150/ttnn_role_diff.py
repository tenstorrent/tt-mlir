# Static before/after diff of matmul program configs in the tt-xla CI TTNN dumps,
# grouped by projection role.
#
# The paired CI runs skipped device perf, so there are no per-op timings to compare.
# What the TTNN dumps do carry is the compile decision itself: for every matmul, which
# program config tt-mlir chose, its in0_block_w, and the in0 shard geometry. This
# reports that per projection role (down / gate+up / qkv / o_proj / lm_head), per model,
# so the model-level e2e delta can be attributed to the shapes that actually moved.
#
# Roles are inferred from shape and instance count exactly as in ../by_projection.py --
# the IR carries no names.
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from static_matmul_survey import largest_divisor_upto, survey  # noqa: E402

# bytes per element for the tile dtypes that appear in ttnn_layout memrefs
BYTES = {"bfp_bf8": 1.0625, "bfp_bf4": 0.5625, "bfp_bf2": 0.3125,
         "bf16": 2.0, "f32": 4.0, "u32": 4.0, "si32": 4.0, "u16": 2.0, "u8": 1.0}
ORDER = ["down", "gate/up", "qkv", "o_proj", "lm_head", "other"]


def roles(counts):
    """(K,N) -> role, from per-layer instance counts. Mirrors ../by_projection.py."""
    per_layer = [k for k, n in counts.items() if n > 1]
    if not per_layer:
        return {k: "other" for k in counts}
    L = min(counts[k] for k in per_layer)
    tall = [k for k in per_layer if counts[k] == L and k[0] > k[1]]
    down = max(tall, key=lambda k: k[0]) if tall else None
    wide = [k for k in per_layer if counts[k] == L and k[0] < k[1]]
    hidden = min(k[0] for k in wide) if wide else None
    out = {}
    for k in counts:
        K, N = k
        if counts[k] == 1:
            out[k] = "lm_head"
        elif counts[k] == 2 * L:
            out[k] = "gate/up"
        elif k == down:
            out[k] = "down"
        elif K >= N:
            out[k] = "o_proj"
        elif hidden and K == hidden and N > K:
            out[k] = "qkv"
        else:
            out[k] = "other"
    return out


def find_graph(model_dir, gidx):
    hits = sorted(Path(model_dir).rglob(f"ttnn_runtime_*_g{gidx}_*.mlir"))
    return hits[0] if hits else None


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["Kw"], r["N"])].append(r)
    return g


def summarize(rs):
    """One shape group in one variant."""
    kinds = Counter(r["kind"] for r in rs)
    ops = Counter(r["op"] for r in rs)
    ws = Counter(r["in0_block_w"] for r in rs)
    return {
        "n": len(rs),
        "kind": kinds.most_common(1)[0][0],
        "kind_mix": ",".join(f"{k}x{v}" for k, v in kinds.most_common()),
        "w": ws.most_common(1)[0][0],
        "kpc": rs[0]["k_per_core"],
        "pcn": rs[0]["per_core_n"],
        "in0_cores": rs[0]["in0_cores"],
        "w_space": rs[0]["w_space"],
        "w_memlayout": rs[0]["w_memlayout"],
        "dtype": rs[0]["w_dtype"],
        # ttnn.linear carries a bias; DS is never chosen for one (bias would have to be
        # DRAM width-sharded to be read per bank)
        "biased": int(ops.get("linear", 0) > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="raw")
    ap.add_argument("--base", default="before", help="variant dir used as the 'before' side")
    ap.add_argument("--new", default="after", help="variant dir used as the 'after' side")
    ap.add_argument("--graph", default="1", help="graph index (1 = decode, 0 = prefill)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root)

    keys = sorted({p.name for v in (args.base, args.new)
                   for p in (root / v).iterdir() if p.is_dir()})
    out = []
    skipped = []
    for k in keys:
        fb = find_graph(root / args.base / k, args.graph)
        fa = find_graph(root / args.new / k, args.graph)
        if not fb or not fa:
            skipped.append((k, args.base if not fb else args.new))
            continue
        rb, ra = survey(fb), survey(fa)
        if not rb or not ra:
            skipped.append((k, "no matmuls"))
            continue
        gb, ga = group(rb), group(ra)
        counts = {key: len(v) for key, v in ga.items()}
        rl = roles(counts)
        M_b = Counter(r["M"] for r in rb).most_common(1)[0][0]
        M_a = Counter(r["M"] for r in ra).most_common(1)[0][0]
        for key in sorted(set(gb) | set(ga)):
            K, N = key
            B = summarize(gb[key]) if key in gb else None
            A = summarize(ga[key]) if key in ga else None
            ref = A or B
            bpe = BYTES.get(ref["dtype"], 1.0625)
            n = ref["n"]
            row = {
                "model": k, "graph": f"g{args.graph}",
                "M_before": M_b, "M_after": M_a,
                "role": rl.get(key, "other"),
                "K": K, "N": N, "n": n,
                "weight_MB_each": K * N * bpe / 1e6,
                "weight_MB_total": n * K * N * bpe / 1e6,
                "dtype": ref["dtype"],
                "biased": ref["biased"],
                "kind_before": B["kind"] if B else "",
                "kind_after": A["kind"] if A else "",
                "kind_mix_before": B["kind_mix"] if B else "",
                "kind_mix_after": A["kind_mix"] if A else "",
                "w_before": B["w"] if B else "",
                "w_after": A["w"] if A else "",
                "kpc_after": A["kpc"] if A else "",
                "pcn_after": A["pcn"] if A else "",
                "in0_cores_after": A["in0_cores"] if A else "",
                "w_space_before": B["w_space"] if B else "",
                "w_space_after": A["w_space"] if A else "",
                "w_memlayout_before": B["w_memlayout"] if B else "",
                "w_memlayout_after": A["w_memlayout"] if A else "",
            }
            kpc = A["kpc"] if A else None
            row["best_w_le8"] = largest_divisor_upto(kpc, 8) if kpc else ""
            row["moved_to_ds"] = int(bool(A and A["kind"] == "DS"
                                          and not (B and B["kind"] == "DS")))
            row["w_collapsed"] = int(bool(A and A["kind"] == "DS" and A["w"] == 1
                                          and kpc and kpc > 1))
            out.append(row)

    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wr.writeheader()
        wr.writerows(out)
    print(f"wrote {args.out}: {len(out)} shape groups over "
          f"{len({r['model'] for r in out})} models")
    for k, why in skipped:
        print(f"  skipped {k}: {why}")


if __name__ == "__main__":
    main()
