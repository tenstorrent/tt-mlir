# Per-projection DS vs no-DS comparison, pivoted so each projection role can be
# read across the whole fleet.
#
# Roles are inferred from shape and instance count, not from names (the IR has none):
#   lm_head   n == 1 (N is the vocab)
#   gate/up   n == 2L, where L is the per-layer count -- gate and up are the SAME
#             K x N and get the same config, so they are indistinguishable here and
#             are reported as one group of 2 per layer
#   down      among n == L with K > N, the one with the largest K (K == intermediate)
#   o_proj    the other n == L with K > N, or K == N (K == heads*head_dim)
#   qkv       n == L with K == hidden, N > hidden
import csv, glob, re, sys
from collections import defaultdict
from pathlib import Path

D = Path(sys.argv[1]); MODELS = sys.argv[2:]
MODELS_DIR = D.parent / "models"
BYTES = {"BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625, "BFLOAT16": 2.0, "FLOAT32": 4.0}
ORDER = ["down", "gate/up", "qkv", "o_proj", "lm_head", "other"]

def load(model, v):
    p = D / f"{model}__{v}.matmuls.csv"
    if not p.exists(): return None
    g = defaultdict(list)
    for r in csv.DictReader(open(p)): g[(int(r["K_w"]), int(r["N"]))].append(r)
    return g

def blockw(model):
    f = glob.glob(f"{MODELS_DIR}/ds/{model}/**/ttnn_runtime_{model}*_g1_*.mlir", recursive=True)
    if not f: return {}
    txt = Path(f[0]).read_text(); out = {}
    for m in re.finditer(r'"ttnn\.(?:matmul|linear)".*?in0_block_w = (\d+).*?: \((.*?)\)\s*->', txt, re.S):
        t = re.findall(r"tensor<([\dx]+)x(?:!ttcore\.tile<[^>]*>|\w+), #\w+>", m.group(2))
        if len(t) >= 2:
            b = [int(x) for x in t[1].split("x")]
            if len(b) >= 2: out.setdefault((b[-2], b[-1]), int(m.group(1)))
    return out

def roles(groups):
    counts = {k: len(v) for k, v in groups.items()}
    per_layer = [k for k, n in counts.items() if n > 1]
    if not per_layer: return {k: "other" for k in groups}
    L = min(counts[k] for k in per_layer)
    out = {}
    tall = [k for k in per_layer if counts[k] == L and k[0] > k[1]]
    down = max(tall, key=lambda k: k[0]) if tall else None
    hidden = None
    wide = [k for k in per_layer if counts[k] == L and k[0] < k[1]]
    if wide: hidden = min(k[0] for k in wide)
    for k in groups:
        K, N = k
        if counts[k] == 1:                       out[k] = "lm_head"
        elif counts[k] == 2 * L:                 out[k] = "gate/up"
        elif k == down:                          out[k] = "down"
        elif K > N or K == N:                    out[k] = "o_proj"
        elif hidden and K == hidden and N > K:   out[k] = "qkv"
        else:                                    out[k] = "other"
    return out

rows = defaultdict(list)
for m in MODELS:
    ds, nd, bw = load(m, "ds"), load(m, "nods"), blockw(m)
    if not ds or not nd: continue
    rl = roles(ds)
    for k in ds:
        if k not in nd: continue
        A, B = ds[k], nd[k]; K, N = k
        mb = K*N*BYTES.get(A[0].get("in1_dtype","BFLOAT8_B"),1.0625)/1e6
        aa = sum(float(r["ns"]) for r in A)/len(A); bb = sum(float(r["ns"]) for r in B)/len(B)
        rows[rl[k]].append({
            "model": m, "shape": f"{K}x{N}", "n": len(A), "w": bw.get(k),
            "ds": aa/1e3, "nd": bb/1e3, "gds": mb*1e3/(aa/1e3), "gnd": mb*1e3/(bb/1e3),
            "pen": aa/bb, "on_ds": A[0]["cfg"] == "DRAM-sharded",
            "tot_ds": sum(float(r["ns"]) for r in A)/1e3,
            "tot_nd": sum(float(r["ns"]) for r in B)/1e3,
        })

for role in ORDER:
    if role not in rows: continue
    rs = sorted(rows[role], key=lambda r: r["ds"])
    print(f"\n### {role}\n")
    print("| model | K x N | n | on DS? | w | DS us | DS GB/s | noDS us | noDS GB/s | penalty | total delta us |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        print(f"| {r['model']} | {r['shape']} | {r['n']} | {'yes' if r['on_ds'] else 'no'} | "
              f"{r['w'] if r['w'] else '-'} | {r['ds']:.1f} | {r['gds']:.1f} | {r['nd']:.1f} | "
              f"{r['gnd']:.1f} | {r['pen']:.2f}x | {r['tot_ds']-r['tot_nd']:+.1f} |")
    on = [r for r in rs if r["on_ds"]]
    if on:
        pens = [r["pen"] for r in on]
        print(f"\nOn the DS path: {len(on)}/{len(rs)} shapes, penalty {min(pens):.2f}x-{max(pens):.2f}x, "
              f"DS {min(r['gds'] for r in on):.0f}-{max(r['gds'] for r in on):.0f} GB/s vs "
              f"noDS {min(r['gnd'] for r in on):.0f}-{max(r['gnd'] for r in on):.0f} GB/s, "
              f"total {sum(r['tot_ds']-r['tot_nd'] for r in on):+.1f} us")
    else:
        print(f"\nNever on the DS path in any model ({len(rs)} shapes), all ~1.00x — these are the controls.")
