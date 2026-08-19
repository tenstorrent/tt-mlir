# Attribute the DS-vs-noDS delta per model: how much is the matmuls themselves,
# and how much is layout/reshard work introduced alongside them. Then classify
# every matmul shape by *why* it differs.
import csv, glob, re, sys
from collections import defaultdict
from pathlib import Path

D = Path(sys.argv[1]); MODELS = sys.argv[2:]
MODELS_DIR = D.parent / "models"
LAYOUT_OPS = {"ReshardDeviceOperation", "ShardedToInterleavedDeviceOperation",
              "InterleavedToShardedDeviceOperation", "ToMemoryConfigDeviceOperation",
              "ToLayoutDeviceOperation", "CopyDeviceOperation"}
BYTES = {"BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625, "BFLOAT16": 2.0, "FLOAT32": 4.0}

def percore(model, v):
    p = D / f"{model}__{v}.percore.csv"
    if not p.exists(): return None
    cls = defaultdict(float)
    for r in csv.DictReader(open(p)):
        if not r["replay"].strip(): continue        # traced region only
        n = r["op_name"]; ns = float(r["duration_ns"])
        k = "matmul" if n == "MatmulDeviceOperation" else ("layout" if n in LAYOUT_OPS else "other")
        cls[k] += ns
    return cls

def matmuls(model, v):
    p = D / f"{model}__{v}.matmuls.csv"
    if not p.exists(): return None
    g = defaultdict(list)
    for r in csv.DictReader(open(p)):
        g[(int(r["K_w"]), int(r["N"]))].append(r)
    return g

def blockw(model):
    """(K,N) -> in0_block_w from the DS graph's IR."""
    f = glob.glob(f"{MODELS_DIR}/ds/{model}/**/ttnn_runtime_{model}*_g1_*.mlir", recursive=True)
    if not f: return {}
    txt = Path(f[0]).read_text(); out = {}
    for m in re.finditer(r'"ttnn\.(?:matmul|linear)".*?in0_block_w = (\d+).*?: \((.*?)\)\s*->', txt, re.S):
        w = int(m.group(1))
        t = re.findall(r"tensor<([\dx]+)x(?:!ttcore\.tile<[^>]*>|\w+), #\w+>", m.group(2))
        if len(t) >= 2:
            b = [int(x) for x in t[1].split("x")]
            if len(b) >= 2: out.setdefault((b[-2], b[-1]), w)
    return out

print("## Where the delta comes from (traced decode step, us)\n")
print("| model | matmul DS | matmul noDS | d matmul | layout DS | layout noDS | d layout | other d | step d |")
print("|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    a, b = percore(m, "ds"), percore(m, "nods")
    if not a or not b: continue
    dm = (a["matmul"]-b["matmul"])/1e3; dl = (a["layout"]-b["layout"])/1e3
    do = (a["other"]-b["other"])/1e3
    print(f"| {m} | {a['matmul']/1e3:.1f} | {b['matmul']/1e3:.1f} | {dm:+.1f} | "
          f"{a['layout']/1e3:.1f} | {b['layout']/1e3:.1f} | {dl:+.1f} | {do:+.1f} | {dm+dl+do:+.1f} |")

print("\n## Per-matmul classification\n")
print("| model | K x N | n | w | DS us | DS GB/s | noDS us | noDS GB/s | penalty | class |")
print("|---|---|---|---|---|---|---|---|---|---|")
tally = defaultdict(int)
for m in MODELS:
    ds, nd, bw = matmuls(m, "ds"), matmuls(m, "nods"), blockw(m)
    if not ds or not nd: continue
    for key in sorted(ds, key=lambda k: -sum(float(r["ns"]) for r in ds[k])):
        if key not in nd: continue
        A, B = ds[key], nd[key]
        K, N = key
        mb = K*N*BYTES.get(A[0].get("in1_dtype","BFLOAT8_B"),1.0625)/1e6
        aa = sum(float(r["ns"]) for r in A)/len(A); bb = sum(float(r["ns"]) for r in B)/len(B)
        ga, gb = mb*1e3/(aa/1e3), mb*1e3/(bb/1e3)
        pen = aa/bb
        on_ds = A[0]["cfg"] == "DRAM-sharded"
        w = bw.get(key)
        if not on_ds:              cls = "control (not on DS)"
        elif pen < 0.98:           cls = "**DS faster**"
        elif pen <= 1.05:          cls = "neutral"
        elif ga < 200:             cls = "**(a) config pathology**"
        else:                      cls = "(b) DS ceiling"
        tally[cls] += 1
        print(f"| {m} | {K}x{N} | {len(A)} | {w if w else '-'} | {aa/1e3:.1f} | {ga:.1f} | "
              f"{bb/1e3:.1f} | {gb:.1f} | {pen:.2f}x | {cls} |")
print("\n### tally")
for k, v in sorted(tally.items(), key=lambda kv: -kv[1]): print(f"- {k}: {v}")
