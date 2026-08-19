# Test the burst hypothesis against the fleet: does DS bandwidth track
# burst = in0_block_w * per-bank shard width (tiles)?
import csv, glob, re, sys
from collections import defaultdict
from pathlib import Path
D=Path(sys.argv[1]); MODELS=sys.argv[2:]; MD=D.parent/"models"
BYTES={"BFLOAT8_B":1.0625,"BFLOAT4_B":0.5625,"BFLOAT16":2.0,"FLOAT32":4.0}
def bw_ir(model):
    f=glob.glob(f"{MD}/ds/{model}/**/ttnn_runtime_{model}*_g1_*.mlir",recursive=True)
    if not f: return {}
    txt=Path(f[0]).read_text(); out={}
    for m in re.finditer(r'"ttnn\.(?:matmul|linear)".*?in0_block_w = (\d+).*?: \((.*?)\)\s*->',txt,re.S):
        t=re.findall(r"tensor<([\dx]+)x(?:!ttcore\.tile<[^>]*>|\w+), #\w+>",m.group(2))
        if len(t)>=2:
            b=[int(x) for x in t[1].split("x")]
            if len(b)>=2: out.setdefault((b[-2],b[-1]),int(m.group(1)))
    return out
rows=[]
for m in MODELS:
    p=D/f"{m}__ds.matmuls.csv"
    if not p.exists(): continue
    g=defaultdict(list)
    for r in csv.DictReader(open(p)): g[(int(r["K_w"]),int(r["N"]))].append(r)
    bw=bw_ir(m)
    for (K,N),A in g.items():
        if A[0]["cfg"]!="DRAM-sharded": continue
        w=bw.get((K,N))
        if not w: continue
        padded=-(-N//256)*256                      # pad N to 32 * 8 banks
        shard_n=padded//8//32                      # tiles per bank
        mb=K*N*BYTES.get(A[0].get("in1_dtype","BFLOAT8_B"),1.0625)/1e6
        avg=sum(float(r["ns"]) for r in A)/len(A)
        rows.append((w*shard_n,w,shard_n,mb*1e3/(avg/1e3),m,f"{K}x{N}"))
rows.sort()
print(f"{'burst':>6s} {'w':>3s} {'shard_n':>7s} {'GB/s':>7s}  model / shape")
for b,w,s,g,m,sh in rows: print(f"{b:6d} {w:3d} {s:7d} {g:7.1f}  {m} {sh}")
print("\n=== grouped by burst band ===")
bands=[(0,16),(16,48),(48,80),(80,160),(160,208),(208,300),(300,1000)]
for lo,hi in bands:
    sel=[r for r in rows if lo<=r[0]<hi]
    if sel: print(f"  burst {lo:4d}-{hi-1:<4d} n={len(sel):2d}  GB/s {min(r[3] for r in sel):5.1f} - {max(r[3] for r in sel):5.1f}")
print("\n=== same burst, different (w, shard_n) factorisation ===")
by=defaultdict(list)
for r in rows: by[r[0]].append(r)
for b,v in sorted(by.items()):
    fac={(r[1],r[2]) for r in v}
    if len(fac)>1:
        print(f"  burst {b}: " + " | ".join(f"w={r[1]},sn={r[2]} -> {r[3]:.1f}" for r in v))
