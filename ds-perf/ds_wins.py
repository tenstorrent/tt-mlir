# Exhaustive check: every shape group where DS achieved MORE GB/s than its no-DS
# counterpart of the same shape. Separates shapes actually on the DS path from
# controls (same 1D-mcast config in both compiles), where any delta is noise.
import csv, sys
from collections import defaultdict
from pathlib import Path
D = Path(sys.argv[1]); MODELS = sys.argv[2:]
BYTES = {"BFLOAT8_B":1.0625,"BFLOAT4_B":0.5625,"BFLOAT16":2.0,"FLOAT32":4.0}
def load(m,v):
    p = D/f"{m}__{v}.matmuls.csv"
    if not p.exists(): return None
    g=defaultdict(list)
    for r in csv.DictReader(open(p)): g[(int(r["K_w"]),int(r["N"]))].append(r)
    return g
rows=[]
for m in MODELS:
    ds,nd = load(m,"ds"), load(m,"nods")
    if not ds or not nd: continue
    for k in ds:
        if k not in nd: continue
        A,B=ds[k],nd[k]; K,N=k
        mb=K*N*BYTES.get(A[0].get("in1_dtype","BFLOAT8_B"),1.0625)/1e6
        aa=sum(float(r["ns"]) for r in A)/len(A); bb=sum(float(r["ns"]) for r in B)/len(B)
        ga,gb = mb*1e3/(aa/1e3), mb*1e3/(bb/1e3)
        rows.append((m,f"{K}x{N}",len(A),A[0]["cfg"]=="DRAM-sharded",ga,gb,ga/gb,
                     (sum(float(r["ns"]) for r in A)-sum(float(r["ns"]) for r in B))/1e3))
tot=len(rows); on=[r for r in rows if r[3]]
wins_on=[r for r in on if r[4]>r[5]]
wins_ctl=[r for r in rows if not r[3] and r[4]>r[5]]
print(f"shape groups compared: {tot}   on DS path: {len(on)}   controls: {tot-len(on)}\n")
print(f"### DS achieved higher GB/s, and the shape IS on the DS path: {len(wins_on)} of {len(on)}\n")
print("| model | K x N | n | DS GB/s | noDS GB/s | DS advantage | total delta us |")
print("|---|---|---|---|---|---|---|")
for r in sorted(wins_on,key=lambda r:-r[6]):
    print(f"| {r[0]} | {r[1]} | {r[2]} | {r[4]:.1f} | {r[5]:.1f} | +{100*(r[6]-1):.1f}% | {r[7]:+.1f} |")
print(f"\n### DS higher but shape NOT on DS path (same config both sides -> noise): {len(wins_ctl)}\n")
print("| model | K x N | DS GB/s | noDS GB/s | delta |")
print("|---|---|---|---|---|")
for r in sorted(wins_ctl,key=lambda r:-r[6]):
    print(f"| {r[0]} | {r[1]} | {r[4]:.1f} | {r[5]:.1f} | +{100*(r[6]-1):.1f}% |")
print(f"\nOn-DS shapes where DS lost: {len(on)-len(wins_on)}. "
      f"Net over all on-DS shapes: {sum(r[7] for r in on):+.1f} us. "
      f"Net over just the DS wins: {sum(r[7] for r in wins_on):+.1f} us.")
