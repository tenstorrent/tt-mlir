"""Per-core matmul duration from each isolated run's device log, then GB/s vs the
512 GB/s p150 spec. Per-core because DEVICE KERNEL DURATION is corrupt on this card."""
import csv, re
from collections import defaultdict
from pathlib import Path
D=Path("/home/bmalesevic/.claude/jobs/30b23368/tmp/iso")
SPEC=512.0
# name -> (K, N, in0_cores, in0_block_w, shard_n, burst_tiles, tile_bytes, bytes/elem)
CFG={
 "q3b_down_c8_w43": (11008,2048, 8,43, 8,344,1088,1.0625),
 "q3b_down_c43_w8": (11008,2048,43, 8, 8, 64,1088,1.0625),
 "q3b_gateup_c8_w8":(2048,11008, 8, 8,43,344,1088,1.0625),
 "f7b_down_c24_w30":(23040,3072,24,30,12,360,1088,1.0625),
 "f7b_down_c8_w18": (23040,3072, 8,18,12,216,1088,1.0625),
 "f7b_gateup_c8_w2":(3072,23040, 8, 2,90,180,1088,1.0625),
 "q8b_down_c8_w16": (12288,4096, 8,16,16,256,1088,1.0625),
 "q8b_gateup_c8_w4":(4096,12288, 8, 4,48,192,1088,1.0625),
}
def matmul_ns(f):
    fh=open(f); freq=float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)",fh.readline()).group(1))
    rd=csv.reader(fh); cols=[c.strip() for c in next(rd)]
    i={c:n for n,c in enumerate(cols)}
    span=defaultdict(lambda:[None,None]); zones=defaultdict(set)
    for row in rd:
        if len(row)<=i["type"]: continue
        z=row[i["zone name"]].strip()
        if not z.endswith("-KERNEL"): continue
        try: op=int(row[i["run host ID"]])
        except ValueError: continue
        t=int(row[i["time[cycles since reset]"]]); c=(row[i["core_x"]],row[i["core_y"]])
        s=span[(op,c)]
        s[0]=t if s[0] is None else min(s[0],t); s[1]=t if s[1] is None else max(s[1],t)
        zones[op].add(c)
    per_op={}
    for (op,c),(a,b) in span.items():
        per_op[op]=max(per_op.get(op,0),(b-a)/freq*1000)   # ns, max over cores
    # the matmul is the op on the most cores with the longest duration; keep all for sanity
    return per_op, {o:len(v) for o,v in zones.items()}
print(f"{'config':<18s} {'burst KB':>9s} {'w':>3s} {'C':>4s} {'n runs':>7s} "
      f"{'median us':>10s} {'GB/s':>7s} {'% of 512':>9s}")
print("-"*80)
res={}
for n,(K,N,C,w,sn,bt,tb,bpe) in CFG.items():
    f=D/f"{n}.device.csv"
    if not f.exists(): print(f"{n:<18s} MISSING"); continue
    per_op,cores=matmul_ns(f)
    # keep ops running on >= 8 cores (the DS matmul grid); drop tiny layout ops
    cand=sorted([v for o,v in per_op.items() if cores[o]>=8 and v>1000])
    if not cand: print(f"{n:<18s} NO MATMUL OP FOUND (ops: {cores})"); continue
    med=cand[len(cand)//2]
    gbs=K*N*bpe/med
    res[n]=(bt*tb/1024,gbs,med/1e3,len(cand))
    print(f"{n:<18s} {bt*tb/1024:>8.0f} {w:>3d} {C:>4d} {len(cand):>7d} "
          f"{med/1e3:>10.1f} {gbs:>7.1f} {100*gbs/SPEC:>8.1f}%")
if res:
    b=max(res.items(),key=lambda kv:kv[1][1])
    print(f"\nBest: {b[0]} -> {b[1][1]:.1f} GB/s = {100*b[1][1]/SPEC:.1f}% of 512 "
          f"(burst {b[1][0]:.0f} KB)")
    print(f"Multicast's achieved 390 GB/s = 76.2%.  "
          f"{'EXCEEDED' if 100*b[1][1]/SPEC > 76.2 else 'NOT exceeded'}")
