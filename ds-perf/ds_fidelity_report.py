"""LoFi vs HiFi2 across the fleet's DS shapes: speed and PCC together."""
import csv, re
from collections import defaultdict
from pathlib import Path
D=Path("/home/bmalesevic/.claude/jobs/30b23368/tmp/fidsweep")
SPEC=512.0
SH=[("q3b_down_w43","qwen_2_5_3b","down",11008,2048,"bfp8",43),
    ("q3b_down","qwen_2_5_3b","down",11008,2048,"bfp8",1),
    ("q3b_gateup","qwen_2_5_3b","gate/up",2048,11008,"bfp8",4),
    ("q3b_oproj","qwen_2_5_3b","o_proj",2048,2048,"bfp8",8),
    ("l1b_down","llama_3_2_1b","down",8192,2048,"bfp8",32),
    ("l1b_gateup","llama_3_2_1b","gate/up",2048,8192,"bfp8",8),
    ("q8b_down","qwen_3_8b","down",12288,4096,"bfp8",16),
    ("q8b_gateup","qwen_3_8b","gate/up",4096,12288,"bfp8",4),
    ("f7b_down","falcon3_7b","down",23040,3072,"bfp8",18),
    ("f7b_gateup","falcon3_7b","gate/up",3072,23040,"bfp8",2),
    ("l8b_down","llama_3_1_8b","down",14336,4096,"bfp8",14),
    ("l8b_gateup","llama_3_1_8b","gate/up",4096,14336,"bfp4",8)]
BPE={"bfp8":1.0625,"bfp4":0.5625}
def dur(f):
    fh=open(f); freq=float(re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)",fh.readline()).group(1))
    rd=csv.reader(fh); cols=[c.strip() for c in next(rd)]; i={c:n for n,c in enumerate(cols)}
    sp=defaultdict(lambda:[None,None])
    for row in rd:
        if len(row)<=i["type"] or not row[i["zone name"]].strip().endswith("-KERNEL"): continue
        try: op=int(row[i["run host ID"]])
        except ValueError: continue
        t=int(row[i["time[cycles since reset]"]]); c=(row[i["core_x"]],row[i["core_y"]])
        s=sp[(op,c)]; s[0]=t if s[0] is None else min(s[0],t); s[1]=t if s[1] is None else max(s[1],t)
    per=defaultdict(float)
    for (op,c),(a,b) in sp.items(): per[op]=max(per[op],(b-a)/freq*1000)
    v=sorted(x for x in per.values() if x>1000)
    return (v[len(v)//2] if v else None)
def pcc(tag):
    p=D/f"{tag}.out"
    if not p.exists(): return None,None
    m=re.findall(r"PCC=([0-9.naN]+)\s+mean_rel_err=([0-9.naN]+)", p.read_text())
    return (float(m[-1][0]), float(m[-1][1])) if m else (None,None)
print(f"{'model':<13s} {'role':<8s} {'K x N':>12s} {'dt':>5s} {'w':>3s} | "
      f"{'LoFi us':>8s} {'GB/s':>6s} {'%512':>5s} {'PCC':>9s} | "
      f"{'HiFi2 us':>9s} {'GB/s':>6s} {'%512':>5s} {'PCC':>9s} | {'speedup':>8s} {'ΔPCC':>9s}")
print("-"*135)
rows=[]
for sid,m,role,K,N,dt,w in SH:
    lu=dur(D/f"{sid}__LoFi.device.csv") if (D/f"{sid}__LoFi.device.csv").exists() else None
    hu=dur(D/f"{sid}__HiFi2.device.csv") if (D/f"{sid}__HiFi2.device.csv").exists() else None
    lp,_=pcc(f"{sid}__LoFi"); hp,_=pcc(f"{sid}__HiFi2")
    by=K*N*BPE[dt]
    lg=by/lu if lu else None; hg=by/hu if hu else None
    f=lambda v,p=1,s="": "—" if v is None else f"{v:.{p}f}{s}"
    sp_=f"{hu/lu:.2f}x" if (lu and hu) else "—"
    dp=f"{lp-hp:+.6f}" if (lp is not None and hp is not None) else "—"
    print(f"{m:<13s} {role:<8s} {K:5d}x{N:<6d} {dt:>5s} {w:>3d} | "
          f"{f(lu/1e3 if lu else None):>8s} {f(lg,0):>6s} {f(100*lg/SPEC if lg else None,0,'%'):>5s} "
          f"{f(lp,6):>9s} | {f(hu/1e3 if hu else None):>9s} {f(hg,0):>6s} "
          f"{f(100*hg/SPEC if hg else None,0,'%'):>5s} {f(hp,6):>9s} | {sp_:>8s} {dp:>9s}")
    if lu and hu: rows.append((hu/lu, lg, lp, hp))
if rows:
    print(f"\nspeedup: min {min(r[0] for r in rows):.2f}x  max {max(r[0] for r in rows):.2f}x")
    print(f"LoFi GB/s: max {max(r[1] for r in rows):.0f} = {100*max(r[1] for r in rows)/SPEC:.0f}% of spec")
    print(f"worst LoFi PCC {min(r[2] for r in rows):.6f};  worst HiFi2 PCC {min(r[3] for r in rows):.6f}")
    print(f"largest PCC loss from dropping to LoFi: {min(r[2]-r[3] for r in rows):+.6f}")
