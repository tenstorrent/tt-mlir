"""Join sweep results to the plan and report GB/s against the 512 GB/s p150 spec."""
import csv, json
from pathlib import Path
T=Path("/home/bmalesevic/.claude/jobs/30b23368/tmp"); R=T/"sweep/results"
TILE={"bfp8":1088,"bfp4":576}; BPE={"bfp8":1.0625,"bfp4":0.5625}; SPEC=512.0
plan={p["name"]:p for p in json.load(open(T/"plan.json"))}
rows=[]
for n,p in plan.items():
    f=R/f"{n}.percore.csv"
    if not f.exists(): rows.append((p,None,None)); continue
    ns=[float(r["duration_ns"]) for r in csv.DictReader(open(f))
        if r["op_name"]=="MatmulDeviceOperation"]
    if not ns: rows.append((p,None,None)); continue
    ns=sorted(ns); med=ns[len(ns)//2]
    by=p["K"]*p["N"]*BPE[p["dt"]]
    rows.append((p,med,by/med))
print(f"{'config':<20s} {'model':<13s} {'role':<8s} {'C':>3s} {'w':>3s} {'burst KB':>9s} "
      f"{'us':>9s} {'GB/s':>7s} {'% of 512':>9s} {'vs control':>11s}")
print("-"*118)
best={}
for p,med,gbs in sorted(rows,key=lambda r:(r[0]["model"],r[0]["role"],-r[0]["burst"])):
    tag=f"{p['model']}/{p['role']}"
    if gbs and (tag not in best or gbs>best[tag][0]): best[tag]=(gbs,p["name"])
ctl={}
for p,med,gbs in rows:
    if p["ctl"] and gbs: ctl[f"{p['model']}/{p['role']}"]=gbs
for p,med,gbs in sorted(rows,key=lambda r:(r[0]["model"],r[0]["role"],-r[0]["burst"])):
    tag=f"{p['model']}/{p['role']}"
    if med is None:
        print(f"{p['name']:<20s} {p['model']:<13s} {p['role']:<8s} {p['C']:>3d} {p['w']:>3d} "
              f"{p['burst']/1024:>8.0f} {'REJECTED/FAIL':>9s}")
        continue
    rel=f"{gbs/ctl[tag]:.2f}x" if tag in ctl else "—"
    mark=" <<<" if best.get(tag,(0,))[1]==p["name"] else ""
    print(f"{p['name']:<20s} {p['model']:<13s} {p['role']:<8s} {p['C']:>3d} {p['w']:>3d} "
          f"{p['burst']/1024:>8.0f} {med/1e3:>9.1f} {gbs:>7.1f} {100*gbs/SPEC:>8.1f}% "
          f"{rel:>11s}{'  CONTROL' if p['ctl'] else ''}{mark}")
ok=[(p,g) for p,m,g in rows if g]
if ok:
    mx=max(ok,key=lambda x:x[1])
    print(f"\nBest DS anywhere in the sweep: {mx[1]:.1f} GB/s = {100*mx[1]/SPEC:.1f}% of spec "
          f"({mx[0]['name']}, burst {mx[0]['burst']/1024:.0f} KB)")
    print(f"Target to beat (multicast's achieved 390 GB/s): 76.2% of spec")
    print(f"{'REACHED' if 100*mx[1]/SPEC>76.2 else 'NOT REACHED'}")
