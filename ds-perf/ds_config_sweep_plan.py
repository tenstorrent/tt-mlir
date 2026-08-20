"""Build the sweep plan: for each fleet on-DS shape, the compiler's own config as a
control plus every legal higher-burst point, ranked. Banks fixed at 8."""
import json
TILE={"bfp8":1088,"bfp4":576}
L1=1572864; RESERVED=180_000; BANKS=8
def divs(n): return [d for d in range(1,n+1) if n%d==0]
SHAPES=[
 ("q3b_down",  "qwen_2_5_3b","down",   11008, 2048,"bfp8", 1),
 ("q3b_gateup","qwen_2_5_3b","gate/up", 2048,11008,"bfp8", 4),
 ("f7b_down",  "falcon3_7b","down",    23040, 3072,"bfp8",18),
 ("f7b_gateup","falcon3_7b","gate/up",  3072,23040,"bfp8", 2),
 ("q8b_down",  "qwen_3_8b","down",     12288, 4096,"bfp8",16),
 ("q8b_gateup","qwen_3_8b","gate/up",   4096,12288,"bfp8", 4),
 ("l1b_gateup","llama_3_1_8b","gate/up", 4096,14336,"bfp4", 8),
]
plan=[]; rows=[]
for tag,m,role,K,N,dt,cw in SHAPES:
    kT,nT=K//32,-(-N//32); sn=-(-nT//BANKS); tb=TILE[dt]; budget=L1-RESERVED
    pcn=sn                                     # DS: per_core_n == per-bank shard width
    legal=[]
    for C in divs(kT):
        kpc=kT//C
        if kpc*2048>budget: continue
        for wv in divs(kpc):
            if wv*sn*tb*3 + wv*2048*2 > budget: continue
            legal.append((wv*sn*tb, C, wv))
    legal.sort(reverse=True)
    seen=set(); picks=[]
    for burst,C,wv in legal:                   # distinct w, biggest burst first
        if wv in seen: continue
        seen.add(wv); picks.append((burst,C,wv))
        if len(picks)==3: break
    # always include the compiler's own choice at C=8 as the control
    ctl=[(cw*sn*tb,8,cw)] if kT%8==0 and (kT//8)%cw==0 else []
    for burst,C,wv in ctl+picks:
        name=f"{tag}_c{C}_w{wv}"
        if any(p["name"]==name for p in plan): continue
        plan.append(dict(name=name,K=K,N=N,C=C,w=wv,pcn=pcn,dt=dt,
                         model=m,role=role,burst=burst,sn=sn,
                         ctl=(C==8 and wv==cw)))
        rows.append((name,m,role,f"{K}x{N}",dt,C,wv,sn,burst/1024,
                     "control" if (C==8 and wv==cw) else ""))
json.dump(plan,open("/home/bmalesevic/.claude/jobs/30b23368/tmp/plan.json","w"),indent=1)
print(f"{len(plan)} configs across {len(SHAPES)} fleet shapes\n")
print(f"{'name':<20s} {'model':<13s} {'role':<8s} {'K x N':>12s} {'dt':>4s} "
      f"{'C':>4s} {'w':>4s} {'shard_n':>7s} {'burst KB':>9s}")
for r in rows:
    print(f"{r[0]:<20s} {r[1]:<13s} {r[2]:<8s} {r[3]:>12s} {r[4]:>4s} "
          f"{r[5]:>4d} {r[6]:>4d} {r[7]:>7d} {r[8]:>8.0f}  {r[9]}")
