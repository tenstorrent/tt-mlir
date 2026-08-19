# Score candidate DS-decline criteria against measured outcomes.
#
# Baseline is DS as currently emitted. Declining a shape sends it to mcast, so the
# gain from declining shape s is +delta(s) if DS was losing, and -|delta(s)| if DS
# was winning. "Oracle" declines exactly the losers.
import csv, os, glob, re
from collections import defaultdict
from pathlib import Path
D=Path(os.environ.get("FLEET", "fleet")); MD=Path(os.environ.get("GRAPHS", str(D.parent/"models")))
M="qwen_2_5_0_5b qwen_3_0_6b llama_3_2_1b falcon3_1b falcon3_3b qwen_2_5_3b falcon3_7b qwen_3_8b llama_3_1_8b".split()
B={"BFLOAT8_B":1.0625,"BFLOAT4_B":0.5625}; T={"BFLOAT8_B":1088,"BFLOAT4_B":576}
def bwir(m):
    f=glob.glob(f"{MD}/ds/{m}/**/ttnn_runtime_{m}*_g1_*.mlir",recursive=True)
    txt=Path(f[0]).read_text(); o={}
    for x in re.finditer(r'"ttnn\.(?:matmul|linear)".*?in0_block_w = (\d+).*?: \((.*?)\)\s*->',txt,re.S):
        t=re.findall(r"tensor<([\dx]+)x(?:!ttcore\.tile<[^>]*>|\w+), #\w+>",x.group(2))
        if len(t)>=2:
            b=[int(v) for v in t[1].split("x")]
            if len(b)>=2: o.setdefault((b[-2],b[-1]),int(x.group(1)))
    return o
S=[]
for m in M:
    g={}
    for v in ("ds","nods"):
        d=defaultdict(list)
        for r in csv.DictReader(open(D/f"{m}__{v}.matmuls.csv")): d[(int(r["K_w"]),int(r["N"]))].append(r)
        g[v]=d
    bw=bwir(m)
    for k,A in g["ds"].items():
        if A[0]["cfg"]!="DRAM-sharded" or k not in g["nods"] or k not in bw: continue
        K,N=k; dt=A[0]["in1_dtype"]; Bn=g["nods"][k]
        ta=sum(float(r["ns"]) for r in A)/1e3; tb=sum(float(r["ns"]) for r in Bn)/1e3
        S.append({"m":m,"sh":f"{K}x{N}","K":K,"N":N,"n":len(A),
                  "kpc":(K//32)//8,"w":bw[k],
                  "burstKB":bw[k]*((-(-N//256)*256)//8//32)*T[dt]/1024,
                  "mb":K*N*B[dt]/1e6,               # weight MB per instance
                  "per_op_mb":K*N*B[dt]/1e6,
                  "delta":ta-tb, "pen":(ta/len(A))/(tb/len(Bn))})
loss=sum(s["delta"] for s in S if s["delta"]>0)
win =sum(-s["delta"] for s in S if s["delta"]<0)
print(f"32 on-DS shapes. DS-as-is costs {sum(s['delta'] for s in S):+.1f} us")
print(f"  losses total {loss:.1f} us over {sum(1 for s in S if s['delta']>0)} shapes")
print(f"  wins   total {win:.1f} us over {sum(1 for s in S if s['delta']<0)} shapes")
print(f"  oracle (decline exactly the losers) improvement = {loss:.1f} us\n")
def score(name, pred):
    dec=[s for s in S if pred(s)]
    gain=sum(s["delta"] for s in dec)
    fp=sum(-s["delta"] for s in dec if s["delta"]<0)
    print(f"  {name:44s} declines {len(dec):2d}/32  net {gain:+9.1f} us  "
          f"({100*gain/loss:5.1f}% of oracle)  wrongly-declined {fp:6.1f} us")
print("candidate criteria:")
score("patch: in0_block_w*2 < kPerCore", lambda s: s["w"]*2 < s["kpc"])
score("patch variant: <= (decline at boundary)", lambda s: s["w"]*2 <= s["kpc"])
for x in (100,150,200,250):
    score(f"burst < {x} KB", lambda s,x=x: s["burstKB"]<x)
for x in (5,10,15,20,30):
    score(f"weight >= {x} MB", lambda s,x=x: s["mb"]>=x)
for x in (5,10,15):
    score(f"weight >= {x} MB  OR  in0_block_w*2 < kPerCore",
          lambda s,x=x: s["mb"]>=x or s["w"]*2<s["kpc"])
score("DS off entirely", lambda s: True)
score("ORACLE (decline iff measured loss)", lambda s: s["delta"]>0)
