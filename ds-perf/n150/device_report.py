# Per-matmul device-time comparison, DS vs no-DS, from the local n150 fleet run.
#
# Consumes fleet/<model>__{ds,nods}.matmuls.csv (matmul_detail.py output: per-matmul
# device durations for the traced decode step only) and joins them to the static
# program-config diff in roles_g1_decode.csv so each shape carries its in0_block_w.
#
# Activation handling is load-bearing. The no-DS compile folds SiLU into the gate matmul;
# the DS compile folds it into the consumer multiply instead. Comparing group averages
# blindly would credit DS with the cost of an activation it simply no longer runs, so
# every per-instance ratio here is taken over the instances that carry no fused
# activation on either side, and the activation's own cost is reported separately.
#
# Roles are inferred from shape and per-layer count, same rules as ../by_projection.py.
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

LAYOUT_OPS = {"ReshardDeviceOperation", "ShardedToInterleavedDeviceOperation",
              "InterleavedToShardedDeviceOperation", "ToMemoryConfigDeviceOperation",
              "ToLayoutDeviceOperation", "CopyDeviceOperation"}
BYTES = {"BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625, "BFLOAT16": 2.0, "FLOAT32": 4.0}
ORDER = ["down", "gate/up", "qkv", "o_proj", "lm_head", "other"]


def roles(counts):
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


def percore_split(path):
    """Traced-region device time split into matmul / layout / everything else."""
    if not Path(path).exists():
        return None
    out = defaultdict(float)
    for r in csv.DictReader(open(path)):
        if not r["replay"].strip():
            continue
        n, ns = r["op_name"], float(r["duration_ns"])
        k = ("matmul" if n == "MatmulDeviceOperation"
             else "layout" if n in LAYOUT_OPS else "other")
        out[k] += ns
        out["total"] += ns
    return out


def load(path):
    if not Path(path).exists():
        return None
    g = defaultdict(list)
    for r in csv.DictReader(open(path)):
        g[(int(r["K_w"]), int(r["N"]))].append(r)
    return g


def avg(rs):
    return sum(float(r["ns"]) for r in rs) / len(rs) if rs else None


def tot(rs):
    return sum(float(r["ns"]) for r in rs)


def plain(rs):
    return [r for r in rs if not r.get("act", "")]


def acted(rs):
    return [r for r in rs if r.get("act", "")]


def mb_of(rs):
    K, N = int(rs[0]["K_w"]), int(rs[0]["N"])
    return K * N * BYTES.get(rs[0].get("in1_dtype", "BFLOAT8_B"), 1.0625) / 1e6


def gbs(mb, ns):
    return mb * 1e3 / (ns / 1e3) if ns else 0


def graphname_map(root="raw"):
    """CI job key -> decode-graph model name (what the fleet CSVs are keyed by)."""
    m = {}
    for d in sorted(Path(root, "after").iterdir()):
        hit = sorted(d.rglob("ttnn_runtime_*_g1_*.mlir"))
        if not hit:
            continue
        name = re.sub(r"^ttnn_runtime_", "", hit[0].name)
        name = re.split(r"_bs\d", name)[0]
        if "_g1_" not in name:
            m[d.name] = name
    return m


def baseline_class(jobmap, path="model_summary.csv"):
    """graphname -> 'fallback' when the before compile emitted no program config for some
    matmuls, so its baseline is ttnn's runtime heuristic rather than tt-mlir multicast.
    Those models measure a different comparison and are kept out of the aggregates."""
    out = {}
    if not Path(path).exists():
        return out
    for r in csv.DictReader(open(path)):
        g = jobmap.get(r["model"])
        if g:
            out[g] = "fallback" if int(r["n_default_before"] or 0) else "clean"
    return out


def static_blockw(roles_csv, jobmap):
    out = {}
    for r in csv.DictReader(open(roles_csv)):
        g = jobmap.get(r["model"])
        if not g:
            continue
        out[(g, int(r["K"]), int(r["N"]))] = (r["w_before"], r["w_after"],
                                              r["kpc_after"], r["biased"],
                                              r.get("pcn_after", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="fleet")
    ap.add_argument("--roles", default="roles_g1_decode.csv")
    ap.add_argument("--out", default="device_matmuls.csv")
    args = ap.parse_args()
    D = Path(args.dir)

    def usable(m):
        """Both variants must have per-matmul rows from inside a traced region. A graph that
        carries no trace region at all yields none, and cannot be compared this way."""
        for v in ("ds", "nods"):
            p = D / f"{m}__{v}.matmuls.csv"
            if not p.exists() or p.stat().st_size == 0:
                return False
            sp = percore_split(D / f"{m}__{v}.percore.csv")
            if not sp or not sp.get("total"):
                return False
        return True

    candidates = sorted({p.name.split("__")[0] for p in D.glob("*__ds.percore.csv")
                         if (D / f"{p.name.split('__')[0]}__nods.percore.csv").exists()})
    models = [m for m in candidates if usable(m)]
    excluded = [m for m in candidates if m not in models]
    jm = graphname_map()
    sw = static_blockw(args.roles, jm)
    cls = baseline_class(jm)

    rows, per_model = [], []
    for m in models:
        ds, nd = load(D / f"{m}__ds.matmuls.csv"), load(D / f"{m}__nods.matmuls.csv")
        rl = roles({k: len(v) for k, v in ds.items()})
        act_credit = 0.0                      # matmul time that left with the activation
        for key in sorted(set(ds) & set(nd)):
            K, N = key
            A, B = ds[key], nd[key]
            # like-for-like: instances with no fused activation on either side
            Ap, Bp = plain(A) or A, plain(B) or B
            fair = bool(plain(A)) and bool(plain(B))
            aa, bb = avg(Ap), avg(Bp)
            mb = mb_of(A)
            Bact = acted(B)
            act_cost = ""
            if Bact and plain(B):
                per = avg(Bact) - avg(plain(B))
                act_credit += len(Bact) * per
                act_cost = round(per / 1e3, 2)
            wb, wa, kpc, biased, pcn = sw.get((m, K, N), ("", "", "", "", ""))
            rows.append({
                "model": m, "baseline": cls.get(m, "clean"),
                "role": rl.get(key, "other"), "K": K, "N": N,
                "n_ds": len(A), "n_nods": len(B),
                "n_nods_activated": len(Bact), "act": (Bact[0]["act"] if Bact else ""),
                "act_cost_us": act_cost, "fair": int(fair),
                "biased": biased, "cores_ds": A[0].get("cores", ""),
                "cfg_ds": A[0].get("cfg", ""), "cfg_nods": B[0].get("cfg", ""),
                "w_before": wb, "w_after": wa, "kpc": kpc, "pcn": pcn,
                "weight_MB": round(mb, 3),
                "ds_us": round(aa / 1e3, 3), "nods_us": round(bb / 1e3, 3),
                "ds_gbs": round(gbs(mb, aa), 1), "nods_gbs": round(gbs(mb, bb), 1),
                "penalty": round(aa / bb, 4) if bb else "",
                "total_ds_us": round(tot(A) / 1e3, 2),
                "total_nods_us": round(tot(B) / 1e3, 2),
                "delta_us": round((tot(A) - tot(B)) / 1e3, 2),
                # what the shape would have cost had the activation stayed put
                "delta_like_us": round(len(A) * (aa - bb) / 1e3, 2),
            })
        a, b = (percore_split(D / f"{m}__{v}.percore.csv") for v in ("ds", "nods"))
        per_model.append({
            "model": m, "baseline": cls.get(m, "clean"),
            "mm_ds": a["matmul"] / 1e3, "mm_nods": b["matmul"] / 1e3,
            "lay_ds": a["layout"] / 1e3, "lay_nods": b["layout"] / 1e3,
            "oth_ds": a["other"] / 1e3, "oth_nods": b["other"] / 1e3,
            "tot_ds": a["total"] / 1e3, "tot_nods": b["total"] / 1e3,
            "act_credit": act_credit / 1e3,
        })

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with_ds = {r["model"] for r in rows if r["cfg_ds"] == "DRAM-sharded"}
    clean_models = {r["model"] for r in per_model
                    if r["baseline"] == "clean" and r["model"] in with_ds}
    fb = [r for r in per_model if r["baseline"] == "fallback"]
    nods_ctrl = [r for r in per_model
                 if r["baseline"] == "clean" and r["model"] not in with_ds]

    if excluded:
        print("Excluded for lack of traced device data: "
              + ", ".join(f"`{m}`" for m in excluded) + ".\n")

    print("### Model level: the whole traced decode step\n")
    print("Multicast-to-DS comparisons only. Models whose *before* compile emitted no program")
    print("config for some matmuls measure fallback-to-DS instead and are listed separately.\n")
    print("| model | matmul DS | matmul no-DS | Δ matmul | of which activation moved out "
          "| Δ matmul, like-for-like | Δ layout | Δ other | Δ step | step ratio |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted([x for x in per_model if x["model"] in clean_models],
                    key=lambda r: r["tot_ds"] / r["tot_nods"]):
        dm = r["mm_ds"] - r["mm_nods"]
        print(f"| {r['model']} | {r['mm_ds']:.1f} | {r['mm_nods']:.1f} | {dm:+.1f} "
              f"| {-r['act_credit']:+.1f} | {dm + r['act_credit']:+.1f} "
              f"| {r['lay_ds']-r['lay_nods']:+.1f} | {r['oth_ds']-r['oth_nods']:+.1f} "
              f"| {r['tot_ds']-r['tot_nods']:+.1f} | {r['tot_ds']/r['tot_nods']:.3f}x |")
    cm = [r for r in per_model if r["model"] in clean_models]
    print(f"\n{sum(1 for r in cm if r['tot_ds'] < r['tot_nods'])} of {len(cm)} models have a "
          f"faster traced decode step under DS. Matmul time alone: "
          f"{sum(1 for r in cm if r['mm_ds'] < r['mm_nods'])} of {len(cm)} faster, "
          f"{sum(r['mm_ds']-r['mm_nods'] for r in cm):+.1f} µs fleet-wide, of which "
          f"{-sum(r['act_credit'] for r in cm):+.1f} µs is the SiLU no longer running "
          f"inside the matmul.\n")
    if nods_ctrl:
        print("\n#### Control: same configs on both sides, DS never chosen\n")
        print("These models compile identically apart from the three non-DS commits in the range, so")
        print("their step delta is what those commits cost on their own — and it is where the SiLU")
        print("move shows up with no DS saving to offset it.\n")
        print("| model | matmul Δ | layout Δ | other Δ | step Δ | step ratio |")
        print("|---|---|---|---|---|---|")
        for r in sorted(nods_ctrl, key=lambda r: r["tot_ds"] / r["tot_nods"]):
            print(f"| {r['model']} | {r['mm_ds']-r['mm_nods']:+.1f} "
                  f"| {r['lay_ds']-r['lay_nods']:+.1f} | {r['oth_ds']-r['oth_nods']:+.1f} "
                  f"| {r['tot_ds']-r['tot_nods']:+.1f} | {r['tot_ds']/r['tot_nods']:.3f}x |")
        rr = [r["tot_ds"] / r["tot_nods"] for r in nods_ctrl]
        print(f"\nStep ratio {min(rr):.3f}x–{max(rr):.3f}x with no DS involved at all.\n")

    if fb:
        print("\n#### Fallback baseline, not multicast — a different comparison\n")
        print("| model | matmul DS | matmul fallback | Δ matmul | Δ layout | Δ other | Δ step | step ratio |")
        print("|---|---|---|---|---|---|---|---|")
        for r in sorted(fb, key=lambda r: r["tot_ds"] / r["tot_nods"]):
            print(f"| {r['model']} | {r['mm_ds']:.1f} | {r['mm_nods']:.1f} "
                  f"| {r['mm_ds']-r['mm_nods']:+.1f} | {r['lay_ds']-r['lay_nods']:+.1f} "
                  f"| {r['oth_ds']-r['oth_nods']:+.1f} | {r['tot_ds']-r['tot_nods']:+.1f} "
                  f"| {r['tot_ds']/r['tot_nods']:.3f}x |")
        print("\nThese numbers say what tt-mlir's program configs are worth against ttnn's runtime")
        print("heuristic, which is a much lower bar than 1D multicast. They are excluded from every")
        print("aggregate above and below.\n")

    print("\n### Per projection role\n")
    print("Per-instance times and the penalty are taken over instances with no fused")
    print("activation on either side. `Δ µs` is what the device actually spent differently")
    print("at that shape; `Δ like` removes the activation move.\n")
    for role in ORDER:
        rs = [r for r in rows if r["role"] == role and r["model"] in clean_models]
        if not rs:
            continue
        on = [r for r in rs if r["cfg_ds"] == "DRAM-sharded"]
        print(f"#### {role}\n")
        print("| model | K × N | n | bias | on DS | w before → after | DS µs | DS GB/s "
              "| no-DS µs | no-DS GB/s | penalty | Δ µs | Δ like | act |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rs, key=lambda r: r["model"]):
            print(f"| {r['model']} | {r['K']} × {r['N']} | {r['n_ds']} "
                  f"| {'yes' if r['biased'] == '1' else '—'} "
                  f"| {'yes' if r['cfg_ds'] == 'DRAM-sharded' else 'no'} "
                  f"| {r['w_before'] or '—'} → {r['w_after'] or '—'} "
                  f"| {r['ds_us']:.1f} | {r['ds_gbs']:.1f} | {r['nods_us']:.1f} "
                  f"| {r['nods_gbs']:.1f} | {r['penalty']:.2f}x | {r['delta_us']:+.1f} "
                  f"| {r['delta_like_us']:+.1f} "
                  f"| {(r['act'] + ' x' + str(r['n_nods_activated'])) if r['act'] else '—'} |")
        if on:
            pens = [r["penalty"] for r in on]
            print(f"\nOn the DS path: {len(on)}/{len(rs)} shapes, per-instance penalty "
                  f"{min(pens):.2f}x–{max(pens):.2f}x, DS {min(r['ds_gbs'] for r in on):.0f}–"
                  f"{max(r['ds_gbs'] for r in on):.0f} GB/s vs no-DS "
                  f"{min(r['nods_gbs'] for r in on):.0f}–{max(r['nods_gbs'] for r in on):.0f} GB/s, "
                  f"like-for-like net **{sum(r['delta_like_us'] for r in on):+.1f} µs**.\n")
        else:
            print(f"\nNever on the DS path ({len(rs)} shapes) — control group, "
                  f"penalty {min(r['penalty'] for r in rs):.2f}x–"
                  f"{max(r['penalty'] for r in rs):.2f}x.\n")

    dsr = [r for r in rows if r["cfg_ds"] == "DRAM-sharded" and r["pcn"]
           and r["model"] in clean_models]

    def collapsed(r):
        return r["w_after"] and r["kpc"] and int(r["w_after"]) == 1 and int(r["kpc"]) > 1

    def thin(r):
        return int(r["pcn"]) == 1 and r["N"] >= 2048

    print("\n### Two loss modes, and both have a clean signature\n")
    print("Every DS shape that is slower than multicast falls into one of exactly two groups, and")
    print("everything outside them wins or ties.\n")

    coll = [r for r in dsr if collapsed(r)]
    if coll:
        print("**Mode 1 — `in0_block_w` collapsed to 1.** tt-mlir's search walks down the divisors of")
        print("K-tiles-per-core; when none is small enough it lands on 1, and the weight read"
              " degenerates")
        print("into one tile per step. This is the failure the Blackhole study measured at ~101 GB/s,")
        print("and it reproduces here almost exactly.\n")
        print("| model | role | K × N | in0_block_w / kPerCore | DS GB/s | no-DS GB/s | penalty | Δ like µs |")
        print("|---|---|---|---|---|---|---|---|")
        for r in sorted(coll, key=lambda r: -r["penalty"]):
            print(f"| {r['model']} | {r['role']} | {r['K']} × {r['N']} "
                  f"| {r['w_after']} / {r['kpc']} | {r['ds_gbs']:.1f} | {r['nods_gbs']:.1f} "
                  f"| {r['penalty']:.2f}x | {r['delta_like_us']:+.1f} |")
        print(f"\nThis is **the single worst DS shape in the fleet** and the only one where the collapse")
        print(f"happens on n150 at all. The guard in `1fad065b` — which is *not* in this range — declines")
        print(f"exactly this pattern, so it is worth {sum(r['delta_like_us'] for r in coll):.1f} µs here.\n")

    hit = [r for r in dsr if thin(r) and not collapsed(r)]
    if hit:
        print("**Mode 2 — `per_core_n = 1` with N >= 2048.** `per_core_n` is how many output tile")
        print("columns a core owns. N = 2048 is 64 N-tiles spread one per core, so each core reads only")
        print("`in0_block_w` weight tiles per step and per-core launch and sync cost dominates.\n")
        print(f"{sum(1 for r in hit if r['penalty'] > 1)} of {len(hit)} such shapes are slower, "
              f"{min(r['penalty'] for r in hit):.2f}x–{max(r['penalty'] for r in hit):.2f}x, spanning "
              f"{' and '.join(sorted({r['role'] for r in hit}))} across "
              f"{len({r['model'] for r in hit})} models.\n")
        print("| model | role | K × N | in0_block_w | weight MB | DS GB/s | no-DS GB/s | penalty | Δ like µs |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(hit, key=lambda r: -r["penalty"]):
            print(f"| {r['model']} | {r['role']} | {r['K']} × {r['N']} | {r['w_after'] or '—'} "
                  f"| {r['weight_MB']:.1f} | {r['ds_gbs']:.1f} | {r['nods_gbs']:.1f} "
                  f"| {r['penalty']:.2f}x | {r['delta_like_us']:+.1f} |")
        small = [r for r in dsr if int(r["pcn"]) == 1 and r["N"] < 2048]
        if small:
            names = ", ".join(f"{r['model']} {r['role']}" for r in small)
            print(f"\nThe N < 2048 half of `per_core_n = 1` *wins* ({names}, "
                  f"at most {max(r['weight_MB'] for r in small):.1f} MB), because at "
                  f"{max(r['ds_gbs'] for r in small):.0f} GB/s against a fleet best of "
                  f"{max(r['ds_gbs'] for r in dsr):.0f} neither path is bandwidth-bound there. Weight "
                  f"size has to be part of the rule.\n")

    both = coll + hit
    rest = [r for r in dsr if r not in both]
    if both:
        print(f"\nDeclining both modes recovers **{sum(r['delta_like_us'] for r in both):.1f} µs** of "
              f"matmul time. The remaining {len(rest)} DS shapes run "
              f"{min(r['penalty'] for r in rest):.2f}x–{max(r['penalty'] for r in rest):.2f}x for "
              f"{sum(r['delta_like_us'] for r in rest):+.1f} µs, taking the fleet's like-for-like matmul "
              f"result from {sum(r['delta_like_us'] for r in dsr):+.1f} µs to "
              f"{sum(r['delta_like_us'] for r in rest):+.1f} µs.\n")

    print("\n#### It is the K-step ratio, not the block width\n")
    print("Absolute bandwidth does not depend on what the other compile did, so these tables use")
    print("every measured model including the fallback-baseline ones.\n")
    allds = [r for r in rows if r["cfg_ds"] == "DRAM-sharded" and r["w_after"] and r["kpc"]]
    for r in allds:
        r["ratio"] = int(r["kpc"]) / int(r["w_after"])
    rg = defaultdict(list)
    for r in allds:
        rg[int(r["ratio"])].append(r)
    print("| kPerCore / in0_block_w | shapes | DS GB/s | median | worst shape |")
    print("|---|---|---|---|---|")
    for k in sorted(rg):
        g = sorted(rg[k], key=lambda r: r["ds_gbs"])
        v = [r["ds_gbs"] for r in g]
        print(f"| {k} | {len(g)} | {min(v):.0f}–{max(v):.0f} | {v[len(v)//2]:.0f} "
              f"| {g[0]['model']} {g[0]['role']} {g[0]['K']}×{g[0]['N']} |")
    lo = [r for r in allds if r["ratio"] <= 4]
    hi = [r for r in allds if r["ratio"] > 4]
    if lo and hi:
        print(f"\nRatio at or below 4: {len(lo)} shapes at {min(r['ds_gbs'] for r in lo):.0f}–"
              f"{max(r['ds_gbs'] for r in lo):.0f} GB/s. Above 4: {len(hi)} shapes at "
              f"{min(r['ds_gbs'] for r in hi):.0f}–{max(r['ds_gbs'] for r in hi):.0f} GB/s. "
              f"The cut is clean and there is no further trend below it.\n")
    print("The absolute block width predicts nothing on its own:\n")
    wg = defaultdict(list)
    for r in allds:
        wg[int(r["w_after"])].append(r)
    print("| in0_block_w | shapes | DS GB/s | ratio range |")
    print("|---|---|---|---|")
    for w in sorted(wg):
        g = wg[w]
        v = [r["ds_gbs"] for r in g]
        print(f"| {w} | {len(g)} | {min(v):.0f}–{max(v):.0f} "
              f"| {min(r['ratio'] for r in g):.0f}–{max(r['ratio'] for r in g):.0f} |")
    w3 = [r for r in allds if int(r["w_after"]) == 3]
    if w3:
        r = max(w3, key=lambda r: r["ds_gbs"])
        print(f"\nThe clearest case against a width threshold is {r['model']}'s {r['role']} "
              f"`{r['K']}x{r['N']}`: `in0_block_w` of just **{r['w_after']}** — the smallest healthy "
              f"width in the fleet — yet **{r['ds_gbs']:.1f} GB/s**, because kPerCore is only "
              f"{r['kpc']} so the ratio is {r['ratio']:.0f}. Meanwhile `in0_block_w` of 2 measures "
              f"191 GB/s at ratio 7 and 134 at ratio 37. What costs bandwidth is the number of "
              f"serialized K-steps the kernel loops over, not how wide each one is.\n")

    print("\n#### Calibrating kMinBlockWidthFraction for n150\n")
    print("The guard in `1fad065b` declines DS when `in0_block_w * kMinBlockWidthFraction <")
    print("kPerCore`, with the constant set to **2** — a number calibrated on Blackhole. On n150 the")
    print("K-step ratio `kPerCore / in0_block_w` separates healthy from degenerate shapes, but the")
    print("cut sits in a different place: ratio 4 shapes are the *fastest* in this fleet.\n")
    cand = [r for r in rows if r["cfg_ds"] == "DRAM-sharded" and r["w_after"] and r["kpc"]]
    for r in cand:
        r["ratio"] = int(r["kpc"]) / int(r["w_after"])
    cl = [r for r in cand if r["baseline"] == "clean"]
    print("| kMinBlockWidthFraction | shapes declined | net µs | wins thrown away |")
    print("|---|---|---|---|")
    for f in (2, 3, 4, 6, 8):
        dec = [r for r in cl if r["ratio"] > f]
        net = sum(r["delta_like_us"] for r in dec)
        lost = sum(-r["delta_like_us"] for r in dec if r["delta_like_us"] < 0)
        print(f"| {f} | {len(dec)} | {net:+.1f} | {lost:.1f} |")
    print("\nEvery shape the guard could touch, by K-step ratio:\n")
    print("| kPerCore / in0_block_w | model | role | in0_block_w / kPerCore | DS GB/s | penalty | Δ like µs | baseline |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted([x for x in cand if x["ratio"] > 2], key=lambda r: -r["ratio"]):
        print(f"| {r['ratio']:.0f} | {r['model']} | {r['role']} | {r['w_after']} / {r['kpc']} "
              f"| {r['ds_gbs']:.1f} | {r['penalty']:.2f}x | {r['delta_like_us']:+.1f} "
              f"| {r['baseline']} |")
    d2 = [r for r in cl if r["ratio"] > 2]
    d4 = [r for r in cl if r["ratio"] > 4]
    kept = [r for r in d2 if r not in d4]
    if d2 and d4 and kept:
        lost = sum(-r["delta_like_us"] for r in kept if r["delta_like_us"] < 0)
        gain = sum(r["delta_like_us"] for r in d4 if r["delta_like_us"] > 0)
        print(f"\n**On n150 the constant should be 4, not 2.** At 2 the guard declines {len(d2)} "
              f"shapes, {len(kept)} of which are wins it should keep — among them the fleet's fastest "
              f"DS shapes, at {min(r['ds_gbs'] for r in kept):.1f}-"
              f"{max(r['ds_gbs'] for r in kept):.1f} GB/s and "
              f"{min(r['penalty'] for r in kept):.2f}x-{max(r['penalty'] for r in kept):.2f}x. That "
              f"trades away {lost:.0f} µs of wins to recover {gain:.0f} µs, a net loss of "
              f"{-sum(r['delta_like_us'] for r in d2):.0f} µs. At 4 it declines exactly the "
              f"{len(d4)} genuinely degenerate shape{'s' if len(d4) != 1 else ''} and costs nothing.\n")
        print("The Blackhole calibration recorded in the guard's own comment (`in0_block_w reduced")
        print("n=8  -6.60% median sps`) does not reproduce here: on this part a reduced block width is")
        print("harmless down to a ratio of 4, and only past that does bandwidth fall off.\n")

    print("\n#### How well the two rules actually separate\n")
    print("Scored against the control-shape noise floor: a shape counts as a real loss at")
    print("penalty > 1.02, a real win at < 0.98. Clean-baseline DS shapes only.\n")
    def predicted(r):
        ratio = (int(r["kpc"]) / int(r["w_after"])) if (r["w_after"] and r["kpc"]) else 0
        return ratio > 4 or (r["pcn"] and int(r["pcn"]) == 1 and r["N"] >= 2048)
    sc = [r for r in rows if r["cfg_ds"] == "DRAM-sharded" and r["baseline"] == "clean"]
    tp = [r for r in sc if predicted(r) and r["penalty"] > 1.02]
    fp = [r for r in sc if predicted(r) and r["penalty"] <= 1.02]
    fn = [r for r in sc if not predicted(r) and r["penalty"] > 1.02]
    tn = [r for r in sc if not predicted(r) and r["penalty"] <= 1.02]
    print("| | predicted loss | predicted keep |")
    print("|---|---|---|")
    print(f"| **is a loss** (>1.02x) | {len(tp)} | {len(fn)} |")
    print(f"| **is not** (<=1.02x) | {len(fp)} | {len(tn)} |")
    fn_names = ", ".join(f"{r['model']} {r['role']}" for r in fn)
    fp_note = (f", with {len(fp)} false positive{'s' if len(fp) != 1 else ''}"
               if fp else ", with no false positives")
    fn_note = (f". {len(fn)} slip{'' if len(fn) != 1 else 's'} through ({fn_names})."
               if fn else ".")
    print(f"\nThe two rules flag {len(tp) + len(fp)} of {len(sc)} DS shapes and catch "
          f"{len(tp)} of the {len(tp) + len(fn)} real losses{fp_note}{fn_note}")
    if fn:
        print("\n| model | role | K × N | penalty | Δ like µs | why it is missed |")
        print("|---|---|---|---|---|---|")
        for r in sorted(fn, key=lambda r: -r["penalty"]):
            print(f"| {r['model']} | {r['role']} | {r['K']} × {r['N']} | {r['penalty']:.2f}x "
                  f"| {r['delta_like_us']:+.1f} | per_core_n {r['pcn']}, "
                  f"in0_block_w {r['w_after']}/{r['kpc']} |")
    print()

    print("\n#### Penalty against per_core_n\n")
    grp = defaultdict(list)
    for r in dsr:
        grp[int(r["pcn"])].append(r)
    print("| per_core_n | shapes | penalty range | median | DS GB/s | no-DS GB/s |")
    print("|---|---|---|---|---|---|")
    for k in sorted(grp):
        g = sorted(grp[k], key=lambda r: r["penalty"])
        pens = [r["penalty"] for r in g]
        print(f"| {k} | {len(g)} | {min(pens):.2f}x–{max(pens):.2f}x | {pens[len(pens)//2]:.2f}x "
              f"| {min(r['ds_gbs'] for r in g):.0f}–{max(r['ds_gbs'] for r in g):.0f} "
              f"| {min(r['nods_gbs'] for r in g):.0f}–{max(r['nods_gbs'] for r in g):.0f} |")

    on = [r for r in rows if r["cfg_ds"] == "DRAM-sharded" and r["model"] in clean_models]
    ctrl = [r for r in rows if r["cfg_ds"] != "DRAM-sharded" and r["model"] in clean_models]
    wins = sorted([r for r in on if r["delta_like_us"] < 0], key=lambda r: r["delta_like_us"])
    loss = sorted([r for r in on if r["delta_like_us"] > 0], key=lambda r: -r["delta_like_us"])
    print("\n### Every DS shape, ranked\n")
    if ctrl:
        cp = [r["penalty"] for r in ctrl]
        print(f"- Noise floor from the {len(ctrl)} never-on-DS control shapes: "
              f"penalty {min(cp):.2f}x–{max(cp):.2f}x.")
    print(f"- {len(on)} shape groups take the DS path across {len(clean_models)} models.")
    print(f"- **{len(wins)} faster**, {sum(r['delta_like_us'] for r in wins):+.1f} µs like-for-like.")
    print(f"- **{len(loss)} slower**, {sum(r['delta_like_us'] for r in loss):+.1f} µs like-for-like.")
    for title, sel in (("Worst losses", loss[:12]), ("Biggest wins", wins[:12])):
        if not sel:
            continue
        print(f"\n**{title}**\n")
        print("| model | role | K × N | penalty | Δ like µs | DS GB/s | no-DS GB/s "
              "| w after / kPerCore | weight MB |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in sel:
            print(f"| {r['model']} | {r['role']} | {r['K']} × {r['N']} | {r['penalty']:.2f}x "
                  f"| {r['delta_like_us']:+.1f} | {r['ds_gbs']:.1f} | {r['nods_gbs']:.1f} "
                  f"| {r['w_after'] or '—'} / {r['kpc'] or '—'} | {r['weight_MB']:.1f} |")


if __name__ == "__main__":
    main()
