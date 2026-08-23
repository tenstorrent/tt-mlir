# Emit the markdown tables for the n150 DS-matmul A/B report.
#
# Inputs: roles_g1_decode.csv / roles_g0_prefill.csv (from ttnn_role_diff.py) and the
# raw perf report JSONs. Everything printed here is measured from the CI artifacts of
# the two runs; nothing is modelled.
import csv
import glob
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from static_matmul_survey import survey  # noqa: E402

ORDER = ["down", "gate/up", "qkv", "o_proj", "lm_head", "other"]


def rep(variant, model):
    f = glob.glob(f"raw/{variant}/{model}/perf_report_*.json")
    if not f:
        return None
    r = json.load(open(f[0]))
    d = {x["measurement_name"]: x["value"] for x in r.get("measurements", [])}
    if not d.get("total_time"):
        return None
    bs = r.get("batch_size") or 1
    s = d["total_samples"]
    tokens = s / bs if s / bs >= 16 else s   # vllm harness reports tokens*batch
    if tokens <= 1:
        return None
    return {
        "ttft": d.get("ttft"), "layers": r.get("num_layers"), "bs": bs,
        "step_ms": 1000 * (d["total_time"] - (d.get("ttft") or 0) / 1000) / (tokens - 1),
    }


def census(variant, model):
    f = glob.glob(f"raw/{variant}/{model}/**/ttnn_runtime_*_g1_*.mlir", recursive=True)
    return Counter(r["kind"] for r in survey(f[0])) if f else None


def classify(models):
    """Split models into comparison classes. Only 'clean' isolates mcast -> DS."""
    out = defaultdict(list)
    for m in models:
        cb, ca = census("before", m), census("after", m)
        b, a = rep("before", m), rep("after", m)
        if not (cb and ca):
            out["no_ir"].append((m, b, a, cb, ca))
        elif not (b and a):
            out["ir_only"].append((m, b, a, cb, ca))
        elif cb.get("default", 0):
            out["nocfg_before"].append((m, b, a, cb, ca))
        elif ca.get("DS", 0):
            out["clean"].append((m, b, a, cb, ca))
        else:
            out["control"].append((m, b, a, cb, ca))
    return out


def pct(a, b):
    return 100 * (a - b) / b if b else float("nan")


def main():
    rows = list(csv.DictReader(open("roles_g1_decode.csv")))
    pre = list(csv.DictReader(open("roles_g0_prefill.csv")))
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    cls = classify(sorted(by))

    def mb(m, role=None, ds_only=False):
        return sum(float(r["weight_MB_total"]) for r in by[m]
                   if (role is None or r["role"] == role)
                   and (not ds_only or r["kind_after"] == "DS"))

    print("## Decode-step outcome, by comparison class\n")
    for name, title in [
        ("clean", "A. Clean A/B — every matmul had a 1D-multicast config before, DS after"),
        ("control", "B. Control — configs identical before/after, DS never chosen"),
        ("nocfg_before", "C. Before compile emitted no program config for some matmuls"),
    ]:
        if not cls[name]:
            continue
        print(f"### {title}\n")
        print("| model | L | ops | on DS | MB/step | MB on DS | step ms before | step ms after | step Δ | ttft Δ |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        ds, tf = [], []
        for m, b, a, cb, ca in cls[name]:
            d = pct(a["step_ms"], b["step_ms"])
            ds.append(d)
            t = pct(a["ttft"], b["ttft"]) if (b["ttft"] and a["ttft"]) else None
            if t is not None:
                tf.append(t)
            print(f"| {m} | {b['layers'] if b['layers'] and b['layers'] > 0 else '—'} "
                  f"| {sum(int(r['n']) for r in by[m])} | {ca.get('DS', 0)} "
                  f"| {mb(m):.0f} | {mb(m, ds_only=True):.0f} "
                  f"| {b['step_ms']:.2f} | {a['step_ms']:.2f} | {d:+.2f}% "
                  f"| {f'{t:+.2f}%' if t is not None else '—'} |")
        print(f"\nstep Δ: mean {st.mean(ds):+.2f}%, median {st.median(ds):+.2f}%, "
              f"range {min(ds):+.2f}%..{max(ds):+.2f}%"
              + (f" · ttft Δ: mean {st.mean(tf):+.2f}%, median {st.median(tf):+.2f}%\n" if tf else "\n"))

    if cls["ir_only"]:
        print("### D. Config diff available, no paired end-to-end number (one side's job failed)\n")
        print("| model | ops | on DS | MB/step | MB on DS | missing side |")
        print("|---|---|---|---|---|---|")
        for m, b, a, cb, ca in cls["ir_only"]:
            print(f"| {m} | {sum(int(r['n']) for r in by[m])} | {ca.get('DS', 0)} "
                  f"| {mb(m):.0f} | {mb(m, ds_only=True):.0f} | {'before' if not b else 'after'} |")
        print()

    ds_models = {r["model"] for r in rows if r["kind_after"] == "DS"}
    print("\n## What moved, per projection role (decode graph, models where DS was chosen anywhere)\n")
    print("| role | shape groups | → DS | ops | ops on DS | MB/step | MB on DS | % | in0_block_w before | in0_block_w after |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for role in ORDER:
        rs = [r for r in rows if r["role"] == role and r["model"] in ds_models]
        if not rs:
            continue
        on = [r for r in rs if r["kind_after"] == "DS"]
        wb = sorted({int(r["w_before"]) for r in on if r["w_before"]})
        wa = sorted({int(r["w_after"]) for r in on if r["w_after"]})
        tot = sum(float(r["weight_MB_total"]) for r in rs)
        tds = sum(float(r["weight_MB_total"]) for r in on)
        print(f"| {role} | {len(rs)} | {len(on)} | {sum(int(r['n']) for r in rs)} "
              f"| {sum(int(r['n']) for r in on)} | {tot:.0f} | {tds:.0f} | {100*tds/tot:.1f}% "
              f"| {','.join(map(str, wb)) or '—'} | {','.join(map(str, wa)) or '—'} |")

    print("\n\n## Every matmul shape, per role\n")
    for role in ORDER:
        rs = [r for r in rows if r["role"] == role and r["model"] in ds_models]
        if not rs:
            continue
        print(f"### {role}\n")
        print("| model | K × N | ops | bias | config before | config after | in0_block_w before | in0_block_w after | K-tiles/core | MB/step |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(rs, key=lambda r: r["model"]):
            print(f"| {r['model']} | {r['K']} × {r['N']} | {r['n']} "
                  f"| {'yes' if r.get('biased') == '1' else '—'} | {r['kind_before']} | {r['kind_after']} "
                  f"| {r['w_before'] or '—'} | {r['w_after'] or '—'} | {r['kpc_after'] or '—'} "
                  f"| {float(r['weight_MB_total']):.0f} |")
        print()

    dsg = [r for r in rows if r["kind_after"] == "DS" and r["w_after"] and r["kpc_after"]]
    eq = sum(1 for r in dsg if int(r["w_after"]) == int(r["kpc_after"]))
    print(f"\n## in0_block_w on n150\n")
    print(f"- `in0_block_w == K-tiles-per-core` in **{eq}/{len(dsg)}** DS shape groups "
          f"(max {max(int(r['w_after']) for r in dsg)}).")
    collapsed = [f"{r['model']} {r['role']}" for r in dsg if int(r["w_after"]) == 1]
    print(f"- collapsed to 1: **{len(collapsed)}** "
          f"({', '.join(collapsed)})")
    red = [r for r in dsg if int(r["w_after"]) < int(r["kpc_after"])]
    print(f"- reduced below K-tiles-per-core: **{len(red)}** groups, all in 7B-plus models:")
    for r in sorted(red, key=lambda r: -int(r["kpc_after"]) / int(r["w_after"])):
        print(f"  - {r['model']} {r['role']} {r['K']}×{r['N']}: "
              f"kPerCore {r['kpc_after']} → in0_block_w {r['w_after']} "
              f"({int(r['kpc_after'])//int(r['w_after'])}× fewer tiles per burst)")

    nb = [r for r in rows if r.get("biased") == "1"]
    print(f"\n## Bias\n")
    print(f"- {len(nb)} of {len(rows)} shape groups carry a bias (`ttnn.linear`); "
          f"**{sum(1 for r in nb if r['kind_after'] == 'DS')}** of them are on DS.")

    npre = len({r["model"] for r in pre if r["kind_after"] == "DS"})
    print(f"\n## Prefill\n\n- DS chosen in **{npre}/{len({r['model'] for r in pre})}** models' "
          f"prefill graph (g0). Prefill is a DS-free control, so its ttft delta measures "
          f"only the non-DS commits in the range.")


if __name__ == "__main__":
    main()
