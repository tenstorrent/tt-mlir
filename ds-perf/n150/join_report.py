# Join the static per-matmul config diff (roles_g1_decode.csv) to the model-level
# e2e numbers from the CI perf reports, and report per projection role.
#
# The paired runs skipped "Run Device Perf", so there are no per-op device timings.
# Per-role *timings* therefore cannot be measured from these artifacts; what can be
# measured is (a) the config change per role/shape and the weight traffic it covers,
# and (b) the model-level decode-step delta those changes add up to.
import argparse
import csv
import glob
import json
from collections import defaultdict

ORDER = ["down", "gate/up", "qkv", "o_proj", "lm_head", "other"]


def e2e(variant, model):
    f = glob.glob(f"raw/{variant}/{model}/perf_report_*.json")
    if not f:
        return None
    r = json.load(open(f[0]))
    d = {x["measurement_name"]: x["value"] for x in r.get("measurements", [])}
    if "ttft" not in d or not d.get("total_time"):
        return None
    bs = r.get("batch_size") or 1
    s = d["total_samples"]
    # vllm harness reports tokens*batch (~3616); the direct harness reports tokens (~110)
    tokens = s / bs if s / bs >= 16 else s
    steps = tokens - 1
    return {
        "sps": s / d["total_time"], "ttft": d["ttft"], "steps": steps,
        "step_ms": 1000 * (d["total_time"] - d["ttft"] / 1000) / steps,
        "layers": r.get("num_layers"), "bs": bs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="roles_g1_decode.csv")
    ap.add_argument("--out-model", default="model_summary.csv")
    ap.add_argument("--out-role", default="role_summary.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.roles)))
    by = defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)

    models = []
    for m in sorted(by):
        rs = by[m]
        b, a = e2e("before", m), e2e("after", m)
        kinds_b = defaultdict(int)
        kinds_a = defaultdict(int)
        for r in rs:
            kinds_b[r["kind_before"]] += int(r["n"])
            kinds_a[r["kind_after"]] += int(r["n"])
        mb_tot = sum(float(r["weight_MB_total"]) for r in rs)
        mb_ds = sum(float(r["weight_MB_total"]) for r in rs if r["kind_after"] == "DS")
        models.append({
            "model": m, "n_mm": sum(int(r["n"]) for r in rs),
            "n_ds": kinds_a.get("DS", 0),
            "n_default_before": kinds_b.get("default", 0),
            "n_default_after": kinds_a.get("default", 0),
            "mb_total": mb_tot, "mb_ds": mb_ds, "pct_mb_ds": 100 * mb_ds / mb_tot,
            "step_ms_before": b["step_ms"] if b else None,
            "step_ms_after": a["step_ms"] if a else None,
            "sps_before": b["sps"] if b else None,
            "sps_after": a["sps"] if a else None,
            "layers": (b or a or {}).get("layers"),
        })

    print("## Model level (decode graph g1)\n")
    hdr = (f"{'model':26s} {'L':>3s} {'mm':>4s} {'onDS':>4s} {'MB/step':>8s} {'MB DS':>8s} {'%':>5s} "
           f"{'ms B':>7s} {'ms A':>7s} {'d ms':>7s} {'d%':>7s} {'GB/s B':>7s} {'GB/s A':>7s} {'note':>16s}")
    print(hdr)
    print("-" * len(hdr))
    for r in models:
        note = ""
        if r["n_default_before"]:
            note = "no cfg before"
        elif r["n_ds"] == 0:
            note = "no DS (control)"
        if r["step_ms_before"] and r["step_ms_after"]:
            d = r["step_ms_after"] - r["step_ms_before"]
            gb = r["mb_total"] / r["step_ms_before"]
            ga = r["mb_total"] / r["step_ms_after"]
            print(f"{r['model']:26s} {str(r['layers'] or ''):>3s} {r['n_mm']:4d} {r['n_ds']:4d} "
                  f"{r['mb_total']:8.0f} {r['mb_ds']:8.0f} {r['pct_mb_ds']:5.1f} "
                  f"{r['step_ms_before']:7.2f} {r['step_ms_after']:7.2f} {d:+7.2f} "
                  f"{100*d/r['step_ms_before']:+6.1f}% {gb:7.1f} {ga:7.1f} {note:>16s}")
        else:
            miss = "before missing" if not r["step_ms_before"] else "after missing"
            print(f"{r['model']:26s} {str(r['layers'] or ''):>3s} {r['n_mm']:4d} {r['n_ds']:4d} "
                  f"{r['mb_total']:8.0f} {r['mb_ds']:8.0f} {r['pct_mb_ds']:5.1f} "
                  f"{'--':>7s} {'--':>7s} {'--':>7s} {'--':>7s} {'--':>7s} {'--':>7s} {miss:>16s}")

    with open(args.out_model, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(models[0].keys()))
        w.writeheader()
        w.writerows(models)

    # per-role fleet aggregate over the models where DS was actually chosen somewhere
    ds_models = {r["model"] for r in rows if r["kind_after"] == "DS"}
    print("\n## Per-role fleet aggregate (models where DS was chosen anywhere)\n")
    hdr2 = (f"{'role':9s} {'shapes':>6s} {'->DS':>5s} {'ops':>5s} {'ops DS':>6s} "
            f"{'MB/step':>9s} {'MB on DS':>9s} {'%':>5s} {'w before':>18s} {'w after':>18s}")
    print(hdr2)
    print("-" * len(hdr2))
    role_rows = []
    for role in ORDER:
        rs = [r for r in rows if r["role"] == role and r["model"] in ds_models]
        if not rs:
            continue
        on = [r for r in rs if r["kind_after"] == "DS"]
        wb = sorted({r["w_before"] for r in on if r["w_before"]}, key=lambda x: int(x))
        wa = sorted({r["w_after"] for r in on if r["w_after"]}, key=lambda x: int(x))
        mb = sum(float(r["weight_MB_total"]) for r in rs)
        mbd = sum(float(r["weight_MB_total"]) for r in on)
        rec = {"role": role, "shapes": len(rs), "shapes_ds": len(on),
               "ops": sum(int(r["n"]) for r in rs), "ops_ds": sum(int(r["n"]) for r in on),
               "mb": mb, "mb_ds": mbd, "pct": 100 * mbd / mb if mb else 0,
               "w_before": ",".join(wb), "w_after": ",".join(wa)}
        role_rows.append(rec)
        print(f"{role:9s} {rec['shapes']:6d} {rec['shapes_ds']:5d} {rec['ops']:5d} {rec['ops_ds']:6d} "
              f"{mb:9.0f} {mbd:9.0f} {rec['pct']:5.1f} {rec['w_before']:>18s} {rec['w_after']:>18s}")

    with open(args.out_role, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(role_rows[0].keys()))
        w.writeheader()
        w.writerows(role_rows)


if __name__ == "__main__":
    main()
