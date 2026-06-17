# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Graph the softmax perf story from the device-profiling data and warm
wall-clock, and prove where the gaps come from.

Inputs:
  - device per-op DEVICE KERNEL DURATION from perf_artifacts/<fb>/perf/ops_perf_results.csv
    (produced by `devperf.py` / `ttrt perf`, and the ident/exp/softmax microbench)
  - warm host submit->wait wall-clock, measured live here (no profiler)

Output: softmax_perf.png (4 panels) + a text summary.

Panels:
  A  wall-clock = device-kernel + host-dispatch overhead   (proves the big gap is dispatch)
  B  device-kernel softmax-math vs #tiles                  (compute scaling, ttnn vs d2m)
  C  marginal ns per added tile                            (the ~20x smoking gun)
  D  d2m softmax-compute decomposed: copy-floor / exp / other-math vs ttnn
                                                           (separates structure vs exp-fidelity)
Run:  python3 plot_perf.py            (needs device for the wall-clock panel; FB_DIR over/ridable)
      python3 plot_perf.py --no-wall  (device CSVs only, skip wall-clock panel)
"""
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FB = os.environ.get("FB_DIR", "_fb")
ART = "perf_artifacts"
TILES = {"1x1": 1, "2x2": 4, "3x3": 9}
SIDE = {"1x1": 32, "2x2": 64, "3x3": 96}


def dev_ops(fb):
    p = os.path.join(ART, fb, "perf", "ops_perf_results.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    return [
        (r.get("OP CODE", ""), int(r.get("DEVICE KERNEL DURATION [ns]") or 0))
        for r in rows
    ]


def ttnn_stages(sz):
    ops = dev_ops(f"ttnn_softmax_{sz}.ttnn") or []
    d = {"tilize": 0, "math": 0, "untilize": 0}
    for code, ns in ops:
        if "Tilize" in code:
            d["tilize"] = ns
        elif "Untilize" in code:
            d["untilize"] = ns
        elif "Softmax" in code:
            d["math"] = ns
    return d


def d2m_compute(fb):
    """The compute generic = 3rd op in exec order (zeros, tilize, COMPUTE, untilize)."""
    ops = dev_ops(fb)
    if not ops:
        return None
    return ops[2][1] if len(ops) >= 4 else ops[0][1]


def d2m_unfused_stages(sz):
    ops = dev_ops(f"d2m_unfused_{sz}_prof.ttm") or []
    if len(ops) < 4:
        return None
    return {
        "init": ops[0][1],
        "tilize": ops[1][1],
        "math": ops[2][1],
        "untilize": ops[3][1],
    }


def measure_wall(fb, loops=300, warmup=30):
    import torch
    import ttrt.binary as binary
    import ttrt.runtime as rt

    DT = {"Float32": rt.DataType.Float32, "UInt32": rt.DataType.UInt32}
    TDT = {"Float32": torch.float32, "UInt32": torch.int32}
    fbb = binary.load_binary_from_path(os.path.join(FB, fb))
    rt.set_compatible_device_runtime(fbb)
    opts = rt.MeshDeviceOptions()
    opts.mesh_shape = list(fbb.get_program_mesh_shape(0))
    dev = rt.open_mesh_device(opts)
    try:
        specs = json.loads(fbb.get_program_inputs_as_json(0))
        ins = []
        for i, s in enumerate(specs):
            d = s["desc"]
            shape = [int(x) for x in d["shape"]]
            dtype = d["layout"]["memory_desc"]["data_type"]
            t = (
                torch.ones(shape, dtype=torch.int32)
                if dtype == "UInt32"
                else torch.randn(shape).to(TDT[dtype])
            ).contiguous()
            ht = rt.create_borrowed_host_tensor(
                t.data_ptr(),
                list(t.shape),
                list(t.stride()),
                t.element_size(),
                DT[dtype],
            )
            ins.append(rt.to_layout(ht, dev, rt.get_layout(fbb, 0, i), True))

        def one():
            o = rt.submit(dev, fbb, 0, ins)
            rt.wait(o)
            for x in o:
                rt.deallocate_tensor(x, force=True)

        for _ in range(warmup):
            one()
        t0 = time.perf_counter()
        for _ in range(loops):
            one()
        return (time.perf_counter() - t0) / loops * 1e6
    finally:
        rt.close_mesh_device(dev)


def main():
    no_wall = "--no-wall" in sys.argv
    szs = ["1x1", "2x2", "3x3"]

    # ---- gather device data ----
    ttnn = {sz: ttnn_stages(sz) for sz in szs}
    unf = {sz: d2m_unfused_stages(sz) for sz in szs}
    fused_total = {
        sz: sum(n for _, n in (dev_ops(f"d2m_fused_{sz}.ttm") or [])) for sz in szs
    }
    ident_c = {sz: d2m_compute(f"d2m_ident_{sz}_prof.ttm") for sz in szs}
    exp_c = {sz: d2m_compute(f"d2m_exp_{sz}_prof.ttm") for sz in szs}
    sm_c = {sz: d2m_compute(f"d2m_softmax_{sz}_prof.ttm") for sz in szs}

    # ---- wall clock (live, no profiler) ----
    wall = {}
    if not no_wall:
        for fam, fb in [
            ("ttnn", "ttnn_softmax_{}.ttnn"),
            ("d2m-fused", "d2m_fused_{}.ttm"),
            ("d2m-unfused", "d2m_unfused_{}_prof.ttm"),
        ]:
            for sz in szs:
                try:
                    wall[(fam, sz)] = measure_wall(fb.format(sz))
                except Exception as e:
                    print(f"wall {fam} {sz} FAIL: {str(e).splitlines()[0][:60]}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    x = np.array([TILES[s] for s in szs])

    # ---- Panel A: wall-clock = device + host dispatch ----
    a = ax[0, 0]
    if wall:
        fams = ["ttnn", "d2m-fused", "d2m-unfused"]
        dev_tot = {
            "ttnn": [sum(ttnn[s].values()) for s in szs],
            "d2m-fused": [fused_total[s] for s in szs],
            "d2m-unfused": [sum(unf[s].values()) for s in szs],
        }
        w = 0.25
        xs = np.arange(len(szs))
        for i, fam in enumerate(fams):
            walls = [wall.get((fam, s), 0) for s in szs]
            devus = [dev_tot[fam][j] / 1000.0 for j in range(len(szs))]
            host = [max(walls[j] - devus[j], 0) for j in range(len(szs))]
            a.bar(
                xs + (i - 1) * w, devus, w, color=f"C{i}", label=f"{fam} device-kernel"
            )
            a.bar(
                xs + (i - 1) * w,
                host,
                w,
                bottom=devus,
                color=f"C{i}",
                alpha=0.35,
                hatch="//",
                label=f"{fam} host-dispatch",
            )
        a.set_xticks(xs)
        a.set_xticklabels(szs)
        a.set_ylabel("warm submit->wait (µs)")
        a.set_title(
            "A. Wall-clock = device-kernel (solid) + host-dispatch (hatched)\n"
            "the big gaps are dispatch, not device"
        )
        a.legend(fontsize=7, ncol=3, loc="upper left")
    else:
        a.set_title("A. (wall-clock skipped: --no-wall)")

    # ---- Panel B: device softmax-math scaling ----
    b = ax[0, 1]
    b.plot(x, [ttnn[s]["math"] for s in szs], "o-", label="ttnn SoftmaxDeviceOperation")
    b.plot(x, [sm_c[s] for s in szs], "s-", label="d2m softmax compute generic")
    b.plot(x, [unf[s]["math"] for s in szs], "x--", label="d2m unfused (cross-check)")
    b.set_xlabel("tiles")
    b.set_ylabel("device kernel duration (ns)")
    b.set_title("B. Softmax MATH only (tilize/untilize excluded both sides)")
    b.set_xticks(x)
    b.legend(fontsize=8)
    b.grid(alpha=0.3)

    # ---- Panel C: marginal ns per added tile ----
    c = ax[1, 0]

    def slope(d):
        ys = np.array([d[s] for s in szs], float)
        return np.polyfit(x, ys, 1)[0]

    labels = ["d2m\nident-copy", "d2m\nexp (1 op)", "d2m\nsoftmax", "ttnn\nsoftmax"]
    vals = [
        slope(ident_c),
        slope(exp_c),
        slope(sm_c),
        slope({s: ttnn[s]["math"] for s in szs}),
    ]
    bars = c.bar(labels, vals, color=["#bbb", "C1", "C3", "C0"])
    for bar, v in zip(bars, vals):
        c.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    c.set_ylabel("ns per ADDED tile (slope, 1->9 tiles)")
    c.set_title(f"C. Marginal cost per tile: d2m softmax {vals[2]/vals[3]:.0f}x ttnn")

    # ---- Panel D: decompose d2m softmax compute: copy / exp / other-math ----
    d = ax[1, 1]
    xs = np.arange(len(szs))
    floor = np.array([ident_c[s] for s in szs], float)
    expc = np.array([exp_c[s] - ident_c[s] for s in szs], float)
    other = np.array([sm_c[s] - exp_c[s] for s in szs], float)
    d.bar(xs - 0.2, floor, 0.4, label="copy floor (loop overhead)", color="#bbb")
    d.bar(xs - 0.2, expc, 0.4, bottom=floor, label="exp (fidelity lever)", color="C1")
    d.bar(
        xs - 0.2,
        other,
        0.4,
        bottom=floor + expc,
        label="other math: reduce/sub/div/fill (structure lever)",
        color="C3",
    )
    d.bar(
        xs + 0.2,
        [ttnn[s]["math"] for s in szs],
        0.4,
        label="ttnn softmax (total math)",
        color="C0",
    )
    d.set_xticks(xs)
    d.set_xticklabels(szs)
    d.set_ylabel("device kernel duration (ns)")
    d.set_title(
        "D. d2m softmax compute decomposed vs ttnn\n"
        "structure (red) ALONE >> ttnn; exp (orange) is a second lever"
    )
    d.legend(fontsize=7)

    fig.tight_layout()
    out = os.path.join(HERE, "softmax_perf.png")
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")

    # ---- text summary ----
    print("\n=== marginal ns/added-tile ===")
    for lab, v in zip(["ident-copy", "exp", "softmax(d2m)", "softmax(ttnn)"], vals):
        print(f"  {lab:16} {v:8.0f}")
    print(f"  d2m/ttnn softmax marginal ratio: {vals[2]/vals[3]:.1f}x")
    print("\n=== d2m softmax compute decomposition (ns) ===")
    for s in szs:
        print(
            f"  {s}: copy-floor={ident_c[s]:6}  exp={exp_c[s]-ident_c[s]:6}  "
            f"other-math={sm_c[s]-exp_c[s]:6}  | ttnn-math={ttnn[s]['math']:6}"
        )
    if wall:
        print("\n=== wall-clock vs device (µs) ===")
        for fam in ["ttnn", "d2m-fused", "d2m-unfused"]:
            for s in szs:
                w = wall.get((fam, s))
                if w is None:
                    continue
                dv = (
                    sum(ttnn[s].values())
                    if fam == "ttnn"
                    else fused_total[s]
                    if fam == "d2m-fused"
                    else sum(unf[s].values())
                ) / 1000.0
                print(
                    f"  {fam:12} {s}: wall={w:8.1f}  device={dv:6.1f}  host={w-dv:8.1f}"
                )


if __name__ == "__main__":
    main()
