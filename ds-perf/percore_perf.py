# Per-op device kernel durations computed per core, from the raw profiler log.
#
# Why not use ops_perf_results.csv: its DEVICE KERNEL DURATION for any multi-core
# op is computed as (max ZONE_END across cores) - (min ZONE_START across cores).
# On this card some cores' cycle counters were zeroed at device init and others
# were not, so those two values come from different epochs and every multi-core op
# gets a constant ~1.4288e13-cycle (~2.9 h) offset. Single-core ops are unaffected,
# which is why exactly the CORE COUNT >= 8 rows are corrupt.
#
# Each core's own start/end are in one epoch, so per-core durations are sound.
# Op duration = max over cores of (core's last kernel ZONE_END - first ZONE_START),
# matching metal's definition while never mixing two cores' clocks.

import argparse
from collections import defaultdict
import csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-log", required=True)
    ap.add_argument("--ops-data", required=True, help="tracy_ops_data.csv, for op names")
    ap.add_argument("--freq-mhz", type=float, default=None)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import re
    names = {}
    with open(args.ops_data, errors="replace") as fh:
        for line in fh:
            m = re.search(r'TT_DNN_DEVICE_OP:\s*"([^"]+)",\s*\d+,\s*\d+,\s*\w+,\s*(\d+)', line)
            if m:
                names[int(m.group(2))] = m.group(1)

    freq = args.freq_mhz
    with open(args.device_log) as fh:
        header = fh.readline()
        if freq is None:
            m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+)", header)
            freq = float(m.group(1)) if m else 1000.0
        reader = csv.reader(fh)
        cols = [c.strip() for c in next(reader)]
        idx = {c: i for i, c in enumerate(cols)}
        I_X, I_Y = idx["core_x"], idx["core_y"]
        I_T = idx["time[cycles since reset]"]
        I_RUN = idx["run host ID"]
        I_TRACE_CTR = idx["trace id counter"]
        I_ZONE, I_TYPE = idx["zone name"], idx["type"]

        # (op, replay, core) -> [min start, max end]
        span = {}
        for row in reader:
            if len(row) <= I_TYPE:
                continue
            zone = row[I_ZONE].strip()
            if not zone.endswith("-KERNEL"):
                continue
            try:
                op = int(row[I_RUN])
            except ValueError:
                continue
            if op == 0:
                continue  # idle/framework rows carry no op id
            t = int(row[I_T])
            key = (op, row[I_TRACE_CTR].strip(), row[I_X], row[I_Y])
            kind = row[I_TYPE].strip()
            cur = span.get(key)
            if cur is None:
                span[key] = [t, t]
            elif kind == "ZONE_START":
                cur[0] = min(cur[0], t)
            else:
                cur[1] = max(cur[1], t)

    print(f"chip freq: {freq} MHz")
    print(f"(op, replay, core) spans: {len(span)}")

    # Per op: duration = max over cores; also track core count.
    per_op = defaultdict(lambda: {"cycles": 0, "cores": 0})
    for (op, replay, _x, _y), (s, e) in span.items():
        d = e - s
        if d < 0:
            continue
        a = per_op[(op, replay)]
        a["cycles"] = max(a["cycles"], d)
        a["cores"] += 1

    print(f"distinct (op, replay) pairs: {len(per_op)}")

    rows = []
    for (op, replay), a in per_op.items():
        rows.append({
            "op_id": op,
            "replay": replay,
            "op_name": names.get(op, "<unnamed>"),
            "cores": a["cores"],
            "duration_ns": a["cycles"] * 1000.0 / freq,
        })

    ns = [r["duration_ns"] for r in rows]
    ns.sort()
    print(f"\nduration sanity: min={ns[0]:.0f} ns  median={ns[len(ns)//2]:.0f} ns  max={ns[-1]:.0f} ns")
    over = sum(1 for v in ns if v > 1e6)
    print(f"durations > 1 ms: {over} / {len(ns)}" + ("  <-- still suspect" if over else "  <-- all plausible"))

    agg = defaultdict(lambda: {"n": 0, "ns": 0.0, "max": 0.0, "cores": 0})
    for r in rows:
        a = agg[r["op_name"]]
        a["n"] += 1
        a["ns"] += r["duration_ns"]
        a["max"] = max(a["max"], r["duration_ns"])
        a["cores"] = max(a["cores"], r["cores"])
    total = sum(a["ns"] for a in agg.values())

    print(f"\n=== top {args.top} ops by total device kernel time ===")
    print(f"{'OP':44s} {'n':>5s} {'cores':>6s} {'total us':>10s} {'avg ns':>9s} {'max ns':>9s} {'%':>6s}")
    for name, a in sorted(agg.items(), key=lambda kv: -kv[1]["ns"])[: args.top]:
        print(f"{name[:44]:44s} {a['n']:5d} {a['cores']:6d} {a['ns']/1e3:10.1f} "
              f"{a['ns']/a['n']:9.0f} {a['max']:9.0f} {100*a['ns']/total:5.1f}%")
    print(f"\ntotal device kernel time across all ops: {total/1e3:.1f} us ({total/1e6:.3f} ms)")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["op_id", "replay", "op_name", "cores", "duration_ns"])
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: -r["duration_ns"]))
        print(f"per-op CSV -> {args.out}")


if __name__ == "__main__":
    main()
