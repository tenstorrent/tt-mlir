#!/usr/bin/env python3
"""
detect_cb_overlap.py — scan a ttnn graph-capture report.json for L1 clashes:
a circular buffer whose L1 address range overlaps a live L1 buffer, i.e. the
same condition tt-metal reports at runtime as
  "Statically allocated circular buffers clash with L1 buffers ...
   L1 buffer allocated at <A> and static circular buffer region ends at <B>".

Liveness-aware: replays buffer_allocate / buffer_deallocate (type == "L1") in
capture order and, at each circular_buffer_allocate, checks the CB's
[address, address+size) against every L1 buffer live at that instant on the
same device.

Usage:  python detect_cb_overlap.py <report.json> [--max N]
Exit code 1 if any overlap is found, else 0.
"""
import argparse
import json
import sys


def ranges_overlap(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="graph-capture report.json")
    ap.add_argument("--max", type=int, default=20, help="max overlaps to print")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument(
        "--ignore-device",
        action="store_true",
        help="compare every CB against every live L1 buffer regardless of "
        "device_id (forces the mesh-artifact fallback on)",
    )
    grp.add_argument(
        "--strict-device",
        action="store_true",
        help="only compare CBs and L1 buffers that share a device_id "
        "(disables the mesh-artifact fallback)",
    )
    args = ap.parse_args()

    graph = json.load(open(args.report))["graph"]

    # Mesh/SPMD graph captures tag circular_buffer_allocate and (L1)
    # buffer_allocate events with DIFFERENT device_ids even though the same
    # program runs on every device at the same L1 addresses. When the two
    # device-id sets are disjoint, matching per-device would never find any
    # overlap, so fall back to device-agnostic comparison. A genuine
    # multi-device capture with aligned ids keeps per-device matching.
    cb_devs = {
        n["params"]["device_id"]
        for n in graph
        if n.get("node_type") == "circular_buffer_allocate"
    }
    l1_devs = {
        n["params"]["device_id"]
        for n in graph
        if n.get("node_type") == "buffer_allocate"
        and n["params"].get("type") == "L1"
    }
    disjoint = bool(cb_devs and l1_devs and cb_devs.isdisjoint(l1_devs))
    if args.strict_device:
        ignore_device = False
    elif args.ignore_device:
        ignore_device = True
    else:
        ignore_device = disjoint

    # (device_id, address) -> size, for currently-live L1 buffers.
    live_l1 = {}
    overlaps = []
    n_cb = 0
    n_l1_alloc = 0

    for node in graph:
        nt = node.get("node_type")
        p = node.get("params", {})
        if nt == "buffer_allocate" and p.get("type") == "L1":
            n_l1_alloc += 1
            live_l1[(p["device_id"], p["address"])] = p["size"]
        elif nt == "buffer_deallocate" and p.get("type") == "L1":
            live_l1.pop((p["device_id"], p["address"]), None)
        elif nt == "circular_buffer_allocate":
            n_cb += 1
            dev = p["device_id"]
            cb0 = p["address"]
            cb1 = cb0 + p.get("size", 0)
            for (bdev, baddr), bsize in live_l1.items():
                if not ignore_device and bdev != dev:
                    continue
                if ranges_overlap(cb0, cb1, baddr, baddr + bsize):
                    overlaps.append(
                        {
                            "counter": node.get("counter"),
                            "device": dev,
                            "l1_device": bdev,
                            "cb": [cb0, cb1, p.get("size")],
                            "cb_cores": p.get("core_range_set"),
                            "l1_buffer": [baddr, baddr + bsize, bsize],
                        }
                    )

    print(f"scanned: {n_l1_alloc} L1 buffer allocs, {n_cb} circular-buffer allocs")
    print(
        f"CB device_ids={sorted(cb_devs)} L1 device_ids={sorted(l1_devs)} -> "
        f"comparison={'device-agnostic' if ignore_device else 'per-device'}"
        + (
            "  (mesh-artifact fallback: CB/L1 device_ids are disjoint)"
            if ignore_device and disjoint and not args.ignore_device
            else ""
        )
    )

    # NO_DISPATCH captures record no real addresses (everything at 0), which
    # makes every CB "overlap" every L1 buffer. Detect and reject that case so
    # the overlap count isn't taken at face value.
    nonzero = [o for o in overlaps if not (o["cb"][0] == 0 and o["l1_buffer"][0] == 0)]
    if overlaps and not nonzero:
        print(
            "WARNING: all allocations are at address 0 — this looks like a "
            "NO_DISPATCH capture with no real L1 placement. Overlaps below are "
            "meaningless; re-capture in NORMAL mode for real addresses."
        )
    print(f"OVERLAPS FOUND: {len(overlaps)} ({len(nonzero)} with real addresses)")
    for o in overlaps[: args.max]:
        cb0, cb1, cbs = o["cb"]
        b0, b1, bs = o["l1_buffer"]
        l1dev = (
            f"(dev{o['l1_device']}) " if o["l1_device"] != o["device"] else ""
        )
        print(
            f"  [op#{o['counter']} dev{o['device']}] "
            f"CB [{cb0},{cb1}) size={cbs} cores={o['cb_cores']}  "
            f"CLASHES  L1 buffer {l1dev}[{b0},{b1}) size={bs}"
        )
    if len(overlaps) > args.max:
        print(f"  ... and {len(overlaps) - args.max} more")

    return 1 if overlaps else 0


if __name__ == "__main__":
    sys.exit(main())
