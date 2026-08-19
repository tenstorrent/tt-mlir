# Static survey of matmul program configs across model decode graphs.
#
# Reads a TTNN-dialect .mlir, resolves the #ttnn_layout aliases its matmul operands
# reference, and reports for each matmul: shapes, config kind, in0_block_w, the in0
# shard core count, and K-tiles-per-core.
#
# The point: on the DRAM-sharded path metal requires
#   (in0 per-core shard width in tiles) % in0_block_w == 0
# and tt-mlir's search starts at kPerCore and walks *down its divisors*
# (MatmulProgramConfig.cpp:370-390). So when kPerCore has no small divisors, the
# search is forced to in0_block_w=1, which measures ~101 GB/s instead of ~280.
# This finds every shape in that trap without running anything.

import argparse
import re
from collections import defaultdict
from pathlib import Path

LAYOUT = re.compile(
    r"^#(\w+) = #ttnn\.ttnn_layout<.*?<(\d+)x(\d+)>, memref<(\d+)x(\d+)x"
    r"(?:!ttcore\.tile<\d+x\d+, (\w+)>|(\w+))[^>]*, #(\w+)>(?:, <(\w+)>)?",
    re.M)
# operands and result of a matmul/linear, plus its program config
OP = re.compile(
    r'"ttnn\.(matmul|linear)"\((.*?)\)\s*<\{(.*?)\}>\s*(?:\{[^}]*\}\s*)?: '
    r'\((.*?)\)\s*->\s*(tensor<[^>]*>(?:, #\w+>)?)', re.S)
TENSOR = re.compile(r"tensor<([\dx]+)x(?:!ttcore\.tile<\d+x\d+, (\w+)>|(\w+)), #(\w+)>")


def parse_layouts(text):
    out = {}
    for m in LAYOUT.finditer(text):
        name, gy, gx, my, mx, tiledt, plaindt, space, memlayout = m.groups()
        out[name] = {
            "grid": (int(gy), int(gx)),
            "memref": (int(my), int(mx)),
            "dtype": tiledt or plaindt,
            "space": space,
            "memlayout": memlayout or "interleaved",
        }
    return out


def cfg_kind(attrs):
    if "dram_sharded_program_config" in attrs:
        return "DS"
    if "multi_cast_1d_program_config" in attrs:
        return "mcast1d"
    if "multi_cast_program_config" in attrs:
        return "mcast2d"
    if "matmul_program_config" in attrs:
        return "other"
    return "default"


def intval(attrs, key):
    m = re.search(rf"{key}\s*=\s*(\d+)", attrs)
    return int(m.group(1)) if m else None


def largest_divisor_upto(n, cap):
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def survey(path):
    text = Path(path).read_text()
    layouts = parse_layouts(text)
    rows = []
    for m in OP.finditer(text):
        opname, _operands, attrs, operand_types, _res = m.groups()
        ts = TENSOR.findall(operand_types)
        if len(ts) < 2:
            continue
        (a_shape, a_tdt, a_pdt, a_lay), (b_shape, b_tdt, b_pdt, b_lay) = ts[0], ts[1]
        a_dims = [int(x) for x in a_shape.split("x")]
        b_dims = [int(x) for x in b_shape.split("x")]
        if len(a_dims) < 2 or len(b_dims) < 2:
            continue
        M, K = a_dims[-2], a_dims[-1]
        Kw, N = b_dims[-2], b_dims[-1]
        la, lb = layouts.get(a_lay), layouts.get(b_lay)
        in0_cores = la["grid"][1] if la else None
        k_per_core = la["memref"][1] if la else None       # tiles along K per in0 core
        w_banks = lb["grid"][1] if lb else None
        w_shard_n = lb["memref"][1] if lb else None
        rows.append({
            "op": opname, "M": M, "K": K, "N": N, "Kw": Kw,
            "kind": cfg_kind(attrs),
            "in0_block_w": intval(attrs, "in0_block_w"),
            "per_core_n": intval(attrs, "per_core_n"),
            "in0_cores": in0_cores, "k_per_core": k_per_core,
            "w_banks": w_banks, "w_shard_n": w_shard_n,
            "w_dtype": b_tdt or b_pdt,
            "w_space": lb["space"] if lb else None,
            "w_memlayout": lb["memlayout"] if lb else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--only-ds", action="store_true")
    args = ap.parse_args()
    for f in args.files:
        rows = survey(f)
        name = Path(f).name
        groups = defaultdict(list)
        for r in rows:
            groups[(r["kind"], r["Kw"], r["N"], r["in0_block_w"], r["in0_cores"])].append(r)
        print(f"\n=== {name}  ({len(rows)} matmul/linear ops) ===")
        if not rows:
            continue
        print(f"{'kind':8s} {'K x N':>14s} {'n':>4s} {'in0cores':>8s} {'k/core':>7s} "
              f"{'w':>3s} {'pcn':>4s} {'wspace':>7s} {'best w<=8':>9s} {'trapped':>7s}")
        for key in sorted(groups, key=lambda k: (k[0], -len(groups[k]))):
            g = groups[key]
            r = g[0]
            if args.only_ds and r["kind"] != "DS":
                continue
            kpc = r["k_per_core"]
            best = largest_divisor_upto(kpc, 8) if kpc else None
            trapped = ""
            if r["kind"] == "DS" and r["in0_block_w"] == 1 and kpc and kpc > 1:
                trapped = "YES" if best == 1 else "no"
            print(f"{r['kind']:8s} {r['Kw']:6d}x{r['N']:<7d} {len(g):4d} "
                  f"{str(r['in0_cores']):>8s} {str(kpc):>7s} {str(r['in0_block_w']):>3s} "
                  f"{str(r['per_core_n']):>4s} {str(r['w_space']):>7s} {str(best):>9s} {trapped:>7s}")


if __name__ == "__main__":
    main()
