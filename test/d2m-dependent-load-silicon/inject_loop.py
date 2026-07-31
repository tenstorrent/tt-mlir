#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Inject a *looped* dependent load into pipeline-generated pre-split D2M IR.

Same idea as inject.py, but the scalar read happens inside an scf.for so the two
wait/pop cadences d2m-insert-scalar-access-cb can produce are exercised on device:

  refill  -- the index transfer is inside the loop, so the pair must go inside it
             and balance per iteration. The CB has 2 pages and the loop runs 4
             times, so without the pop the third reserve_back blocks forever.
  hoisted -- the index transfer is outside the loop and only the reads are inside,
             so the pair must bracket the whole loop: one push, one wait, four
             reads, one pop. A pair placed inside would wait a second time on a CB
             pushed once and hang.

Both variants sum the four reads and recover the same row, so the expected output
is the same reversal inject.py produces:

  index[r][c] = r*128 + c  (ttrt --init arange)
  core (i,j) reads local element [0,k] = index[32i][32j+k] = 4096i + 32j + k
  sum over k=0..3          = 4*(4096i + 32j) + 6
  divui 4                  = 4096i + 32j + 1     (32j+1 <= 97)
  divui 4096               = i
  row = 7 - i
"""
import re
import sys

if len(sys.argv) != 4 or sys.argv[3] not in ("refill", "hoisted"):
    sys.exit("usage: inject_loop.py <presplit.mlir> <out.mlir> refill|hoisted")
src, dst, variant = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(src).read().split("\n")


def find(pred, what):
    hits = [i for i, l in enumerate(lines) if pred(l)]
    assert hits, f"not found: {what}"
    return hits[0]


def indent(i):
    return re.match(r"\s*", lines[i]).group(0)


def type_of(line):
    return line.rsplit(" : ", 1)[1].strip()


lines = [
    l.replace("%arg1: memref<256x128xsi32>", "%arg1: memref<256x128xi32>")
    for l in lines
]

i_w_dram = find(lambda l: "%alloc_0 = memref.alloc()" in l, "weights DRAM alloc")
i_f32_cb = find(lambda l: "%alloc_1 = memref.alloc()" in l, "weights L1 CB alloc")
i_ins = find(lambda l: l.strip().startswith("ins(%alloc_0 "), "generic ins")
i_addargs = find(
    lambda l: l.strip().startswith("additionalArgs(%alloc_1, %alloc_2 "),
    "generic additionalArgs",
)
i_load = find(
    lambda l: "d2m.remote_load %alloc_1 %alloc_0[%core0, %core1]" in l,
    "weights remote_load",
)
i_dealloc = find(lambda l: "memref.dealloc %alloc_1 " in l, "weights CB dealloc")

W_DRAM = type_of(lines[i_w_dram])
F32_CB = type_of(lines[i_f32_cb])
IX_DRAM = W_DRAM.replace("x32x32xf32", "x32x32xi32")
IX_CB = F32_CB.replace("32x32xf32", "32x32xi32")
ind = indent(i_w_dram)
b = indent(i_load)

lines[i_dealloc + 1 : i_dealloc + 1] = [
    f"{ind}memref.dealloc %ixcb : {IX_CB}",
    f"{ind}memref.dealloc %ixdram : {IX_DRAM}",
]

ix_transfer = f"d2m.remote_load %ixcb %ixdram[%core0, %core1] : {IX_CB}, {IX_DRAM}"
body = [
    f"{b}%cst0 = arith.constant 0 : index",
    f"{b}%cst1 = arith.constant 1 : index",
    f"{b}%cst4 = arith.constant 4 : index",
    f"{b}%cst7 = arith.constant 7 : index",
    f"{b}%zero = arith.constant 0 : i32",
    f"{b}%c4i = arith.constant 4 : i32",
    f"{b}%c4096i = arith.constant 4096 : i32",
]
if variant == "hoisted":
    body.append(f"{b}{ix_transfer}")
body += [
    f"{b}%acc = scf.for %k = %cst0 to %cst4 step %cst1 "
    f"iter_args(%a = %zero) -> (i32) {{",
]
if variant == "refill":
    body.append(f"{b}  {ix_transfer}")
body += [
    f"{b}  %v = memref.load %ixcb[%cst0, %k] : {IX_CB}",
    f"{b}  %s = arith.addi %a, %v : i32",
    f"{b}  scf.yield %s : i32",
    f"{b}}}",
    f"{b}%q = arith.divui %acc, %c4i : i32",
    f"{b}%permrow = arith.divui %q, %c4096i : i32",
    f"{b}%permidx = arith.index_cast %permrow : i32 to index",
    f"{b}%row = arith.subi %cst7, %permidx : index",
    lines[i_load].replace("[%core0, %core1]", "[%row, %core1]"),
]
lines[i_load : i_load + 1] = body

assert lines[i_addargs].rstrip().endswith(")")
lines[i_addargs] = (
    lines[i_addargs]
    .replace(
        "additionalArgs(%alloc_1, %alloc_2 :",
        "additionalArgs(%alloc_1, %alloc_2, %ixcb :",
    )
    .replace(")", f", {IX_CB})", 1)
)
lines[i_ins] = (
    lines[i_ins]
    .replace("ins(%alloc_0 : ", "ins(%alloc_0, %ixdram : ")
    .replace(W_DRAM + ")", W_DRAM + ", " + IX_DRAM + ")")
)

layout = re.search(r"layout = (\S+)", lines[i_w_dram + 1]).group(1)
lines[i_f32_cb:i_f32_cb] = [
    f"{ind}%ixdram = memref.alloc() {{address = 1085504 : i64, alignment = 32 : i64}} : {IX_DRAM}",
    f"{ind}d2m.to_device %arg1, %ixdram layout = {layout} : memref<256x128xi32> into {IX_DRAM}",
    f"{ind}%ixcb = memref.alloc() {{address = 122048 : i64, alignment = 16 : i64, "
    f"d2m.synchronized_buffer = 2 : i32}} : {IX_CB}",
]

open(dst, "w").write("\n".join(lines))
print(f"injected looped dependent load ({variant})")
