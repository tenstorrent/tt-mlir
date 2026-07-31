#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Inject a dependent load into pipeline-generated pre-split D2M IR.

The weights tilize generic streams shard [core0, core1] from DRAM. We add an
i32 index buffer as a second `ins` operand plus its own L1 CB, scalar-read one
i32 out of that CB, and use it to pick the *row* the weights transfer reads.

With `ttrt run --init arange`:
  index[r][c] = r*128 + c, so core (i,j) reads index[32i][32j] = 4096i + 32j
  perm = 4096i + 32j;  divui 4096 -> i  (32j <= 96 < 4096)
  row  = 7 - i                            <- a reversal, not the identity
So output tile-row i must come from weights tile-row 7-i.

Works off the line contents in the dump so it is agnostic to indentation and to
whichever type aliases the printer chose.
"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")


def find(pred, what):
    hits = [i for i, l in enumerate(lines) if pred(l)]
    assert hits, f"not found: {what}"
    return hits[0]


def indent(i):
    return re.match(r"\s*", lines[i]).group(0)


def type_of(line):
    """The memref type after the final ' : ' on an alloc line."""
    return line.rsplit(" : ", 1)[1].strip()


# The index input arrives as si32 from TTIR normalization; the scalar-load path
# needs a signless integer (arith on the loaded value is signless-only), so the
# fixture declares it signless.
lines = [
    l.replace("%arg1: memref<256x128xsi32>", "%arg1: memref<256x128xi32>")
    for l in lines
]

# Locate the pieces of the weights tilize generic.
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
bind = indent(i_load)

# Insert in descending line order so earlier indices stay valid.

# 4. Free the added buffers alongside the originals.
lines[i_dealloc + 1 : i_dealloc + 1] = [
    f"{ind}memref.dealloc %ixcb : {IX_CB}",
    f"{ind}memref.dealloc %ixdram : {IX_DRAM}",
]

# 3. The dependent load itself, replacing the static row index.
lines[i_load : i_load + 1] = [
    f"{bind}%cst0 = arith.constant 0 : index",
    f"{bind}%cst7 = arith.constant 7 : index",
    f"{bind}%cst4096 = arith.constant 4096 : i32",
    f"{bind}d2m.remote_load %ixcb %ixdram[%core0, %core1] : {IX_CB}, {IX_DRAM}",
    f"{bind}%perm = memref.load %ixcb[%cst0, %cst0] : {IX_CB}",
    f"{bind}%permrow = arith.divui %perm, %cst4096 : i32",
    f"{bind}%permidx = arith.index_cast %permrow : i32 to index",
    f"{bind}%row = arith.subi %cst7, %permidx : index",
    lines[i_load].replace("[%core0, %core1]", "[%row, %core1]"),
]

# 2. Add the index buffer to the generic's ins / additionalArgs.
lines[i_addargs] = (
    lines[i_addargs]
    .replace(
        "additionalArgs(%alloc_1, %alloc_2 :",
        "additionalArgs(%alloc_1, %alloc_2, %ixcb :",
    )
    .replace(")", f", {IX_CB})", 1)
    if lines[i_addargs].rstrip().endswith(")")
    else None
)
assert lines[i_addargs], "additionalArgs rewrite failed"
lines[i_ins] = (
    lines[i_ins]
    .replace("ins(%alloc_0 : ", "ins(%alloc_0, %ixdram : ")
    .replace(W_DRAM + ")", W_DRAM + ", " + IX_DRAM + ")")
)

# 1. Allocate the index DRAM buffer + its L1 CB, and stage the host input in.
#    Addresses are past the existing buffers: DRAM allocs here run to ~1085504,
#    L1 CBs sit at 105664 and 113856 (8192 apart).
layout = re.search(r"layout = (\S+)", lines[i_w_dram + 1]).group(1)
lines[i_f32_cb:i_f32_cb] = [
    f"{ind}%ixdram = memref.alloc() {{address = 1085504 : i64, alignment = 32 : i64}} : {IX_DRAM}",
    f"{ind}d2m.to_device %arg1, %ixdram layout = {layout} : memref<256x128xi32> into {IX_DRAM}",
    f"{ind}%ixcb = memref.alloc() {{address = 122048 : i64, alignment = 16 : i64, "
    f"d2m.synchronized_buffer = 2 : i32}} : {IX_CB}",
]

open(dst, "w").write("\n".join(lines))
print("injected dependent load")
