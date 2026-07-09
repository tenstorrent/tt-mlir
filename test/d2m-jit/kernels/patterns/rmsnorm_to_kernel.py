# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-core RMSNorm as a d2m-jit pattern (milestone 1 of Llama prefill bring-up).

Shards the hidden dim across a 1xG grid and reduces it *across cores* with no
semaphores: each core computes a partial sum-of-squares over its hidden shard,
then a cross-core all-reduce (every core `remote_load`s all partials and sums;
the sequential-kernel ordering is the read-after-write barrier) yields the
per-row inv-rms on every core, which each core uses to scale its own shard.

sum-of-squares is computed as `diag(x_shard @ x_shard^T)` (a single multi-K
transpose_b matmul, output shard-1) to avoid a multi-tile elementwise square
(which the region shard-size-1 rule rejects). `use_tile_matmul=False` is
required for transpose_b.

Math: fold 1/H into `gamma' = gamma*sqrt(H)` on host; eps is added as H*eps to
the reduced sum-of-squares before rsqrt, so the result equals torch RMSNorm and
PCC is a true correctness check.

Two pieces here:
  - KERNEL_BENCHES: the multi-core kernels above (rms_partial / rms_allreduce4 /
    rms_scale), validated on device (test_patterns.py -k rmsnorm, PCC vs torch).
  - @d2m.pattern(root=ttir.RMSNormOp) + PATTERN_TESTS: a single-core lowering of
    ttir.rms_norm, checked with FileCheck. It materializes the identity / 1-over-H
    / eps constants in the rewrite scope (enabled by RewriteScope.add_host_input)
    and expands the 1-D weight to [1, hidden] with tensor.expand_shape.

Follow-ups:
  - e2e=True (run the rewritten module on device, PCC vs TTNN baseline) is left
    off: the 1-D weight rank change isn't lowerable by the d2m-jit device
    pipeline (ttir.reshape stays unbufferized; memref.expand_shape fails to
    legalize; a metadata view can't add a tiling dim). Needs reshape lowering in
    the d2m-jit pipeline. On-device numerics are covered by KERNEL_BENCHES.
  - arbitrary G: the all-reduce is unrolled for G=4 (a loop-carried add hits the
    scf.for bufferization limit; remote_store is tile-granular).
"""
import torch

import d2m_jit as d2m
from d2m_jit.testing import InputSpec, KernelBench, PatternTest
from ttmlir import ir
from ttmlir.dialects import ttir, tensor


@d2m.kernel
def rms_partial(x, ident, part):
    c = core_index(1)
    a = remote_load(x, [0, c])  # this core's [1, KC] hidden shard
    xxt = matmul(a, a, transpose_b=True)  # [1,1]; diagonal = partial sum of squares
    i = remote_load(ident, [0, 0])
    remote_store(part, [0, c], reduce_sum(xxt * i, 1))


@d2m.kernel
def rms_allreduce4(part, eps, inv):
    # Rootless all-reduce over G=4 partials (straight-line adds; a loop-carried
    # add would re-hit the scf.for bufferization limit). `eps` holds H*eps so
    # rsqrt(sum_sq + H*eps) == 1/sqrt(mean(x^2)+eps) after the sqrt(H) fold.
    c = core_index(1)
    p0 = remote_load(part, [0, 0])
    p1 = remote_load(part, [0, 1])
    p2 = remote_load(part, [0, 2])
    p3 = remote_load(part, [0, 3])
    e = remote_load(eps, [0, 0])
    remote_store(inv, [0, c], rsqrt(((p0 + p1) + (p2 + p3)) + e))


@d2m.kernel
def rms_scale(x, gammap, inv, out, kc):
    c = core_index(1)
    n_off = c * kc
    for n in range(kc):
        t = remote_load(x, [0, n_off + n])
        g = tile_bcast_row(remote_load(gammap, [0, n_off + n]))
        iv = tile_bcast_col(remote_load(inv, [0, c]))
        remote_store(out, [0, n_off + n], t * g * iv)


EPS = 1e-5


# --- @d2m.pattern lowering (ttir.rms_norm -> fused d2m subgraph) -------------
# Single-core lowering, used for the rewrite/e2e path. Uses rewrite-scope
# constants (identity, 1/H, eps tiles) now that RewriteScope can materialize
# them. 1/H is applied via the invh tile (no host sqrt(H) fold, since gamma is
# a graph value here).


@d2m.kernel
def rms_reduce_sc(x, ident, invh, eps, inv):
    a = remote_load(x, [0, 0])  # [1, KH] hidden block
    xxt = matmul(a, a, transpose_b=True)  # [1,1]; diag = sum of squares
    i = remote_load(ident, [0, 0])
    ss = reduce_sum(xxt * i, 1)  # [seq,1] sum_h x^2
    ih = remote_load(invh, [0, 0])  # 1/H
    e = remote_load(eps, [0, 0])  # eps
    remote_store(inv, [0, 0], rsqrt(ss * ih + e))


@d2m.kernel
def rms_scale_sc(x, gamma, inv, out, kh):
    for n in range(kh):
        t = remote_load(x, [0, n])
        g = tile_bcast_row(remote_load(gamma, [0, n]))
        iv = tile_bcast_col(remote_load(inv, [0, 0]))
        remote_store(out, [0, n], t * g * iv)


@d2m.pattern(root=ttir.RMSNormOp, benefit=10)
def lower_rmsnorm(op, rewriter):
    x_val = op.operands[0]
    w_val = op.operands[1]
    xt = ir.RankedTensorType(x_val.type)
    seq, hidden = int(xt.shape[-2]), int(xt.shape[-1])
    kh = hidden // 32
    eps = float(ir.FloatAttr(op.attributes["epsilon"]).value)

    # ttir.rms_norm weight is 1-D tensor<hidden>; expand to [1, hidden] in-graph.
    # Use tensor.expand_shape (bufferizes through the standard pipeline; a raw
    # ttir.reshape would be left unbufferized by the post-legalization d2m-jit
    # pipeline).
    with rewriter.ip, op.location:
        w2d_ty = ir.RankedTensorType.get([1, hidden], xt.element_type)
        i64 = ir.IntegerType.get_signless(64)
        reassoc = ir.ArrayAttr.get(
            [ir.ArrayAttr.get([ir.IntegerAttr.get(i64, 0), ir.IntegerAttr.get(i64, 1)])]
        )
        static_shape = ir.DenseI64ArrayAttr.get([1, hidden])
        w2d = tensor.ExpandShapeOp(w2d_ty, w_val, reassoc, [], static_shape).result

    x_1k = d2m.Layout(
        shape=(seq, hidden), dtype=d2m.float32, block_shape=[1, kh], grid_shape=[1, 1]
    )
    x_tiles = d2m.Layout(
        shape=(seq, hidden), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    ident_l = d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    scal_l = d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    gamma_l = d2m.Layout(
        shape=(1, hidden), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    inv_l = d2m.Layout(
        shape=(seq, 1), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    out_l = d2m.Layout(
        shape=(seq, hidden), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )

    x = d2m.to_layout(d2m.from_value(x_val), x_1k)
    ident = d2m.to_layout(torch.eye(32, dtype=torch.float32), ident_l)
    invh = d2m.to_layout(
        torch.full((32, 32), 1.0 / hidden, dtype=torch.float32), scal_l
    )
    eps_t = d2m.to_layout(torch.full((32, 32), eps, dtype=torch.float32), scal_l)
    inv = d2m.empty(inv_l)
    rms_reduce_sc(x, ident, invh, eps_t, inv, grid=(1, 1))

    xg = d2m.to_layout(d2m.from_value(x_val), x_tiles)
    gamma = d2m.to_layout(d2m.from_value(w2d), gamma_l)
    out = d2m.empty(out_l)
    rms_scale_sc(xg, gamma, inv, out, kh, grid=(1, 1))
    return d2m.from_device(out)


PATTERN_TESTS = [
    PatternTest(
        name="rmsnorm_lowers",
        # e2e (run rewritten module on device, PCC vs TTNN baseline) is BLOCKED:
        # ttir.rms_norm's 1-D weight needs a 1-D->2-D rank change, and the
        # d2m-jit device pipeline lowers no reshape (ttir.reshape stays
        # unbufferized; memref.expand_shape fails to legalize; a metadata view
        # can't add a tiling dim). FileCheck below still proves the rewrite fires
        # and materializes rewrite-scope constants. On-device numerics are
        # covered by KERNEL_BENCHES.
        ttir="""
        module {
          func.func @forward(%x: tensor<32x128xf32>, %w: tensor<128xf32>) -> tensor<32x128xf32> {
            %r = "ttir.rms_norm"(%x, %w) <{epsilon = 1.0e-05 : f32, normalized_shape = array<i64: 128>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<32x128xf32>, tensor<128xf32>) -> tensor<32x128xf32>
            return %r : tensor<32x128xf32>
          }
        }
        """,
        check="""
        CHECK-LABEL: func.func @forward
        CHECK-NOT:   ttir.rms_norm
        CHECK:       arith.constant dense
        CHECK:       d2m.generic
        """,
    ),
]


def _golden(x, gamma):
    ms = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(ms + EPS) * gamma


def rmsnorm_run(kernel, inputs, cfg):
    """Chain partial -> all-reduce -> scale on a 1xG grid. inputs=[x, gamma]."""
    x, gamma = inputs
    seq, hidden = x.shape
    g = cfg["grid_shape"][1]
    kh = hidden // 32
    kc = kh // g
    gammap = gamma * (hidden**0.5)
    ident = torch.eye(32, dtype=x.dtype)
    eps_t = torch.full((32, 32), hidden * EPS, dtype=x.dtype)  # H*eps

    x_shard = d2m.Layout(
        shape=(seq, hidden), dtype=d2m.float32, block_shape=[1, kc], grid_shape=[1, g]
    )
    x_tiles = d2m.Layout(
        shape=(seq, hidden), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, g]
    )
    ident_l = d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    eps_l = d2m.Layout(
        shape=(32, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
    )
    part_l = d2m.Layout(
        shape=(seq, g * 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, g]
    )
    gamma_l = d2m.Layout(
        shape=(1, hidden), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, g]
    )
    out_l = d2m.Layout(
        shape=(seq, hidden), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, g]
    )

    old = d2m.config.use_tile_matmul
    d2m.config.use_tile_matmul = False  # transpose_b needs the block lowering
    try:
        part = d2m.empty(part_l)
        rms_partial(
            d2m.to_layout(x, x_shard), d2m.to_layout(ident, ident_l), part, grid=(1, g)
        )
        inv = d2m.empty(part_l)
        rms_allreduce4(part, d2m.to_layout(eps_t, eps_l), inv, grid=(1, g))
        out = d2m.empty(out_l)
        rms_scale(
            d2m.to_layout(x, x_tiles),
            d2m.to_layout(gammap, gamma_l),
            inv,
            out,
            kc,
            grid=(1, g),
        )
        return out.to_host()
    finally:
        d2m.config.use_tile_matmul = old


# On-device numerics: anchor hidden=3200 sharded across a 1x4 grid.
KERNEL_BENCHES = [
    KernelBench(
        name="rmsnorm_multicore_h3200_g4",
        kernel=rms_scale,  # nominal entrypoint; rmsnorm_run orchestrates all 3
        golden=_golden,
        input_shapes=[(32, 3200), (1, 3200)],
        run=rmsnorm_run,
        inputs=InputSpec("randn"),
        default_cfg=dict(block_shape=[1, 1], grid_shape=[1, 4], dtype="float32"),
    ),
]


if __name__ == "__main__":
    from d2m_jit.testing import run_bench

    actual, expected = run_bench(KERNEL_BENCHES[0])
    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
    print(
        "actual[0,:3]=",
        [round(v, 3) for v in actual[0, :3].tolist()],
        "expected[0,:3]=",
        [round(v, 3) for v in expected[0, :3].tolist()],
        "pcc=",
        round(pcc, 5),
    )
    assert pcc >= 0.99, f"PCC {pcc} < 0.99"
    print("rmsnorm_to_kernel: PCC OK")
