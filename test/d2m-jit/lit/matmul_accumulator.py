# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""IR-shape coverage for d2m-jit matmul accumulator initialization."""

import d2m_jit as d2m
import d2m_jit._src.builder as builder_mod

from d2m_jit._src.builder import _Builder, _emit_returns_and_finalise, _run_pipeline


@d2m.kernel
def k_standalone_matmul(lhs, rhs, out):
    a = remote_load(lhs, [0, 0])
    b = remote_load(rhs, [0, 0])
    c = a @ b
    remote_store(out, [0, 0], c)


@d2m.kernel
def k_loop_carried_matmul(lhs, rhs, out, k_blocks):
    c = zeros([1, 1])
    for k in range(k_blocks):
        a = remote_load(lhs, [0, k])
        b = remote_load(rhs, [k, 0])
        c += a @ b
    remote_store(out, [0, 0], c)


@d2m.kernel
def k_tiled_loop_carried_matmul(lhs, rhs, out, m_blocks, n_blocks, k_blocks):
    for m in range(m_blocks):
        for n in range(n_blocks):
            acc = zeros([1, 1])
            for k in range(k_blocks):
                a = remote_load(lhs, [m, k])
                b = remote_load(rhs, [k, n])
                acc += a @ b
            remote_store(out, [m, n], acc)


def _layout(shape, block_shape):
    return d2m.Layout(
        shape=shape, dtype=d2m.float32, block_shape=list(block_shape), grid_shape=[1, 1]
    )


def _finalize(label, out):
    builder = _Builder.get()
    _emit_returns_and_finalise(builder, [out])
    builder.module.operation.verify()
    print(label)
    print(builder.module)
    _Builder.reset()


def _compile_pipeline(label, out):
    builder = _Builder.get()
    _emit_returns_and_finalise(builder, [out])
    builder.module.operation.verify()

    old_use_tile_matmul = d2m.config.use_tile_matmul
    old_get_system_desc_path = builder_mod._get_system_desc_path
    d2m.config.use_tile_matmul = True
    builder_mod._get_system_desc_path = lambda: None
    try:
        _run_pipeline(builder)
        builder.module.operation.verify()
    finally:
        d2m.config.use_tile_matmul = old_use_tile_matmul
        builder_mod._get_system_desc_path = old_get_system_desc_path
        _Builder.reset()

    print(label)


lhs = d2m.empty(_layout((32, 32), (1, 1)))
rhs = d2m.empty(_layout((32, 32), (1, 1)))
out = d2m.empty(_layout((32, 32), (1, 1)))
k_standalone_matmul(lhs, rhs, out, grid=(1, 1))
_finalize("STANDALONE_MATMUL_ZERO_INIT_IR", out)

lhs = d2m.empty(_layout((32, 64), (1, 1)))
rhs = d2m.empty(_layout((64, 32), (1, 1)))
out = d2m.empty(_layout((32, 32), (1, 1)))
k_loop_carried_matmul(lhs, rhs, out, 2, grid=(1, 1))
_finalize("LOOP_CARRIED_MATMUL_ACC_IR", out)

lhs = d2m.empty(_layout((32, 64), (1, 1)))
rhs = d2m.empty(_layout((64, 32), (1, 1)))
out = d2m.empty(_layout((32, 32), (1, 1)))
k_loop_carried_matmul(lhs, rhs, out, 2, grid=(1, 1))
_compile_pipeline("LOOP_CARRIED_MATMUL_ACC_PIPELINE", out)

lhs = d2m.empty(_layout((64, 96), (1, 1)))
rhs = d2m.empty(_layout((96, 64), (1, 1)))
out = d2m.empty(_layout((64, 64), (1, 1)))
k_tiled_loop_carried_matmul(lhs, rhs, out, 2, 2, 3, grid=(1, 1))
_compile_pipeline("TILED_LOOP_CARRIED_MATMUL_ACC_PIPELINE", out)

print("PASS matmul accumulator")


# CHECK-LABEL: STANDALONE_MATMUL_ZERO_INIT_IR
# CHECK:       d2m.tile_fill
# CHECK:       "d2m.tile_matmul"

# CHECK-LABEL: LOOP_CARRIED_MATMUL_ACC_IR
# CHECK:       d2m.tile_fill
# CHECK:       scf.for
# CHECK-SAME:  iter_args
# CHECK:       "d2m.tile_matmul"
# CHECK:       scf.yield

# CHECK-LABEL: LOOP_CARRIED_MATMUL_ACC_PIPELINE

# CHECK-LABEL: TILED_LOOP_CARRIED_MATMUL_ACC_PIPELINE
# CHECK:       PASS matmul accumulator
