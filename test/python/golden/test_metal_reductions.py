# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from typing import List

from ttmlir.ir import *

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs, get_board_id

pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


def _sum_atol(shape, dtype):
    per_elem_tol = 0.01 if dtype == torch.bfloat16 else 0.0005
    return math.prod(shape) * per_elem_tol


def _max_atol(dtype):
    return 0.01 if dtype == torch.bfloat16 else 0.0


def _mean_atol(shape, dim_arg, dtype):
    per_elem_tol = 0.01 if dtype == torch.bfloat16 else 0.0005
    reduction_size = math.prod(shape[d] for d in dim_arg)
    return math.prod(shape) * per_elem_tol / reduction_size


def _reduction_atol(reduce_type: str, shape, dim_arg, dtype):
    if reduce_type == "sum":
        return _sum_atol(shape, dtype)
    if reduce_type == "mean":
        return _mean_atol(shape, dim_arg, dtype)
    if reduce_type in ("max", "min"):
        return _max_atol(dtype)
    raise ValueError(f"Unsupported reduce_type: {reduce_type}")


_REDUCE_TYPES = ["sum", "max", "min", "mean"]

_3D_OUTER_REDUCE = {
    combo: _REDUCE_TYPES[i % len(_REDUCE_TYPES)]
    for i, combo in enumerate(
        (b, m, n) for b in [2, 3, 8, 16, 64] for m in [1, 2, 4, 8] for n in [1, 2, 8]
    )
}

_4D_OUTER_REDUCE = {
    combo: _REDUCE_TYPES[i % len(_REDUCE_TYPES)]
    for i, combo in enumerate(
        (a, b, m, n, d)
        for a in [2, 3, 4, 8, 16, 32, 64]
        for b in [2, 4, 8]
        for m in [2]
        for n in [4]
        for d in [0, 1]
    )
}


def create_reductions_constrained_inputs(
    input_shape, reduce_type, dim_arg, keep_dim, dtype
):
    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [dtype])
        def reductions_constrained_inputs(
            in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
        ):
            in_tensor = torch.randn(input_shape, dtype=dtype)
            if dtype == torch.float32:
                # Simulate TF32 truncation in the golden computation
                # TF32 has 10 bits mantissa vs FP32's 23 bits = ~3 decimal digits precision
                scale = 2**13  # Roughly equivalent to TF32 precision
                in_tensor = (in_tensor * scale).round() / scale
            builder.set_goldens(inputs={in0: in_tensor})
            if reduce_type == "sum":
                return builder.sum(in0, dim_arg=dim_arg, keep_dim=keep_dim)
            elif reduce_type == "max":
                return builder.max(
                    in0, dim_arg=dim_arg, keep_dim=keep_dim, unit_attrs=unit_attrs
                )
            elif reduce_type == "mean":
                return builder.mean(in0, dim_arg=dim_arg, keep_dim=keep_dim)
            elif reduce_type == "min":
                return builder.min(
                    in0, dim_arg=dim_arg, keep_dim=keep_dim, unit_attrs=unit_attrs
                )

    return module


@pytest.mark.parametrize("m", [4, 8, 16])
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("dim_arg", [[0], [1], [0, 1]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_sum(
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "sum", dim_arg, keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_sum_atol(shape, dtype),
    )


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("m", [4, 8])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("dim_arg", [[1], [2], [1, 2]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_sum_3d(
    b: int,
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    if len(dim_arg) >= 2 and not keep_dim:
        pytest.skip(
            "keep_dim=False not supported for multi-dim reductions on inner 2 dims because the reshape after the reduction is unsupported due to noc issue: https://github.com/tenstorrent/tt-mlir/issues/6377"
        )

    tile_size = 32
    shape = (
        b,
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "sum", dim_arg, keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_sum_atol(shape, dtype),
    )


@pytest.mark.parametrize("b", [2, 3, 8, 16, 64])
@pytest.mark.parametrize("m", [1, 2, 4, 8])
@pytest.mark.parametrize("n", [1, 2, 8])
@pytest.mark.parametrize("dim_arg", [[0]])
@pytest.mark.parametrize("keep_dim", [True])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_reduce_outer_3d(
    b: int,
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    total_tiles = b * m * n
    if total_tiles >= 128:
        pytest.xfail(
            f"Outer dim reduction incorrect when total input tiles ({total_tiles}) >= 128: "
            "block factor analysis splits the reduction dim, cross-partition "
            "all-reduce is missing, issue here: https://github.com/tenstorrent/tt-mlir/issues/7895"
        )

    reduce_type = _3D_OUTER_REDUCE[(b, m, n)]
    tile_size = 32
    shape = (
        b,
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, reduce_type, dim_arg, keep_dim, dtype
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_reduction_atol(reduce_type, shape, dim_arg, dtype),
    )


@pytest.mark.parametrize("a", [2, 3, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("b", [2, 4, 8])
@pytest.mark.parametrize("m", [2])
@pytest.mark.parametrize("n", [4])
@pytest.mark.parametrize("dim_arg", [[0], [1]])
@pytest.mark.parametrize("keep_dim", [True])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_reduce_outer_4d(
    a: int,
    b: int,
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
    system_desc,
):
    total_tiles = a * b * m * n
    if dim_arg == [0] and total_tiles >= 128:
        pytest.xfail(
            f"Outer dim 0 reduction incorrect when total input tiles ({total_tiles}) >= 128: "
            "block factor analysis splits the reduction dim, issue here: https://github.com/tenstorrent/tt-mlir/issues/7895"
        )
    if dim_arg == [1] and a == 3 and b == 4:
        pytest.xfail(
            "Outer dim 1 reduction incorrect for a=3, b=4: block factor "
            "analysis splits the reduction dim due to odd batch size, issue here: https://github.com/tenstorrent/tt-mlir/issues/7895"
        )
    # TODO: (a=3, b=8, dim_arg=[1]) fails on p150 with L1 OOM after the
    # non-square grid selection changes. Re-enable once grid selection handles
    # this combination without exhausting L1.
    if dim_arg == [1] and a == 3 and b == 8 and get_board_id(system_desc) == "p150":
        pytest.skip("L1 OOM on p150 with current grid selection")

    reduce_type = _4D_OUTER_REDUCE[(a, b, m, n, dim_arg[0])]
    tile_size = 32
    shape = (
        a,
        b,
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, reduce_type, dim_arg, keep_dim, dtype
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_reduction_atol(reduce_type, shape, dim_arg, dtype),
    )


@pytest.mark.parametrize("a", [1, 2])
@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("m", [4, 8])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("dim_arg", [[2], [3], [2, 3]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_sum_4d(
    a: int,
    b: int,
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    if len(dim_arg) >= 2 and not keep_dim:
        pytest.skip(
            "keep_dim=False not supported for multi-dim reductions on inner 2 dims because the reshape after the reduction is unsupported due to noc issue: https://github.com/tenstorrent/tt-mlir/issues/6377"
        )

    tile_size = 32
    shape = (
        a,
        b,
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "sum", dim_arg, keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_sum_atol(shape, dtype),
    )


@pytest.mark.parametrize("m", [4, 8, 16])
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("dim_arg", [[0], [1]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_max(
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "max", dim_arg, keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_max_atol(dtype),
    )


# Unaligned shapes: dimensions that are NOT multiples of the tile size (32).
# These exercise the OOB padding fill values — sum needs zero-fill and max
# needs neg-inf fill so that padded elements don't corrupt the reduction.


@pytest.mark.parametrize(
    "shape",
    [(100, 50), (37, 61), (50, 100), (129, 65)],
)
@pytest.mark.parametrize("dim_arg", [[0], [1], [0, 1]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_sum_unaligned(
    shape: tuple,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "sum", dim_arg, keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_sum_atol(shape, dtype),
    )


@pytest.mark.parametrize(
    "shape",
    [(100, 50), (37, 61), (50, 100), (129, 65)],
)
@pytest.mark.parametrize("dim_arg", [[0], [1]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_max_unaligned(
    shape: tuple,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "max", dim_arg, keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_max_atol(dtype),
    )


@pytest.mark.parametrize("m", [4, 8, 16])
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("dim_arg", [[0], [1], [0, 1]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_mean(
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "mean", dim_arg, keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_mean_atol(shape, dim_arg, dtype),
    )


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("m", [4, 8])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("dim_arg", [[1], [2], [1, 2]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_mean_3d(
    b: int,
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    if len(dim_arg) >= 2 and not keep_dim:
        pytest.skip(
            "keep_dim=False not supported for multi-dim reductions on inner 2 dims because the reshape after the reduction is unsupported due to noc issue: https://github.com/tenstorrent/tt-mlir/issues/6377"
        )

    tile_size = 32
    shape = (
        b,
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "mean", dim_arg, keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_mean_atol(shape, dim_arg, dtype),
    )


@pytest.mark.parametrize("a", [1, 2])
@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("m", [4, 8])
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("dim_arg", [[2], [3], [2, 3]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_mean_4d(
    a: int,
    b: int,
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    if len(dim_arg) >= 2 and not keep_dim:
        pytest.skip(
            "keep_dim=False not supported for multi-dim reductions on inner 2 dims because the reshape after the reduction is unsupported due to noc issue: https://github.com/tenstorrent/tt-mlir/issues/6377"
        )

    tile_size = 32
    shape = (
        a,
        b,
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "mean", dim_arg, keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_mean_atol(shape, dim_arg, dtype),
    )


@pytest.mark.parametrize(
    "shape",
    [(100, 50), (37, 61), (50, 100), (129, 65)],
)
@pytest.mark.parametrize("dim_arg", [[0], [1], [0, 1]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_mean_unaligned(
    shape: tuple,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "mean", dim_arg, keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_mean_atol(shape, dim_arg, dtype),
    )


@pytest.mark.parametrize("m", [4, 8, 16])
@pytest.mark.parametrize("n", [2, 4, 8])
@pytest.mark.parametrize("dim_arg", [[0], [1]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_min(
    m: int,
    n: int,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "min", dim_arg, keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_max_atol(dtype),
    )


@pytest.mark.parametrize(
    "shape",
    [(100, 50), (37, 61), (50, 100), (129, 65)],
)
@pytest.mark.parametrize("dim_arg", [[0], [1]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_min_unaligned(
    shape: tuple,
    dim_arg: List[int],
    keep_dim: bool,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "min", dim_arg, keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_max_atol(dtype),
    )
