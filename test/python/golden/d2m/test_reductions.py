# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from typing import List, Optional

from ttmlir.ir import *

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs, get_board_id

pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


_INTEGER_DTYPES = (torch.int32,)
_REDUCE_TYPES = ["sum", "max", "min", "mean"]
_INT_REDUCE_TYPES = ["sum", "max", "min"]
_FLOAT_DTYPES = [torch.float32, torch.bfloat16]
_DTYPE_IDS = {torch.float32: "f32", torch.bfloat16: "bf16", torch.int32: "i32"}
_KEEP_DIMS = [True, False]
_FLOAT_REDUCTION_VARIANTS = [
    (dtype, reduce_type, keep_dim)
    for reduce_type in _REDUCE_TYPES
    for dtype in _FLOAT_DTYPES
    for keep_dim in _KEEP_DIMS
]
_INT_REDUCTION_VARIANTS = [
    (torch.int32, reduce_type, keep_dim)
    for reduce_type in _INT_REDUCE_TYPES
    for keep_dim in _KEEP_DIMS
]
_REDUCTION_VARIANTS = _FLOAT_REDUCTION_VARIANTS + _INT_REDUCTION_VARIANTS
_OUTER_REDUCTION_VARIANTS = [
    (dtype, reduce_type, True)
    for reduce_type in _REDUCE_TYPES
    for dtype in _FLOAT_DTYPES
] + [(torch.int32, reduce_type, True) for reduce_type in _INT_REDUCE_TYPES]


def _reduction_atol(reduce_type: str, shape, dim_arg, dtype):
    if dtype in _INTEGER_DTYPES:
        return 0.0
    per_elem_tol = 0.01 if dtype == torch.bfloat16 else 0.0005
    if reduce_type == "sum":
        return math.prod(shape) * per_elem_tol
    if reduce_type == "mean":
        reduction_size = math.prod(shape[d] for d in dim_arg)
        return math.prod(shape) * per_elem_tol / reduction_size
    if reduce_type in ("max", "min"):
        return 0.01 if dtype == torch.bfloat16 else 0.0
    raise ValueError(f"Unsupported reduce_type: {reduce_type}")


# Int range small enough that reductions won't overflow int32.
def _int_input_range(reduce_type: str, shape, dim_arg):
    if reduce_type != "sum":
        return -10_000, 10_000
    reduction_size = max(1, math.prod(shape[d] for d in dim_arg))
    bound = min(10_000, max(1, (2**31 - 1) // (8 * reduction_size)))
    return -bound, bound


# Cycle reduce_type/dtype/keep_dim independently across shape/dim combos so
# each combo gets exactly one variant (vs. full cross product).
def _cycled_reduction_params(
    combos,
    *,
    shape_arity=None,
    variants=None,
    reduce_types=_REDUCE_TYPES,
    dtypes=_FLOAT_DTYPES,
    keep_dims=_KEEP_DIMS,
):
    def pick(options, i):
        return options[i % len(options)]

    params = []
    for i, combo in enumerate(combos):
        split_at = len(combo) if shape_arity is None else shape_arity
        shape_values = combo[:split_at]
        attr_values = combo[split_at:]
        if variants is None:
            reduce_type = pick(reduce_types, i)
            dtype = pick(dtypes, i)
            keep_dim = pick(keep_dims, i)
        else:
            dtype, reduce_type, keep_dim = pick(variants, i)
        ids = "-".join(
            "_".join(map(str, x)) if isinstance(x, list) else str(x) for x in combo
        )
        params.append(
            pytest.param(
                *shape_values,
                dtype,
                *attr_values,
                reduce_type,
                keep_dim,
                id=f"{ids}-{reduce_type}-{_DTYPE_IDS[dtype]}-keep{int(keep_dim)}",
            )
        )
    return params


def create_reductions_constrained_inputs(
    input_shape, reduce_type, dim_arg, keep_dim, dtype
):
    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [dtype])
        def reductions_constrained_inputs(
            in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
        ):
            if dtype in _INTEGER_DTYPES:
                lo, hi = _int_input_range(reduce_type, input_shape, dim_arg)
                # torch.randint's high is exclusive.
                in_tensor = torch.randint(lo, hi + 1, input_shape, dtype=dtype)
            else:
                in_tensor = torch.randn(input_shape, dtype=dtype)
                if dtype == torch.float32:
                    # Round golden to ~TF32 precision (10 mantissa bits).
                    scale = 2**13
                    in_tensor = (in_tensor * scale).round() / scale
            builder.set_goldens(inputs={in0: in_tensor})

            kwargs = {"dim_arg": dim_arg, "keep_dim": keep_dim}
            if reduce_type in ("max", "min"):
                kwargs["unit_attrs"] = unit_attrs
            return getattr(builder, reduce_type)(in0, **kwargs)

    return module


_2D_SHAPE_DIM_COMBOS = [
    (m, n, dim_arg)
    for m in [4, 8, 16]
    for n in [2, 4, 8]
    for dim_arg in [[0], [1], [0, 1]]
]


@pytest.mark.parametrize(
    "m,n,dtype,dim_arg,reduce_type,keep_dim",
    _cycled_reduction_params(
        _2D_SHAPE_DIM_COMBOS, shape_arity=2, variants=_REDUCTION_VARIANTS
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_2d(
    m: int,
    n: int,
    dtype: torch.dtype,
    dim_arg: List[int],
    reduce_type: str,
    keep_dim: bool,
    target: str,
    request,
    device,
):
    tile_size = 32
    shape = (m * tile_size, n * tile_size)

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, reduce_type, dim_arg, keep_dim, dtype
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_reduction_atol(reduce_type, shape, dim_arg, dtype),
    )


# Non-tile-aligned shapes exercise OOB padding (0 for sum, +/- inf for
# max/min) so padded elements don't corrupt the reduction.
_2D_UNALIGNED_COMBOS = [
    (shape, dim_arg)
    for shape in [(100, 50), (37, 61), (50, 100), (129, 65), (1, 501)]
    for dim_arg in [[0], [1], [0, 1]]
]


@pytest.mark.parametrize(
    "shape,dtype,dim_arg,reduce_type,keep_dim",
    _cycled_reduction_params(
        _2D_UNALIGNED_COMBOS, shape_arity=1, variants=_REDUCTION_VARIANTS
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_2d_unaligned(
    shape: tuple,
    dtype: torch.dtype,
    dim_arg: List[int],
    reduce_type: str,
    keep_dim: bool,
    target: str,
    request,
    device,
):
    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, reduce_type, dim_arg, keep_dim, dtype
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_reduction_atol(reduce_type, shape, dim_arg, dtype),
    )


_3D_INNER_COMBOS = [
    (b, m, n, dim_arg)
    for b in [1, 2]
    for m in [4, 8]
    for n in [2, 4]
    for dim_arg in [[1], [2], [1, 2]]
]


@pytest.mark.parametrize(
    "b,m,n,dtype,dim_arg,reduce_type,keep_dim",
    _cycled_reduction_params(
        _3D_INNER_COMBOS, shape_arity=3, variants=_REDUCTION_VARIANTS
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_3d_inner(
    b: int,
    m: int,
    n: int,
    dtype: torch.dtype,
    dim_arg: List[int],
    reduce_type: str,
    keep_dim: bool,
    target: str,
    request,
    device,
):
    if len(dim_arg) >= 2 and not keep_dim:
        pytest.skip(
            "keep_dim=False not supported for multi-dim reductions on inner 2 dims because the reshape after the reduction is unsupported due to noc issue: https://github.com/tenstorrent/tt-mlir/issues/6377"
        )

    tile_size = 32
    shape = (b, m * tile_size, n * tile_size)

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, reduce_type, dim_arg, keep_dim, dtype
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_reduction_atol(reduce_type, shape, dim_arg, dtype),
    )


# Outer (batch dim) reductions go through the D2M accumulation rewriter
# instead of per-tile tile_reduce.
_3D_OUTER_COMBOS = [
    (b, m, n) for b in [2, 3, 8, 16, 64] for m in [1, 2, 4, 8] for n in [1, 2, 8]
]


@pytest.mark.parametrize(
    "b,m,n,dtype,reduce_type,keep_dim",
    _cycled_reduction_params(_3D_OUTER_COMBOS, variants=_OUTER_REDUCTION_VARIANTS),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_outer_3d(
    b: int,
    m: int,
    n: int,
    dtype: torch.dtype,
    reduce_type: str,
    keep_dim: bool,
    target: str,
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

    tile_size = 32
    shape = (b, m * tile_size, n * tile_size)

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, reduce_type, [0], keep_dim, dtype),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_reduction_atol(reduce_type, shape, [0], dtype),
    )


_4D_INNER_COMBOS = [
    (a, b, m, n, dim_arg)
    for a in [1, 2]
    for b in [1, 2]
    for m in [4, 8]
    for n in [2, 4]
    for dim_arg in [[2], [3], [2, 3]]
]


@pytest.mark.parametrize(
    "a,b,m,n,dtype,dim_arg,reduce_type,keep_dim",
    _cycled_reduction_params(
        _4D_INNER_COMBOS, shape_arity=4, variants=_REDUCTION_VARIANTS
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_4d_inner(
    a: int,
    b: int,
    m: int,
    n: int,
    dtype: torch.dtype,
    dim_arg: List[int],
    reduce_type: str,
    keep_dim: bool,
    target: str,
    request,
    device,
):
    if len(dim_arg) >= 2 and not keep_dim:
        pytest.skip(
            "keep_dim=False not supported for multi-dim reductions on inner 2 dims because the reshape after the reduction is unsupported due to noc issue: https://github.com/tenstorrent/tt-mlir/issues/6377"
        )

    tile_size = 32
    shape = (a, b, m * tile_size, n * tile_size)

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, reduce_type, dim_arg, keep_dim, dtype
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_reduction_atol(reduce_type, shape, dim_arg, dtype),
    )


_4D_OUTER_COMBOS = [
    (a, b, dim_arg)
    for a in [2, 3, 4, 8, 16, 32, 64]
    for b in [2, 4, 8]
    for dim_arg in [0, 1]
]


@pytest.mark.parametrize(
    "a,b,dtype,reduce_dim,reduce_type,keep_dim",
    _cycled_reduction_params(
        _4D_OUTER_COMBOS, shape_arity=2, variants=_OUTER_REDUCTION_VARIANTS
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_outer_4d(
    a: int,
    b: int,
    dtype: torch.dtype,
    reduce_dim: int,
    reduce_type: str,
    keep_dim: bool,
    target: str,
    request,
    device,
    system_desc,
):
    m, n = 2, 4
    total_tiles = a * b * m * n
    if reduce_dim == 0 and total_tiles >= 128:
        pytest.xfail(
            f"Outer dim 0 reduction incorrect when total input tiles ({total_tiles}) >= 128: "
            "block factor analysis splits the reduction dim, issue here: https://github.com/tenstorrent/tt-mlir/issues/7895"
        )
    if reduce_dim == 1 and a == 3 and b == 4:
        pytest.xfail(
            "Outer dim 1 reduction incorrect for a=3, b=4: block factor "
            "analysis splits the reduction dim due to odd batch size, issue here: https://github.com/tenstorrent/tt-mlir/issues/7895"
        )
    # TODO(#8079): (a=3, b=8, reduce_dim=1) fails on p150 with L1 OOM on the
    # non-square 10x13 grid. Re-enable once grid selection handles non-square
    # grids without inflating per-core L1.
    if reduce_dim == 1 and a == 3 and b == 8 and get_board_id(system_desc) == "p150":
        pytest.skip("L1 OOM on non-square grid (see #8079)")

    tile_size = 32
    shape = (a, b, m * tile_size, n * tile_size)

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, reduce_type, [reduce_dim], keep_dim, dtype
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_reduction_atol(reduce_type, shape, [reduce_dim], dtype),
    )


@pytest.mark.parametrize("m", [8])
@pytest.mark.parametrize("n", [8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("dim_arg", [[0], [1]])
@pytest.mark.parametrize("keep_dim", [True])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_allocate_max(
    m: int,
    n: int,
    dtype: torch.dtype,
    dim_arg: int,
    keep_dim: bool,
    target: str,
    request,
    device,
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    options = [
        # Request the allocator to attempt to minimize stream buffer sizes
        # and reblock streams accordingly.
        "test-buffer-size-policy=min",
    ]

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(shape, "max", dim_arg, keep_dim, dtype),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
    )
