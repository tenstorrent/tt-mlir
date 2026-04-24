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
from conftest import x86_only, get_request_kwargs, get_board_id
from test_utils import Marks, SkipIf, shape_str, shapes_list_str

pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


_INTEGER_DTYPES = (torch.int32,)
_REDUCE_TYPES = ["sum", "max", "min", "mean"]
_INT_REDUCE_TYPES = ["sum", "max", "min"]
_FLOAT_DTYPES = [torch.float32, torch.bfloat16]
_INT_DTYPES = [torch.int32]
_DTYPE_IDS = {torch.float32: "f32", torch.bfloat16: "bf16", torch.int32: "i32"}
_KEEP_DIMS = [True, False]


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
    reduce_types=_REDUCE_TYPES,
    dtypes=_FLOAT_DTYPES,
    keep_dims=_KEEP_DIMS,
):
    def pick(options, i):
        return options[i % len(options)]

    params = []
    for i, combo in enumerate(combos):
        reduce_type = pick(reduce_types, i)
        dtype = pick(dtypes, i)
        keep_dim = pick(keep_dims, i)
        ids = "-".join(
            "_".join(map(str, x)) if isinstance(x, list) else str(x) for x in combo
        )
        params.append(
            pytest.param(
                *combo,
                reduce_type,
                dtype,
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
    "m,n,dim_arg,reduce_type,dtype,keep_dim",
    _cycled_reduction_params(_2D_SHAPE_DIM_COMBOS),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_2d(
    m: int,
    n: int,
    dim_arg: List[int],
    reduce_type: str,
    dtype: torch.dtype,
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
    "shape,dim_arg,reduce_type,dtype,keep_dim",
    _cycled_reduction_params(_2D_UNALIGNED_COMBOS),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_2d_unaligned(
    shape: tuple,
    dim_arg: List[int],
    reduce_type: str,
    dtype: torch.dtype,
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
    "b,m,n,dim_arg,reduce_type,dtype,keep_dim",
    _cycled_reduction_params(_3D_INNER_COMBOS),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_3d_inner(
    b: int,
    m: int,
    n: int,
    dim_arg: List[int],
    reduce_type: str,
    dtype: torch.dtype,
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
    "b,m,n,reduce_type,dtype,keep_dim",
    _cycled_reduction_params(_3D_OUTER_COMBOS, keep_dims=[True]),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_outer_3d(
    b: int,
    m: int,
    n: int,
    reduce_type: str,
    dtype: torch.dtype,
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
    "a,b,m,n,dim_arg,reduce_type,dtype,keep_dim",
    _cycled_reduction_params(_4D_INNER_COMBOS),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_4d_inner(
    a: int,
    b: int,
    m: int,
    n: int,
    dim_arg: List[int],
    reduce_type: str,
    dtype: torch.dtype,
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
    "a,b,reduce_dim,reduce_type,dtype,keep_dim",
    _cycled_reduction_params(_4D_OUTER_COMBOS, keep_dims=[True]),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_outer_4d(
    a: int,
    b: int,
    reduce_dim: int,
    reduce_type: str,
    dtype: torch.dtype,
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


# Int32 reductions: inner via SFPU reduce (tile_reduce_* is float-only),
# outer via the D2M accumulation rewriter. Mean is float-only.


@pytest.mark.parametrize("dim_arg", [[0], [1], [0, 1]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("reduce_type", _INT_REDUCE_TYPES)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
def test_reduce_i32_2d(
    dim_arg: List[int],
    keep_dim: bool,
    reduce_type: str,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    shape = (4 * 32, 2 * 32)

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, reduce_type, dim_arg, keep_dim, dtype
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_reduction_atol(reduce_type, shape, dim_arg, dtype),
    )


_2D_UNALIGNED_INT_COMBOS = [
    (shape, dim_arg)
    for shape in [(100, 50), (37, 61), (50, 100), (129, 65), (1, 501)]
    for dim_arg in [[0], [1], [0, 1]]
]


@pytest.mark.parametrize(
    "shape,dim_arg,reduce_type,dtype,keep_dim",
    _cycled_reduction_params(
        _2D_UNALIGNED_INT_COMBOS,
        reduce_types=_INT_REDUCE_TYPES,
        dtypes=_INT_DTYPES,
    ),
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduce_i32_2d_unaligned(
    shape: tuple,
    dim_arg: List[int],
    reduce_type: str,
    dtype: torch.dtype,
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


@pytest.mark.parametrize("dim_arg", [[1], [2], [1, 2]])
@pytest.mark.parametrize("reduce_type", _INT_REDUCE_TYPES)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
def test_reduce_i32_3d_inner(
    dim_arg: List[int],
    reduce_type: str,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    # keep_dim is pinned to True to avoid the reshape-after-reduce issue
    # (#6377) that affects multi-dim inner reductions.
    shape = (2, 4 * 32, 2 * 32)

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, reduce_type, dim_arg, keep_dim=True, dtype=dtype
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_reduction_atol(reduce_type, shape, dim_arg, dtype),
    )


@pytest.mark.parametrize("b", [2, 8])
@pytest.mark.parametrize("reduce_type", _INT_REDUCE_TYPES)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
def test_reduce_i32_outer_3d(
    b: int,
    reduce_type: str,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    shape = (b, 2 * 32, 2 * 32)

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, reduce_type, dim_arg=[0], keep_dim=True, dtype=dtype
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_reduction_atol(reduce_type, shape, [0], dtype),
    )


@pytest.mark.parametrize("dim_arg", [[2], [3]])
@pytest.mark.parametrize("reduce_type", _INT_REDUCE_TYPES)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
def test_reduce_i32_4d_inner(
    dim_arg: List[int],
    reduce_type: str,
    target: str,
    dtype: torch.dtype,
    request,
    device,
):
    shape = (2, 2, 4 * 32, 2 * 32)

    compile_and_execute_ttir(
        create_reductions_constrained_inputs(
            shape, reduce_type, dim_arg, keep_dim=True, dtype=dtype
        ),
        target=target,
        **get_request_kwargs(request),
        device=device,
        atol=_reduction_atol(reduce_type, shape, dim_arg, dtype),
    )


# ---------------------------------------------------------------------------
# ttmetal-only mirrors of multi-backend tests in
# `ttir_ops/reduction/test_reduction.py`.
# ---------------------------------------------------------------------------

_REDUCTION_OP_CPU_HOISTED_NAMES = [
    "argmax",
    "max",
    "mean",
    "min",
    "prod",
    "reduce_and" | Marks(pytest.mark.skip(reason="Builder test not supported #5792")),
    "reduce_or" | Marks(pytest.mark.skip(reason="Builder test not supported #5792")),
    "sum",
]

_REDUCTION_KEEP_DIM_OPTIONS = [True, False]
_REDUCTION_DIM_ARG_OPTIONS = [[0], [2], [1, 2], None]


@x86_only
@pytest.mark.parametrize("shape", [(32, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
@pytest.mark.parametrize("keep_dim", _REDUCTION_KEEP_DIM_OPTIONS)
@pytest.mark.parametrize("dim_arg", _REDUCTION_DIM_ARG_OPTIONS)
@pytest.mark.parametrize("reduction_op_name", _REDUCTION_OP_CPU_HOISTED_NAMES)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reduction_cpu_hoisted_ops(
    shape,
    dtype: torch.dtype,
    keep_dim: bool,
    dim_arg: Optional[List[int]],
    reduction_op_name: str,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def reduction_op_cpu_hoisted_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            reduction_op_builder_map = {
                "argmax": builder.argmax,
                "max": builder.max,
                "mean": builder.mean,
                "min": builder.min,
                "prod": builder.prod,
                "reduce_and": builder.reduce_and,
                "reduce_or": builder.reduce_or,
                "sum": builder.sum,
            }

            reduction_func = reduction_op_builder_map.get(reduction_op_name)

            return reduction_func(
                in0,
                dim_arg=dim_arg,
                keep_dim=keep_dim,
                unit_attrs=["ttir.should_hoist"],
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@x86_only
@pytest.mark.parametrize(
    "shapes,dim",
    [
        ([(4, 4, 32, 32)], 1),
        ([(2, 8, 16, 16)], 0),
        ([(4, 4, 32, 32)], -1),
    ],
    ids=["dim1", "dim0", "dim_negative"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_hoisted_cumsum(
    shapes: List[Shape],
    dim: int,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def hoisted_cumsum(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.cumsum(in0, dim=dim, unit_attrs=["ttir.should_hoist"])

    compile_and_execute_ttir(
        module,
        test_base=request.node.name,
        target=target,
        device=device,
    )


# ============================================================
# Tests moved from test_ttir_ops.py during TTMetal test
# reorganization.
# ============================================================


@x86_only
@pytest.mark.parametrize("shape", [(4, 4)], ids=shape_str)
@pytest.mark.parametrize("dim_args", [[0]])
@pytest.mark.parametrize("target", ["ttmetal" | SkipIf("sim")])
def test_hoisted_reduce_or(
    shape: Shape, dim_args: List[int], target: str, request, device
):
    """Test the hoisted reduce_or operation with proper dimensions and keep_dim parameter"""

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def hoisted_reduce_or_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.reduce_or(
                in0, dim_arg=dim_args, keep_dim=True, unit_attrs=["ttir.should_hoist"]
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
