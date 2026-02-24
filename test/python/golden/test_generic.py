# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import inspect
import functools
import itertools
from typing import Callable, List

from ttmlir.dialects import d2m, ttcore, memref, linalg, tensor
from ttmlir.ir import *

from builder.base.builder_utils import Operand
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import Marks, shape_str
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("d2m")


def _affine_map_from_lambda(fn):
    class Dim:
        def __init__(self, position, name):
            self.position = position
            self.name = name

    dims = tuple(
        Dim(name, i) for name, i in enumerate(inspect.signature(fn).parameters)
    )
    num_dims = len(dims)
    results = fn(*dims)
    exprs = []
    for result in results:
        if isinstance(result, Dim):
            exprs.append(AffineDimExpr.get(result.position))
        elif isinstance(result, int):
            assert (
                result == 0
            ), "The only integer constant allowed in an indexing_map is 0"
            exprs.append(AffineConstantExpr.get(result))
        else:
            raise TypeError(
                "Unsupported indexing_map result type `{type(result)}` for result `{result}`"
            )
    num_syms = 0
    return AffineMap.get(num_dims, num_syms, exprs)


def _create_generic(
    operands,
    grid,
    block_factors,
    indexing_maps,
    iterator_types,
):
    if (
        isinstance(block_factors, list)
        and len(block_factors) > 0
        and isinstance(block_factors[0], tuple)
    ):
        assert isinstance(block_factors, list)
        assert isinstance(block_factors[0], tuple)
        block_factors = [b for bs in block_factors for b in bs]

    inputs = operands[:-1]
    outputs = operands[-1:]
    assert len(outputs) == 1
    ret_type = outputs[0].type
    ctx = ret_type.context
    threads = ArrayAttr.get([d2m.ir.ThreadAttr.get(ctx, "unified")])
    return d2m.GenericOp(
        [ret_type],
        inputs,
        outputs,
        ttcore.ir.GridAttr.get(ctx, grid),
        block_factors,
        list(map(_affine_map_from_lambda, indexing_maps)),
        ArrayAttr.get(
            list(
                ttcore.ir.IteratorTypeAttr.get(
                    ctx, ttcore.IteratorType[i.title()].value
                )
                for i in iterator_types
            )
        ),
        threads,
        len(threads),
    )


def generic(
    grid=None,
    block_factors=None,
    indexing_maps=None,
    iterator_types=None,
):
    assert grid is not None
    assert (iterator_types is None) or (
        indexing_maps is not None
    ), "if iterator_types is set, indexing_types must also be set"

    if indexing_maps is None:
        indexing_maps = []

    if block_factors is None:
        block_factors = [1] * len(grid)

    if indexing_maps:
        for indexing_map in indexing_maps:
            num_dims = len(inspect.signature(indexing_map).parameters)
            if iterator_types is not None:
                assert num_dims == len(iterator_types)
            assert len(block_factors) == num_dims

    if iterator_types is None:
        iterator_types = []

    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            nonlocal grid
            nonlocal block_factors
            nonlocal indexing_maps
            nonlocal iterator_types

            generic = _create_generic(
                args,
                grid,
                block_factors,
                indexing_maps,
                iterator_types,
            )
            assert len(generic.regions[0].blocks) == 0
            generic.regions[0].blocks.append()
            print(dir(generic.attributes))
            block = generic.regions[0].blocks[0]
            ctx = generic.context
            loc = generic.location
            generic.attributes["d2m.explicit_par"] = UnitAttr.get(ctx)
            grid_rank = len(grid)
            for arg in args:
                arg_type = RankedTensorType.get(
                    arg.type.shape[grid_rank:], arg.type.element_type
                )
                block.add_argument(d2m.ir.CBType.get(ctx, arg_type), loc)
            with InsertionPoint(block):
                f(*args, **kwargs)

            return generic.result

        return _wrapper

    return _decorator


def remote_load(
    src, indices, mcast_start_index=None, mcast_shape=None, mcast_dims=None
):
    dst = tensor.empty(src.type.shape[len(indices) :], src.type.element_type)
    return d2m.remote_load(
        RankedTensorType.get(dst.type.shape, dst.type.element_type),
        src,
        indices,
        mcast_start_index=mcast_start_index,
        mcast_shape=mcast_shape,
        mcast_dims=mcast_dims,
        local_buffer=dst,
    )


@pytest.mark.parametrize(
    "grid",
    [
        (8, 8),
    ],
)
@pytest.mark.parametrize(
    "block_shape,block_factors",
    [
        ((64, 64, 64), (1, 1, 8)),
    ],
)
@pytest.mark.parametrize(
    "interchange",
    [
        "mnk",
    ],
)
@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_generic(
    grid,
    block_shape,
    block_factors,
    interchange,
    dtype,
    target: str,
    request,
    device,
):
    block_m, block_n, block_k = block_shape
    m = block_m * grid[0] * block_factors[0]
    n = block_n * grid[1] * block_factors[1]
    k = block_k * block_factors[2]

    lhs_shape = [m, k]
    rhs_shape = [k, n]
    out_shape = [m, n]

    lhs_grid = [grid[0], block_factors[2]]
    rhs_grid = [block_factors[2], grid[1]]

    lhs_block_shape = [block_m // 32, block_k // 32]
    rhs_block_shape = [block_k // 32, block_n // 32]
    out_block_shape = [block_m // 32, block_n // 32]

    indexing_maps, iterator_types = {
        "mnk": (
            [
                lambda m, n, k: (m, k),
                lambda m, n, k: (k, n),
                lambda m, n, k: (m, n),
            ],
            ["parallel", "parallel", "reduction"],
        ),
        "kmn": (
            [
                lambda k, m, n: (m, k),
                lambda k, m, n: (k, n),
                lambda k, m, n: (m, n),
            ],
            ["reduction", "parallel", "parallel"],
        ),
    }[interchange]

    torch_dtype = {
        "f32": torch.float,
        "bf16": torch.bfloat16,
    }[dtype]

    def generic_module(builder: D2MBuilder):
        lhs_golden = torch.randn(lhs_shape, dtype=torch_dtype)
        rhs_golden = torch.randn(rhs_shape, dtype=torch_dtype)
        out_golden = lhs_golden @ rhs_golden

        @builder.func([lhs_shape, rhs_shape], [torch_dtype, torch_dtype])
        def main(
            lhs: Operand,
            rhs: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            @generic(
                grid=grid,
                block_factors=block_factors,
                indexing_maps=indexing_maps,
                iterator_types=iterator_types,
            )
            def mm(lhs, rhs, out):
                mbi = d2m.block_index(0)
                nbi = d2m.block_index(1)
                kbi = d2m.block_index(2)
                lhs_shard = remote_load(lhs, [mbi, kbi])
                rhs_shard = remote_load(rhs, [kbi, nbi])
                out_shard = tensor.empty(out_block_shape, out.type.element_type)
                d2m.tile_matmul_block(lhs_shard, rhs_shard, out_shard)
                res = d2m.remote_store(
                    out.type, out, [mbi, nbi], local_buffer=out_shard
                )
                d2m.yield_([out_shard])

            device_lhs = builder.to_layout(
                lhs,
                output_type=builder.get_metal_tensor_layout(
                    lhs.type.shape, grid=lhs_grid, tiled=True, dtype=dtype
                ),
                unit_attrs=unit_attrs,
            )
            device_rhs = builder.to_layout(
                rhs,
                output_type=builder.get_metal_tensor_layout(
                    rhs.type.shape, grid=rhs_grid, tiled=True, dtype=dtype
                ),
                unit_attrs=unit_attrs,
            )
            device_out = d2m.empty(
                builder.get_metal_tensor_layout(
                    out_shape, grid=grid, tiled=True, dtype=dtype
                )
            )
            mm_out = mm(device_lhs, device_rhs, device_out)
            res = builder.to_layout(
                mm_out,
                output_type=RankedTensorType.get(out_shape, lhs.type.element_type),
                unit_attrs=unit_attrs,
            )
            builder.set_goldens({lhs: lhs_golden, rhs: rhs_golden}, {res: out_golden})
            return res

    def mm_comparison(builder: D2MBuilder):
        lhs_golden = torch.randn(lhs_shape)
        rhs_golden = torch.randn(rhs_shape)
        out_golden = lhs_golden @ rhs_golden

        @builder.func([lhs_shape, rhs_shape], [torch.float32, torch.float32])
        def main(
            lhs: Operand,
            rhs: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            return builder.matmul(lhs, rhs)

    compile_and_execute_ttir(
        generic_module,
        # mm_comparison,
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline",
        print_ir=True,
        **get_request_kwargs(request),
    )
