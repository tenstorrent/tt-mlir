# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import inspect
import functools
import itertools
from typing import Callable, List

from ttmlir.dialects import d2m, ttcore, memref, linalg
from ttmlir.ir import *

from builder.base.builder_utils import Operand
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import Marks
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
            block = generic.regions[0].blocks[0]
            ctx = generic.context
            loc = generic.location
            grid_rank = len(grid)
            for arg in args:
                arg_type = RankedTensorType.get(arg.type.shape[grid_rank:], arg.type.element_type)
                block.add_argument(d2m.ir.CBType.get(ctx, arg_type), loc)
            with InsertionPoint(block):
                f(*args, **kwargs)

            return generic.result

        return _wrapper

    return _decorator


def remote_load(
    src, indices, mcast_start_index=None, mcast_shape=None, mcast_dims=None
):
    dst = d2m.empty(
        RankedTensorType.get(src.type.shape[len(indices):], src.type.element_type)
    )
    return d2m.remote_load(
        RankedTensorType.get(dst.type.shape, dst.type.element_type),
        src,
        indices,
        mcast_start_index=mcast_start_index,
        mcast_shape=mcast_shape,
        mcast_dims=mcast_dims,
        local_buffer=dst
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_generic(
    target: str,
    request,
    device,
):
    shape = [64, 64]

    def generic_module(builder: D2MBuilder):
        lhs_golden = torch.randn(shape)
        rhs_golden = torch.randn(shape)
        out_golden = lhs_golden @ rhs_golden

        @builder.func([shape, shape], [torch.float32, torch.float32])
        def main(
            lhs: Operand,
            rhs: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            @generic(
                grid=(1, 1),
                block_factors=[1, 1, 1],
                indexing_maps=[
                    lambda m, n, k: (m, k),
                    lambda m, n, k: (k, n),
                    lambda m, n, k: (m, n),
                ],
                iterator_types=["parallel", "parallel", "reduction"],
            )
            def mm(lhs, rhs, out):
                mbi = d2m.block_index(0)
                nbi = d2m.block_index(1)
                kbi = d2m.block_index(2)
                lhs_shard = remote_load(lhs, [mbi, kbi])
                rhs_shard = remote_load(rhs, [kbi, nbi])
                out_shard = d2m.empty(
                    RankedTensorType.get(out.type.shape[2:], out.type.element_type)
                )
                d2m.tile_matmul_block(lhs_shard, rhs_shard, out_shard)
                res = d2m.remote_store(
                    out.type, out, [mbi, nbi], local_buffer=out_shard
                )
                d2m.yield_([out_shard])

            device_lhs = builder.to_layout(
                lhs,
                output_type=builder.get_metal_tensor_layout(lhs.type.shape, tiled=True),
                unit_attrs=unit_attrs,
            )
            device_rhs = builder.to_layout(
                rhs,
                output_type=builder.get_metal_tensor_layout(rhs.type.shape, tiled=True),
                unit_attrs=unit_attrs,
            )
            device_out = d2m.empty(builder.get_metal_tensor_layout(shape, tiled=True))
            res = mm(device_lhs, device_rhs, device_out)
            builder.set_goldens({lhs: lhs_golden, rhs: rhs_golden}, {res: out_golden})
            return res

    compile_and_execute_ttir(
        generic_module,
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline",
        print_ir=True,
        **get_request_kwargs(request),
    )
