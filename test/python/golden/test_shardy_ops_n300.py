# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only
from collections import OrderedDict

from ttmlir.ir import StringAttr
from builder.base.builder import Operand, Shape, TypeInfo
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_utils import compile_stablehlo_to_flatbuffer
from test_utils import Marks, shape_str

pytestmark = pytest.mark.n300


def sharding_constraint(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    tensor_sharding_attr = builder.tensor_sharding_attr(
        mesh_name="mesh",
        dimension_shardings=[
            builder.dimension_sharding_attr(
                axes=[builder.axis_ref_attr(name="x")],
                is_closed=True,
            ),
            builder.dimension_sharding_attr(
                axes=[builder.axis_ref_attr(name="y")],
                is_closed=False,
            ),
        ],
    )

    builder.sharding_constraint(in0, tensor_sharding_attr=tensor_sharding_attr)
    return builder.add(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize(
    "test_fn",
    [
        sharding_constraint,
    ],
)
def test_sharding_constraint(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    request,
):
    compile_stablehlo_to_flatbuffer(
        test_fn,
        inputs_shapes=[shape, shape],
        inputs_types=[dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_dict=OrderedDict([("x", 1), ("y", 2)]),
    )


def reshard(
    in0: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    tensor_sharding_attr = builder.tensor_sharding_attr(
        mesh_name="mesh",
        dimension_shardings=[
            builder.dimension_sharding_attr(
                axes=[builder.axis_ref_attr(name="y")],
                is_closed=True,
            ),
            builder.dimension_sharding_attr(
                axes=[builder.axis_ref_attr(name="x")],
                is_closed=False,
            ),
            builder.dimension_sharding_attr(
                axes=[],
                is_closed=True,
            ),
        ],
    )

    builder.sharding_constraint(in0, tensor_sharding_attr=tensor_sharding_attr)
    out_sharding = builder.tensor_sharding_attr(
        mesh_name="mesh",
        dimension_shardings=[
            builder.dimension_sharding_attr(
                axes=[],
                is_closed=True,
            ),
            builder.dimension_sharding_attr(
                axes=[builder.axis_ref_attr(name="x")],
                is_closed=True,
            ),
            builder.dimension_sharding_attr(
                axes=[builder.axis_ref_attr(name="y")],
                is_closed=True,
            ),
        ],
    )
    print(f"out_sharding: {out_sharding}")
    return builder.reshard(in0, sharding=out_sharding)


@pytest.mark.parametrize("shape", [(8, 8, 8)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize(
    "test_fn",
    [
        reshard,
    ],
)
def test_reshard(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    request,
):
    compile_stablehlo_to_flatbuffer(
        test_fn,
        inputs_shapes=[shape],
        inputs_types=[dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_dict=OrderedDict([("x", 1), ("y", 2)]),
    )


def manual_computation(
    in0: Operand,
    in1: Operand,
    builder: StableHLOBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    dim_sharding_none = builder.dimension_sharding_attr(
        axes=[],
        is_closed=True,
    )
    dim_sharding_x = builder.dimension_sharding_attr(
        axes=[builder.axis_ref_attr(name="x")],
        is_closed=True,
    )
    dim_sharding_y = builder.dimension_sharding_attr(
        axes=[builder.axis_ref_attr(name="y")],
        is_closed=False,
    )
    in_shardings = builder.tensor_sharding_per_value_attr(
        [
            builder.tensor_sharding_attr(
                mesh_name="mesh",
                dimension_shardings=[dim_sharding_x, dim_sharding_y],
            ),
            builder.tensor_sharding_attr(
                mesh_name="mesh",
                dimension_shardings=[dim_sharding_y, dim_sharding_none],
            ),
        ]
    )
    out_shardings = builder.tensor_sharding_per_value_attr(
        [
            builder.tensor_sharding_attr(
                mesh_name="mesh",
                dimension_shardings=[dim_sharding_x, dim_sharding_none],
            ),
        ]
    )
    manual_axes = builder.manual_axes_attr([StringAttr.get("y"), StringAttr.get("x")])
    rtt = builder.create_ranked_tensor_type((128, 128))
    builder.manual_computation(
        [rtt],
        [in0, in1],
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        manual_axes=manual_axes,
    )
    return builder.add(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize(
    "test_fn",
    [
        manual_computation,
    ],
)
def test_manual_computation(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    request,
):
    compile_stablehlo_to_flatbuffer(
        test_fn,
        inputs_shapes=[shape, shape],
        inputs_types=[dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_dict=OrderedDict([("x", 1), ("y", 2)]),
    )
