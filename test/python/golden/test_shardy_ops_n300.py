# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only
from collections import OrderedDict

from builder.base.builder import Operand, Shape, TypeInfo
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_utils import compile_stablehlo_to_flatbuffer
from test_utils import Marks, shape_str

pytestmark = [pytest.mark.frontend("shlo"), pytest.mark.n300]


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
            builder.dimension_sharding_attr(
                axes=[],
                is_closed=False,
            ),
        ],
    )
    tensor_sharding_attr2 = builder.tensor_sharding_attr(
        mesh_name="mesh",
        dimension_shardings=[
            builder.dimension_sharding_attr(
                axes=[],
                is_closed=True,
            ),
            builder.dimension_sharding_attr(
                axes=[builder.axis_ref_attr(name="y")],
                is_closed=False,
            ),
            builder.dimension_sharding_attr(
                axes=[builder.axis_ref_attr(name="x")],
                is_closed=False,
            ),
        ],
    )
    dim_mapping_attr = builder.dim_mapping_attr([0])
    dim_mapping_attr1 = builder.dim_mapping_attr([1])
    dim_mapping_attr2 = builder.dim_mapping_attr([2])
    tensor_mapping_attr = builder.tensor_mapping_attr(
        [dim_mapping_attr, dim_mapping_attr2]
    )
    tensor_mapping_attr1 = builder.tensor_mapping_attr(
        [dim_mapping_attr2, dim_mapping_attr1]
    )
    tensor_mapping_attr2 = builder.tensor_mapping_attr(
        [dim_mapping_attr, dim_mapping_attr1]
    )
    # builder.op_sharding_rule_attr([8,16,8], [tensor_mapping_attr, tensor_mapping_attr1], [tensor_mapping_attr2])
    # print(op_sharding_rule_attr)
    builder.sharding_constraint(in0, tensor_sharding_attr=tensor_sharding_attr)
    add_0 = builder.add(in0, in1, unit_attrs=["sdy.sharding"])
    # builder.tensor_sharding_per_value_attr([tensor_sharding_attr, tensor_sharding_attr2])#, [in0, in1])
    return add_0


@pytest.mark.parametrize("shape", [(8, 8, 8)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
@pytest.mark.parametrize(
    "test_fn",
    [
        sharding_constraint,
    ],
)
def test_sharding_constraint3(
    test_fn: Callable,
    shape: Shape,
    dtype: torch.dtype,
    mesh_shape: Tuple[int, int],
    request,
):
    compile_stablehlo_to_flatbuffer(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
    )
