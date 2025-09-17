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

pytestmark = pytest.mark.frontend("shlo")


@pytest.mark.parametrize("x_constraint", [True])
@pytest.mark.parametrize("y_constraint", [False])
@pytest.mark.parametrize("shape", [(8, 8)], ids=shape_str)
@pytest.mark.parametrize("shape2", [(8, 4, 2)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_sharding_constraint(
    shape: Shape,
    shape2: Shape,
    dtype: torch.dtype,
    x_constraint: bool,
    y_constraint: bool,
    target: str,
    request,
):
    def sharding_constraint(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        y0 = builder.sub_axis_info_attr(pre_size=1, size=2)
        y1 = builder.sub_axis_info_attr(pre_size=2, size=2)
        builder.set_graph_level_check(True)
        tensor_sharding_attr = builder.tensor_sharding_attr(
            mesh_name="mesh",
            dimension_shardings=[
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="x")],
                    is_closed=x_constraint,
                ),
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="y")],
                    is_closed=y_constraint,
                ),
            ],
        )
        tensor_sharding_attr2 = builder.tensor_sharding_attr(
            mesh_name="mesh",
            dimension_shardings=[
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="x")],
                    is_closed=x_constraint,
                ),
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="y", sub_axis_info_attr=y0)],
                    is_closed=True,
                ),
                builder.dimension_sharding_attr(
                    axes=[builder.axis_ref_attr(name="y", sub_axis_info_attr=y1)],
                    is_closed=True,
                ),
            ],
        )
        builder.sharding_constraint(in0, tensor_sharding_attr=tensor_sharding_attr)
        add0 = builder.add(in0, in1, unit_attrs=unit_attrs)
        # builder.sharding_constraint(in2, tensor_sharding_attr=tensor_sharding_attr2)
        # reshape0 = builder.reshape(add0, in2, unit_attrs=unit_attrs)
        return add0

    compile_stablehlo_to_flatbuffer(
        sharding_constraint,
        [shape, shape, shape2],
        [dtype, dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", 1), ("y", 2)]),
        target=target,
    )
