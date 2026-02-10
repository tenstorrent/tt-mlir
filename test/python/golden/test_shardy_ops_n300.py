# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only, get_request_kwargs
from collections import OrderedDict

from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_apis import compile_and_execute_shlo
from test_utils import Marks, shape_str

pytestmark = [pytest.mark.frontend("shlo"), pytest.mark.n300]


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_sharding_constraint(
    shape: Shape,
    dtype: torch.dtype,
    mesh_shape: Tuple[int, int],
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def sharding_constraint(
            in0: Operand,
            in1: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
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
            return builder.add(in0, in1)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
    )
