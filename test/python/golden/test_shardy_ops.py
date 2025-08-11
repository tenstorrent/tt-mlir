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
from builder.base.builder_utils import build_stablehlo_module
from test_utils import Marks, shape_str

from ttmlir.passes import (
    stablehlo_pipeline,
    stablehlo_to_ttir_pipeline,
)


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
    module, shlo_builder = build_stablehlo_module(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", 1), ("y", 2)]),
    )

    stablehlo_pipeline(module)
    stablehlo_to_ttir_pipeline(module)
