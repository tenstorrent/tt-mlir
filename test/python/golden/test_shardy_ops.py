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
from builder.base.builder_utils import compile_and_execute_shlo
from test_utils import Marks, shape_str, sharding_str

pytestmark = pytest.mark.frontend("shlo")


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
    return builder.add(in0, in1, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize(
    "test_fn",
    [
        sharding_constraint,
    ],
)
def test_sharding_constraint(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_shlo(
        test_fn,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", 1), ("y", 1)]),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(2, 4, 8, 16)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("mesh_shape", [(1, 1)], ids=["1x2"])
@pytest.mark.parametrize(
    "sharding",
    [
        [("x", True), ("y", False), ("", False), ("", False)],
    ],
    ids=sharding_str,
)
def test_op_sharding_annotation(
    shape: Shape,
    dtype: torch.dtype,
    mesh_shape: Tuple[int, int],
    sharding: List[Tuple[str, bool]],
    request,
    device,
):
    def op_sharding_annotation(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        sharding_attr = builder.create_sharding_attr_from_tuples("mesh", sharding)
        return builder.add(in0, in1, unit_attrs=unit_attrs, sharding_attr=sharding_attr)

    compile_and_execute_shlo(
        op_sharding_annotation,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("mesh_shape", [(1, 1)])
def test_input_annotation(
    shape: Shape,
    dtype: torch.dtype,
    mesh_shape: Tuple[int, int],
    request,
    device,
):
    def input_annotation(
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
                    is_closed=True,
                ),
            ],
        )
        builder.arg_attrs[in0] = {"sdy.sharding": tensor_sharding_attr}
        return builder.add(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_shlo(
        input_annotation,
        [shape, shape],
        [dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
    )
