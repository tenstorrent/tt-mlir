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
from test_utils import Marks, shape_str, sharding_str

pytestmark = pytest.mark.frontend("shlo")


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_sharding_constraint(
    shape: Shape, dtype: torch.dtype, target: str, request, device
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
        mesh_dict=OrderedDict([("x", 1), ("y", 1)]),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(2, 4, 8, 16)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("mesh_shape", [(1, 1)], ids=["1x2"])
def test_op_sharding_annotation(
    shape: Shape,
    dtype: torch.dtype,
    mesh_shape: Tuple[int, int],
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def op_sharding_annotation(
            in0: Operand,
            in1: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
            sharding_attr = builder.tensor_sharding_attr(
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
                    builder.dimension_sharding_attr(
                        axes=[],
                        is_closed=False,
                    ),
                ],
            )

            return builder.add(
                in0,
                in1,
                sharding_attr=builder.tensor_sharding_per_value_attr([sharding_attr]),
            )

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
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
    def module(builder: StableHLOBuilder):
        @builder.func([shape, shape], [dtype, dtype])
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
            builder.set_arg_attribute(in0, "sdy.sharding", tensor_sharding_attr)
            return builder.add(in0, in1)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
    )


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("mesh_shape", [(1, 1)])
def test_manual_computation_op(
    shape: Shape,
    dtype: torch.dtype,
    mesh_shape: Tuple[int, int],
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def my_modela(in0: Operand, in1: Operand, builder: StableHLOBuilder):
            def single_device_func(
                inner0: Operand, inner1: Operand, builder: StableHLOBuilder
            ):
                add0 = builder.add(inner0, inner1)
                add1 = builder.add(add0, inner1)
                cosine0 = builder.cosine(add1)
                sin0 = builder.sine(add1)
                return cosine0, sin0

            tensor_sharding_attr = builder.tensor_sharding_attr(
                mesh_name="mesh",
                dimension_shardings=[
                    builder.dimension_sharding_attr(
                        axes=[],
                        is_closed=True,
                    ),
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

            manual_computation_op0, manual_computation_op1 = builder.manual_computation(
                single_device_func,
                [in0, in1],
                in_shardings=[tensor_sharding_attr, tensor_sharding_attr],
                out_shardings=[tensor_sharding_attr, tensor_sharding_attr],
                manual_axes=["x", "y"],
            )
            return manual_computation_op0, manual_computation_op1

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_reshard_op(shape: Shape, dtype: torch.dtype, target: str, request, device):
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

            builder.reshard(in0, tensor_sharding_attr=tensor_sharding_attr)
            return builder.add(in0, in1)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", 1), ("y", 1)]),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(2, 4, 8, 16)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("mesh_shape", [(1, 1)], ids=["1x2"])
def test_op_sharding_annotation_with_priority(
    shape: Shape,
    dtype: torch.dtype,
    mesh_shape: Tuple[int, int],
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def op_sharding_annotation(
            in0: Operand,
            in1: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
            sharding_attr = builder.tensor_sharding_attr(
                mesh_name="mesh",
                dimension_shardings=[
                    builder.dimension_sharding_attr(
                        axes=[builder.axis_ref_attr(name="x")],
                        is_closed=True,
                        priority=10,
                    ),
                    builder.dimension_sharding_attr(
                        axes=[builder.axis_ref_attr(name="y")],
                        is_closed=False,
                    ),
                    builder.dimension_sharding_attr(
                        axes=[],
                        is_closed=False,
                    ),
                    builder.dimension_sharding_attr(
                        axes=[],
                        is_closed=False,
                    ),
                ],
            )

            return builder.add(
                in0,
                in1,
                sharding_attr=builder.tensor_sharding_per_value_attr([sharding_attr]),
            )

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
    )


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("mesh_shape", [(1, 1)], ids=["1x2"])
def test_sdy_all_gather(
    shape: Shape,
    dtype: torch.dtype,
    mesh_shape: Tuple[int, int],
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def my_modela(in0: Operand, in1: Operand, builder: StableHLOBuilder):
            tensor_sharding_attr0 = builder.tensor_sharding_attr(
                mesh_name="mesh",
                dimension_shardings=[
                    builder.dimension_sharding_attr(
                        axes=[],
                        is_closed=True,
                    ),
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
            add0 = builder.add(
                in0,
                in1,
                sharding_attr=builder.tensor_sharding_per_value_attr(
                    [tensor_sharding_attr0]
                ),
            )

            tensor_sharding_attr1 = builder.tensor_sharding_attr(
                mesh_name="mesh",
                dimension_shardings=[
                    builder.dimension_sharding_attr(
                        axes=[],
                        is_closed=True,
                    ),
                    builder.dimension_sharding_attr(
                        axes=[],
                        is_closed=True,
                    ),
                    builder.dimension_sharding_attr(
                        axes=[],
                        is_closed=True,
                    ),
                    builder.dimension_sharding_attr(
                        axes=[],
                        is_closed=True,
                    ),
                ],
            )
            axes_ref_list0 = builder.axes_ref_list_attr(axis_ref_list=[])
            axes_ref_list1 = builder.axes_ref_list_attr(axis_ref_list=[])
            axes_ref_list2 = builder.axes_ref_list_attr(
                axis_ref_list=[builder.axis_ref_attr(name="x")]
            )
            axes_ref_list3 = builder.axes_ref_list_attr(
                axis_ref_list=[builder.axis_ref_attr(name="y")]
            )
            gathering_axes = builder.list_of_axis_ref_lists_attr(
                [axes_ref_list0, axes_ref_list1, axes_ref_list2, axes_ref_list3]
            )
            sdy_all_gather0 = builder.sdy_all_gather(
                add0, gathering_axes, tensor_sharding_attr1
            )
            return sdy_all_gather0

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        device=device,
    )
