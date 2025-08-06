# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
import subprocess
import torch
import pytest
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict
from collections import OrderedDict

from ttmlir.dialects import func, sdy

from builder.base.builder import *
from builder.stablehlo.stablehlo_builder import StableHLOBuilder

# ----- Private APIs -----


def _get_target_path(output_path, filename, target):
    target_dir = os.path.join(output_path, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return os.path.join(target_dir, filename)


# ----- Public APIs -----


def build_stablehlo_module(
    fn: Callable,
    inputs_shapes: List[Shape],
    inputs_types: Optional[List[Union[torch.dtype, TypeInfo]]] = None,
    mesh_name: str = "mesh",
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    module_dump: bool = False,
    base: Optional[str] = None,
    output_root: str = ".",
):
    ctx = Context()

    # Grab the location of the test function in python for later debugging
    try:
        fname = inspect.getfile(fn)
        line_no = inspect.getsourcelines(fn)[1]
        loc = Location.file(fname, line_no, 0, ctx)
    except (OSError, TypeError):
        loc = Location.unknown(ctx)

    # Instantiate builder which is passed as the last argument to
    # `fn` so the user can use it to build ops.
    stablehlo_builder = StableHLOBuilder(ctx, loc)

    # Default to all f32s
    if inputs_types is None:
        inputs_types = [torch.float32] * len(inputs_shapes)

    if len(inputs_shapes) != len(inputs_types):
        raise ValueError(
            f"inputs_shapes and inputs_types must have the same length: "
            f"{len(inputs_shapes)} != {len(inputs_types)}"
        )

    with ctx, loc:
        fn_input_types = [
            stablehlo_builder._create_ranked_tensor_type(
                shape,
                stablehlo_builder._get_type_from_torch_dtype(
                    dtype if isinstance(dtype, torch.dtype) else dtype
                ),
            )
            for (shape, dtype) in zip(inputs_shapes, inputs_types)
        ]

        # Wrap everything in a mlir module.
        module = Module.create()
        module.body.append(
            stablehlo_builder.mesh(
                mesh_name=mesh_name,
                mesh_attr=stablehlo_builder._create_mesh_attr_from_ordered_dict(
                    mesh_dict
                ),
            )
        )

        with InsertionPoint(module.body):
            # Wrap everything in a mlir function.
            @func.func(*fn_input_types, name=fn.__name__)
            def decorated_func(*inputs):
                # Randomly generate golden tensors for function inputs.
                input_goldens = []
                for index, (operand, dtype) in enumerate(zip(inputs, inputs_types)):
                    input_goldens.append(
                        stablehlo_builder._generate_input_golden(
                            operand, dtype, index
                        ).tensor
                    )
                result = fn(*inputs, stablehlo_builder)
                output_ops = result if hasattr(result, "__iter__") else (result,)
                output_goldens = [
                    stablehlo_builder._get_golden_tensor(op) for op in output_ops
                ]
                stablehlo_builder.set_graph_input_output(input_goldens, output_goldens)
                return result

        print(f"`{fn.__name__}` sucessfully transformed into a MLIR module.")

        base = fn.__name__ if base is None else base

        filename = _get_target_path(output_root, base + "_shlo.mlir", "shlo")

        if module_dump:
            with open(filename, "w") as f:
                f.write(str(module))
                print(module)

        return module, stablehlo_builder
