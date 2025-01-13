# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import inspect
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable, Dict
from ttmlir import util
from ttmlir.ir import *
from ttmlir.dialects import ttir, tt, ttnn, func, tensor
from ttmlir.passes import *
import torch
from generator_ops import *


def create_flatbuffer_golden_map(golden_map):
    golden_info = {}
    for name, golden_tensor in golden_map.items():
        golden_info[name] = create_golden_tensor(
            name,
            list(golden_tensor.shape),
            list(golden_tensor.stride()),
            DataType.Float32,
            golden_tensor.data_ptr(),
        )
    return golden_info


def print_module(module):
    for op in module.body.operations:
        for region in op.regions:
            for block in region.blocks:
                for op in block.operations:
                    print("-----------------")
                    print(op)
                    print(op.location)
                    print(util.get_loc_name(op.location))
                    print(util.get_loc_full(op.location))

                    for operand in op.operands:
                        print(operand)

                    print("-----------------")


def module_post_processing(module, function_name, golden_map={}):
    print("xxxxxxxxxxxxxxxxxxxxTTIRxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(module)
    print("xxxxxxxxxxxxxxxxxxxxTTIRxxxxxxxxxxxxxxxxxxxxxxxxx")

    print("xxxxxxxxxxxxxxxxxxxxTTIRxxxxxxxxxxxxxxxxxxxxxxxxx")
    print_module(module)
    print("xxxxxxxxxxxxxxxxxxxxTTIRxxxxxxxxxxxxxxxxxxxxxxxxx")

    with open(f"{function_name}_ttir.mlir", "w") as file:
        file.write(str(module))

    ttir_to_ttnn_backend_pipeline(module, f"system-desc-path={SYSTEM_DESC_PATH}")

    print("xxxxxxxxxxxxxxxxxxxxTTNNxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(module)
    print("xxxxxxxxxxxxxxxxxxxxTTNNxxxxxxxxxxxxxxxxxxxxxxxxx")

    print("xxxxxxxxxxxxxxxxxxxxTTNNxxxxxxxxxxxxxxxxxxxxxxxxx")
    print_module(module)
    print("xxxxxxxxxxxxxxxxxxxxTTNNxxxxxxxxxxxxxxxxxxxxxxxxx")

    with open(f"{function_name}_ttnn.mlir", "w") as file:
        file.write(str(module))

    flatbuffer_golden_map = create_flatbuffer_golden_map(golden_map)
    ttnn_to_flatbuffer_file(module, f"{function_name}.ttnn", flatbuffer_golden_map)


def run_ttir_to_ttir_decomposition_pass(module, dump=False):
    ttir_to_ttir_decomposition_pass(module)
    if dump:
        print_module(module)


def run_ttir_load_system_desc(module, dump=False):
    ttir_load_system_desc(module, f"system-desc-path={SYSTEM_DESC_PATH}")
    if dump:
        print_module(module)


def run_ttir_implicit_device(module, dump=False):
    ttir_implicit_device(module)
    if dump:
        print_module(module)


def run_ttir_broadcast_fold(module, dump=False):
    ttir_broadcast_fold(module)
    if dump:
        print_module(module)


def run_ttnn_layout(module, dump=False):
    ttnn_layout(module)
    if dump:
        print_module(module)


def run_convert_ttir_to_ttnn_pass(module, dump=False):
    convert_ttir_to_ttnn_pass(module)
    if dump:
        print_module(module)


def run_remove_dead_values_pass(module, dump=False):
    remove_dead_values_pass(module)
    if dump:
        print_module(module)


def run_ttnn_workarounds(module, dump=False):
    ttnn_workarounds(module)
    if dump:
        print_module(module)


def run_canonicalizer_pass(module, dump=False):
    canonicalizer_pass(module)
    if dump:
        print_module(module)


def run_ttnn_pipeline_layout_decomposition_pass(module, dump=False):
    ttnn_pipeline_layout_decomposition_pass(module)
    if dump:
        print_module(module)


def run_ttnn_pipeline_dealloc_pass(module, dump=False):
    ttnn_pipeline_dealloc_pass(module)
    if dump:
        print_module(module)


def test_relu_decomp():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def relu(inputs):

                ttir_op_res, golden_dict = create_relu(
                    inputs, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        run_ttir_to_ttir_decomposition_pass(module, False)
        run_ttir_load_system_desc(module, False)
        run_ttir_implicit_device(module, False)
        run_ttir_broadcast_fold(module, False)
        run_ttnn_layout(module, True)
        run_convert_ttir_to_ttnn_pass(module, False)
        run_remove_dead_values_pass(module, False)
        run_ttnn_workarounds(module, False)
        run_canonicalizer_pass(module, False)
        run_ttnn_pipeline_layout_decomposition_pass(module, False)
        run_ttnn_pipeline_dealloc_pass(module, False)


def test_relu():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def relu(inputs):

                ttir_op_res, golden_dict = create_relu(
                    inputs, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_sigmoid():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def sigmoid(inputs):
                ttir_op_res, golden_dict = create_sigmoid(
                    inputs, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_exp():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def exp(inputs):
                ttir_op_res, golden_dict = create_exp(
                    inputs, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_abs():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def abs(inputs):
                ttir_op_res, golden_dict = create_abs(
                    inputs, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_logical_not():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def logical_not(inputs):
                ttir_op_res, golden_dict = create_logical_not(
                    inputs, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_neg():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def neg(inputs):
                ttir_op_res, golden_dict = create_neg(
                    inputs, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_sqrt():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def sqrt(inputs):
                ttir_op_res, golden_dict = create_sqrt(
                    inputs, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_rsqrt():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def rsqrt(inputs):
                ttir_op_res, golden_dict = create_rsqrt(
                    inputs, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_reciprocal():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def reciprocal(inputs):
                ttir_op_res, golden_dict = create_reciprocal(
                    inputs, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_add():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128), (128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def add(input_one, input_two):
                ttir_op_res, golden_dict = create_add(
                    input_one, input_two, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_multiply():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128), (128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def multiply(input_one, input_two):
                ttir_op_res, golden_dict = create_multiply(
                    input_one, input_two, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_softmax():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def softmax(inputs):
                ttir_op_res, golden_dict = create_softmax(
                    inputs, [(128, 128)], 1, golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_cos():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def cos(inputs):
                ttir_op_res, golden_dict = create_cos(
                    inputs, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_sin():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def sin(inputs):
                ttir_op_res, golden_dict = create_sin(
                    inputs, [(128, 128)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_transpose():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(1, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def transpose(inputs):
                ttir_op_res, golden_dict = create_transpose(
                    inputs, [(128, 1)], 0, 1, golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_unsqueeze():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(1, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def unsqueeze(inputs):
                ttir_op_res, golden_dict = create_unsqueeze(
                    inputs, [(1, 1, 128)], 0, golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_squeeze():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(1, 1, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def squeeze(inputs):
                ttir_op_res, golden_dict = create_squeeze(
                    inputs, [(1, 128)], 0, golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_concat():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(1, 128), (1, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def concat(input_one, input_two):
                ttir_op_res, golden_dict = create_concat(
                    input_one, input_two, [(1, 128)], 0, golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_mean():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(1, 12, 3200)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def mean(inputs):
                ttir_op_res, golden_dict = create_mean(
                    inputs, [(1, 12, 1)], True, [-1], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_reshape():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(12, 3200)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def reshape(inputs):
                ttir_op_res, golden_dict = create_reshape(
                    inputs, [(1, 12, 32, 100)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_typecast():
    function_name = inspect.currentframe().f_code.co_name

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(1, 12, 3200)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            @func.func(*input_operands, name=f"{function_name}")
            def typecast(inputs):
                return create_typecast(inputs, [(1, 12, 3200)])

        module_post_processing(module, function_name, golden_map)


def test_matmul():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(12, 3200), (3200, 3200)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def matmul(input_one, input_two):
                ttir_op_res, golden_dict = create_matmul(
                    input_one, input_two, [(12, 3200)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_embedding():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(1, 12), (32000, 3200)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def embedding(input_one, input_two):
                ttir_op_res, golden_dict = create_embedding(
                    input_one, input_two, [(1, 12, 3200)], golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]
                return ttir_op_res

        module_post_processing(module, function_name, golden_map)


def test_multi_op():
    function_name = inspect.currentframe().f_code.co_name

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(128, 128)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            @func.func(*input_operands, name=f"{function_name}")
            def relu(inputs):
                res = create_relu(inputs, [(128, 128)])
                res = create_relu(res, [(128, 128)])
                res = create_relu(res, [(128, 128)])
                res = create_relu(res, [(128, 128)])
                res = create_relu(res, [(128, 128)])
                res = create_relu(res, [(128, 128)])
                res = create_relu(res, [(128, 128)])
                res = create_relu(res, [(128, 128)])
                res = create_relu(res, [(128, 128)])
                res = create_relu(res, [(128, 128)])
                res = create_relu(res, [(128, 128)])
                res = create_relu(res, [(128, 128)])
                return create_relu(res, [(128, 128)])

        module_post_processing(module, function_name, golden_map)

def test_llama_attention():
    function_name = inspect.currentframe().f_code.co_name
    golden_map = {}

    with Context() as ctx, Location.name(f"{function_name}"):
        parent_location = Location.name(f"{function_name}")
        tt.register_dialect(ctx)
        ttir.register_dialect(ctx)

        module = Module.create()
        with InsertionPoint(module.body):

            input_shape_list = [(1, 12, 3200),
                                (1, 1, 12, 12),
                                (1, 12),
                                (1, 50, 1),
                                (1, 32, 50, 100),
                                (1),
                                (1, 32, 50, 100),
                                (1, 32, 50, 100),
                                (1),
                                (1, 32, 50, 100),
                                (1),
                                (3200, 3200),
                                (3200, 3200),
                                (3200, 3200),
                                (3200, 3200)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden

            @func.func(*input_operands, name=f"{function_name}")
            def llama_attention(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14):
                ttir_op_res, golden_dict = create_squeeze(
                    inputs, [(12, 3200)], 0, golden_inputs
                )
                golden_map[golden_dict["location"]] = golden_dict["golden_output"]


                return ttir_op_res

        module_post_processing(module, function_name, golden_map)

#test_llama_attention()
#test_relu() # pcc=1
#test_sigmoid() # pcc=0.99
#test_add() # pcc=1
#test_multiply() # pcc=1
#test_softmax() #pcc=0.98965
#test_cos() #pcc=0.999999
#test_sin() #pcc=0.999999
#test_transpose() #PCC=0.03773
#test_unsqueeze() #pcc=-0.110075
#test_squeeze() #pcc=-0.1644
#test_concat() #srcTensor.volume() * srcTensor.element_size() == dstTensor.volume() * dstTensor.element_size()
#test_reshape() #Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=0)] grow to 9929504 B which is beyond max L1 size of 1499136 B
#test_matmul() #Always | FATAL    | ttnn.matmul: The width of the first tensor must be equal to the height of the second tensor (38400 != 1). The shape of first tensor was ttnn.Shape([1[32], 38400]) and the shape of second tensor was ttnn.Shape([1[32], 10240000]))
