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
from ttmlir.util import debug_print_module


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

    ttir_debug = debug_print_module(module)
    with open(f"{function_name}_ttir.mlir", "w") as file:
        file.write(ttir_debug)

    ttir_to_ttnn_backend_pipeline(module, f"system-desc-path={SYSTEM_DESC_PATH}")

    ttnn_debug = debug_print_module(module)
    with open(f"{function_name}_ttnn.mlir", "w") as file:
        file.write(ttnn_debug)

    flatten_golden_map = {}

    for loc, tensor in golden_map.items():
        flatten_golden_map[loc] = tensor.flatten()

    flatbuffer_golden_map = create_flatbuffer_golden_map(flatten_golden_map)
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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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

            input_shape_list = [(1, 12, 32, 100)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

            @func.func(*input_operands, name=f"{function_name}")
            def transpose(inputs):
                ttir_op_res, golden_dict = create_transpose(
                    inputs, [(1, 32, 12, 100)], -3, -2, golden_inputs
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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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

            input_shape_list = [(1, 32, 12, 50), (1, 32, 12, 50)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

            @func.func(*input_operands, name=f"{function_name}")
            def concat(input_one, input_two):
                ttir_op_res, golden_dict = create_concat(
                    input_one, input_two, [(1, 32, 12, 100)], -1, golden_inputs
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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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

            input_shape_list = [(1, 12, 3200)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

            @func.func(*input_operands, name=f"{function_name}")
            def reshape(inputs):
                ttir_op_res, golden_dict = create_reshape(
                    inputs, [(12, 3200)], golden_inputs
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

            input_shape_list = [(32, 12, 12), (32, 12, 100)]

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

            @func.func(*input_operands, name=f"{function_name}")
            def matmul(input_one, input_two):
                ttir_op_res, golden_dict = create_matmul(
                    input_one, input_two, [(32, 12, 100)], golden_inputs
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
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

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

            input_shape_list = [(1, 12, 3200),      # arg0
                                (1, 1, 12, 12),     # arg1
                                (1, 12),            # arg2
                                (1, 50, 1),         # arg3
                                (1, 32, 50, 100),   # arg4
                                (1, 1),             # arg5
                                (1, 32, 50, 100),   # arg6
                                (1, 32, 50, 100),   # arg7
                                (1, 1),             # arg8
                                (1, 32, 50, 100),   # arg9
                                (1, 1),             # arg10
                                (3200, 3200),       # arg11
                                (3200, 3200),       # arg12
                                (3200, 3200),       # arg13
                                (3200, 3200)]       # arg14

            input_operands = []
            for shape in input_shape_list:
                input_operands.append(RankedTensorType.get(shape, F32Type.get()))

            golden_inputs = []
            for index, shape in enumerate(input_shape_list):
                torch_input_golden = torch.randn(shape, dtype=torch.float32)
                golden_inputs.append(torch_input_golden)
                golden_map[f"input_{index}"] = torch_input_golden.flatten()

            @func.func(*input_operands, name=f"{function_name}")
            def llama_attention(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14):
                
                output1, golden_dict1 = create_squeeze(arg0, [(12, 3200)], 0, [golden_inputs[0]])
                golden_map[golden_dict1["location"]] = golden_dict1["golden_output"]

                output3, golden_dict3 = create_matmul(output1, arg11, [(12, 3200)], [golden_dict1["golden_output"], golden_inputs[11]])
                golden_map[golden_dict3["location"]] = golden_dict3["golden_output"]

                output5, golden_dict5 = create_reshape(output3, [(1, 12, 32, 100)], [golden_dict3["golden_output"]])
                golden_map[golden_dict5["location"]] = golden_dict5["golden_output"]

                output7, golden_dict7 = create_transpose(output5, [(1, 32, 12, 100)], -3, -2, [golden_dict5["golden_output"]])
                golden_map[golden_dict7["location"]] = golden_dict7["golden_output"]

                output9, golden_dict9 = create_unsqueeze(arg2, [(1, 1, 12)], 1, [golden_inputs[2]])
                golden_map[golden_dict9["location"]] = golden_dict9["golden_output"]

                output11, golden_dict11 = create_matmul(arg3, output9, [(1, 50, 12)], [golden_inputs[3], golden_dict9["golden_output"]])
                golden_map[golden_dict11["location"]] = golden_dict11["golden_output"]

                output13, golden_dict13 = create_transpose(output11, [(1, 12, 50)], -2, -1, [golden_dict11["golden_output"]])
                golden_map[golden_dict13["location"]] = golden_dict13["golden_output"]

                output15, golden_dict15 = create_concat(output13, output13, [(1, 12, 100)], -1, [golden_dict13["golden_output"], golden_dict13["golden_output"]])
                golden_map[golden_dict15["location"]] = golden_dict15["golden_output"]

                output17, golden_dict17 = create_cos(output15, [(1, 12, 100)], [golden_dict15["golden_output"]])
                golden_map[golden_dict17["location"]] = golden_dict17["golden_output"]

                output19, golden_dict19 = create_unsqueeze(output17, [(1, 1, 12, 100)], 1, [golden_dict17["golden_output"]])
                golden_map[golden_dict19["location"]] = golden_dict19["golden_output"]

                output21, golden_dict21 = create_multiply(output7, output19, [(1, 32, 12, 100)], [golden_dict7["golden_output"], golden_dict19["golden_output"]])
                golden_map[golden_dict21["location"]] = golden_dict21["golden_output"]

                output23, golden_dict23 = create_transpose(output7, [(1, 32, 100, 12)], -2, -1, [golden_dict7["golden_output"]])
                golden_map[golden_dict23["location"]] = golden_dict23["golden_output"]

                output25, golden_dict25 = create_matmul(arg4, output23, [(1, 32, 50, 12)], [golden_inputs[4], golden_dict23["golden_output"]])
                golden_map[golden_dict25["location"]] = golden_dict25["golden_output"]

                output27, golden_dict27 = create_transpose(output25, [(1, 32, 12, 50)], -2, -1, [golden_dict25["golden_output"]])
                golden_map[golden_dict27["location"]] = golden_dict27["golden_output"]

                output29, golden_dict29 = create_multiply(output27, arg5, [(1, 32, 12, 50)], [golden_dict27["golden_output"], golden_inputs[5]])
                golden_map[golden_dict29["location"]] = golden_dict29["golden_output"]
                
                output31, golden_dict31 = create_transpose(output7, [(1, 32, 100, 12)], -2, -1, [golden_dict7["golden_output"]])
                golden_map[golden_dict31["location"]] = golden_dict31["golden_output"]

                output33, golden_dict33 = create_matmul(arg6, output31, [(1, 32, 50, 12)], [golden_inputs[6], golden_dict31["golden_output"]])
                golden_map[golden_dict33["location"]] = golden_dict33["golden_output"]

                output35, golden_dict35 = create_transpose(output33, [(1, 32, 12, 50)], -2, -1, [golden_dict33["golden_output"]])
                golden_map[golden_dict35["location"]] = golden_dict35["golden_output"]

                output37, golden_dict37 = create_concat(output29, output35, [(1, 32, 12, 100)], -1, [golden_dict29["golden_output"], golden_dict35["golden_output"]])
                golden_map[golden_dict37["location"]] = golden_dict37["golden_output"]

                output39, golden_dict39 = create_sin(output15, [(1, 12, 100)], [golden_dict15["golden_output"]])
                golden_map[golden_dict39["location"]] = golden_dict39["golden_output"]

                output41, golden_dict41 = create_unsqueeze(output39, [(1, 1, 12, 100)], 1, [golden_dict39["golden_output"]])
                golden_map[golden_dict41["location"]] = golden_dict41["golden_output"]

                output43, golden_dict43 = create_multiply(output37, output41, [(1, 32, 12, 100)], [golden_dict37["golden_output"], golden_dict41["golden_output"]])
                golden_map[golden_dict43["location"]] = golden_dict43["golden_output"]

                output45, golden_dict45 = create_add(output21, output43, [(1, 32, 12, 100)], [golden_dict21["golden_output"], golden_dict43["golden_output"]])
                golden_map[golden_dict45["location"]] = golden_dict45["golden_output"]

                output47, golden_dict47 = create_squeeze(output45, [(32, 12, 100)], 0, [golden_dict45["golden_output"]])
                golden_map[golden_dict47["location"]] = golden_dict47["golden_output"]

                output49, golden_dict49 = create_matmul(output1, arg12, [(12, 3200)], [golden_dict1["golden_output"], golden_inputs[12]])
                golden_map[golden_dict49["location"]] = golden_dict49["golden_output"]

                output51, golden_dict51 = create_reshape(output49, [(1, 12, 32, 100)], [golden_dict49["golden_output"]])
                golden_map[golden_dict51["location"]] = golden_dict51["golden_output"]

                output53, golden_dict53 = create_transpose(output51, [(1, 32, 12, 100)], -3, -2, [golden_dict51["golden_output"]])
                golden_map[golden_dict53["location"]] = golden_dict53["golden_output"]

                output55, golden_dict55 = create_multiply(output53, output19, [(1, 32, 12, 100)], [golden_dict53["golden_output"], golden_dict19["golden_output"]])
                golden_map[golden_dict55["location"]] = golden_dict55["golden_output"]

                output57, golden_dict57 = create_transpose(output53, [(1, 32, 100, 12)], -2, -1, [golden_dict53["golden_output"]])
                golden_map[golden_dict57["location"]] = golden_dict57["golden_output"]

                output59, golden_dict59 = create_matmul(arg7, output57, [(1, 32, 50, 12)], [golden_inputs[7], golden_dict57["golden_output"]])
                golden_map[golden_dict59["location"]] = golden_dict59["golden_output"]

                output61, golden_dict61 = create_transpose(output59, [(1, 32, 12, 50)], -2, -1, [golden_dict59["golden_output"]])
                golden_map[golden_dict61["location"]] = golden_dict61["golden_output"]

                output63, golden_dict63 = create_multiply(output61, arg8, [(1, 32, 12, 50)], [golden_dict61["golden_output"], golden_inputs[8]])
                golden_map[golden_dict63["location"]] = golden_dict63["golden_output"]

                output65, golden_dict65 = create_transpose(output53, [(1, 32, 100, 12)], -2, -1, [golden_dict53["golden_output"]])
                golden_map[golden_dict65["location"]] = golden_dict65["golden_output"]

                output67, golden_dict67 = create_matmul(arg9, output65, [(1, 32, 50, 12)], [golden_inputs[9], golden_dict65["golden_output"]])
                golden_map[golden_dict67["location"]] = golden_dict67["golden_output"]

                output69, golden_dict69 = create_transpose(output67, [(1, 32, 12, 50)], -2, -1, [golden_dict67["golden_output"]])
                golden_map[golden_dict69["location"]] = golden_dict69["golden_output"]

                output71, golden_dict71 = create_concat(output63, output69, [(1, 32, 12, 100)], -1, [golden_dict63["golden_output"], golden_dict69["golden_output"]])
                golden_map[golden_dict71["location"]] = golden_dict71["golden_output"]

                output73, golden_dict73 = create_multiply(output71, output41, [(1, 32, 12, 100)], [golden_dict71["golden_output"], golden_dict41["golden_output"]])
                golden_map[golden_dict73["location"]] = golden_dict73["golden_output"]

                output75, golden_dict75 = create_add(output55, output73, [(1, 32, 12, 100)], [golden_dict55["golden_output"], golden_dict73["golden_output"]])
                golden_map[golden_dict75["location"]] = golden_dict75["golden_output"]

                output77, golden_dict77 = create_squeeze(output75, [(32, 12, 100)], 0, [golden_dict75["golden_output"]])
                golden_map[golden_dict77["location"]] = golden_dict77["golden_output"]

                output79, golden_dict79 = create_transpose(output77, [(32, 100, 12)], -2, -1, [golden_dict77["golden_output"]])
                golden_map[golden_dict79["location"]] = golden_dict79["golden_output"]

                output81, golden_dict81 = create_matmul(output47, output79, [(32, 12, 12)], [golden_dict47["golden_output"], golden_dict79["golden_output"]])
                golden_map[golden_dict81["location"]] = golden_dict81["golden_output"]

                output83, golden_dict83 = create_unsqueeze(output81, [(1, 32, 12, 12)], 0, [golden_dict81["golden_output"]])
                golden_map[golden_dict83["location"]] = golden_dict83["golden_output"]

                output85, golden_dict85 = create_multiply(output83, arg10, [(1, 32, 12, 12)], [golden_dict83["golden_output"], golden_inputs[10]])
                golden_map[golden_dict85["location"]] = golden_dict85["golden_output"]

                output87, golden_dict87 = create_add(output85, arg1, [(1, 32, 12, 12)], [golden_dict85["golden_output"], golden_inputs[1]])
                golden_map[golden_dict87["location"]] = golden_dict87["golden_output"]

                output89, golden_dict89 = create_softmax(output87, [(1, 32, 12, 12)], -1, [golden_dict87["golden_output"]])
                golden_map[golden_dict89["location"]] = golden_dict89["golden_output"]

                output91, golden_dict91 = create_squeeze(output89, [(32, 12, 12)], 0, [golden_dict89["golden_output"]])
                golden_map[golden_dict91["location"]] = golden_dict91["golden_output"]

                output93, golden_dict93 = create_matmul(output1, arg13, [(12, 3200)], [golden_dict1["golden_output"], golden_inputs[13]])
                golden_map[golden_dict93["location"]] = golden_dict93["golden_output"]

                output95, golden_dict95 = create_reshape(output93, [(1, 12, 32, 100)], [golden_dict93["golden_output"]])
                golden_map[golden_dict95["location"]] = golden_dict95["golden_output"]

                output97, golden_dict97 = create_transpose(output95, [(1, 32, 12, 100)], -3, -2, [golden_dict95["golden_output"]])
                golden_map[golden_dict97["location"]] = golden_dict97["golden_output"]

                output99, golden_dict99 = create_transpose(output97, [(1, 32, 100, 12)], -2, -1, [golden_dict97["golden_output"]])
                golden_map[golden_dict99["location"]] = golden_dict99["golden_output"]

                output101, golden_dict101 = create_squeeze(output99, [(32, 100, 12)], 0, [golden_dict99["golden_output"]])
                golden_map[golden_dict101["location"]] = golden_dict101["golden_output"]

                output103, golden_dict103 = create_transpose(output101, [(32, 12, 100)], -2, -1, [golden_dict101["golden_output"]])
                golden_map[golden_dict103["location"]] = golden_dict103["golden_output"]

                output105, golden_dict105 = create_matmul(output91, output103, [(32, 12, 100)], [golden_dict91["golden_output"], golden_dict103["golden_output"]])
                golden_map[golden_dict105["location"]] = golden_dict105["golden_output"]

                output107, golden_dict107 = create_unsqueeze(output105, [(1, 32, 12, 100)], 0, [golden_dict105["golden_output"]])
                golden_map[golden_dict107["location"]] = golden_dict107["golden_output"]

                output109, golden_dict109 = create_transpose(output107, [(1, 12, 32, 100)], -3, -2, [golden_dict107["golden_output"]])
                golden_map[golden_dict109["location"]] = golden_dict109["golden_output"]

                output111, golden_dict111 = create_reshape(output109, [(12, 3200)], [golden_dict109["golden_output"]])
                golden_map[golden_dict111["location"]] = golden_dict111["golden_output"]

                output113, golden_dict113 = create_matmul(output111, arg14, [(12, 3200)], [golden_dict111["golden_output"], golden_inputs[14]])
                golden_map[golden_dict113["location"]] = golden_dict113["golden_output"]

                output115, golden_dict115 = create_unsqueeze(output113, [(1, 12, 3200)], 0, [golden_dict113["golden_output"]])
                golden_map[golden_dict115["location"]] = golden_dict115["golden_output"]

                return output115

        module_post_processing(module, function_name, golden_map)


#test_llama_attention()


#test_squeeze() #pass
test_matmul()  #pass
#test_reshape() #pass
#test_unsqueeze()  #pass
#test_cos()  #pass
#test_multiply() #pass
#test_sin() #pass
#test_add() #pass
#test_concat()  #pass
#test_transpose() #actual_pcc=0.00531
