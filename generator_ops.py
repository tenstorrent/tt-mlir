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
from ttmlir.passes import ttir_to_ttnn_backend_pipeline, ttnn_to_flatbuffer_file
import torch

Shape = Union[List[int], Tuple[int, ...]]
Operand = Union[Value, OpView, Operation]
SYSTEM_DESC_PATH = os.environ.get("SYSTEM_DESC_PATH", "")


def get_type(input: Operand):
    if isinstance(input, Value):
        type = input.type
    elif isinstance(input, OpView):
        type = input.operation.result.type
    elif isinstance(input, Operation):
        type = input.result.type
    else:
        raise TypeError(f"Invalid input {type(input)}")

    assert isinstance(type, RankedTensorType), "Only ranked tensors are supported"

    return type


def output_type_operands(output_shape_list):
    output_operands = []
    output_type_list = []

    for shape in output_shape_list:
        output_operands.append(create_empty(shape))

    for operand in output_operands:
        output_type_list.append(get_type(operand))

    return output_operands, output_type_list


empty_index = 0


def create_empty(shape):
    global empty_index

    res = tensor.EmptyOp(
        shape, F32Type.get(), loc=Location.name(f"empty_{empty_index}")
    )
    empty_index += 1
    return res


relu_index = 0


def create_relu(inputs, output_shape_list, golden_inputs):
    global relu_index
    location = f"relu_{relu_index}"

    relu_output_operands, relu_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.ReluOp(
        relu_output_type_list,
        [inputs],
        relu_output_operands,
        loc=Location.name(location),
    )
    relu_index += 1

    #print("relu-golden-pre----------start")
    #print()
    a = golden_inputs[0]
    #print(a.shape)
    #print("relu-golden-pre---------end")
    #print()
    #print()

    golden_output = torch.relu(*golden_inputs)
    golden_output.flatten()

    #print("relu-golden-post----------start")
    #print()
    #print(golden_output.shape)
    #print("relu-golden-post---------end")
    #print()
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


sigmoid_index = 0


def create_sigmoid(inputs, output_shape_list, golden_inputs):
    global sigmoid_index
    location = f"sigmoid_{sigmoid_index}"

    sigmoid_output_operands, sigmoid_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.SigmoidOp(
        sigmoid_output_type_list,
        [inputs],
        sigmoid_output_operands,
        loc=Location.name(location),
    )
    sigmoid_index += 1

    #print("sigmoid-golden-pre----------start")
    #print()
    a = golden_inputs[0]
    #print(a.shape)
    #print("sigmoid-golden-pre---------end")
    #print()
    #print()

    golden_output = torch.sigmoid(*golden_inputs)
    golden_output.flatten()

    #print("sigmoid-golden-post----------start")
    #print()
    #print(golden_output.shape)
    #print("sigmoid-golden-post---------end")
    #print()
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


exp_index = 0


def create_exp(inputs, output_shape_list, golden_inputs):
    global exp_index
    location = f"exp_{exp_index}"

    exp_output_operands, exp_output_type_list = output_type_operands(output_shape_list)
    res = ttir.ExpOp(
        exp_output_type_list,
        [inputs],
        exp_output_operands,
        loc=Location.name(location),
    )
    exp_index += 1

    golden_output = torch.exp(*golden_inputs)
    golden_output.flatten()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


abs_index = 0


def create_abs(inputs, output_shape_list, golden_inputs):
    global abs_index
    location = f"abs_{abs_index}"

    abs_output_operands, abs_output_type_list = output_type_operands(output_shape_list)
    res = ttir.AbsOp(
        abs_output_type_list,
        [inputs],
        abs_output_operands,
        loc=Location.name(location),
    )
    abs_index += 1

    golden_output = torch.abs(*golden_inputs)
    golden_output.flatten()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


logical_not_index = 0


def create_logical_not(inputs, output_shape_list, golden_inputs):
    global logical_not_index
    location = f"logical_not_{logical_not_index}"

    logical_not_output_operands, logical_not_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.LogicalNotOp(
        logical_not_output_type_list,
        [inputs],
        logical_not_output_operands,
        loc=Location.name(location),
    )
    logical_not_index += 1

    golden_output = torch.logical_not(*golden_inputs)
    golden_output.flatten()

    return res, {"location": location, "golden_output": golden_output}


neg_index = 0


def create_neg(inputs, output_shape_list, golden_inputs):
    global neg_index
    location = f"neg_{neg_index}"

    neg_output_operands, neg_output_type_list = output_type_operands(output_shape_list)
    res = ttir.NegOp(
        neg_output_type_list,
        [inputs],
        neg_output_operands,
        loc=Location.name(location),
    )
    neg_index += 1

    golden_output = torch.neg(*golden_inputs)
    golden_output.flatten()
    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


sqrt_index = 0


def create_sqrt(inputs, output_shape_list, golden_inputs):
    global sqrt_index
    location = f"sqrt_{sqrt_index}"

    sqrt_output_operands, sqrt_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.SqrtOp(
        sqrt_output_type_list,
        [inputs],
        sqrt_output_operands,
        loc=Location.name(location),
    )
    sqrt_index += 1

    golden_output = torch.sqrt(*golden_inputs)
    golden_output.flatten()
    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


rsqrt_index = 0


def create_rsqrt(inputs, output_shape_list, golden_inputs):
    global rsqrt_index
    location = f"rsqrt_{rsqrt_index}"

    rsqrt_output_operands, rsqrt_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.RsqrtOp(
        rsqrt_output_type_list,
        [inputs],
        rsqrt_output_operands,
        loc=Location.name(location),
    )
    rsqrt_index += 1

    golden_output = torch.rsqrt(*golden_inputs)
    golden_output.flatten()
    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


reciprocal_index = 0


def create_reciprocal(inputs, output_shape_list, golden_inputs):
    global reciprocal_index
    location = f"reciprocal_{reciprocal_index}"

    reciprocal_output_operands, reciprocal_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.ReciprocalOp(
        reciprocal_output_type_list,
        [inputs],
        reciprocal_output_operands,
        loc=Location.name(location),
    )
    reciprocal_index += 1

    golden_output = torch.reciprocal(*golden_inputs)
    golden_output.flatten()
    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


add_index = 0


def create_add(input_one, input_two, output_shape_list, golden_inputs):
    global add_index
    location = f"add_{add_index}"

    add_output_operands, add_output_type_list = output_type_operands(output_shape_list)
    res = ttir.AddOp(
        add_output_type_list,
        [input_one, input_two],
        add_output_operands,
        loc=Location.name(location),
    )
    add_index += 1

    #print("add-golden-pre----------start")
    #print()
    a = golden_inputs[0]
    b = golden_inputs[0]
    #print(a.shape)
    #print(b.shape)
    #print("add-golden-pre---------end")
    #print()
    #print()

    golden_output = torch.add(*golden_inputs)
    golden_output.flatten()

    #print("add-golden-post----------start")
    #print()
    #print(golden_output.shape)
    #print("add-golden-post---------end")
    #print()
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


multiply_index = 0


def create_multiply(input_one, input_two, output_shape_list, golden_inputs):
    global multiply_index
    location = f"multiply_{multiply_index}"

    multiply_output_operands, multiply_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.MultiplyOp(
        multiply_output_type_list,
        [input_one, input_two],
        multiply_output_operands,
        loc=Location.name(location),
    )
    multiply_index += 1

    #print("multiply-golden-pre-----------start")
    #print()
    a = golden_inputs[0]
    b = golden_inputs[1]
    #print(a.shape)
    #print(b.shape)
    #print("multiply-golden-pre-----------end")
    #print()

    golden_output = torch.multiply(*golden_inputs)
    golden_output.flatten()

    #print("multiply-golden-post----------start")
    #print()
    #print(golden_output.shape)
    #print("multiply-golden-post---------end")
    #print()
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


softmax_index = 0


def create_softmax(inputs, output_shape_list, dim, golden_inputs):
    global softmax_index
    location = f"softmax_{softmax_index}"

    softmax_output_operands, softmax_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.SoftmaxOp(
        softmax_output_type_list[0],
        inputs,
        softmax_output_operands[0],
        dim,
        loc=Location.name(location),
    )
    softmax_index += 1

    #print("softmax-golden-pre-----------start")
    #print()
    a = golden_inputs[0]
    #print(a.shape)
    #print("softmax-golden-pre-----------end")
    #print()

    golden_output = torch.softmax(*golden_inputs, dim)
    golden_output.flatten()

    #print("softmax-golden-post-----------start")
    #print()
    #print(golden_output.shape)
    #print("softmax-golden-post-----------end")
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


cos_index = 0


def create_cos(inputs, output_shape_list, golden_inputs):
    global cos_index
    location = f"cos_{cos_index}"

    cos_output_operands, cos_output_type_list = output_type_operands(output_shape_list)
    res = ttir.CosOp(
        cos_output_type_list,
        [inputs],
        cos_output_operands,
        loc=Location.name(location),
    )
    cos_index += 1

    #print("cos-golden-pre------start")
    #print()
    a = golden_inputs[0]
    #print(a.shape)
    #print("cos-golden-pre------end")
    #print()

    golden_output = torch.cos(*golden_inputs)
    golden_output.flatten()

    #print("cos-golden-post------start")
    #print()
    #print(golden_output.shape)
    #print("cos-golden-post------end")
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


sin_index = 0


def create_sin(inputs, output_shape_list, golden_inputs):
    global sin_index
    location = f"sin_{sin_index}"

    sin_output_operands, sin_output_type_list = output_type_operands(output_shape_list)
    res = ttir.SinOp(
        sin_output_type_list,
        [inputs],
        sin_output_operands,
        loc=Location.name(location),
    )
    sin_index += 1

    #print("sin-golden-pre------start")
    #print()
    a = golden_inputs[0]
    #print(a.shape)
    #print("sin-golden-pre------end")
    #print()

    golden_output = torch.sin(*golden_inputs)
    golden_output.flatten()

    #print("sin-golden-post------start")
    #print()
    #print(golden_output.shape)
    #print("sin-golden-post------end")
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


transpose_index = 0


def create_transpose(inputs, output_shape_list, dim0, dim1, golden_inputs):
    global transpose_index
    location = f"transpose_{transpose_index}"

    transpose_output_operands, transpose_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.TransposeOp(
        transpose_output_type_list[0],
        inputs,
        transpose_output_operands[0],
        dim0,
        dim1,
        loc=Location.name(location),
    )
    transpose_index += 1

    #print("transpose-golden-pre------start")
    #print()
    a = golden_inputs[0]
    #print(a.shape)
    #print("transpose-golden-pre------end")
    #print()

    golden_output = torch.transpose(*golden_inputs, dim0, dim1)
    golden_output.flatten()

    #print("transpose-golden-post------start")
    #print()
    #print(golden_output.shape)
    #print("transpose-golden-post------end")
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


unsqueeze_index = 0


def create_unsqueeze(inputs, output_shape_list, dim, golden_inputs):
    global unsqueeze_index
    location = f"unsqueeze_{unsqueeze_index}"

    unsqueeze_output_operands, unsqueeze_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.UnsqueezeOp(
        unsqueeze_output_type_list[0],
        inputs,
        unsqueeze_output_operands[0],
        dim,
        loc=Location.name(location),
    )
    unsqueeze_index += 1

    #print("unsqueeze-golden-pre-----start")
    #print()
    a = golden_inputs[0]
    #print(a.shape)
    #print("unsqueeze-golden-pre-----end")
    #print()

    golden_output = torch.unsqueeze(*golden_inputs, dim)
    golden_output.flatten()

    #print("unsqueeze-golden-post-----start")
    #print()
    #print(golden_output.shape)
    #print("unsqueeze-golden-post-----end")
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


squeeze_index = 0


def create_squeeze(inputs, output_shape_list, dim, golden_inputs):
    global squeeze_index
    location = f"squeeze_{squeeze_index}"

    squeeze_output_operands, squeeze_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.SqueezeOp(
        squeeze_output_type_list[0],
        inputs,
        squeeze_output_operands[0],
        dim,
        loc=Location.name(location),
    )
    squeeze_index += 1

    #print("squeeze-golden-pre-----start")
    #print()
    a = golden_inputs[0]
    #print(a.shape)
    #print("squeeze-golden-pre-----end")
    #print()

    golden_output = torch.squeeze(*golden_inputs, dim)
    golden_output.flatten()

    #print("squeeze-golden-post-----start")
    #print()
    #print(golden_output.shape)
    #print("squeeze-golden-post-----end")
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


concat_index = 0


def create_concat(input_one, input_two, output_shape_list, dim, golden_inputs):
    global concat_index
    location = f"concat_{concat_index}"

    concat_output_operands, concat_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.ConcatOp(
        concat_output_type_list[0],
        [input_one, input_two],
        concat_output_operands[0],
        dim,
        loc=Location.name(location),
    )
    concat_index += 1

    #print("concat-golden-pre-------start")
    #print()
    a = golden_inputs[0]
    b = golden_inputs[1]
    #print(a.shape)
    #print(b.shape)
    #print("concat-golden-pre-------end")
    #print()

    golden_output = torch.concat(golden_inputs, dim)
    golden_output.flatten()

    #print("concat-------start")
    #print()
    #print(golden_output.shape)
    #print("concat-------end")
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


mean_index = 0


def create_mean(inputs, output_shape_list, keep_dim, dim_arg, golden_inputs):
    global mean_index
    location = f"mean_{mean_index}"

    mean_output_operands, mean_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.MeanOp(
        mean_output_type_list[0],
        inputs,
        mean_output_operands[0],
        True,
        dim_arg=dim_arg,
        loc=Location.name(location),
    )
    mean_index += 1

    golden_output = torch.mean(*golden_inputs)
    golden_output.flatten()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


reshape_index = 0


def create_reshape(inputs, output_shape_list, golden_inputs):
    global reshape_index
    location = f"reshape_{reshape_index}"

    reshape_output_operands, reshape_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.ReshapeOp(
        reshape_output_type_list[0],
        inputs,
        reshape_output_operands[0],
        shape=output_shape_list[0],
        loc=Location.name(location),
    )
    reshape_index += 1

    #print("reshape-golden-pre----------start")
    #print()
    a = golden_inputs[0]
    #print(a)
    #print(a.flatten())
    #print(a.shape)
    #print("reshape-golden-pre---------end")
    #print()
    #print()
    golden_output = torch.reshape(*golden_inputs, output_shape_list[0])
    golden_output.flatten()
    #print(golden_output)
    #print(golden_output.flatten())

    #print("reshape-golden-post---------start")
    #print()
    #print(golden_output.shape)
    #print("reshape-golden-post---------end")
    #print()
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


typecast_index = 0


def create_typecast(inputs, output_shape_list):
    global typecast_index
    location = f"typecast_{typecast_index}"

    typecast_output_operands, typecast_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.TypecastOp(
        typecast_output_type_list,
        [inputs],
        typecast_output_operands,
        loc=Location.name(location),
    )
    typecast_index += 1

    return res


matmul_index = 0


def create_matmul(input_one, input_two, output_shape_list, golden_inputs):
    global matmul_index
    location = f"matmul_{matmul_index}"

    matmul_output_operands, matmul_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.MatmulOp(
        matmul_output_type_list[0],
        input_one,
        input_two,
        matmul_output_operands[0],
        loc=Location.name(location),
    )
    matmul_index += 1

    #print("matmul-golden-pre----------start")
    #print()
    a = golden_inputs[0]
    b = golden_inputs[1]
    #print(a.shape)
    #print(b.shape)
    #print("matmul-golden-pre---------end")
    #print()
    #print()

    golden_output = torch.matmul(*golden_inputs)
    golden_output.flatten()

    #print("matmul-golden-post----------start")
    #print()
    #print(golden_output.shape)
    #print("matmul-golden-post---------end")
    #print()
    #print()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}


embedding_index = 0


def create_embedding(input_one, input_two, output_shape_list, golden_inputs):
    global embedding_index
    location = f"embedding_{embedding_index}"

    embedding_output_operands, embedding_output_type_list = output_type_operands(
        output_shape_list
    )
    res = ttir.EmbeddingOp(
        embedding_output_type_list[0],
        input_one,
        input_two,
        embedding_output_operands[0],
        loc=Location.name(location),
    )
    embedding_index += 1

    golden_output = torch.embedding(*golden_inputs)
    golden_output.flatten()

    torch.save(golden_output, f"/code/tt-mlir/builder_goldens/{location}.pt")

    return res, {"location": location, "golden_output": golden_output}
