# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# DISCLAIMER: this file will be removed very soon and is a temporary solution

from ..core.ops import get_op_outputs
from ttmlir import ir
import torch


ttir_dtype_maps = {
    "i32": torch.int32,
    "i64": torch.int64,
    "f32": torch.float32,
    "f64": torch.float64,
    "si32": torch.int32,
    "i1": torch.bool,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
}

ttrt_dtype_maps = {
    "DataType.Float32": torch.float32,
    "DataType.BFloat16": torch.bfloat16,
    "DataType.UInt32": torch.uint32,
    "DataType.UInt16": torch.uint16,
    "DataType.UInt8": torch.uint8,
    "DataType.Int32": torch.int32,
}


def resolve_dense_attr(dense_attr):
    if dense_attr.is_splat:
        value = dense_attr.get_splat_value()
        if dense_attr.type.shape != [1]:
            value = torch.ones(dense_attr.type.shape) * value.value
        return value
    try:
        values = [dense_attr[i] for i in range(len(dense_attr))]
        return values
    except Exception as e:
        print(f"Indexing error: {e}")

    try:
        shape = dense_attr.type.shape
        if len(shape) == 2:
            values = [
                [dense_attr.get(i, j) for j in range(shape[1])] for i in range(shape[0])
            ]
        elif len(shape) == 1:
            values = [dense_attr.get(i) for i in range(shape[0])]
        return values
    except Exception as e:
        print(f"Get method error: {e}")


def handle_dense_elements_attr(x):
    val = resolve_dense_attr(x)
    if hasattr(val, "value"):
        val = val.value
    return val


handle_attr_type = {
    ir.DenseIntElementsAttr: handle_dense_elements_attr,
    ir.IntegerAttr: lambda x: x.value,
    ir.BoolAttr: lambda x: x.value,
    ir.DenseI64ArrayAttr: lambda x: [x[i] for i in range(len(x))],
    ir.DenseFPElementsAttr: handle_dense_elements_attr,
    ir.ArrayAttr: lambda x: [x[i].value for i in range(len(x))],
    ir.StringAttr: lambda x: x.value,
}
