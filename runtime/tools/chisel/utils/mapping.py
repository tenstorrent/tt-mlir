# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# DISCLAIMER: this file will be removed very soon and is a temporary solution

import ttmlir
import torch


ttir_dtype_maps = {
    "i32": torch.int32,
    "i64": torch.int64,
    "f32": torch.float32,
    "f64": torch.float64,
    "i1": torch.bool,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
}


def resolve_dense_attr(dense_attr):
    if dense_attr.is_splat:
        value = dense_attr.get_splat_value()
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


class OpMapping:
    def __init__(self, torch_op, arg_map=None, unpack_inputs=True):
        self.torch_op = torch_op
        self.arg_map = arg_map or {}
        self.unpack_inputs = unpack_inputs

    def __call__(self, ir_op, inputs):
        if isinstance(inputs, list) and len(inputs) > 1:
            result_inputs = inputs
        else:
            result_inputs = (
                inputs[0] if isinstance(inputs, list) and len(inputs) == 1 else inputs
            )

        torch_args = {}
        for k, v in self.arg_map.items():
            if k in ir_op.attributes:
                if isinstance(ir_op.attributes[k], ttmlir.ir.DenseElementsAttr):
                    val = resolve_dense_attr(ir_op.attributes[k])
                    if hasattr(val, "value"):
                        val = val.value
                    torch_args[v] = val
                elif isinstance(ir_op.attributes[k], ttmlir.ir.DenseI64ArrayAttr):
                    torch_args[v] = [x for x in ir_op.attributes[k]]
                elif isinstance(ir_op.attributes[k], ttmlir.ir.DenseI32ArrayAttr):
                    torch_args[v] = [x for x in ir_op.attributes[k]]
                elif isinstance(ir_op.attributes[k], ttmlir.ir.ArrayAttr):
                    torch_args[v] = [x.value for x in ir_op.attributes[k]]
                else:
                    torch_args[v] = ir_op.attributes[k].value

        if ir_op.name == "ttir.constant":
            torch_args["dtype"] = ttir_dtype_maps[
                str(ir_op.attributes[0].attr.type.element_type)
            ]
            torch_args["shape"] = ir_op.attributes[0].attr.type.shape

        if ir_op.name == "ttir.typecast":
            torch_args["dtype"] = ttir_dtype_maps[str(ir_op.output.type.element_type)]

        if not self.unpack_inputs:
            print("torch op", self.torch_op, result_inputs, torch_args)
            result = self.torch_op(result_inputs, **torch_args)
            if result is not None:
                print("torch op result", result.shape, result)
            return result

        result = self.torch_op(*result_inputs, **torch_args)
        print("torch op unpack", self.torch_op, result_inputs, torch_args)
        if result is not None:
            print("torch op result", result.shape, result)
        return result


def custom_broadcast(x, size=None):
    for i in range(len(size)):
        try:
            if size[i] < x.shape[i]:
                size[i] = x.shape[i]
        except Exception as e:
            import pdb

            pdb.set_trace()
            print(f"Broadcasting error: {e}")
    return x.expand(size)


def custom_where(a, b, c):
    a = a.to(torch.bool)
    return torch.where(a, b, c)


def custom_typecast(x, dtype=None):
    if dtype is None:
        return x
    return x.to(dtype)


def custom_constant(*args, **kwargs):
    data = kwargs["data"]
    kwargs.pop("data")
    shape = kwargs.get("shape", None)
    dtype = kwargs.get("dtype", None)

    res = torch.tensor([data])
    if dtype is not None:
        res = res.to(dtype)
    if shape is not None:
        # Special case, if res is a scalar, we broadcast
        if res.numel() == 1:
            while len(shape) - len(res.shape) > 0:
                res = res.unsqueeze(0)
            res = res.broadcast_to(shape)
            return res

        res = res.reshape(shape)
    return res


def custom_conv2d(*args, **kwargs):
    # Convert from channels last (NHWC) to channels first (NCHW) for PyTorch
    I = args[0].permute(0, 3, 1, 2)
    weight = args[1].permute(0, 1, 2, 3)

    print(f"Debug: custom_conv2d {I}\n{weight}")

    # Get and validate kwargs with defaults
    padding = kwargs.get("padding", 0)
    if isinstance(padding, list):
        if all(e == padding[0] for e in padding):
            padding = (padding[0], padding[0])
        else:
            raise ValueError("Unsupported padding format")

    stride = kwargs.get("stride", 1)
    if isinstance(stride, list):
        if all(e == stride[0] for e in stride):
            stride = stride[0]
        else:
            raise ValueError("Unsupported stride format")

    dilation = kwargs.get("dilation", 1)
    if isinstance(dilation, list):
        if all(e == dilation[0] for e in dilation):
            dilation = dilation[0]
        else:
            raise ValueError("Unsupported dilation format")

    groups = kwargs.get("groups", 1)
    if isinstance(groups, list):
        if len(groups) == 1:
            groups = groups[0]
        else:
            raise ValueError("Unsupported groups format")

    result = torch.nn.functional.conv2d(
        I, weight, stride=stride, padding=padding, dilation=dilation, groups=groups
    )

    # Convert back to channels last (NHWC)
    return result.permute(0, 2, 3, 1)


def custom_reduce_and(x, dim=None, keepdim=False):
    if dim is None:
        return torch.all(x)
    return torch.all(x, dim=dim, keepdim=keepdim)


def custom_prod(x, dim=None, keepdim=False):
    if dim is None:
        return torch.prod(x)
    dim = reversed(sorted(dim))
    for d in dim:
        x = torch.prod(x, d, keepdim=keepdim)
    return x


def custom_max(x, dim=None, keepdim=False):
    if dim is None:
        values, _ = torch.max(x)
        return values
    dim = reversed(sorted(dim))
    for d in dim:
        x, _ = torch.max(x, d, keepdim=keepdim)
    return x


def custom_argmax(x, dim=None, keepdim=False):
    if dim is None:
        return torch.argmax(x)
    dim = reversed(sorted(dim))
    for d in dim:
        x = torch.argmax(x, d, keepdim=keepdim)
    return x


def custom_slice(x, begins=None, ends=None, step=None):
    if begins is None and ends is None and step is None:
        return x
    if begins is None:
        begins = [0] * len(x.shape)
    if ends is None:
        ends = list(x.shape)
    if step is None:
        step = [1] * len(x.shape)

    tmp1 = list(zip(begins, ends, step))
    slice = ""
    for b, e, s in tmp1:
        slice = slice + f"{b}:{e}:{s},"  # holy smokes this is ugly

    slice = slice[:-1]
    slice = "x[" + slice + "]"
    result = eval(slice)

    return result


def custom_max_pool2d(*args, **kwargs):
    I = args[0]  # Input is already in NHWC: [B, H, W, C]
    # Convert to NCHW for PyTorch
    I = I.permute(0, 3, 1, 2)
    # Extract pooling parameters
    kernel_size = [kwargs["kernel_height"], kwargs["kernel_width"]]
    stride = [kwargs["stride_height"], kwargs["stride_width"]]
    dilation = [kwargs["dilation_height"], kwargs["dilation_width"]]
    pt, pb = kwargs["padding_top"], kwargs["padding_bottom"]
    pl, pr = kwargs["padding_left"], kwargs["padding_right"]
    ceil_mode = kwargs["ceil_mode"]
    if (pt == pb) and (pl == pr):
        padding = [pt, pl]
        out = torch.nn.functional.max_pool2d(
            I,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )
    else:
        pad = [pl, pr, pt, pb]  # left, right, top, bottom
        I = torch.nn.functional.pad(I, pad)
        out = torch.nn.functional.max_pool2d(
            I,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )
    # Convert back to NHWC
    return out.permute(0, 2, 3, 1)


def custom_avg_pool2d(*args, **kwargs):
    I = args[0]  # Input is already in NHWC: [B, H, W, C]
    # Convert to NCHW for PyTorch
    I = I.permute(0, 3, 1, 2)
    # Extract pooling parameters
    kernel_size = [kwargs["kernel_height"], kwargs["kernel_width"]]
    stride = [kwargs["stride_height"], kwargs["stride_width"]]
    pt, pb = kwargs["padding_top"], kwargs["padding_bottom"]
    pl, pr = kwargs["padding_left"], kwargs["padding_right"]
    if (pt == pb) and (pl == pr):
        padding = [pt, pl]
        out = torch.nn.functional.avg_pool2d(
            I,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
    else:
        pad = [pl, pr, pt, pb]  # left, right, top, bottom
        I = torch.nn.functional.pad(I, pad)
        out = torch.nn.functional.avg_pool2d(
            I,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
    # Convert back to NHWC
    return out.permute(0, 2, 3, 1)


def custom_embedding(*args, **kwargs):
    # Extract the input tensor and the weight tensor
    input_tensor = args[0]
    weight_tensor = args[1]

    # Convert the input tensor to long type if it is not already
    if input_tensor.dtype != torch.long:
        input_tensor = input_tensor.long()

    # Perform the embedding lookup
    result = torch.nn.functional.embedding(input_tensor, weight_tensor)

    return result


ttir_to_torch_mapping = {
    # do nothing
    "ttir.empty": OpMapping(lambda x=None, *args, **kwargs: None),
    "func.return": OpMapping(lambda x=None, *args, **kwargs: x),
    "ttir.add": OpMapping(torch.add),
    "ttir.arange": OpMapping(
        torch.arange, {"start": "start", "end": "end", "step": "step"}
    ),
    "ttir.broadcast": OpMapping(
        custom_broadcast, {"broadcast_dimensions": "size"}, unpack_inputs=False
    ),
    "ttir.constant": OpMapping(custom_constant, {"value": "data"}),
    "ttir.div": OpMapping(torch.div),
    "ttir.exp": OpMapping(torch.exp, unpack_inputs=False),
    "ttir.pow": OpMapping(torch.pow),
    "ttir.ge": OpMapping(torch.ge),
    "ttir.gt": OpMapping(torch.gt),
    "ttir.le": OpMapping(torch.le),
    "ttir.ne": OpMapping(torch.ne),
    "ttir.logical_and": OpMapping(torch.logical_and),
    "ttir.log": OpMapping(torch.log, unpack_inputs=False),
    "ttir.lt": OpMapping(torch.lt),
    "ttir.matmul": OpMapping(torch.matmul),
    "ttir.max": OpMapping(
        custom_max, {"dim_arg": "dim", "keep_dim": "keepdim"}, unpack_inputs=False
    ),
    "ttir.maximum": OpMapping(torch.maximum),
    "ttir.minimum": OpMapping(torch.minimum),
    "ttir.multiply": OpMapping(torch.multiply),
    "ttir.cumsum": OpMapping(torch.cumsum, {"dim": "dim"}, unpack_inputs=False),
    "ttir.permute": OpMapping(
        torch.permute, {"permutation": "dims"}, unpack_inputs=False
    ),
    "ttir.prod": OpMapping(
        custom_prod, {"dim_arg": "dim", "keep_dim": "keepdim"}, unpack_inputs=False
    ),
    "ttir.argmax": OpMapping(
        custom_argmax, {"dim_arg": "dim", "keep_dim": "keepdim"}, unpack_inputs=False
    ),
    "ttir.reshape": OpMapping(torch.reshape, {"shape": "shape"}, unpack_inputs=False),
    "ttir.rsqrt": OpMapping(torch.rsqrt, unpack_inputs=False),
    "ttir.sqrt": OpMapping(torch.sqrt, unpack_inputs=False),
    "ttir.subtract": OpMapping(torch.subtract),
    "ttir.sum": OpMapping(
        torch.sum, {"dim_arg": "dim", "keep_dim": "keepdim"}, unpack_inputs=False
    ),
    "ttir.tanh": OpMapping(torch.tanh, unpack_inputs=False),
    "ttir.typecast": OpMapping(
        custom_typecast, {"dtype": "dtype"}, unpack_inputs=False
    ),
    "ttir.where": OpMapping(custom_where),
    "ttir.concat": OpMapping(torch.concat, {"dim": "dim"}, unpack_inputs=False),
    "ttir.embedding": OpMapping(custom_embedding),
    "ttir.slice": OpMapping(
        custom_slice,
        {"begins": "begins", "ends": "ends", "step": "step"},
        unpack_inputs=False,
    ),
    "ttir.reduce_and": OpMapping(
        custom_reduce_and,
        {"dim_arg": "dim", "keep_dim": "keepdim"},
        unpack_inputs=False,
    ),
    "ttir.neg": OpMapping(torch.neg, unpack_inputs=False),
    "ttir.abs": OpMapping(torch.abs, unpack_inputs=False),
    "ttir.eq": OpMapping(torch.eq),
    "ttir.conv2d": OpMapping(
        custom_conv2d,
        {
            "stride": "stride",
            "padding": "padding",
            "dilation": "dilation",
            "groups": "groups",
        },
    ),
    "ttir.max_pool2d": OpMapping(
        custom_max_pool2d,
        {
            "kernel_height": "kernel_height",
            "kernel_width": "kernel_width",
            "stride_height": "stride_height",
            "stride_width": "stride_width",
            "padding_top": "padding_top",
            "padding_bottom": "padding_bottom",
            "padding_left": "padding_left",
            "padding_right": "padding_right",
            "dilation_height": "dilation_height",
            "dilation_width": "dilation_width",
            "ceil_mode": "ceil_mode",
        },
        unpack_inputs=False,
    ),
    "ttir.avg_pool2d": OpMapping(
        custom_avg_pool2d,
        {
            "kernel_height": "kernel_height",
            "kernel_width": "kernel_width",
            "stride_height": "stride_height",
            "stride_width": "stride_width",
            "padding_top": "padding_top",
            "padding_bottom": "padding_bottom",
            "padding_left": "padding_left",
            "padding_right": "padding_right",
            "dilation_height": "dilation_height",
            "dilation_width": "dilation_width",
        },
        unpack_inputs=False,
    ),
}
