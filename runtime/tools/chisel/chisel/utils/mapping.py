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
    "ui32": torch.uint32,
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
    ir.FloatAttr: lambda x: x.value,
    ir.DenseI64ArrayAttr: lambda x: [x[i] for i in range(len(x))],
    ir.DenseI32ArrayAttr: lambda x: [x[i] for i in range(len(x))],
    ir.DenseFPElementsAttr: handle_dense_elements_attr,
    ir.ArrayAttr: lambda x: [x[i].value for i in range(len(x))],
    ir.StringAttr: lambda x: x.value,
}


class OpMapping:
    def __init__(self, torch_op, arg_map=None, unpack_inputs=True):
        self.torch_op = torch_op
        self.arg_map = arg_map or {}
        self.unpack_inputs = unpack_inputs

    def _process_args(self, op: ir.Operation):
        args = {}
        for attr in op.attributes:
            args[attr.name] = handle_attr_type[type(attr.attr)](attr.attr)
        return args

    def __call__(self, op, inputs):
        if isinstance(inputs, list) and len(inputs) > 1:
            result_inputs = inputs
        else:
            result_inputs = (
                inputs[0] if isinstance(inputs, list) and len(inputs) == 1 else inputs
            )

        op_args = self._process_args(op)

        torch_args = {}
        for arg_name, arg_value in op_args.items():
            if arg_name in self.arg_map:
                if self.arg_map[arg_name] == "":
                    continue
                torch_args[self.arg_map[arg_name]] = arg_value
            else:
                torch_args[arg_name] = arg_value

        if op.name == "ttir.constant":
            torch_args["dtype"] = ttir_dtype_maps[
                str(get_op_outputs(op)[0].type.element_type)
            ]
            torch_args["shape"] = get_op_outputs(op)[0].type.shape

        if op.name == "ttir.typecast":
            torch_args["dtype"] = ttir_dtype_maps[
                str(get_op_outputs(op)[0].type.element_type)
            ]

        if not self.unpack_inputs:
            result = self.torch_op(result_inputs, **torch_args)
            return result

        result = self.torch_op(*result_inputs, **torch_args)
        return result


def custom_broadcast(x, size=None):
    for i in range(len(size)):
        try:
            if size[i] < x.shape[i]:
                size[i] = x.shape[i]
        except Exception as e:
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

    if isinstance(data, torch.Tensor):
        res = data
    else:
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


def custom_mean(x, dim=None, keepdim=False):
    if dim is None:
        return torch.mean(x)
    dim = reversed(sorted(dim))
    for d in dim:
        x = torch.mean(x, d, keepdim=keepdim)
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

    slices = tuple(slice(b, e, s) for b, e, s in tmp1)
    result = x[slices]

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


def custom_matmul(x, y, transpose_a=False, transpose_b=False):
    if transpose_a:
        x = torch.transpose(x.clone(), -1, -2)
    if transpose_b:
        y = torch.transpose(y.clone(), -1, -2)
    return torch.matmul(x, y)


def custom_transpose(x, dim0, dim1):
    return x.transpose(dim0, dim1)


def custom_fill_cache(
    cache: torch.Tensor, input: torch.Tensor, batch_offset: int = 0
) -> torch.Tensor:
    """
    Reference implementation of ttir.fill_cache (FillCacheOp).

    • `cache`  – destination tensor  [B, S, …]
    • `input`  – values to copy      [B′, S′, …]  (B′ ≤ B, S′ ≤ S)
    • `batch_offset` – starting index on the batch axis

    The function returns a *new* tensor with the relevant slice replaced; the
    original `cache` is left untouched.
    """
    # Clone so we don’t mutate the original tensor held by the executor.
    result = cache.clone()

    b_end = (
        batch_offset + input.shape[0]
    )  # batch range to update                         # length on the seq axis

    # We assume layout [batch, sequence, *rest]; trailing dims must match.
    result[:, :, batch_offset : input.shape[-2]] = input
    return result


def custom_full(*args, **kwargs):
    return torch.full(kwargs["size"], kwargs["fill_value"])


def custom_update_cache(
    cache: torch.Tensor,
    input: torch.Tensor,
    update_index: torch.Tensor,
    *,
    batch_offset: int = 0,
) -> torch.Tensor:
    result = cache.clone()

    idx = update_index.to(dtype=torch.long)
    result[..., idx, :] = input
    return result


def custom_embeding(input, weight):
    return torch.nn.functional.embedding(input.long(), weight)


def custom_comparison_operator(input: torch.Tensor, other: torch.Tensor, torch_op):
    return torch_op(input, other).to(dtype=input.dtype)


def custom_argmax(x, dim=[], keepdim=False):
    # PyTorch only supports single-dimension argmax, so we iteratively apply it
    if not dim:
        # If no dimension is specified, flatten the tensor and find the global argmax
        x = x.flatten()
        result = torch.argmax(x)
        if keepdim:
            # If keepdim is True, we need to return a tensor with the same number of dimensions
            result = result.unsqueeze(0)
        return result

    dim = sorted(dim)
    for d in dim:
        x = x.argmax(d, keepdim=keepdim)
    return x


def custom_batch_norm(
    input, running_mean, running_var, weight, bias, *, epsilon=1e-5, training=False, **_
):
    # Flatten 4D tensors to 1D if necessary
    if running_mean.ndim != 1:
        running_mean = running_mean.flatten()
    if running_var.ndim != 1:
        running_var = running_var.flatten()
    if weight is not None and weight.ndim != 1:
        weight = weight.flatten()
    if bias is not None and bias.ndim != 1:
        bias = bias.flatten()

    out = torch.nn.functional.batch_norm(
        input, running_mean, running_var, weight, bias, training=training, eps=epsilon
    )
    return out


ttir_to_torch_mapping = {
    # do nothing
    "ttir.empty": OpMapping(lambda x=None, *args, **kwargs: None),
    "func.return": OpMapping(lambda x=None, *args, **kwargs: x),
    "ttir.add": OpMapping(torch.add),
    "ttir.arange": OpMapping(
        torch.arange,
        {"start": "start", "end": "end", "step": "step", "arange_dimension": ""},
    ),
    "ttir.broadcast": OpMapping(
        custom_broadcast, {"broadcast_dimensions": "size"}, unpack_inputs=False
    ),
    "ttir.constant": OpMapping(custom_constant, {"value": "data"}),
    "ttir.div": OpMapping(torch.div),
    "ttir.exp": OpMapping(torch.exp, unpack_inputs=False),
    "ttir.pow": OpMapping(torch.pow),
    "ttir.ge": OpMapping(
        lambda input, other: custom_comparison_operator(input, other, torch.ge)
    ),
    "ttir.gt": OpMapping(
        lambda input, other: custom_comparison_operator(input, other, torch.gt)
    ),
    "ttir.le": OpMapping(
        lambda input, other: custom_comparison_operator(input, other, torch.le)
    ),
    "ttir.lt": OpMapping(
        lambda input, other: custom_comparison_operator(input, other, torch.lt)
    ),
    "ttir.ne": OpMapping(
        lambda input, other: custom_comparison_operator(input, other, torch.ne)
    ),
    "ttir.logical_and": OpMapping(torch.logical_and),
    "ttir.matmul": OpMapping(custom_matmul),
    "ttir.max": OpMapping(
        custom_max, {"dim_arg": "dim", "keep_dim": "keepdim"}, unpack_inputs=False
    ),
    "ttir.mean": OpMapping(
        custom_mean, {"dim_arg": "dim", "keep_dim": "keepdim"}, unpack_inputs=False
    ),
    "ttir.maximum": OpMapping(torch.maximum),
    "ttir.multiply": OpMapping(torch.multiply),
    "ttir.cumsum": OpMapping(torch.cumsum, {"dim": "dim"}, unpack_inputs=False),
    "ttir.permute": OpMapping(
        torch.permute, {"permutation": "dims"}, unpack_inputs=False
    ),
    "ttir.prod": OpMapping(
        custom_prod, {"dim_arg": "dim", "keep_dim": "keepdim"}, unpack_inputs=False
    ),
    "ttir.reshape": OpMapping(torch.reshape, {"shape": "shape"}, unpack_inputs=False),
    "ttir.rsqrt": OpMapping(torch.rsqrt, unpack_inputs=False),
    "ttir.sqrt": OpMapping(torch.sqrt, unpack_inputs=False),
    "ttir.cos": OpMapping(torch.cos, unpack_inputs=False),
    "ttir.unsqueeze": OpMapping(torch.unsqueeze, unpack_inputs=False),
    "ttir.squeeze": OpMapping(torch.squeeze, unpack_inputs=False),
    "ttir.repeat_interleave": OpMapping(torch.repeat_interleave, unpack_inputs=False),
    "ttir.sin": OpMapping(torch.sin, unpack_inputs=False),
    "ttir.clamp_scalar": OpMapping(torch.clamp, unpack_inputs=False),
    "ttir.clamp_tensor": OpMapping(torch.clamp, unpack_inputs=False),
    "ttir.softmax": OpMapping(
        torch.nn.functional.softmax,
        {"dimension": "dim", "stable": ""},
        unpack_inputs=False,
    ),
    "ttir.sigmoid": OpMapping(torch.sigmoid, unpack_inputs=False),
    "ttir.subtract": OpMapping(torch.sub),
    "ttir.sum": OpMapping(
        torch.sum, {"dim_arg": "dim", "keep_dim": "keepdim"}, unpack_inputs=False
    ),
    "ttir.transpose": OpMapping(custom_transpose, unpack_inputs=False),
    "ttir.reciprocal": OpMapping(torch.reciprocal, unpack_inputs=False),
    "ttir.tanh": OpMapping(torch.tanh, unpack_inputs=False),
    "ttir.typecast": OpMapping(
        custom_typecast,
        {"dtype": "dtype", "conservative_folding": ""},
        unpack_inputs=False,
    ),
    "ttir.where": OpMapping(custom_where),
    "ttir.concat": OpMapping(torch.concat, {"dim": "dim"}, unpack_inputs=False),
    "ttir.full": OpMapping(
        custom_full, {"shape": "size", "fill_value": "fill_value"}, unpack_inputs=False
    ),
    "ttir.embedding": OpMapping(custom_embeding),
    "ttir.fill_cache": OpMapping(
        custom_fill_cache,
        {"batch_offset": "batch_offset"},
    ),
    "ttir.update_cache": OpMapping(
        custom_update_cache,
        {"batch_offset": "batch_offset"},
    ),
    "ttir.slice_static": OpMapping(
        custom_slice,
        {"begins": "begins", "ends": "ends", "step": "step"},
        unpack_inputs=False,
    ),
    "ttir.reduce_and": OpMapping(
        custom_reduce_and,
        {"dim_arg": "dim", "keep_dim": "keepdim"},
        unpack_inputs=False,
    ),
    "ttir.relu": OpMapping(torch.nn.functional.relu, unpack_inputs=False),
    "ttir.neg": OpMapping(torch.neg, unpack_inputs=False),
    "ttir.abs": OpMapping(torch.abs, unpack_inputs=False),
    "ttir.sign": OpMapping(torch.sign, unpack_inputs=False),
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
    "ttir.batch_norm_inference": OpMapping(custom_batch_norm),
    "ttir.argmax": OpMapping(
        custom_argmax, {"dim_arg": "dim", "keep_dim": "keepdim"}, unpack_inputs=False
    ),
}
