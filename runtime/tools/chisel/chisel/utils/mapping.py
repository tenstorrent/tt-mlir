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
            attr_type = type(attr.attr)
            if attr_type in handle_attr_type:
                args[attr.name] = handle_attr_type[attr_type](attr.attr)
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


def custom_batch_norm_inference(*args, **kwargs):
    """
    Custom implementation of batch normalization inference operation.

    This applies batch normalization to an input tensor in inference mode.

    Args:
    - args[0]: Input tensor for batch normalization
    - args[1]: Scale/weight tensor for batch normalization
    - args[2]: Offset/bias tensor for batch normalization
    - args[3]: Mean tensor for batch normalization
    - args[4]: Variance tensor for batch normalization

    Kwargs:
    - dimension: Dimension to apply batch normalization over (default: 1)
    - epsilon: Small value to avoid division by zero (default: 1e-5)

    Returns:
    - Result of batch normalization
    """
    if len(args) == 1 and isinstance(args[0], list):
        # Handle case where all inputs are in a list
        input_tensor = args[0][0]
        scale = args[0][1]
        offset = args[0][2]
        mean = args[0][3]
        variance = args[0][4]
    else:
        # Handle case where inputs are separate arguments
        input_tensor = args[0]
        scale = args[1]
        offset = args[2]
        mean = args[3]
        variance = args[4]

    dimension = kwargs.get("dimension", 1)
    epsilon = kwargs.get("epsilon", 1e-5)

    # Validate that mean and variance sizes match the dimension we're normalizing over
    norm_dim_size = input_tensor.shape[dimension]
    mean_size = mean.numel()

    print(
        f"Batch norm: input shape={input_tensor.shape}, dimension={dimension}, norm_dim_size={norm_dim_size}, mean_size={mean_size}"
    )

    # Only proceed if sizes match
    if norm_dim_size != mean_size:
        # Return input unchanged if sizes don't match
        print(
            f"WARNING: Batch norm dimension size mismatch: {norm_dim_size} != {mean_size}, returning input unchanged"
        )
        return input_tensor

    # Create permutation to move the normalization dimension to position 1 (after batch)
    ndim = input_tensor.ndim
    perm = list(range(ndim))
    if dimension != 1:
        perm[1], perm[dimension] = perm[dimension], perm[1]
        input_permuted = input_tensor.permute(perm)
        print(f"Permuted input shape: {input_permuted.shape} with perm={perm}")
    else:
        input_permuted = input_tensor
        print(f"No permutation needed, input_permuted shape: {input_permuted.shape}")

    # Apply batch normalization (inference mode: training=False)
    print(
        f"Before batch_norm: shape={input_permuted.shape}, mean.shape={mean.shape}, variance.shape={variance.shape}"
    )
    result = torch.nn.functional.batch_norm(
        input_permuted,
        running_mean=mean,
        running_var=variance,
        weight=scale,
        bias=offset,
        training=False,
        eps=epsilon,
    )
    print(f"After batch_norm: result shape={result.shape}")

    # Inverse permutation to restore original dimension order
    if dimension != 1:
        # Compute the true inverse permutation
        # If perm[i] = j, then inv_perm[j] = i
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        print(f"Applying inverse permutation={inv_perm} to shape {result.shape}")
        result = result.permute(inv_perm)
        print(f"After inverse permutation: result shape={result.shape}")

    return result


def custom_pad(*args, **kwargs):
    """
    Custom implementation of pad operation with dimension reformatting.

    Args:
    - args[0]: Input tensor to pad

    Kwargs:
    - padding: Padding specification as array of integers
    - value: Value to use for padding

    Returns:
    - Padded tensor
    """
    if len(args) == 1 and isinstance(args[0], list):
        # Handle case where inputs are in a list
        input_tensor = args[0][0]
    else:
        # Handle case where input is a single argument
        input_tensor = args[0]

    padding = kwargs.get("padding", [])
    value = kwargs.get("value", 0)

    # Reformat padding dimensions from TTIR format to PyTorch format
    # TTIR format: [d0_start, d0_end, d1_start, d1_end, ..., dn_start, dn_end]
    # PyTorch format: [dn_start, dn_end, ..., d1_start, d1_end, d0_start, d0_end] (reversed)
    golden_padding = []
    for i in range(len(padding) // 2):
        golden_padding.append(padding[-((2 * i) + 2)])
        golden_padding.append(padding[-((2 * i) + 1)])

    return torch.nn.functional.pad(
        input_tensor, pad=golden_padding, mode="constant", value=value
    )


def custom_dot_general(*args, **kwargs):
    """
    Custom implementation of generalized dot product operation.

    Args:
    - args[0]: Left-hand side tensor
    - args[1]: Right-hand side tensor

    Kwargs:
    - batch_dims_lhs: Batch dimensions for left tensor (list of ints)
    - contract_dims_lhs: Contraction dimensions for left tensor (list of ints)
    - batch_dims_rhs: Batch dimensions for right tensor (list of ints)
    - contract_dims_rhs: Contraction dimensions for right tensor (list of ints)

    Returns:
    - Result of generalized dot product
    """
    if len(args) == 1 and isinstance(args[0], list):
        # Handle case where inputs are in a list
        lhs = args[0][0]
        rhs = args[0][1]
    else:
        # Handle case where inputs are separate arguments
        lhs = args[0]
        rhs = args[1]

    batch_dims_lhs = kwargs.get("batch_dims_lhs", [])
    contract_dims_lhs = kwargs.get("contract_dims_lhs", [])
    batch_dims_rhs = kwargs.get("batch_dims_rhs", [])
    contract_dims_rhs = kwargs.get("contract_dims_rhs", [])

    # Ensure dimensions are lists
    if not isinstance(batch_dims_lhs, (list, tuple)):
        batch_dims_lhs = (
            list(batch_dims_lhs)
            if hasattr(batch_dims_lhs, "__iter__")
            else [batch_dims_lhs]
        )
    if not isinstance(contract_dims_lhs, (list, tuple)):
        contract_dims_lhs = (
            list(contract_dims_lhs)
            if hasattr(contract_dims_lhs, "__iter__")
            else [contract_dims_lhs]
        )
    if not isinstance(batch_dims_rhs, (list, tuple)):
        batch_dims_rhs = (
            list(batch_dims_rhs)
            if hasattr(batch_dims_rhs, "__iter__")
            else [batch_dims_rhs]
        )
    if not isinstance(contract_dims_rhs, (list, tuple)):
        contract_dims_rhs = (
            list(contract_dims_rhs)
            if hasattr(contract_dims_rhs, "__iter__")
            else [contract_dims_rhs]
        )

    # Compute non-batch dimensions
    non_batch_dims_lhs = [d for d in range(lhs.dim()) if d not in batch_dims_lhs]
    non_batch_dims_rhs = [d for d in range(rhs.dim()) if d not in batch_dims_rhs]

    # Compute output shape
    lhs_shape = list(lhs.shape)
    rhs_shape = list(rhs.shape)
    batch_shape = [lhs_shape[d] for d in batch_dims_lhs]
    non_contract_lhs = [d for d in non_batch_dims_lhs if d not in contract_dims_lhs]
    non_contract_rhs = [d for d in non_batch_dims_rhs if d not in contract_dims_rhs]
    out_shape = (
        batch_shape
        + [lhs_shape[d] for d in non_contract_lhs]
        + [rhs_shape[d] for d in non_contract_rhs]
    )

    # For the simple case with no batch dimensions, use tensordot directly
    if len(batch_shape) == 0:
        result = torch.tensordot(
            lhs,
            rhs,
            dims=(contract_dims_lhs, contract_dims_rhs),
        )
    else:
        # Reorder dimensions: batch dimensions first, then non-batch dimensions
        transposed_lhs = torch.permute(lhs, (batch_dims_lhs + non_batch_dims_lhs))
        transposed_rhs = torch.permute(rhs, (batch_dims_rhs + non_batch_dims_rhs))

        # Reshape to combine batch dimensions
        lhs_reshaped = transposed_lhs.reshape(
            batch_shape + [lhs_shape[d] for d in non_batch_dims_lhs]
        )
        rhs_reshaped = transposed_rhs.reshape(
            batch_shape + [rhs_shape[d] for d in non_batch_dims_rhs]
        )

        # Perform tensordot along the contraction dimensions
        # Adjust contraction dimensions for the reshaped tensors
        contract_dims_lhs_adj = [
            d + len(batch_shape) for d in contract_dims_lhs if d in non_batch_dims_lhs
        ]
        contract_dims_rhs_adj = [
            d + len(batch_shape) for d in contract_dims_rhs if d in non_batch_dims_rhs
        ]

        result = torch.tensordot(
            lhs_reshaped,
            rhs_reshaped,
            dims=(contract_dims_lhs_adj, contract_dims_rhs_adj),
        )

    return result.reshape(out_shape)


def custom_convolution(*args, **kwargs):
    """
    Custom implementation of generalized convolution operation.

    This decomposes ttir.convolution into permutations and conv_general_dilated following
    the TTIRToTTIRDecomposition pattern, with support for arbitrary tensor layouts.

    Args:
    - args[0]: Input tensor
    - args[1]: Weight tensor
    - args[2]: (Optional) Bias tensor

    Kwargs (from convolution_layout and other attributes):
    - input_batch: Input batch dimension
    - input_feature: Input feature dimension
    - input_spatial_dimensions: Input spatial dimensions (list)
    - kernel_output_feature: Kernel output feature dimension
    - kernel_input_feature: Kernel input feature dimension
    - kernel_spatial_dimensions: Kernel spatial dimensions (list)
    - output_batch: Output batch dimension
    - output_feature: Output feature dimension
    - output_spatial_dimensions: Output spatial dimensions (list)
    - window_strides: Convolution strides
    - padding: Padding for each spatial dimension
    - weight_dilation: Weight dilation
    - feature_group_count: Number of feature groups

    Returns:
    - Result of convolution with original layout preserved
    """
    if len(args) >= 2:
        input_tensor = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
    else:
        # Handle case where inputs are in a list
        input_tensor = args[0][0]
        weight = args[0][1]
        bias = args[0][2] if len(args[0]) > 2 else None

    # Extract layout information
    input_batch = kwargs.get("input_batch", 0)
    input_feature = kwargs.get("input_feature", 1)
    input_spatial_dims = kwargs.get("input_spatial_dimensions", [2, 3])
    kernel_output_feature = kwargs.get("kernel_output_feature", 0)
    kernel_input_feature = kwargs.get("kernel_input_feature", 1)
    kernel_spatial_dims = kwargs.get("kernel_spatial_dimensions", [2, 3])
    output_batch = kwargs.get("output_batch", 0)
    output_feature = kwargs.get("output_feature", 1)
    output_spatial_dims = kwargs.get("output_spatial_dimensions", [2, 3])

    # Extract convolution parameters
    window_strides = kwargs.get("window_strides", [1, 1])
    padding = kwargs.get("padding", [0, 0, 0, 0])
    weight_dilation = kwargs.get("weight_dilation", [1, 1])
    feature_group_count = int(kwargs.get("feature_group_count", 1))

    # Get tensor ranks
    input_rank = input_tensor.dim()
    weight_rank = weight.dim()

    # Ensure layout indices are valid for the tensor ranks
    input_batch = (
        min(input_batch, input_rank - 1) if input_batch >= 0 else input_rank - 1
    )
    input_feature = (
        min(input_feature, input_rank - 1) if input_feature >= 0 else input_rank - 1
    )
    input_spatial_dims = [
        min(d, input_rank - 1) if d >= 0 else i + 2
        for i, d in enumerate(input_spatial_dims)
    ]

    kernel_output_feature = (
        min(kernel_output_feature, weight_rank - 1)
        if kernel_output_feature >= 0
        else weight_rank - 1
    )
    kernel_input_feature = (
        min(kernel_input_feature, weight_rank - 1)
        if kernel_input_feature >= 0
        else weight_rank - 1
    )
    kernel_spatial_dims = [
        min(d, weight_rank - 1) if d >= 0 else i + 2
        for i, d in enumerate(kernel_spatial_dims)
    ]

    output_batch = (
        min(output_batch, input_rank - 1) if output_batch >= 0 else input_rank - 1
    )
    output_feature = (
        min(output_feature, input_rank - 1) if output_feature >= 0 else input_rank - 1
    )
    output_spatial_dims = [
        min(d, input_rank - 1) if d >= 0 else i + 2
        for i, d in enumerate(output_spatial_dims)
    ]

    # Determine number of spatial dimensions
    num_spatial = len([s for s in input_spatial_dims if s is not None])

    # Build input permutation to standard layout: [batch, channel, *spatial]
    input_perm = [input_batch, input_feature] + input_spatial_dims[:num_spatial]

    # Validate and apply input permutation
    if len(set(input_perm)) == len(input_perm) and all(
        0 <= idx < input_rank for idx in input_perm
    ):
        input_permuted = input_tensor.permute(*input_perm)
    else:
        input_permuted = input_tensor

    # Build weight permutation to standard layout: [output_feature, input_feature, *spatial]
    weight_perm = [kernel_output_feature, kernel_input_feature] + kernel_spatial_dims[
        :num_spatial
    ]

    # Validate and apply weight permutation
    if len(set(weight_perm)) == len(weight_perm) and all(
        0 <= idx < weight_rank for idx in weight_perm
    ):
        weight_permuted = weight.permute(*weight_perm)
    else:
        weight_permuted = weight

    # Parse padding for spatial dimensions
    spatial_padding = []
    for i in range(num_spatial):
        pad_start = padding[2 * i] if len(padding) > 2 * i else 0
        pad_end = padding[2 * i + 1] if len(padding) > 2 * i + 1 else 0
        # Use symmetric padding (first start, then end)
        spatial_padding.append((pad_start, pad_end))

    # Get strides and dilations for spatial dimensions (may need to handle variable sizes)
    strides = (
        window_strides[:num_spatial]
        if isinstance(window_strides, (list, tuple))
        else [window_strides] * num_spatial
    )
    dilations = (
        weight_dilation[:num_spatial]
        if isinstance(weight_dilation, (list, tuple))
        else [weight_dilation] * num_spatial
    )

    try:
        # For 2D convolution
        if num_spatial == 2:
            result = torch.nn.functional.conv2d(
                input_permuted,
                weight_permuted,
                bias=bias,
                stride=strides,
                padding=(
                    spatial_padding[0][0],
                    spatial_padding[1][0],
                ),  # Use start padding
                dilation=dilations,
                groups=feature_group_count,
            )
        # For 1D convolution
        elif num_spatial == 1:
            result = torch.nn.functional.conv1d(
                input_permuted,
                weight_permuted,
                bias=bias,
                stride=strides,
                padding=spatial_padding[0][0],
                dilation=dilations,
                groups=feature_group_count,
            )
        # For 3D convolution
        elif num_spatial == 3:
            result = torch.nn.functional.conv3d(
                input_permuted,
                weight_permuted,
                bias=bias,
                stride=strides,
                padding=(
                    spatial_padding[0][0],
                    spatial_padding[1][0],
                    spatial_padding[2][0],
                ),
                dilation=dilations,
                groups=feature_group_count,
            )
        else:
            # Fallback: return input unchanged if unsupported spatial dims
            return input_tensor
    except Exception as e:
        # Fallback on error
        return input_tensor

    # Build output permutation to restore original layout
    # We need to build a permutation that transforms from standard layout [batch, channel, *spatial]
    # back to the original layout specified by [output_batch, output_feature, output_spatial_dims]
    result_rank = result.dim()

    # Only apply output permutation if we have valid layout information and result has expected rank
    if result_rank == input_rank:
        # Create the target layout: where each dimension should go
        output_layout = [output_batch, output_feature] + output_spatial_dims[
            :num_spatial
        ]

        # Create inverse permutation: from standard layout positions to original layout positions
        # If standard position i needs to go to original position output_layout[i],
        # then we need to find where output_layout[i] currently is and move it to position i
        if len(output_layout) >= result_rank:
            # Build the permutation: for each position in result, where did it come from in standard layout?
            output_perm = []
            for orig_pos in range(result_rank):
                # Find which standard position should go to orig_pos
                # This is the position of orig_pos in output_layout
                try:
                    std_pos = output_layout.index(orig_pos)
                    output_perm.append(std_pos)
                except ValueError:
                    # orig_pos not in output_layout, just keep it in place
                    output_perm.append(orig_pos)

            # Validate permutation
            if len(output_perm) == result_rank and len(set(output_perm)) == result_rank:
                if all(0 <= idx < result_rank for idx in output_perm):
                    result = result.permute(*output_perm)

    return result


def custom_pooling(*args, **kwargs):
    """
    Custom implementation of generalized pooling operation.

    This decomposes ttir.pooling into pool operations following the
    PoolingToPool2dPattern, with proper permutations to handle arbitrary layouts and spatial dimensions.

    Args:
    - args[0]: Input tensor

    Kwargs:
    - pooling_method: "Max", "Average", or "Sum"
    - window_dimensions: Kernel size for each dimension
    - window_strides: Stride for each dimension
    - padding: Padding for each dimension [d0_start, d0_end, d1_start, d1_end, ...]
    - window_dilations: Dilation for each dimension (optional)

    Returns:
    - Result of pooling with original layout preserved
    """
    # Handle both cases: args is a single tensor or a list of tensors
    if isinstance(args, tuple) and len(args) > 0:
        input_tensor = args[0]
    else:
        input_tensor = args[0] if isinstance(args, (list, tuple)) else args

    # Extract pooling parameters
    pooling_method = kwargs.get("pooling_method", "Max")
    window_dimensions = kwargs.get("window_dimensions", [1, 1, 3, 3])
    window_strides = kwargs.get("window_strides", [1, 1, 1, 1])
    padding = kwargs.get("padding", [0, 0, 0, 0, 0, 0, 0, 0])
    window_dilations = kwargs.get("window_dilations", [1, 1, 1, 1])

    # Ensure all are lists
    if not isinstance(window_dimensions, (list, tuple)):
        window_dimensions = (
            list(window_dimensions)
            if hasattr(window_dimensions, "__iter__")
            else [window_dimensions]
        )
    if not isinstance(window_strides, (list, tuple)):
        window_strides = (
            list(window_strides)
            if hasattr(window_strides, "__iter__")
            else [window_strides]
        )
    if not isinstance(padding, (list, tuple)):
        padding = list(padding) if hasattr(padding, "__iter__") else [padding]
    if not isinstance(window_dilations, (list, tuple)):
        window_dilations = (
            list(window_dilations)
            if hasattr(window_dilations, "__iter__")
            else [window_dilations]
        )

    # Get tensor rank
    rank = input_tensor.dim()

    # Find spatial dimensions (dimensions where window size > 1)
    spatial_dim_indices = []
    for i, dim_size in enumerate(window_dimensions):
        if i < rank and dim_size > 1:  # Only consider indices within tensor rank
            spatial_dim_indices.append(i)

    # Default to last two dimensions if no spatial dimensions found
    if len(spatial_dim_indices) == 0:
        spatial_dim_indices = [max(0, rank - 2), max(1, rank - 1)]

    # Limit to actual spatial dimensions available
    num_spatial = len(spatial_dim_indices)
    spatial_dim_indices = spatial_dim_indices[:num_spatial]

    # Ensure spatial dimensions are within rank
    spatial_dim_indices = [min(idx, rank - 1) for idx in spatial_dim_indices]

    # Build permutation to move spatial dimensions to the end in order
    # Standard layout for pooling: [batch, channel, *spatial]
    # We want: [batch/channel dims..., spatial dims...]
    non_spatial_dims = [i for i in range(rank) if i not in spatial_dim_indices]
    permutation = non_spatial_dims + spatial_dim_indices

    # Verify permutation is valid
    if len(permutation) != rank or len(set(permutation)) != rank:
        permutation = list(range(rank))

    # Apply permutation
    input_permuted = (
        input_tensor.permute(*permutation)
        if len(set(permutation)) == rank
        else input_tensor
    )

    # Compute inverse permutation for later
    inverse_perm = [0] * rank
    for i, p in enumerate(permutation):
        inverse_perm[p] = i

    # Save the original input shape (before permutation) for restoration
    original_input_shape = tuple(input_tensor.shape)

    # Extract kernel size, stride, padding, and dilation for spatial dimensions
    kernel_params = []
    stride_params = []
    dilation_params = []

    for i, spatial_idx in enumerate(spatial_dim_indices):
        kernel = (
            int(window_dimensions[spatial_idx])
            if spatial_idx < len(window_dimensions)
            else 1
        )
        stride = (
            int(window_strides[spatial_idx]) if spatial_idx < len(window_strides) else 1
        )
        dilation = (
            int(window_dilations[spatial_idx])
            if spatial_idx < len(window_dilations)
            else 1
        )

        kernel_params.append(kernel)
        stride_params.append(stride)
        dilation_params.append(dilation)

    # Extract padding for spatial dimensions
    pad_params = []
    for i, spatial_idx in enumerate(spatial_dim_indices):
        pad_start = (
            int(padding[2 * spatial_idx]) if len(padding) > 2 * spatial_idx else 0
        )
        pad_end = (
            int(padding[2 * spatial_idx + 1])
            if len(padding) > 2 * spatial_idx + 1
            else 0
        )
        pad_params.append((pad_start, pad_end))

    # Reshape input to [batch, channel, *spatial] for PyTorch pooling
    # Move channels to position 1 if they're not already there
    input_shape = input_permuted.shape
    batch_channel_dims = rank - num_spatial

    # Flatten batch and channel dimensions into first dimension for pooling
    input_reshaped = input_permuted.reshape(-1, *input_shape[batch_channel_dims:])

    # Apply pooling based on method and number of spatial dimensions
    try:
        if num_spatial == 1:
            # Reshape for 1D pooling: [batch*channel, 1, spatial]
            input_for_pool = input_reshaped.unsqueeze(1)

            if pooling_method == "Max":
                result = torch.nn.functional.max_pool1d(
                    input_for_pool,
                    kernel_size=kernel_params[0],
                    stride=stride_params[0],
                    padding=pad_params[0][0],
                    dilation=dilation_params[0],
                )
            elif pooling_method == "Average":
                result = torch.nn.functional.avg_pool1d(
                    input_for_pool,
                    kernel_size=kernel_params[0],
                    stride=stride_params[0],
                    padding=pad_params[0][0],
                )
            elif pooling_method == "Sum":
                result = torch.nn.functional.avg_pool1d(
                    input_for_pool,
                    kernel_size=kernel_params[0],
                    stride=stride_params[0],
                    padding=pad_params[0][0],
                )
                result = result * kernel_params[0]

            result = result.squeeze(1)

        elif num_spatial == 2:
            if pooling_method == "Max":
                result = torch.nn.functional.max_pool2d(
                    input_reshaped,
                    kernel_size=tuple(kernel_params),
                    stride=tuple(stride_params),
                    padding=(pad_params[0][0], pad_params[1][0]),
                    dilation=tuple(dilation_params),
                )
            elif pooling_method == "Average":
                result = torch.nn.functional.avg_pool2d(
                    input_reshaped,
                    kernel_size=tuple(kernel_params),
                    stride=tuple(stride_params),
                    padding=(pad_params[0][0], pad_params[1][0]),
                )
            elif pooling_method == "Sum":
                result = torch.nn.functional.avg_pool2d(
                    input_reshaped,
                    kernel_size=tuple(kernel_params),
                    stride=tuple(stride_params),
                    padding=(pad_params[0][0], pad_params[1][0]),
                )
                result = result * (kernel_params[0] * kernel_params[1])

        elif num_spatial == 3:
            if pooling_method == "Max":
                result = torch.nn.functional.max_pool3d(
                    input_reshaped,
                    kernel_size=tuple(kernel_params),
                    stride=tuple(stride_params),
                    padding=(pad_params[0][0], pad_params[1][0], pad_params[2][0]),
                    dilation=tuple(dilation_params),
                )
            elif pooling_method == "Average":
                result = torch.nn.functional.avg_pool3d(
                    input_reshaped,
                    kernel_size=tuple(kernel_params),
                    stride=tuple(stride_params),
                    padding=(pad_params[0][0], pad_params[1][0], pad_params[2][0]),
                )
            elif pooling_method == "Sum":
                result = torch.nn.functional.avg_pool3d(
                    input_reshaped,
                    kernel_size=tuple(kernel_params),
                    stride=tuple(stride_params),
                    padding=(pad_params[0][0], pad_params[1][0], pad_params[2][0]),
                )
                result = result * (
                    kernel_params[0] * kernel_params[1] * kernel_params[2]
                )
        else:
            # Unsupported number of spatial dimensions
            return input_tensor

        # Reshape result back to proper dimensions
        # Result is currently [flattened_batch_channel, *output_spatial]
        # We need to unflatten to [batch, channel, *output_spatial]
        output_spatial_shape = result.shape[1:]
        batch_channel_shape = input_shape[:batch_channel_dims]
        result = result.reshape(*batch_channel_shape, *output_spatial_shape)

    except Exception as e:
        # Fallback on error
        return input_tensor

    # Apply inverse permutation to restore original layout
    # First reshape to match the permuted shape structure
    # The permuted tensor has structure: [non_spatial dims..., spatial dims...]
    # We need to convert back to: [all dims...]
    if len(set(permutation)) == rank:
        result = result.permute(*inverse_perm)

    return result


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
    "ttir.argmax": OpMapping(
        custom_argmax, {"dim_arg": "dim", "keep_dim": "keepdim"}, unpack_inputs=False
    ),
    "ttir.batch_norm_inference": OpMapping(
        custom_batch_norm_inference,
        {
            "dimension": "dimension",
            "epsilon": "epsilon",
        },
        unpack_inputs=False,
    ),
    "ttir.pad": OpMapping(
        custom_pad,
        {
            "padding": "padding",
            "value": "value",
        },
        unpack_inputs=False,
    ),
    "ttir.dot_general": OpMapping(
        custom_dot_general,
        {
            "batch_dims_lhs": "batch_dims_lhs",
            "contract_dims_lhs": "contract_dims_lhs",
            "batch_dims_rhs": "batch_dims_rhs",
            "contract_dims_rhs": "contract_dims_rhs",
        },
        unpack_inputs=False,
    ),
    "ttir.convolution": OpMapping(custom_convolution),
    "ttir.pooling": OpMapping(custom_pooling),
}
