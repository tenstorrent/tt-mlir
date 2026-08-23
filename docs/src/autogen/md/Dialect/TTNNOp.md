# TTNN Dialect Ops

Auto-generated reference of operations in the `TTNN` dialect (187 ops).

## `ttnn.abs`

Eltwise absolute.

Eltwise absolute operation.

## `ttnn.acos`

Eltwise arccosine op.

Performs an elementwise arccosine (`acos`) on the input tensor; result
range [0, π] for inputs in [-1, 1]. Floating-point tensors.

## `ttnn.adamw`

AdamW optimizer step (ttml).

Fused AdamW optimizer step, backed by the ttml metal op
`ttml::metal::adamw`. Updates the parameter tensor from its gradient and the
first/second moment estimates, and updates the moments in place.

`param`, `exp_avg`, `exp_avg_sq` and the optional `max_exp_avg_sq` (used for
AMSGrad) are all mutated in place, so the op has no results. `amsgrad` is
implied by the presence of `max_exp_avg_sq`.

## `ttnn.add`

Eltwise add.

Eltwise add operation.

## `ttnn.aggregate_tensor`

Aggregate distributed tensor back to host

Aggregates a multi-device tensor back into a single host-side tensor
according to the specified composer configuration.

This operation takes a distributed tensor and creates a single-device
host tensor aggregated according to the MeshComposerConfig configuration.

// Note: In the ttnn API, the aggregate_tensor op takes an input tensor and a MeshToTensor object as its arguments.
// To create a MeshToTensor object, the API create_mesh_composer is used, which requires both MeshComposerConfig and MeshDevice.
// Instead of modeling create_mesh_composer as a separate op in the IR, these two strongly coupled APIs are combined into one aggregate_tensor op
// to keep the IR simple and less cluttered. If a clear need arises in the future, splitting them into separate ops can be considered.

Example:
```mlir
%result = "ttnn.aggregate_tensor"(%input, %device) <{
  composer_config = #ttnn.mesh_composer_config<
    dims = [1],
    mesh_shape_override = #ttnn.mesh_shape<2x2>
  >
}> : (tensor<32x64xbf16>, !ttnn.device) -> tensor<32x64xbf16>
```

## `ttnn.all_gather`

All gather op.

Tensor All Gather operation

## `ttnn.all_reduce`

All reduce op.

Tensor All Reduce operation

## `ttnn.all_reduce_async`

Asynchronous all reduce op.

Asynchronous all-reduce collective communication operation. Performs a
reduction (e.g. sum) across all devices in a mesh and distributes the
result back to all devices, using async execution to overlap
communication with computation.

## `ttnn.all_to_all_combine`

Combine expert outputs back to original token positions.

Inverse of dispatch: gathers expert computation results from expert devices
and restores tokens to their original device and order.

## `ttnn.all_to_all_dispatch`

Dispatch tokens to expert devices for MoE computation.

Routes tokens to devices holding their selected experts via all-to-all
communication. Used before sparse_matmul in the MoE dispatch/combine flow.

## `ttnn.all_to_all_dispatch_metadata`

Dispatch tokens with metadata to expert devices for MoE computation.

Routes tokens along with expert scores to devices holding their selected
experts via all-to-all communication within a ring (cluster_axis).
Returns dispatched tokens (sparse — only routed token slots are filled),
all-gathered expert indices, and all-gathered expert scores.

Dimensions:
  M = tokens per ring device
  K = selected experts per token
  D = total devices, E = total experts

Input shapes (per device, after mesh sharding):
  - input_tensor:   [1, 1, M, H]
  - expert_indices: [1, 1, M, K]
  - expert_scores:  [1, 1, M, K]
  - expert_mapping: [D, E]  entry [d, e] = device ID owning expert e

Output shapes (per device, num_devices * M = tokens_global):
  - dispatched: [1, tokens_global, H]
  - indices:    [1, tokens_global, K]  (all-gathered across ring)
  - scores:     [1, tokens_global, K]  (all-gathered across ring)

This op runs in tt-metal persistent mode only: the three output buffers
(`dispatched_buffer`/`indices_buffer`/`scores_buffer`) and the
`cross_device_semaphore` are materialized in the function prelude by the
`DistributedOpInterface` hooks and passed to the kernel. They are modeled
as `Optional` only because conversion emits them unbound and the prelude
pass binds them later; a finalized op always has all four bound. The drain
core (where indices/scores are sharded) is fixed internally, so it is not
an op input — the kernel derives it from the shard spec.

## `ttnn.alloc`

Alloc op.

Tensor Alloc operation

## `ttnn.allocate_moe_compute_semaphore`

Allocate the moe_compute A2A combine semaphore.

Creates the cross-device global semaphore used by `ttnn.moe_compute`'s
A2A selective-reduce-combine. The combine cores are placed dynamically by
tt-metal, so this op carries the placement inputs as attributes and its
runtime handler queries the cores (`get_moe_combine_cores`) and creates
the semaphore on them.

Example:
```mlir
%semaphore = "ttnn.allocate_moe_compute_semaphore"(%device) <{output_height_shard_dim = 2 : ui32, hidden_size = 1280 : ui32, initial_value = 0 : ui32, mux_core_range_set = #ttnn.core_range_set<[#ttnn.core_range<(1,1), (3,3)>]>}> : (!ttnn.device) -> !ttnn.global_semaphore
```

## `ttnn.arange`

Arange operation.

Tensor arange operation.

Produces a (1, 1, 1, N)-shaped tensor with values from `start` to `end` (exclusive) with a step size of `step`.

Examples:
  %0 = "ttnn.arange"() {start = 0 : i64, end = 5 : i64 step = 1 : i64} : () -> tensor<1x1x1x5xi64>
  // %0: [[[[0, 1, 2, 3, 4]]]]

  %1 = "ttnn.arange"() {start = 0 : i64, end = 10 : i64, step = 2 : i64} : () -> tensor<1x1x1x5xf32>
  // %1: [[[[0.0, 2.0, 4.0, 6.0, 8.0]]]]

## `ttnn.argmax`

Argmax reduction op.

Determine the indices of the maximum values along a specified dimension of a tensor or over all elements in a tensor.

Parameters:
  - `input`: The input tensor.
  - `dim`: Specifies the dimension along which the argmax is applied.
  - `keep_dim`: If set to true, the output tensor will have the same number of dimensions as the input tensor.

IR usage:
// Input tensor of shape (128, 28, 28, 64)
%input = ... : tensor<128x28x28x64xbf16>

%empty = "ttnn.empty"(%0) <{....}> : -> tensor<128x28x28xi32>
%4 = "ttnn.argmax"(%input, %empty) <{dim = 3 : i32}> : (tensor<128x28x28xbf16>, tensor<128x28x28xi32) -> tensor<128x28x28xi32>

Example:
  input: [[1, 5, 3],
          [2, 4, 6]]

  // Computing along dim 0
  output: [1, 0, 1]

  // Computing along dim 1
  output: [1, 2]

  // Computing for entire tensor
  output: 5

## `ttnn.asin`

Eltwise arcsine op.

Performs an elementwise arcsine (`asin`) on the input tensor; result
range [-π/2, π/2] for inputs in [-1, 1]. Floating-point tensors.

## `ttnn.asinh`

Eltwise inverse hyperbolic sine op.

Performs an elementwise inverse hyperbolic sine (`asinh`) on the input
tensor. Accepts all real-valued inputs. Floating-point tensors.

## `ttnn.assign`

Assign Tensor

Returns a new tensor which is a new copy of input tensor.
Alternatively, copies input tensor ``input`` to ``optional_output_tensor``
if their shapes and memory layouts match, and returns input_b tensor.
Input tensors can be of any data type.
Output tensor will be of same data type as Input tensor.

## `ttnn.atan`

Eltwise arctangent op.

Performs an elementwise arctangent (`atan`) operation on the input tensor.
This operation computes the inverse tangent of each element, returning
values in the range [-π/2, π/2]. Supports floating-point tensor types.

Example:

```mlir
%input = tensor<4xf32> {1.0, 0.5, 0.0, -1.0}
%result = "ttir.atan"(%input) : (tensor<4xf32>) -> tensor<4xf32>
```

Given the input `[1.0, 0.5, 0.0, -1.0]`, the result would be approximately:
`[0.785, 0.464, 0.0, -0.785]` (values in radians).

## `ttnn.atan2`

Eltwise atan2 OP.

Performs element-wise atan2 operation on lhs and rhs tensor and produces a result
tensor.

Example:
```
  // %lhs: [0.0, 1.0, -1.0]
  // %rhs: [1.0, 0.0, 0.0]
  %result = "ttnn.atan2"(%lhs, %rhs) : (tensor<3xf64>, tensor<3xf64>) -> tensor<3xf64>
  // %result: [0.0, 1.57079637, -1.57079637] // [0.0, pi/2, -pi/2]
```

## `ttnn.avg_pool2d`

Applies a 2D average pooling over an input signal composed of several input planes.

It is a downsampling operation to reduce the spatial dimensions (height and width) of a input tensor by computing averages with in a window.

Example:
  // 3x3 input tensor
  input: [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]
  kernel_height: 2
  kernel_width: 2
  stride_height: 1
  stride_width: 1
  dilation_height: 1
  dilation_width: 1
  output: [[3, 4],
           [6, 7]]

## `ttnn.batch_norm_inference`

Batch normalization inference op.

Batch normalization operation for inference over each channel on input tensor.
Uses pre-computed mean and variance.

## `ttnn.batch_norm_training`

Batch normalization training op.

Batch normalization operation for training over each channel on input tensor.
Computes batch statistics and updates running mean and variance.

## `ttnn.begin_trace_capture`

Begin trace capture.

Begins trace capture. Returns a scalar tensor containing the trace id.
Inputs:
  - `device` TTNN_Device: The device to capture the trace on.
  - `cq_id` ui32: The command queue to capture the trace with. Must be 0 or 1.
Outputs:
  - `trace_id` AnyRankedTensor: The scalar trace id tensor containing the trace id.

## `ttnn.bitcast_convert`

Bitcast_convert op.

This op reinterprets the bits of each element in
the input tensor as a data type of the output tensor.

The output data type is derived from the result tensor's TTNNLayoutAttr
encoding via the TTNN_DtypeOpInterface.

## `ttnn.bitwise_and`

Eltwise bitwise AND.

Performs element-wise bitwise AND of two tensors `lhs` and `rhs`
and produces a `result` tensor.

Example:
    // %lhs: [[1, 2], [3, 4]]
    // %rhs: [[5, 6], [7, 8]]
    %result = "ttnn.bitwise_and"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    // %result: [[1, 2], [3, 0]]

## `ttnn.bitwise_not`

Eltwise bitwise NOT.

Performs element-wise NOT of tensor `operand` and produces a `result` tensor.

Example:
    // Bitwise operation with with integer tensors
    // %operand: [[1, 2], [3, 4]]
    %result = "ttnn.bitwise_not"(%operand) : (tensor<2x2xi32>) -> tensor<2x2xi32>
    // %result: [[-2, -3], [-4, -5]]

## `ttnn.bitwise_or`

Eltwise bitwise OR.

Performs element-wise bitwise OR of two tensors `lhs` and `rhs`
and produces a `result` tensor.

Example:
    // %lhs: [[1, 2], [3, 4]]
    // %rhs: [[5, 6], [7, 8]]
    %result = "ttnn.bitwise_or"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    // %result: [[5, 6], [7, 12]]

## `ttnn.bitwise_xor`

Eltwise bitwise XOR.

Performs element-wise bitwise XOR of two tensors `lhs` and `rhs`
and produces a `result` tensor.

Example:
  // %lhs: [[1, 2], [3, 4]]
  // %rhs: [[5, 6], [7, 8]]
  %result = "ttnn.bitwise_xor"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  // %result: [[4, 4], [4, 12]]

## `ttnn.capture_or_execute_trace`

Capture or execute trace.

Captures or executes the trace. Will have read/write memory effects on the cached trace data.
If the trace data exists (meaning the trace was captured previously), it will be executed with
the execute_callee function. Otherwise, the trace is captured.

Capturing is split across two functions so that the persistent input/output slots are
allocated only once - on the very first capture - and reused for every recapture:
  - `allocate_slots_callee` allocates the persistent device input/output slots. It runs
    exactly once per trace.
  - `capture_callee` takes those slots as arguments and captures the trace against them.

A cached trace that has gone stale is recaptured by invoking `capture_callee` again with the
same slots. Because the slots are never reallocated, a recapture leaves the device allocator
state untouched and therefore cannot invalidate any other cached trace. The initial capture and
every recapture run the identical program, so anything established by the first capture (compiled
programs, program cache entries) holds for every recapture by construction.

Inputs:
  - `device` TTNN_Device: The device where the trace was captured.
  - `allocate_slots_callee` FlatSymbolRefAttr: The symbol of the slot allocation function.
  - `capture_callee` FlatSymbolRefAttr: The symbol of the capture trace function.
  - `execute_callee` FlatSymbolRefAttr: The symbol of the execute trace function.
  - `inputs` Variadic<AnyRankedTensor>: The input tensors to the trace function.
  - `semaphore_inputs` Variadic<TTNN_GlobalSemaphore>: Global semaphores forwarded into the capture/execute programs.
Outputs:
  - `results` Variadic<AnyRankedTensor>: The output tensors from the trace function.

## `ttnn.cbrt`

Eltwise cubic root.

Eltwise cubic root operation.

## `ttnn.ceil`

Eltwise ceil.

Eltwise ceil operation.

## `ttnn.chunked_scaled_dot_product_attention`

Chunked prefill scaled dot product attention over a paged KV cache.

Chunked-prefill attention. A prefill chunk of `query` attends
causally over the prefix `[0, chunk_start_idx + chunk_len)` resident in the
paged K/V cache, read on device via `page_table`, with the prefix offset given
by the device tensor `chunk_start_idx` (`[1]` int32).

Because the prefix offset is a device tensor (not a host scalar) and the page
table is consumed on device, the op holds no host-side state: it can be
captured inside a trace and replayed across invocations with a different
`chunk_start_idx` without recompiling.

## `ttnn.clamp_scalar`

Clamp op.

Clamp tensor values to a specified range.

Example:
  min: 2.000000+00
  input: [[0, 1, 2, 3, 4, 5, 6, 7]]
  max: 5.000000+00

  "ttnn.clamp_scalar"(%arg0) <{max = 2.000000e+00 : f32, min = 5.000000e+00 : f32}>
  -> %out = [[2, 2, 2, 3, 4, 5, 5, 5]]

## `ttnn.clamp_tensor`

Clamp op.

Clamp tensor values to a specified range using min/max as tensor.

Example:
  min:   [[2, 2, 2, 3, 3, 3, 0, 0]]
  input: [[0, 1, 2, 3, 4, 5, 6, 7]]
  max:   [[5, 5, 5, 9, 9, 9, 6, 6]]

  "ttnn.clamp_tensor"(%input, %min, %max)
  %out:  [[2, 2, 2, 3, 4, 5, 6, 6]]

## `ttnn.concat`

Concat op.

Concat tensors along a given dimension.

## `ttnn.concatenate_heads`

Concatenate heads op used in attention layer.

Takes in a tensor of shape [batch_size, num_heads, sequence_size, head_size],
concatenates heads back along the width dimension and returns the tensor
of shape [batch_size, sequence_size, num_heads * head_size].

## `ttnn.constant`

Constant op.

Produces tensor filled with given constant value.

Examples:
  %0 = "ttnn.constant"() {value = dense<[[3, 4, 2], [1, 7, 8]]> : tensor<2x3xui16>} : () -> tensor<2x3xui16>
  // %0: [[3, 4, 2], [1, 7, 8]]
  %1 = "ttnn.constant"() {value = dense<[0.2, 1.3]> : tensor<2xf32>} : () -> tensor<2xf32>
  // %1: [0.2, 1.3]

## `ttnn.conv1d`

Conv1d operation.

Applies a 1D convolution over an input signal composed of several input planes.

This op models the `ttnn::conv1d` library function, which is itself a thin wrapper
around `ttnn::conv2d` (the input is reshaped to a height-1 image and the 1D
parameters are mapped to their 2D equivalents).

Inputs:
- `input` (AnyRankedTensor): expected in the following channels-last format (N, L_in, C) where:
  - N is the batch size
  - L_in is the length of the input signal
  - C is the number of input channels
- `weight` (AnyRankedTensor): expected in the following format (O, C/G, K) where:
  - O is the number of output channels
  - C is the number of input channels
  - G is the number of groups
  - K is the length of the kernel
- `bias` (Optional<AnyRankedTensor>): expected in the following format (1, 1, 1, O)
  (matching `ttnn::conv2d`, which `ttnn::conv1d` delegates to).

Attributes:
- `in_channels` (i32): The number of input channels.
- `out_channels` (i32): The number of output channels.
- `batch_size` (i32): The batch size.
- `input_length` (i32): The length of the input signal.
- `kernel_size` (i32): The length of the kernel K.
- `stride` (i32): The stride of the kernel window.
- `padding` (array<2xi32>): [pL, pR] padding added to the left and right of the input.
- `dilation` (i32): The spacing between kernel elements.
- `groups` (i32): Number of blocked connections from input channels to output channels. Input and output channels must both be divisible by groups.

Outputs:
- `result` (AnyRankedTensor): returned in `ttnn::conv2d`'s flattened layout
  (1, 1, N * L_out, O) where:
  - `L_out = (L_in + pL + pR - dilation * (K - 1) - 1) / stride + 1`
  A reshape back to (N, L_out, O) is inserted during lowering from TTIR.

## `ttnn.conv2d`

Conv2d operation.

Applies a 2D convolution over an input image composed of several input planes.

Inputs:
- `input` (AnyRankedTensor): expected in the following flattened format (1, 1, N * H_in * W_in, C) where:
  - N is the batch size
  - H_in is the height of the input planes
  - W_in is the width of the input planes
  - C is the number of channels
- `weight` (AnyRankedTensor): expected in the following format (O, C/G, K_H, K_W).
- `bias` (Optional<AnyRankedTensor>): expected in the following format (1, 1, 1, O) where:
  - C is the number of input channels
  - O is the number of output channels
  - G is the number of groups
  - K_H is the height of the kernel
  - K_W is the width of the kernel

Attributes:
- `in_channels` (i32): The number of input channels.
- `out_channels` (i32): The number of output channels.
- `batch_size` (i32): The batch size.
- `input_height` (i32): The input height.
- `input_width` (i32): The input width.
- `kernel_size` (array<2xi32>): [K_H, K_W] where K_H is the kernel height and K_W is the kernel width.
- `stride` (array<2xi32>): [sH, sW] where sH is stride for height and sW is stride for width.
- `padding` (array<2xi32> | array<4xi32>):
  - array<2xi32>: [pH, pW] where pH is padding for height (top/bottom) and pW is padding for width (left/right).
  - array<4xi32>: [pT, pB, pL, pR] for top, bottom, left, and right padding respectively.
- `dilation` (array<2xi32>): [dH, dW] where dH is dilation for height and dW is dilation for width.
- `groups` (i32): Number of blocked connections from input channels to output channels. Input and output channels must both be divisible by groups.

Outputs:
- `result` (AnyRankedTensor): returned in the following flattened format (1, 1, N * H_out * W_out, O) where:
  - `H_out = (H_in + pT + pB - dH * (K_H - 1) - 1) / sH + 1`
  - `W_out = (W_in + pL + pR - dW * (K_W - 1) - 1) / sW + 1`

Example:
  %input = ttir.empty() : () -> tensor<1x1x1024x64xbf16>
  %weight = ttir.empty() : () -> tensor<64x64x3x3xbf16>
  %bias = ttir.empty() : () -> tensor<1x1x1x64xbf16>
  %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %0 = "ttnn.conv2d"(%input, %weight, %bias, %device)
    <{
      in_channels = 64: i32,
      out_channels = 64: i32,
      batch_size = 1: i32,
      input_height = 32: i32,
      input_width = 32: i32,
      kernel_size = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      padding = array<i32: 0, 0>,
      dilation = array<i32: 1, 1>,
      groups = 1: i32
    }> : (tensor<1x1x1024x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, !ttnn.device) -> tensor<1x1x900x64xbf16>

## `ttnn.conv3d`

Conv3d operation.

Applies a 3D convolution over an input volume composed of several input planes.

Inputs:
- `input` (AnyRankedTensor): expected in the following format (N, D, H, W, C) where:
  - N is the batch size
  - D is the depth of the input volume
  - H is the height of the input planes
  - W is the width of the input planes
  - C is the number of input channels
- `weight` (AnyRankedTensor): expected in the following format (K_D * K_H * K_W * C / G, O) where:
  - K_D is the depth of the kernel
  - K_H is the height of the kernel
  - K_W is the width of the kernel
  - C is the number of input channels
  - O is the number of output channels
  - G is the number of groups
  The spatial kernel dimensions and input channels are flattened together into a 2D tensor.
- `bias` (Optional<AnyRankedTensor>): expected in the following format (1, O) where:
  - O is the number of output channels

Attributes:
- `in_channels` (i32): The number of input channels.
- `out_channels` (i32): The number of output channels.
- `batch_size` (i32): The batch size.
- `input_depth` (i32): The input depth.
- `input_height` (i32): The input height.
- `input_width` (i32): The input width.
- `kernel_size` (array<3xi32>): [K_D, K_H, K_W] where K_D is the kernel depth, K_H is the kernel height and K_W is the kernel width.
- `stride` (array<3xi32>): [sD, sH, sW] where sD is stride for depth, sH for height and sW is stride for width.
- `padding` (array<3xi32>): [pD, pH, pW] where pD is padding for depth, pH is padding for height and pW is padding for width.
  Padding is symmetric (same on both sides of each dimension).
- `padding_mode` (StrAttr): "zeros" or "replicate" - padding fill strategy.
- `groups` (i32): Number of blocked connections from input channels to output channels. Input and output channels must both be divisible by groups.

Outputs:
- `result` (AnyRankedTensor): returned in the following format (N, D_out, H_out, W_out, O) where:
  - `D_out = (D_in + 2*pD - K_D) / sD + 1`
  - `H_out = (H_in + 2*pH - K_H) / sH + 1`
  - `W_out = (W_in + 2*pW - K_W) / sW + 1`

Example:
  %input = ttir.empty() : () -> tensor<1x28x28x28x32xbf16>
  %weight = ttir.empty() : () -> tensor<864x64xbf16>  // 864 = 3*3*3*32 (2D tensor)
  %bias = ttir.empty() : () -> tensor<1x64xbf16>      // 2D tensor
  %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %0 = "ttnn.conv3d"(%input, %weight, %bias, %device)
    <{
      in_channels = 32: i32,
      out_channels = 64: i32,
      batch_size = 1: i32,
      input_depth = 28: i32,
      input_height = 28: i32,
      input_width = 28: i32,
      kernel_size = array<i32: 3, 3, 3>,
      stride = array<i32: 1, 1, 1>,
      padding = array<i32: 0, 0, 0>,
      padding_mode = "zeros",
      groups = 1: i32
    }> : (tensor<1x28x28x28x32xbf16>, tensor<864x64xbf16>, tensor<1x64xbf16>, !ttnn.device) -> tensor<1x26x26x26x64xbf16>

## `ttnn.conv_transpose2d`

ConvTranspose2d operation.

Applies a 2D transposed convolution operator over an input image composed of several input planes.

Inputs:
  - `input` AnyRankedTensor: expected in the following format (N, H_in, W_in, C) where:
    - N is the batch size
    - H_in is the height of the input planes
    - W_in is the width of the input planes
    - C is the number of channels

  - `weight` AnyRankedTensor: expected in the following format (C, O/G, K_H, K_W).
  - `bias` Optional<AnyRankedTensor>: expected in the following format (1, 1, 1, O) where:
    - C is the number of input channels
    - O is the number of output channels
    - G is the number of groups
    - K_H is the height of the kernel
    - K_W is the width of the kernel

  - `output` AnyRankedTensor: expected in the following format (N, H_out, W_out, O) where:
    - H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (K_H - 1) + output_padding[0] + 1
    - W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (K_W - 1) + output_padding[1] + 1

Attributes:
  - `in_channels` i32: The number of input channels.
  - `out_channels` i32: The number of output channels.
  - `batch_size` i32: The batch size.
  - `input_height` i32: The input height.
  - `input_width` i32: The input width.
  - `kernel_size` array<2xi32>: The kernel size.
  - `stride` array<2xi32>: Controls the stride for the cross-correlation.
  - `padding` array<2xi32>: Controls the amount of implicit zero padding on both sides for dilation * (kernel_size - 1) - padding number of points.
  - `output_padding` array<2xi32>: Controls the additional size added to one side of the output shape.
  - `dilation` array<2xi32>: Controls the spacing between the kernel points
  - `groups` i32: Controls the connections between inputs and outputs. Must be divisible by input and output channels.

Example:
  // %input: tensor<3x8x8x256xbf16>
  // %weight: tensor<256x256x3x3xbf16>
  // %bias: tensor<1x1x1x256xbf16>
  // %output: tensor<3x10x10x256xbf16>
  %0 = "ttnn.conv_transpose2d"(%input, %weight, %bias, %output, %device)
    <{
      batch_size = 3: i32,
      dilation = array<i32: 1, 1>,
      groups = 1: i32,
      in_channels = 256: i32,
      input_height = 8: i32,
      input_width = 8: i32,
      kernel_size = array<i32: 3, 3>,
      out_channels = 256: i32,
      output_padding = array<i32: 0, 0>,
      padding = array<i32: 0, 0>,
      stride = array<i32: 1, 1>
    }> : (tensor<3x8x8x256xbf16>, tensor<256x256x3x3xbf16>, tensor<1x1x1x256xbf16>, tensor<3x10x10x256xbf16>) -> tensor<3x10x10x256xbf16>

## `ttnn.copy`

Device to device copy op.

Copies `src` into the pre-allocated device tensor `dst`. Both tensors must be
on device and have identical types.

Inputs:
  - `src` AnyRankedTensor: The device tensor to copy from.
  - `dst` AnyRankedTensor: The pre-allocated device tensor to copy into.

## `ttnn.cos`

Eltwise cosine.

Eltwise cosine operation.

## `ttnn.create_global_semaphore`

Create global semaphore op.

Creates a global semaphore with the specified core range and initial value.

Global semaphores are similar to normal semaphores but differ in a couple of ways:
1) their lifetime exists beyond the scope of an op (and thus must created/deallocated separately like tensors)
2) they can be initialized before an op is even dispatched (whereas normal semaphores are initialized at dispatch time)

This is needed for CCLs (and multichip communication in general) to synchronize devices or for normal signalling where normal semaphores
cannot be used. This is because the different devices in a mesh are not running ops in a synchronized manner
(devices can be running different ops in the program) so the existence/initialization of normal semaphores
cannot be guaranteed on a different device that we need to communicate with.

For example, if we have to run a matmul then an all gather,
device x could be doing the matmul whereas device y has completed the matmul and begun an all gather.
Now, if device y has to do a remote semaphore increment on device x as part of the all gather,
this would lead to undefined behaviour
since device x is still executing the matmul and it's semaphore for the all gather op does not even exist yet.

Example:
```mlir
%semaphore = "ttnn.create_global_semaphore"(%device) <{core_range_set = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,7)>]>, initial_value = 0 : ui32}> : (!ttnn.device) -> !ttnn.global_semaphore
```

## `ttnn.cumprod`

Cumulative product op.

Computes the cumulative product of elements of a tensor along specified dimension.

Example:
  input: [[2, 3, 4],
          [5, 6, 7]]

  // Cumulative product along dim=0:
  output: [[2, 3, 4],
           [10, 18, 28]]

  // Cumulative product along dim=1:
  output: [[2, 6, 24],
           [5, 30, 210]]

## `ttnn.cumsum`

Cumulative sum op.

Computes the cumulative sum of elements of a tensor along specified dimension.

Example:
  input: [[1, 2, 3],
          [4, 5, 6]]

  // Cumulative sum along dim=0:
  output: [[1, 2, 3],
           [5, 7, 9]]

  // Cumulative sum along dim=1:
  output: [[1, 3, 6],
           [4, 9, 15]]

## `ttnn.d2m_subgraph`

Dispatch D2M compiled subgraph.

References a D2M-compiled subgraph function containing ttnn.generic ops.
The function is a private function in the same module.

Before TTNNMaterializeD2M runs, the referenced function contains a TTNN subgraph
to be compiled via D2M.

After TTNNMaterializeD2M runs, the referenced function contains:
- ttnn.generic ops (the compiled subgraph)
- Kernel functions are generated at module scope

TTNNCollaspeD2M will inline the D2M function body at the call site.

Example:
```mlir
// Before D2M compilation
%result = ttnn.d2m_subgraph @d2m_subgraph
    ins(%a : tensor<...>)
    outs(%out : tensor<...>) : tensor<...>

func.func private @d2m_subgraph(...) -> tensor<...> {
  // ttnn subgraph
}

// After D2M compilation
%result = ttnn.d2m_subgraph @d2m_subgraph
    ins(%a : tensor<...>)
    outs(%out : tensor<...>) : tensor<...>

func.func private @d2m_subgraph(...) -> tensor<...> {
  ttnn.generic ...
  // more generic ops if subgraph didn't fully fuse
}
func.func private @kernel0() { ... }
... other kernel functions at module scope
```

## `ttnn.deallocate`

Deallocate op.

Tensor Deallocate operation

## `ttnn.dequantize`

Dequantize operation.

Applies dequantization to the input tensor.

Inputs:
  - `input` AnyRankedTensor: The input tensor to be dequantized. Must have quantized element type.
  - `scale` AnyRankedTensor: The scale factor (or factors for per-axis quantization).
  - `zero_point` AnyRankedTensor: The zero point value (or values for per-axis quantization). Must be in range of the quantized storage type.
  - `axis` Optional<i32>: The axis along which quantization is applied. Must be in range [0, rank) where rank is the rank of the input tensor.
  - `output_dtype` Optional<TTCore_DataTypeAttr>: The data type of the output tensor.
```
// For per-tensor dequantization:
output[i] = (input[i] - zero_point) * scale
// For per-axis dequantization:
output[i0, i1, ..., ia, ..., in] = (input[i0, i1, ..., ia, ..., in] - zero_point[ia]) * scale[ia]
```
Example:
```mlir
%input = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
%output = ttir.empty() : () -> tensor<64x128xf32>
%dequantized = "ttnn.dequantize"(%input, %output) : (tensor<64x128x!quant.uniform<i32:f32, 0.1>, tensor<64x128xf32>) -> tensor<64x128xf32>
```

## `ttnn.distribute_tensor`

Distribute tensor across mesh devices

Distributes a host-side tensor across multiple devices in a mesh
according to the specified mapping configuration.

This operation takes a single-device tensor and creates a multi-device
tensor distributed according to the MeshMapperConfig configuration.

// Note: In the ttnn API, the distribute_tensor op takes an input tensor and a TensorToMesh object as its arguments.
// To create a TensorToMesh object, the API create_mesh_mapper is used, which requires both MeshMapperConfig and MeshDevice.
// Instead of modeling create_mesh_mapper as a separate op in the IR, these two strongly coupled APIs are combined into one distribute_tensor op
// to keep the IR simple and less cluttered. If a clear need arises in the future, splitting them into separate ops can be considered.

Example:
```mlir
%result = "ttnn.distribute_tensor"(%input, %device) <{
  mapper_config = #ttnn.mesh_mapper_config<
    placements = [#ttnn.placement<replicate>, #ttnn.placement<shard, 1>],
    mesh_shape_override = #ttnn.mesh_shape<2x2>
  >,
  cq_id = 0 : ui32
}> : (tensor<32x64xbf16>, !ttnn.device) -> tensor<32x64xbf16>
```

## `ttnn.distributed_layer_norm`

Distributed layer normalization with all-gather op.

Intermediate TTNN op for distributed layer normalization across mesh devices.

Always decomposes into:
  layer_norm_pre_all_gather + all_gather + layer_norm_post_all_gather.

Does not reach serialization — no flatbuffer/runtime/EmitC needed.

Inputs:
  - input: Input tensor, width-sharded across mesh devices along the
      normalized (last) dimension.
  - weight: Optional gamma (scale) tensor applied after normalization.
  - bias: Optional beta (shift) tensor applied after normalization.
  - residual: Optional residual tensor to add to input before
      normalization. norm_input = input + residual.

Attributes:
  - cluster_axis: Mesh dimension (0 or 1) along which to all-gather
      the partial statistics across devices.
  - epsilon: Small constant added to the denominator for numerical
      stability. Defaults to 1e-05.

## `ttnn.distributed_rms_norm`

Distributed RMS normalization with all-gather op.

Fused distributed RMS normalization operation across mesh devices.
Computes local RMS statistics (the mean of squared values, E(x²)),
all-gathers the statistics along the specified cluster_axis to obtain
globally-correct values, then normalizes each element by
x / sqrt(E(x²) + epsilon) and applies optional weight scaling locally
on each device.

Only statistics are communicated across devices — the input data itself
is not all-gathered. Each device's output shape equals its input shape.

Maps to ttnn::fused_rms_minimal at runtime.

This operation requires the input tensor to be width-sharded across devices.

Inputs:
  - input: Input tensor. Must be width-sharded in L1 with shape
      (1,1,32,M) where M is a multiple of 32. Tiled layout required.
  - weight: Optional gamma (scale) tensor applied after normalization.
      Must be in ROW_MAJOR layout with width equal to tile_width (32),
      i.e. reshaped from 1D (N,) to 2D (N/32, 32).
  - residual: Optional residual tensor to add to input before
      normalization (x + residual). Must have the same shard spec as
      input.
  - stats: Scratch tensor for intermediate RMS statistics exchanged
      across devices via all-gather. Shape (1,1,32,32), width-sharded
      on core (0,0) in L1. Dtype is Float32 when fp32_dest_acc_en is
      set, otherwise BFloat16.

Attributes:
  - cluster_axis: Mesh dimension (0 or 1) along which to all-gather
      the RMS statistics across devices.
  - epsilon: Small constant added to the denominator for numerical
      stability. Defaults to 1e-12.
  - sub_device_id: Optional sub-device targeting for kernel placement.
  - num_links: Optional number of links for the all-gather
      communication.
  - topology: CCL topology for all-gather (Linear or Ring).
  - compute_config: Device compute kernel configuration. Controls math
      fidelity, fp32_dest_acc_en, and packer L1 accumulation.
  - program_config: LayerNormShardedMultiCoreProgramConfig derived from
      the input's shard spec (core grid, block_h, block_w).

## `ttnn.dit_rms_norm_unary_fused`

Fused RMSNorm + unary activation op.

Fused RMS (Root Mean Square) normalization followed by an optional unary
activation (e.g. SiLU, GELU), computed in a single kernel pass. This is
equivalent to:

  y = <activation>(rms_norm(input + residual_input, weight, bias, epsilon))

but avoids materializing the intermediate normalized tensor. Targets DiT
(Diffusion Transformer) blocks. Maps to
`ttnn::experimental::dit_rms_norm_unary_fused` at runtime.

Normalization is performed over the last dimension of the input tensor,
matching the TTNN runtime implementation.

## `ttnn.divide`

Eltwise divide.

Eltwise divide operation.

## `ttnn.dropout`

Dropout operation.

Applies dropout to the input tensor element-wise.

Example:
  %result = "ttnn.dropout"(%input) <{prob = 0.2 : f32, scale = 1.25 : f32, seed = 42 : ui32}> : (tensor<64x128xbf16>) -> tensor<64x128xbf16>

Attributes:
  - `prob` (Float): Dropout probability. Elements are zeroed with this probability [Default: 0.0].
  - `scale` (Float): Scale factor applied to non-zeroed elements. Typically 1/(1-prob) [Default: 1.0].
  - `seed` (Integer): Seed for the random number generator [Default: 0].
  - `use_per_device_seed` (Bool): Whether to use a different seed per device [Default: true].

Inputs:
  - `input` (Tensor): The input tensor.

Output:
  - `result` (Tensor): The output tensor with dropout applied.

## `ttnn.dump_tensor`

Saves a tensor to disk in the TTNN binary format

Saves a tensor to disk in the TTNN binary format. Files must use the `.tensorbin` extension.

Inputs:
  - `file_path` StrAttr: Path of the file where tensor should be dumped. Must end with `.tensorbin` extension.
  - `input` AnyRankedTensor: Tensor to serialize.

## `ttnn.embedding`

Embedding op.

Embedding operation.

## `ttnn.embedding_bw`

Embedding backward op.

Embedding backward operation. Generates the gradient of the embedding operation with respect to the input.

The gradient is returned as a 4D tensor of shape (1, 1, dictionary_size, embedding_size), matching
what tt-metal produces; the two leading unit dimensions have to be reshaped away by the producer of
this op if the consumer expects the 2D weight shape.

## `ttnn.empty`

Empty op.

Tensor empty operation

## `ttnn.end_trace_capture`

End trace capture.

Ends trace capture for the given trace id. Consumes a scalar tensor containing the trace id.
Has no output, but will have memory effects on the trace region of the device, modelled by
trace resource in the compiler.
Inputs:
  - `device` TTNN_Device: The device to end the trace capture on.
  - `trace_id` AnyRankedTensor: The trace id tensor to end the capture for. Must be a scalar.
  - `cq_id` ui32: The command queue to end the capture with. Must be 0 or 1.

## `ttnn.eq`

Eltwise equal to.

Eltwise equal to operation.

## `ttnn.erf`

Eltwise erf op.

Eltwise erf operation. Calculates erf(x) for each element of the input tensor.

## `ttnn.erfc`

Eltwise erfc op.

Eltwise erfc operation. Calculates erfc(x) for each element of the input tensor.

## `ttnn.execute_trace`

Execute trace.

Executes the captured trace. Consumes a scalar tensor containing the trace id.
Has no output, but will have read/write memory effects on the cached trace input/output tensors
created when capturing the trace.
Inputs:
  - `device` TTNN_Device: The device where the trace was captured.
  - `trace_id` AnyRankedTensor: The trace id tensor to execute. Must be a scalar.
  - `cq_id` ui32: The command queue to execute the trace with. Must be 0 or 1.
  - `blocking` bool: Whether the trace should be executed synchronously.

## `ttnn.exp`

Eltwise exponential.

Eltwise exponential operation.

## `ttnn.expm1`

Performs element-wise exponential minus one operation on `operand` tensor
and stores the result in the output tensor.

Example:
    %a: [[0, 1], [0, 0]]
    "ttnn.exmp1"(%a, %out) -> %out: [[0, 1.71828], [0, 0]]

## `ttnn.fill_cache`

Fill static cache tensor.

Fills the `cache` tensor in-place with values from `input` at `batch_offset`.

## `ttnn.flash_mla_prefill`

Flash Multi-head Latent Attention prefill operation.

Multi-head Latent Attention (MLA) prefill. Mirrors
`ttnn::transformer::flash_mla_prefill`.

Shapes use `B` (batch), `Hq`/`Hkv` (query/kv heads), `Sq` (sequence
length; must equal `Sk` for prefill), `dh_qk` (Q/K head size), and
`head_dim_v` (V/output head size).

Args:
    query (AnyRankedTensor): `[B x Hq x Sq x dh_qk]`.
    key (AnyRankedTensor): `[B x Hkv x Sq x dh_qk]`.
    value (AnyRankedTensor, optional): `[B x Hkv x Sq x head_dim_v]`.
        When absent (MLA-from-latent), V is taken from the first
        `head_dim_v` features of K.
    attention_mask (AnyRankedTensor, optional): `[1|B x 1 x Sq x Sq]`.
        Only valid when `is_causal` is `false`.
    head_dim_v (uint): Head dimension of V/output.
    is_causal (bool): Defaults to `true`. Cannot be `true` when
        `attention_mask` is provided.
    scale (float, optional): Softmax scale. Defaults to `1 / sqrt(dh_qk)`.

Returns:
    AnyRankedTensor: `[B x Hq x Sq x head_dim_v]` (same dtype as `query`).

## `ttnn.floor`

Eltwise floor op.

Eltwise floor operation.

## `ttnn.from_device`

FromDevice op.

This op retrieves the input tensor from the given device.

## `ttnn.full`

Creates a tensor filled with the specified value

Tensor operation to create a tensor filled with a specified value.

Given a `shape` and a `fill_value`, produces a tensor with the shape, filled
with the specified value.

Example:
  %0 = "ttnn.full"() <{
    fill_value = 7 : i32,
    shape = #ttnn.shape<64x128>
  }> : () -> tensor<64x128xui32, #ttnn_layout>
  // %0: [[[7, 7, 7, ..., 7], [7, 7, 7, ..., 7], ..., [7, 7, 7, ..., 7]]]

## `ttnn.gather`

Gather op.

Gathers values from the input tensor along the given dimension using an
index tensor. This corresponds to torch.gather semantics and maps
directly to the tt-metal `ttnn::gather` device op.

Parameters:
  - `input` (ttnn.Tensor): The source tensor to gather from.
  - `index` (ttnn.Tensor): Indices specifying which values to gather.
  - `dim` (int32_t): The dimension along which to gather.

## `ttnn.ge`

Eltwise greater than or equal to.

Eltwise greater than or equal to operation.

## `ttnn.gelu`

Eltwise GELU.

Eltwise GELU operation.

## `ttnn.gelu_bw`

Backward pass operation for the GELU activation function.

Computes the gradient of the GELU (Gaussian Error Linear Unit) activation
function with respect to its input during backpropagation.

This operation corresponds to ttnn.experimental.gelu_bw.

## `ttnn.generic`

Generic operation.

Generic operation capable of running a program with custom kernels. Each kernel is described with a
symbol reference to its function in EmitC dialect plus compile and runtime arguments. Generic operation
is supplied with concatenated input and output `ios` tensors.

Inputs:
  - `inputs_and_outputs` Variadic<AnyRankedTensor>: The input and output tensors.
  - `program` ProgramAttr: Program descriptor that includes a description of each kernels, array of CBs and array of semaphores.

## `ttnn.get_device`

Get Device op.

This op returns a submesh carved out from the parent runtime device.
Mesh shape and mesh offset define the size and offset of the submesh.

## `ttnn.global_avg_pool2d`

A global average pooling 2d operation

The `global_avg_pool2d` operation applies global average pooling over the spatial dimensions
(height and width) of a 4D input tensor. In essence, it should be realised as the sum-reduce style operation
under the hood, for performance reasons (since we include all elements, there is no need for kernel allocation).
It reduces spatial dimensions to 1.

Example:
```mlir
%device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

%result = "ttnn.global_avg_pool2d"(%input)
          : (tensor<1x128x128x32xbf16>) -> tensor<1x1x1x32xbf16>
```

Inputs:
- `input`: 4D tensor with shape [N, H, W, C] where N is batch size, H is height, W is width, and C is channels

Outputs:
- `result`: 4D tensor with shape [N, 1, 1, C] containing the global average pooled values

Note: The operation reduces spatial dimensions (H, W) to (1, 1) by computing the average across
all spatial locations for each channel independently.

## `ttnn.group_norm`

Group normalization op.

Computes group normalization over the input tensor. The input tensor's channels
are split into groups, and mean and variance are computed per group. The result
is normalized by subtracting the mean and dividing by the standard deviation,
then optionally scaled and shifted by weight (gamma) and bias (beta).

## `ttnn.gt`

Eltwise greater than.

Eltwise greater than operation.

## `ttnn.hardsigmoid`

Eltwise hardsigmoid.

Eltwise hardsigmoid operation. Computes hardsigmoid(x) = max(0, min(1, (x + 3) / 6)).

## `ttnn.indexer_score_dsa`

DeepSeek Sparse Attention lightning-indexer scorer.

Computes DSA (Deepseek Sparse Attention) lightning-indexer
scores. Corresponds to `ttnn.experimental.indexer_score_dsa`
(`ttnn::experimental::indexer_score_dsa`).

For every query position `s` and key position `t`:
```
score[b, s, t] = sum_h relu(q[b, h, s, :] . k[b, t, :]) * weights[b, h, s]
```
Causality is controlled by `chunk_start_idx`: key `t` is visible to query
`s` iff `t <= chunk_start_idx + s`. Masked (future) positions are set to
`-inf`.

Shapes use `B` (batch, must be 1), `Hi` (indexer heads), `Sq` (queries),
`T` (keys), and `D` (indexer head dim).

Args:
    query (AnyRankedTensor): `[B x Hi x Sq x D]`.
    key (AnyRankedTensor): `[B x 1 x T x D]`.
    weights (AnyRankedTensor): `[B x Hi x Sq x 1]` per-head gate scales.
    chunk_start_idx (uint): Causal offset of the first query within the
        full key sequence. Defaults to `0`.
    cluster_axis (optional uint): Mesh axis carrying the query sequence
        shard. On a mesh, the op offsets each device's causal window by
        `rank * Sq`, where `rank` is that device's index along this axis.
        Unset leaves the op on a flat row-major enumeration over all of
        the query's devices, which is only correct when the query
        sequence is sharded across every device.

Returns:
    AnyRankedTensor: `[B x 1 x Sq x T]` scores.

Note: this op is only supported on Blackhole.

## `ttnn.isfinite`

Eltwise isfinite op.

Eltwise isfinite operation.

## `ttnn.layer_norm`

Layer normalization op.

Performs layer normalization on the input tensor. This operation normalizes
the input tensor by computing the mean and variance of elements across
the specified dimensions, then normalizes by subtracting the mean and
dividing by the standard deviation, optionally scaling and shifting the result.

This operation performs normalization over the last dimension of the input tensor,
matching the TTNN runtime implementation.

## `ttnn.layer_norm_post_all_gather`

Layer normalization post-all-gather op.

Applies normalization using gathered statistics from a distributed
layer normalization pipeline. Takes the original input tensor and
the all-gathered partial statistics, computes the final normalized
output, and optionally scales by weight and shifts by bias.

Inputs:
  - input: Original input tensor.
  - stats: All-gathered statistics tensor from pre-all-gather phase.
  - weight: Optional scale (gamma) tensor.
  - bias: Optional shift (beta) tensor.

Attributes:
  - epsilon: Small constant for numerical stability (default 1e-12).
  - compute_config: Optional device compute kernel configuration.
  - program_config: Optional LayerNormShardedMultiCoreProgramConfig.

## `ttnn.layer_norm_pre_all_gather`

Layer normalization pre-all-gather op.

Computes local partial statistics (Welford) for distributed layer
normalization on a width-sharded input tensor. Produces partial
statistics that should be all-gathered across devices before being
consumed by a post-all-gather normalization op.

Inputs:
  - input: Input tensor (width-sharded in L1).
  - residual_input: Optional residual tensor to add before computing
      statistics. Must have the same shape as input.
  - recip: Optional precomputed reciprocal tensor.

Attributes:
  - compute_config: Optional device compute kernel configuration.
  - program_config: Optional LayerNormShardedMultiCoreProgramConfig.

## `ttnn.le`

Eltwise less than or equal to.

Eltwise less than or equal to operation.

## `ttnn.leaky_relu`

Eltwise leaky relu operation.

The Leaky ReLU (Rectified Linear Unit) operation computes an element-wise
activation function over its input tensor. It is defined as:

y = x if x > 0
y = parameter * x if x <= 0

where `parameter` is a small, user-defined constant that determines the slope for
negative inputs.

Attributes:
- `parameter` (float): The slope for negative values.

Inputs:
- `input` (Tensor): The input tensor to be activated.

Outputs:
- `output` (Tensor): The tensor after applying the Leaky ReLU activation.

## `ttnn.linear`

Linear transformation of inputs.

Produces the matmul of tensors `a` and `b` with optional addition with `bias`.

Example:
  // %a = [[1., 2.], [2., 1.]]
  // %b = [[0., 1.], [1., 0.]]
  // %bias = [[1.]]
  "ttnn.linear"(%a, %b, %bias, %result) : (tensor<2x2xf16>, tensor<2x2xf16>, tensor<1xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
  // %result = [[3., 2.], [2., 3.]]

## `ttnn.load_tensor`

Loads a tensor from disk

Loads a tensor from disk, optionally placing it directly on a device.

Inputs:
  - `file_path` StrAttr: Path of the file of the serialized tensor. Must end with `.tensorbin` extension.
  - `device` Optional<TTNN_Device>: Device where tensor should be deserialized. It has to be provided iff the serialized tensor is a device tensor.
Outputs:
  - `result` AnyRankedTensor: Deserialized tensor from the `file_path`.

## `ttnn.log`

Eltwise logarithm.

Eltwise logarithm operation.

## `ttnn.log1p`

Eltwise log1p operation.

Performs element-wise logarithm plus one operation on `operand` tensor and
puts the result in the output tensor.

Example:
  %a: [0.0, -0.999, 7.0, 6.38905621, 15.0]
  "ttnn.logp1"(%a, %out) -> %out: [0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]

## `ttnn.logical_and`

Eltwise logical and.

Eltwise logical and operation.

## `ttnn.logical_left_shift`

Eltwise Logical Left Shift operation

The `logical_left_shift` operation performs an elementwise logical left shift
on the elements of the first tensor by the corresponding shift amounts in the
second tensor.

## `ttnn.logical_not`

Eltwise logical not op.

Eltwise logical not operation.

## `ttnn.logical_or`

Eltwise logical or.

Eltwise logical or operation.

## `ttnn.logical_right_shift`

Eltwise Logical Right Shift operation

The `logical_right_shift` operation performs an elementwise logical right shift
on the elements of the first tensor by the corresponding shift amounts in the
second tensor.

## `ttnn.logical_xor`

Eltwise logical xor.

Eltwise logical xor operation.

## `ttnn.lt`

Eltwise less than.

Eltwise less than operation.

## `ttnn.matmul`

## `ttnn.max`

Max reduction op.

Max reduction op.

## `ttnn.max_pool2d`

Applies a 2D max pooling over an input signal composed of several input planes.

Applies a 2D max pooling over an input signal composed of several input planes.

## `ttnn.max_pool2d_with_indices`

Applies a 2D max pooling over an input signal composed of several input planes, returning both values and indices.

Applies a 2D max pooling over an input signal composed of several input planes.
Returns both the maximum values and the indices of where those values were found in the input tensor.
The indices can be used for unpooling operations or gradient computation.

## `ttnn.maximum`

Eltwise maximum OP.

Calculates maximum of input tensors' values element-wise and stores result in output tensor.

Example:
  %lhs: [[3, 2, 7], [1, 4, 4]]
  %rhs: [[1, 4, 2], [1, 2, 3]]
  "ttnn.maximum"(%lhs, %rhs, %out) -> %out: [[3, 4, 7], [1, 4, 4]]

## `ttnn.mean`

Mean reduction op.

Mean reduction op.

## `ttnn.mesh_partition`

Mesh partition operation.

Mesh partition op.

## `ttnn.min`

Min reduction op.

This op computes the minimum of all elements of the tensor or along
specified dimension.

Example:
  input: [[1, 5, 3],
          [4, 2, 6]]

  // Computing along dim 0
  output: [1, 2, 3]

  // Computing along dim 1
  output: [1, 2]

  // Computing for entire tensor
  output: 1

## `ttnn.minimum`

Eltwise minimum OP.

Calculates minimum of input tensors' values element-wise and stores result
in output tensor.

Example:
  %lhs: [[3, 2, 7], [1, 4, 4]]
  %rhs: [[1, 4, 2], [1, 2, 3]]
  "ttnn.minimum"(%lhs, %rhs, %out) -> %out: [[1, 2, 2], [1, 2, 3]]

## `ttnn.mish`

Eltwise Mish.

Eltwise Mish operation.

## `ttnn.moe_compute`

Fused MoE expert compute (selective tilize + experts + combine).

Composite operation that performs the MoE expert FFN in a single fused
kernel: selective tilize of dispatched tokens, the gate/up matmul (W0/W1)
followed by SILU or SwiGLU activation, the down matmul (W2), and an
A2A selective-reduce-combine. Maps to
`ttnn::experimental::moe_compute` at runtime.

`optional_output_tensor` is the combine-output buffer and
`cross_device_semaphore` synchronizes the A2A combine across devices; both
are bound in the function prelude by this op's `DistributedOpInterface`
hooks. `cluster_axis` selects the mesh axis the combine reduces over;
`mux_core_range_set` provides the fabric-mux cores.

See `TTIR_MoeComputeOp` for input/output shapes and dtypes.

## `ttnn.moe_expert_token_remap`

Remap global expert routing to local device experts with sparsity.

Converts global expert routing scores to local per-device expert mapping
and creates a sparsity pattern for efficient sparse_matmul.

## `ttnn.moe_gpt`

Fused MoE compute kernel for GPT-OSS models.

Fused Mixture-of-Experts compute kernel for GPT-OSS-style models. Runs
the full MoE compute pipeline in a single kernel: selective tilize,
gate/up projection (w0_w1), SwiGLU activation, all-to-all ring exchange
along cluster_axis, down projection (w2), and combine. Typically follows
ttnn.all_to_all_dispatch_metadata in the MoE dispatch/compute/combine flow.

Dimensions:
  T         = total_tokens across the dispatch ring
  K         = hidden_size
  N         = intermediate_size
  K_sel     = selected experts per token (top-k)
  E         = experts per device
  D         = total devices
  C_dram    = DRAM-bank-aligned compute cores used for weight sharding
  C_worker  = worker cores in the compute grid
  L         = layers per weight tensor (typically 1)
  G         = weight groups per core (gate/up interleaved pairs)
  TILE_SIZE = compute tile size
  L1_ALIGN  = L1 alignment

Input shapes and layouts (per device, after mesh sharding; enforced by
TTNNWorkaroundsPass to match kernel expectations):
- input_tensor:   [T, K]                             bf16,   ROW_MAJOR
    (kernel tilizes internally)
- expert_indices: [T, K_sel]                         uint16, ROW_MAJOR, L1 HEIGHT_SHARDED
- expert_scores:  [T, K_sel]                         bf16,   ROW_MAJOR, L1 HEIGHT_SHARDED
- expert_mapping: [1, E*D]                           uint16, ROW_MAJOR
    (flattened mapping, entry [0, i] = device ID owning expert i;
     the tilize_reader kernel casts the buffer to uint16_t*)
- w0_w1_tensor:   [C_dram, L, E, G, K, 4*TILE_SIZE]  (interleaved gate+up)
- w2_tensor:      [C_dram, L, E, 2, N, 4*TILE_SIZE]  (down projection)

Output shapes:
- token_counts:       [1, align(E*sizeof(u32), L1_ALIGN) / sizeof(u32)]                      uint32
- activation_records: [1, (T+1) * align((2*E+1)*sizeof(u32), L1_ALIGN) / sizeof(u32)]        uint32
- token_indices:      [E, (T+1) * align(sizeof(u32), L1_ALIGN) / sizeof(u32)]                uint32
- tilize_out:         [C_worker, 2, TILE_SIZE, K]    bf16, TILE
- tilize_out_rm:      [C_worker, 2, TILE_SIZE, K]    bf16, ROW_MAJOR
where align(n, a) = ceil(n/a) * a.

## `ttnn.multiply`

Eltwise multiply.

Eltwise multiply operation.

## `ttnn.ne`

Eltwise not equal to.

Eltwise not equal to operation.

## `ttnn.neg`

Eltwise negate.

Eltwise negate operation.

## `ttnn.nlp_concat_heads`

nlp_concat_heads op in TTNN dialect.

"This op targets specific case of concatenate heads operation where input tensor
[B, num_heads, S, head_dim] is permuted and reshaped into [B, 1, S, num_heads * head_dim]."

## `ttnn.nlp_concat_heads_decode`

Concatenate heads op used in attention layer.

Shuffles [S=1, B=32, 32(num_heads), head_dim] tensor into tensor with shape [S=1, 1, B=32, num_heads * head_dim].
This operation assumes that input num_heads is padded to at most 32. When invoking this op,
we specify the actual num_heads via the attribute `num_heads` and it should be less than input padded num_heads.
Operation will unpad the input num_heads to the actual num_heads.
The output is default width sharded by num heads.

## `ttnn.nlp_create_qkv_heads_decode`

nlp_create_qkv_heads_decode op in TTNN dialect.

Shuffles [1, S=1, B, head_dim * (num_heads + 2*num_kv_heads)] fused qkv matrix into Q, K, and V heads with shape [S, B, num_heads, head_dim] for Q and [S, B, num_kv_heads, head_dim] for K and V, where num_heads and num_kv_heads will be padded to nearest 32.
  - Input must be sharded, B=32 and S=1.
  - overlap_qk_coregrid is a boolean flag that determines whether the output Q and K heads are on same core grid. If true, then Q, K, and V heads are on the same core grid. If false, the Q and K heads are on non-overlapping core-grid useful for processing Q and K in parallel.
  - Batch offset is used to fuse batch slicing. If provided slice size must also be provided in which batch dim of QKV output will be slice_size.

## `ttnn.ones`

Creates a tensor filled with ones.

Tensor operation to create a tensor filled with ones.

Given a ShapeAttr `shape`, produces a tensor with the same shape, filled with ones.

Example:
  %0 = "ttnn.ones"() <{shape = array<i32:64, 28, 28>}> : () -> tensor<64x28x28xbf16>
  // %0: [[[1, 1, 1, ..., 1], [1, 1, 1, ..., 1], ..., [1, 1, 1, ..., 1]]]

## `ttnn.pad`

Pad op.

Pad input tensor by padding the input_shape to output_shape using the provided value.

The `padding` attribute must be a sequence of integers that is twice the size as the rank of the input.
Each pair of integers in the padding attribute represents the amount of padding to add to the low and high of that dimension.
I.e: an input tensor of shape <1x30x30x64xf32> with padding attribute <0, 0, 1, 1, 1, 1, 0, 0> will return a tensor of shape <1x32x32x64xf32>,
and so will a padding attribute of <0, 0, 0, 2, 0, 2, 0, 0>.

## `ttnn.paged_fill_cache`

Paged fill cache op.

Fills the `cache` tensor in-place with values from `input` at `batch_offset`.

## `ttnn.paged_flash_multi_latent_attention_decode`

Paged flash multi-latent attention decode operation.

Paged flash Multi-Latent Attention (MLA) for the decode phase. Combines
flash attention with multi-latent attention and paged KV cache support,
optimized for single-token decode.

Key difference from PagedScaledDotProductAttentionDecode: the value tensor
is optional (may be null for latent-only MLA) and head_dim_v specifies the
value head dimension separately.

The key/value cache is laid out as `[max_num_blocks, nkv, block_size,
head_dim]`. MLA keeps a single compressed latent KV cache shared across all
query heads, so the number of KV heads `nkv` (dim 1) must be 1.

## `ttnn.paged_scaled_dot_product_attention_decode`

Paged scaled dot product attention decode operation.

Paged scaled dot product attention decode operation.

## `ttnn.paged_update_cache`

Paged update cache op.

Inputs:
  - `cache`: The cache tensor to be updated. This tensor is modified in place [max_num_blocks, num_heads, block_size, head_dim]
  - `input`: The input tensor containing new values. [1, num_users, num_heads (padded to 32), head_dim]
  - `update_index`: Indices specifying where to update the cache. [num_users]
  - `share_cache`: Whether the cache tensors share memory regions. Defaults to False.
  - `page_table`: The page table for managing memory regions during updates. [num_users, max_num_blocks_per_seq]

## `ttnn.permute`

Permute operation.

Permute input tensor dimensions.

Attributes:
  - `permutation` array<i64>: The permutation of the input tensor dimensions.

Example:
%a = ttir.empty() : () -> tensor<2x3x4xi32>
%0 = "ttir.permute"(%a) {permutation = array<i64: 1, 2, 0>} : (tensor<2x3x4xi32>) -> tensor<3x4x2xi32>

## `ttnn.point_to_point`

Point To Point operation.

Performs point-to-point communication by copying a tensor shard from one device to another
within a multi-device mesh. This operation is typically used for explicit data movement in
distributed tensor computations, where a specific device (send_coord) sends its local tensor
data to a target device (receive_coord).

If `optional_output_tensor` is not provided, a new output tensor will be allocated automatically
at the receiver. If provided, the data will be written into the specified output tensor.

The operation returns a multi-device tensor whose buffer layout follows the mesh configuration.

## `ttnn.pow_scalar`

Eltwise power OP.

The `pow_scalar` operation performs an exponentiation of each element of an
input tensor with a scalar exponent and returns the result.

Example:
```mlir
%result = ttnn.pow_scalar(%input) <{exponent = 2.0 : f32}> : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
// Input tensors:
// %input: [2.0, 3.0, 4.0, 5.0]  // Bases
// %exponent: 2.0  // Power
// Output tensor: [4.0, 9.0, 16.0, 25.0]
```

Restriction: TTNN API supports exponent ≥ 0 only.

## `ttnn.pow_tensor`

Eltwise power OP.

Performs element-wise exponentiation of lhs tensor by rhs tensor and produces a
result tensor. Tensors must be of same shape.

Example:
```
  %result = "ttnn.pow_tensor"(%lhs, %rhs) : (tensor<6xf64>, tensor<6xf64>) -> tensor<6xf64>

  %lhs: [-2.0, -0.0, -36.0, 5.0, 3.0, 10000.0]
  %rhs: [2.0, 2.0, 1.1, 2.0, -1.0, 10.0]
  %result: [4.0, 0.0, -nan, 25.0, 0.333333343, inf]
```

## `ttnn.prepare_conv2d_bias`

Prepares conv2d bias so that it can be consumed by the conv2d op.

## `ttnn.prepare_conv2d_weights`

Prepares conv2d weights so that they can be consumed by the conv2d op.

## `ttnn.prepare_conv3d_weights`

Prepares conv3d weights so that they can be consumed by the conv3d op.

## `ttnn.prepare_conv_transpose2d_bias`

Prepares conv_transpose2d bias so that it can be consumed by the conv_transpose2d op.

## `ttnn.prepare_conv_transpose2d_weights`

Prepares conv_transpose2d weights so that they can be consumed by the conv_transpose2d op.

## `ttnn.prod`

Product reduction op.

This op computes the product of all elements of the tensor (full product)
or along a specific dimension.

Example:
  input: [[1, 2, 3],
          [4, 5, 6]]

  // Computing along dim 0
  output: [4, 10, 18]

  // Computing along dim 1
  output: [6, 120]

  // Computing full product
  output: 720

## `ttnn.quantize`

Quantize operation.

Applies quantization to the input tensor.

Inputs:
  - `input` AnyRankedTensor: The input tensor to be quantized. Must have floating-point element type.
  - `scale` AnyRankedTensor: The scale factor (or factors for per-axis quantization). Must be either a scalar (for per-tensor quantization) or a 1D tensor with size matching the dimension of the specified axis (for per-axis quantization).
  - `zero_point` AnyRankedTensor: The zero point value (or values for per-axis quantization). Must be in range of the quantized storage type.
  - `axis` Optional<i32>: The axis along which quantization is applied. Must be in range [0, rank) where rank is the rank of the input tensor.
  - `output_dtype` Optional<TTCore_DataTypeAttr>: The data type of the output tensor.

```
// For per-tensor quantization:
output[i] = round(input[i] / scale) + zero_point
// For per-axis quantization:
output[i0, i1, ..., ia, ..., in] = round(input[i0, i1, ..., ia, ..., in] / scale[ia]) + zero_point[ia]
```
Example:
```mlir
%input = ttir.empty() : () -> tensor<64x128xf32>
%output = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
%quantized = "ttir.quantize"(%input, %output) : (tensor<64x128xf32>, tensor<64x128x!quant.uniform<i32:f32, 0.1>>) -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
```

## `ttnn.rand`

Random number generation operation.

Returns a tensor filled with random numbers drawn from a uniform distribution over given interval [low, high) [Default: [0, 1)].

Example:
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.rand"(%0) <{high = 1.000000e+00 : f32, layout = #ttnn.layout<tile>, low = 0.000000e+00 : f32, seed = 0 : ui32, size = [32 : i32, 32 : i32]}> : (!ttnn.device) -> tensor<32x32xbf16, #ttnn_layout1>

Attributes:
  - `size` (TTNN_ShapeAttr): The shape of the tensor to create.
  - `device` (TTNN_Device): The device where the trace was captured.
  - `low` (Float): The lower bound of the range (inclusive) [Default: 0.0].
  - `high` (Float): The upper bound of the range (exclusive) [Default: 1.0].
  - `seed` (Integer): Value to initialize the random number generator for reproducible results [Default: 0].

Outputs:
  - `result` (Tensor): The generated tensor containing the random values.

## `ttnn.reciprocal`

Eltwise reciprocal.

Eltwise reciprocal operation.

## `ttnn.reduce_scatter`

Reduce scatter op.

Tensor Reduce Scatter operation

## `ttnn.relu`

Eltwise ReLU.

Eltwise ReLU operation.

## `ttnn.relu6`

Eltwise ReLU6.

Eltwise ReLU6 operation.

## `ttnn.remainder`

Eltwise remainder.

Performs element-wise remainder of dividend lhs and divisor rhs tensors and produces a
result tensor.

Example:

// %lhs: [17, -17, 17, -17]
// %rhs: [3, 3, -3, -3]
%result = "ttnn.remainder"(%lhs, %rhs) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
// %result: [2, -2, 2, -2]

## `ttnn.repeat`

Repeat op.

Returns a new tensor filled with repetition of input tensor according to number of times specified in repeat_dims.

Parameters:
  - `input_tensor` (ttnn.Tensor): the input tensor.
  - `repeat_dims` (number): The number of repetitions for each element.

## `ttnn.repeat_interleave`

Repeat interleave op.

Repeats elements of a tensor along a specified dimension.
It allows for flexible repetition patterns, where each element can be repeated a different number of times.
This is particularly useful for tasks that require duplicating elements in a non-uniform manner.

Parameters:
- `input`: The input tensor.
- `repeats`: Specifies the number of repetitions for each element, each element is repeated that number of times.
- `dim`: The dimension along which to repeat values.

## `ttnn.requantize`

Requantize operation.

Applies requantization to the input tensor.

Inputs:
  - `input` AnyRankedTensor: The input tensor to be requantized. Must have quantized element type.
  - `in_scale` AnyRankedTensor: The input scale factor (or factors for per-axis quantization). Must be either a scalar (for per-tensor quantization) or a 1D tensor with size matching the dimension of the specified axis (for per-axis quantization).
  - `in_zero_point` AnyRankedTensor: The input zero point value (or values for per-axis quantization). Must be in range of the quantized storage type.
  - `out_scale` AnyRankedTensor: The output scale factor (or factors for per-axis quantization). Must be either a scalar (for per-tensor quantization) or a 1D tensor with size matching the dimension of the specified axis (for per-axis quantization).
  - `out_zero_point` AnyRankedTensor: The output zero point value (or values for per-axis quantization). Must be in range of the quantized storage type.
  - `axis` Optional<i32>: The axis along which quantization is applied. Must be in range [0, rank) where rank is the rank of the input tensor.
  - `output_dtype` Optional<TTCore_DataTypeAttr>: The data type of the output tensor.
```
// For per-tensor requantization:
output[i] = round((input[i] - input_zero_point) * (input_scale / output_scale)) + output_zero_point
// For per-axis requantization:
output[i0, i1, ..., ia, ..., in] = round((input[i0, i1, ..., ia, ..., in] - in_zero_point[ia]) * (in_scale[ia] / out_scale[ia])) + out_zero_point[ia]
```
Example:
```mlir
%input = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
%output = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.2>>
%requantized = "ttnn.requantize"(%input, %output) : (tensor<64x128x!quant.uniform<i32:f32, 0.1>, tensor<64x128x!quant.uniform<i32:f32, 0.2>>) -> tensor<64x128x!quant.uniform<i32:f32, 0.2>>
```

## `ttnn.reset_global_semaphore`

Reset global semaphore op.

Resets a global semaphore to the specified value.

Example:
```mlir
"ttnn.reset_global_semaphore"(%semaphore) <{value = 0 : ui32}> : (!ttnn.global_semaphore) -> ()
```

## `ttnn.reshape`

Reshape op.

Reshape tensor.

## `ttnn.rms_norm`

RMS normalization op.

RMS (Root Mean Square) normalization operation over the input tensor.
Normalizes the input by computing the root mean square of elements and
dividing by that value, optionally scaling and shifting the result.

This operation performs normalization over the last dimension of the input tensor,
matching the TTNN runtime implementation.

## `ttnn.rms_norm_pre_all_gather`

Pre all-gather RMS normalization op.

Computes local partial statistics for distributed RMS normalization across mesh devices.
This op performs the local pre-processing stage of RMSNorm when the normalized dimension
is sharded across devices. It computes the local RMS statistics (partial/local sum(x) and
sum(x²)) needed to form the globally-correct RMS normalization factor,
before an all-gather collects these statistics along the specified cluster_axis.

The gathered statistics are then consumed by the corresponding post-all-gather op, which
computes the final normalization factor ( x / sqrt(E(x²) + epsilon) ) and applies normalization
and optional weight scaling locally on each device.

Only statistics are communicated across devices; the input tensor itself is not all-gathered.

Maps to ttnn::rms_norm_pre_all_gather at runtime.

This operation requires the input tensor to be width-sharded across devices.

Inputs:
  - input: Input tensor. Must be width-sharded in L1. Tiled layout required.
  - residual: Optional Residual Input Tensor to add to input
    before normalization (x + residual). If provided, it must match
    input's padded shape and sharding.

Outputs:
  - result: Intermediate/local statistics tensor that is later aggregated across
    devices via all-gather. The result is in TILE layout.

Attributes:
  - compute_config: Device compute kernel configuration. Controls math
      fidelity, fp32_dest_acc_en, and packer L1 accumulation.
  - program_config: LayerNormShardedMultiCoreProgramConfig derived from
      the input's shard spec (core grid, block_h, block_w).
  - use_2d_core_grid: Optional flag controlling 2D core-grid execution.
     Defaults to None

## `ttnn.rotary_embedding`

Rotary embedding op in TTNN dialect.

Applies rotary embedding to the input tensor using precomputed cosine and sine caches.
Formula used:
  x_rotated = x * cos + rotate_half(x) * sin
  where rotate_half(x) swaps the first and second halves of the last dimension of x.

Example:
  ```mlir
  %result = ttnn.rotary_embedding(%input, %cos_cache, %sin_cache)
    : tensor<1x32x1024x64xf16>, tensor<1x1x1024x64xf16>, tensor<1x1x1024x64xf16>
    -> tensor<1x32x1024x64xf16>
  ```

## `ttnn.rotary_embedding_llama`

Rotary embedding llama operation.

Applies rotary embedding to the input tensor using precomputed cosine and sine caches along with a transformation matrix.

The operation supports both prefill and decode modes:
- Prefill mode: Uses interleaved memory layout
- Decode mode: Uses height-sharded memory layout

Example:
```mlir
%result = ttnn.rotary_embedding_llama(%input, %cos_cache, %sin_cache, %trans_mat)
  {is_decode_mode = false}
  : tensor<1x32x128xbf16>, tensor<1x32x128xbf16>, tensor<1x32x128xbf16>,
  tensor<1x1x32x32xbf16> -> tensor<1x32x128xbf16>
```

## `ttnn.rsqrt`

Eltwise rsqrt.

Eltwise rsqrt operation.

## `ttnn.sampling`

Sampling operation.

Performs fused top-k + top-p + multinomial sampling on pre-filtered
candidate logits using the ttnn::sampling kernel.

The op accepts two SHAPE forms; SamplingOpRank2RewritePattern (a
decomposition workaround) transforms the first into the second so the
ttnn::sampling kernel sees the rank it requires. Dtype and layout
adaptation are handled separately by operand workarounds.

Pre-decomposition (matches TTIR view):
  - `input_values`:  [batch, candidates]      bf16
  - `input_indices`: [batch, candidates]      int32
  - `result`:        [batch]                  int32

Post-decomposition (rank matches the kernel; dtype/layout are still the
IR's user-facing types — operand workarounds insert any needed to_layout
and typecast ops, e.g. retyping the result to uint32 to match the kernel
and converting back to int32 for consumers):
  - `input_values`:  [1, 1, batch, candidates]  bf16
  - `input_indices`: [1, 1, batch, candidates]  int32
  - `result`:        [1, 1, 1, batch]           int32

Common to both forms:
  - `k`: Per-request top-k values [batch] uint32
  - `p`: Per-request top-p values [batch] bf16
  - `temp`: Per-request temperature values [batch] bf16
  - `seed`: Optional random seed (uint32 scalar attribute)

## `ttnn.scaled_dot_product_attention`

Scaled dot product attention operation.

FlashAttention-2 SDPA. Supports MHA, MQA, and GQA.

Shapes use `B` (batch), `Hq`/`Hkv` (query/kv heads), `Sq`/`Sk` (query/kv
seq len), `D` (head size).

Args:
    query (AnyRankedTensor): `[B x Hq x Sq x D]`.
    key (AnyRankedTensor): `[B x Hkv x Sk x D]`.
    value (AnyRankedTensor): `[B x Hkv x Sk x D]`. Same type as `key`.
    attention_mask (AnyRankedTensor, optional): `[1|B x 1|Hq x Sq x Sk]`.
        Dim 0 broadcasts batch, dim 1 broadcasts heads. Only valid when
        `is_causal` is `false`. Defaults to `None`.
    is_causal (bool): Defaults to `true`. Requires `Sq == Sk` and no
        `attention_mask`.
    scale (float, optional): Softmax scale. Defaults to `1 / sqrt(D)`.
    sliding_window_size (uint, optional): If `is_causal`, attends to the
        last N tokens; otherwise attends to a window of size N centered at
        the current position. Defaults to `None`.
    attention_sink (AnyRankedTensor, optional): `[1 x Hq x 1 x 1]`, one
        value per query head broadcast across batch and tile dims.
        Defaults to `None`.

Returns:
    AnyRankedTensor: `[B x Hq x Sq x D]` (same type as `query`).

## `ttnn.scaled_dot_product_attention_decode`

A version of scaled dot product attention specifically for decode.

Flash-Decode SDPA for single-query-token decode. Supports MQA and GQA.

Shapes use `B` (batch), `Hq`/`Hkv` (query/kv heads), `Sk` (kv seq len),
`D` (head size).

Args:
    query (AnyRankedTensor): `[1 x B x Hq x D]`. Dim 0 must be 1.
    key (AnyRankedTensor): `[B x Hkv x Sk x D]`.
    value (AnyRankedTensor): `[B x Hkv x Sk x D]`. Same type as `key`.
    is_causal (bool, optional): Defaults to `true`. Mutually exclusive
        with `attention_mask`.
    attention_mask (AnyRankedTensor, optional): `[1|B x 1 x 1|Hq x Sk]`.
        Dim 0 broadcasts batch, dim 2 broadcasts heads. Only valid when
        `is_causal` is `false`.
    cur_pos_tensor (AnyRankedTensor): 1D `[B]` integer tensor of decode
        positions.
    attention_sink (AnyRankedTensor, optional): `[Hq x 32]`, single tile
        wide.
    scale (float, optional): Softmax scale. Defaults to `1 / sqrt(D)`.
    sliding_window_size (uint, optional): Restrict attention to the last
        N keys, anchored at the decode position (`cur_pos`, defaulting to
        the last kv position when not provided). Defaults to `None`.

Returns:
    AnyRankedTensor: `[1 x B x Hq x D]` (same type as `query`).

## `ttnn.scatter`

Scatter op.

Embeds the values of the source tensor into the input tensor at locations specified by the index tensor along the given dimension.

Parameters:
  - `input` (ttnn.Tensor): The tensor being updated.
  - `index` (ttnn.Tensor): Indices where values will be written to.
  - `source` (ttnn.Tensor): The values to scatter into the input tensor.
  - `dim` (int32_t): The dimension along which to scatter.
  - `scatter_reduce_type` (Enum): The scatter reduce type to use (SUM, PROD, MIN, MAX, INVALID).

## `ttnn.sdpa_bw`

Scaled dot-product attention backward (ttml).

Training-oriented fused scaled dot-product attention backward pass, backed
by the ttml metal op `ttml::metal::sdpa_bw`. Given the upstream gradient
and the forward pass tensors (output, Q, K, V and log-sum-exp
intermediates), computes the gradients w.r.t. query, key and value.

All tensors are rank 4 `(B, H, S, D)`. `attention_mask` is only valid when
`mask_type` is `arbitrary`.

## `ttnn.sdpa_fw`

Scaled dot-product attention forward (ttml).

Training-oriented fused scaled dot-product attention forward pass, backed
by the ttml metal op `ttml::metal::sdpa_fw`. Computes
`softmax(Q @ K^T / sqrt(D) + mask) @ V` and, when `return_intermediates`
is set, also returns the per-row log-sum-exp intermediates used by the
backward pass.

All tensors are rank 4 `(B, H, S, D)`. `attention_mask` is only valid when
`mask_type` is `arbitrary`.

## `ttnn.selective_reduce_combine`

Reduce and combine phase of the MoE pipeline.

Takes dense blocks of expert-computed tokens from the MoE compute kernel,
sparsifies them, and sends tokens back to their originating devices via
fabric.

## `ttnn.sigmoid`

Eltwise sigmoid.

Eltwise sigmoid operation.

## `ttnn.sign`

Eltwise sign operation.

Returns the sign of the `operand` element-wise and produces a `result`
tensor.

Example:
  %a: [[3, -2, 0], [1, -4, 4]]
  "ttnn.sign"(%a, %out) -> %out: [[1, -1, 0], [1, -1, 1]]

## `ttnn.silu`

Eltwise SiLU.

Eltwise SiLU (Swish) operation.

## `ttnn.sin`

Eltwise sine.

Eltwise sine operation.

## `ttnn.slice_dynamic`

Dynamic slice op.

Extract a portion of a tensor based on the specified start (`begins`), stop (`ends`), and step
indices for each dimension. Maps to ttnn::slice.

## `ttnn.slice_static`

Slice op.

Extract a portion of a tensor based on the specified start (`begins`), stop (`ends`), and step
indices for each dimension. The `begins` and `ends` parameters are attributes with fixed values.
Maps to ttnn::slice.

## `ttnn.softmax`

Softmax op.

Softmax operation.

## `ttnn.sort`

Sort op.

Sorts elements of a tensor along a given dimension.

Input:
  - input: AnyRankedTensor

Attributes:
  - dim (int8): The dimension to sort along (default: -1, the last dim).
  - descending (bool): If True, sort in descending order (default: False).
  - stable (bool): If True, ensures stable sort (equal elements keep order).

Returns a tuple:
  - values: the sorted tensor.
  - indices: the original indices of the sorted values.

## `ttnn.sparse_matmul`

Sparse block matrix multiplication with sparsity mask.

The `sparse_matmul` operation performs batched matrix multiplication where
computation is selectively skipped for blocks marked as zero in the sparsity
tensor. Input `b` is organized as a collection of weight matrices indexed by
a block dimension (dim 1), and the sparsity tensor controls which blocks
participate in the computation.

Supported Modes:
- is_input_a_sparse=false, is_input_b_sparse=true (column-parallel):
  a: [A, B, M, K], b: [1, E, K, N], sparsity: [A, B, 1, E]
  -> output: [A, B, 1, E, M, N]
- is_input_a_sparse=true, is_input_b_sparse=false (row-parallel):
  a: [A, E, M, K], b: [1, E, K, N], sparsity: [1, 1, A, E]
  -> output: [A, E, M, N]
- is_input_a_sparse=true, is_input_b_sparse=true (both sparse):
  a: [1, E, M, K], b: [1, E, K, N], sparsity: [1, 1, 1, E]
  -> output: [1, E, M, N]

Example:
```mlir
%result = "ttnn.sparse_matmul"(%activations, %weights, %sparsity) <{
    is_input_a_sparse = false, is_input_b_sparse = true, nnz = 2
}> : (tensor<2x4x32x2880xbf16>, tensor<1x4x2880x5760xbf16>,
      tensor<2x4x1x4xbf16>) -> tensor<2x4x1x4x32x5760xbf16>
```

## `ttnn.split_query_key_value_and_split_heads`

Split query, key, values and split heads op used in attention layer.

Splits input_tensor of shape [batch_size, sequence_size, 3 * hidden_size] into 3 tensors (Query, Key, Value) of shape [batch_size, sequence_size, hidden_size]. Then, reshapes and permutes the output tensors, to make them ready for computing attention scores.
If kv_input_tensor is passed in, then input_tensor of shape [batch_size, sequence_size, hidden_size] is only used for Query, and kv_input_tensor of shape [batch_size, sequence_size, 2 * hidden_size] is used for Key and Value.
For the sharded implementation, the input query, key and value are expected to be concatenated such that the heads are interleaved (q1 k1 v1…qn kn vn).

## `ttnn.sqrt`

Eltwise sqrt.

Eltwise sqrt operation.

## `ttnn.subtract`

Eltwise subtract.

Eltwise subtract operation.

## `ttnn.sum`

Sum reduction op.

Sum reduction op.

## `ttnn.tan`

Eltwise tan op.

Eltwise tan operation.

## `ttnn.tanh`

Eltwise tanh op.

Eltwise tanh operation.

## `ttnn.to_device`

ToDevice op.

This op sends the input tensor to the given device with the given memory config.

## `ttnn.to_layout`

ToLayout op.

This is the narrow layout op: it may only change the page layout
(tile <-> row-major) of the input tensor and, optionally, its data type.
It doesn't change the memory config (buffer type / tensor memory layout /
grid) or device placement - those aggregate changes belong to
ttnn.to_tensor_spec.

ttnn.to_layout is produced by the TTNNDecomposeLayouts pass (alongside
to_device / to_memory_config / typecast) when it breaks down an aggregate
ttnn.to_tensor_spec.

The output data type is derived from the result tensor's TTNNLayoutAttr
encoding via the TTNN_DtypeOpInterface. The target page layout (tile vs
row-major) is also derived from the encoding. Whether this op actually
changes dtype can be queried via `hasDtypeChange()`.

## `ttnn.to_memory_config`

ToMemoryConfig op.

This op converts the memory config of the input tensor based on the given memory config.
It handles:
  - Dram to L1
  - L1 to Dram
  - Interleaved to sharded
  - Sharded to interleaved
  - Sharded to sharded (reshard)

## `ttnn.to_tensor_spec`

ToTensorSpec op.

This op wraps all layout information gathered from ttir.toLayout (data type,
page layout, memory config and device placement). It is the aggregate op
that ttir.to_layout lowers to; it is used/updated by the optimizer to perform
optimizations, and later broken down into specific memory/layout operations
(to_layout, to_device, to_memory_config, typecast) by the TTNNDecomposeLayouts
pass. It never reaches flatbuffer serialization.

The output data type, target page layout (tile vs row-major) and memory config
are all derived from the result tensor's TTNNLayoutAttr encoding via the
TTNN_TensorSpecInterface. Whether this op actually changes dtype can be queried
via `hasDtypeChange()`.

## `ttnn.topk`

Top-K selection operation.

Returns the `k` largest or `k` smallest elements of the `input_tensor` along a given dimension `dim`.
If `dim` is not provided, the last dimension of the input_tensor is used.
If `largest` is True, the `k` largest elements are returned. Otherwise, the `k` smallest elements are returned.
The boolean option `sorted` if True, will make sure that the returned `k` elements are sorted.

## `ttnn.topk_router_gpt`

Fused router linear layer for GPT-style mixture-of-experts models.

A fused multi-core matmul + top-k for the GPT-OSS MoE router. Computes the
router logits via a linear projection (input @ weight + bias) and returns
the top-k expert indices and weights for each token.

Inputs:
- input:  [B, hidden_dim] bf16 hidden states
- weight: [hidden_dim, num_experts] bf16 router weight matrix
- bias:   [B, num_experts] bf16 router bias (pre-broadcast across batch)

Outputs:
- expert_indices: [B, k] ui16 top-k expert indices (ROW_MAJOR)
- expert_weights: [B, k] bf16 top-k router weights (ROW_MAJOR)

num_experts must be 128. B must be 32.

## `ttnn.transpose`

Transpose op.

Transpose tensor along two given dimensions.

## `ttnn.typecast`

Typecast op.

This op converts the data type of the input tensor based on the given data type.
It handles:
  - conversions of data types.

The output data type is derived from the result tensor's TTNNLayoutAttr
encoding via the TTNN_DtypeOpInterface.

## `ttnn.update_cache`

Update static cache tensor.

Updates the `cache` tensor in-place with values from `input` at `update_index` and `batch_offset`.

## `ttnn.upsample`

Upsample 2D op.

Upsample 2D operation. Input tensor is assumed to be in NHWC format.

Attributes:
- `scale_factor` (si32 | array<i32>): The scale factor for upsampling in H and W dimensions respectively.
- `mode` (str): The upsampling algorithm. Currently only "nearest" and "bilinear" are supported. Default is "nearest".

Example:
  // %a: tensor<10x64x32xbf16>
  %0 = "ttnn.upsample"(%a) <{scale_factor = array<i32: 2, 4>}> : (tensor<10x64x32x3xbf16>) -> tensor<10x128x128x3xbf16>

## `ttnn.where`

Eltwise where.

Eltwise where operation.

## `ttnn.write_tensor`

Write tensor op.

Copies host_tensor data into device_tensor through cq_id.
Memory copy is done in place, thus no output is returned.
Inputs:
  - `host_tensor` AnyRankedTensor: The host tensor to copy.
  - `device_tensor` AnyRankedTensor: The device tensor to copy into.
  - `blocking` bool: Whether the copy should be executed synchronously.
  - `cq_id` i32: The command queue to copy the tensor with. Must be 0 or 1.

## `ttnn.zeros`

Creates a tensor filled with zeros.

Tensor operation to create a tensor filled with zeros.

Given a ShapeAttr `shape`, produces a tensor with the same shape, filled with zeros.

Example:
  %0 = "ttnn.zeros"() <{shape = array<i32:64, 28, 28>}> : () -> tensor<64x28x28xbf16>
  // %0: [[[0, 0, 0, ..., 0], [0, 0, 0, ..., 0], ..., [0, 0, 0, ..., 0]]]

