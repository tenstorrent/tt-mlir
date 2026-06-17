# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import get_request_kwargs
import os
from typing import List, Optional
from builder.base.builder_utils import Operand, Shape, get_artifact_dir
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


def check_op(mlir_file: str, op_name: str) -> bool:
    with open(mlir_file, "r") as f:
        for line in f:
            if f"ttnn.{op_name}" in line:
                return True
    return False


def slice_precedes_matmul(mlir_file: str, mm_ops: tuple = ("matmul", "linear")) -> bool:
    # After PermuteSliceAfterMatmul fusion the output slice is pushed up into a
    # matmul/linear operand, so a slice now feeds the op instead of consuming it.
    # (A linear lowers to ttnn.linear for 2D and to ttnn.matmul + add when
    # batched, so accept either as the consuming op.)
    #
    # This is a textual line-order heuristic: it discriminates fusion only
    # because the graphs under test have no slice upstream of the matmul/linear.
    # If fusion did not fire, the only slice consumes the op's output and lands
    # after it. Don't reuse this on a graph whose matmul input is itself sliced.
    first_slice = first_mm = None
    with open(mlir_file, "r") as f:
        for idx, line in enumerate(f):
            if first_slice is None and "ttnn.slice" in line:
                first_slice = idx
            if first_mm is None and any(f"ttnn.{op}" in line for op in mm_ops):
                first_mm = idx
    return first_slice is not None and first_mm is not None and first_slice < first_mm


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),
            (64, 32, 3, 3),
            (1, 1, 1, 64),
            (16,),  # batch_norm scale (gamma)
            (16,),  # batch_norm offset (beta)
            (16,),  # batch_norm mean
            (16,),  # batch_norm variance
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 7])
@pytest.mark.parametrize(
    "stride,padding,dilation,groups", [([2, 1], [2, 1], [2, 1], 2)]
)
@pytest.mark.parametrize("dimension", [1])  # channel dimension for NCHW format
@pytest.mark.parametrize("epsilon", [0.0])
def test_batch_norm_decomposition(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    dimension: int,
    epsilon: float,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def conv2d_batch_norm(
            input_tensor: Operand,
            conv_weight: Operand,
            conv_bias: Operand,
            bn_scale: Operand,
            bn_offset: Operand,
            bn_mean: Operand,
            bn_variance: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Create input tensor with random data
            input_tensor_data = torch.randn(shapes[0], dtype=dtypes[0])

            # Create conv2d weights and bias
            conv_weight_data = torch.randn(shapes[1], dtype=dtypes[1])
            conv_bias_data = torch.randn(shapes[2], dtype=dtypes[2])

            # Create batch norm parameters
            bn_scale_data = torch.randn(shapes[3], dtype=dtypes[3])
            bn_offset_data = torch.randn(shapes[4], dtype=dtypes[4])
            bn_mean_data = torch.randn(shapes[5], dtype=dtypes[5])
            bn_variance_data = (
                torch.abs(torch.randn(shapes[6], dtype=dtypes[6])) + 1e-5
            )  # Ensure positive variance

            input_tensor_data_rs = input_tensor_data.transpose(-2, -1).transpose(-3, -2)
            conv_result = torch.nn.functional.conv2d(
                input_tensor_data_rs,
                conv_weight_data,
                conv_bias_data.squeeze(),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            conv_result = conv_result.transpose(-3, -2).transpose(-2, -1)

            golden_output = torch.nn.functional.batch_norm(
                conv_result,
                bn_mean_data,
                bn_variance_data,
                bn_scale_data,
                bn_offset_data,
                eps=epsilon,
            )

            conv2d_0 = builder.conv2d(
                input_tensor,
                conv_weight,
                conv_bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                unit_attrs=unit_attrs,
            )

            batch_norm_0 = builder.batch_norm_inference(
                conv2d_0,
                bn_scale,
                bn_offset,
                bn_mean,
                bn_variance,
                epsilon=epsilon,
                dimension=dimension,
            )

            builder.set_goldens(
                {
                    input_tensor: input_tensor_data,
                    conv_weight: conv_weight_data,
                    conv_bias: conv_bias_data,
                    bn_scale: bn_scale_data,
                    bn_offset: bn_offset_data,
                    bn_mean: bn_mean_data,
                    bn_variance: bn_variance_data,
                },
                {batch_norm_0: golden_output},
            )
            builder.set_operand_goldens({conv2d_0: conv_result})
            return batch_norm_0

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
        pipeline_options=["enable-fusing-conv2d-with-multiply-pattern=true"],
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    assert check_op(output_path, "conv2d") and not check_op(output_path, "batch_norm")


@pytest.mark.xfail(
    reason="Compile error: is_floating_point(): argument 'input' (position 1) must be Tensor, not NoneType"
)
@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),  # input
            (64, 64, 3, 3),  # conv weight
            (1, 1, 1, 64),  # conv bias
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3])
@pytest.mark.parametrize("stride", [[1, 1]])
@pytest.mark.parametrize("padding", [[1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("activation", ["relu", "relu6", "silu"])
def test_conv_activation_fusing(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    activation: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def conv2d_activation(
            input_tensor: Operand,
            conv_weight: Operand,
            conv_bias: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Create input tensor with random data
            input_tensor_data = torch.randn(shapes[0], dtype=dtypes[0])

            # Create conv2d weights and bias
            conv_weight_data = torch.randn(shapes[1], dtype=dtypes[1])
            conv_bias_data = torch.randn(shapes[2], dtype=dtypes[2])

            # Calculate golden output_path using torch operations
            input_tensor_data_rs = input_tensor_data.transpose(-2, -1).transpose(-3, -2)
            conv_result = torch.nn.functional.conv2d(
                input_tensor_data_rs,
                conv_weight_data,
                conv_bias_data.squeeze(),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            conv_result = conv_result.transpose(-3, -2).transpose(-2, -1)

            # Apply activation based on parameter
            if activation == "relu":
                golden_output = torch.nn.functional.relu(conv_result)
            elif activation == "relu6":
                golden_output = torch.nn.functional.relu6(conv_result)
            elif activation == "silu":
                golden_output = torch.nn.functional.silu(conv_result)

            # Create conv2d builder op
            conv = builder.conv2d(
                input_tensor,
                conv_weight,
                conv_bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                unit_attrs=unit_attrs,
            )

            # Add activation builder op based on parameter
            if activation == "relu":
                activation_op = builder.relu(conv)
            elif activation == "relu6":
                activation_op = builder.relu6(conv)
            elif activation == "silu":
                activation_op = builder.silu(conv)

            builder.set_goldens(
                {
                    input_tensor: input_tensor_data,
                    conv_weight: conv_weight_data,
                    conv_bias: conv_bias_data,
                },
                {conv: golden_output},
            )
            return activation_op

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    assert check_op(output_path, "conv2d") and not check_op(output_path, activation)


@pytest.mark.xfail(
    reason="Compile error: is_floating_point(): argument 'input' (position 1) must be Tensor, not NoneType"
)
@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),  # input
            (64, 64, 3, 3),  # conv weight
            (1, 1, 1, 64),  # conv bias
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3])
@pytest.mark.parametrize("stride", [[1, 1]])
@pytest.mark.parametrize("padding", [[1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("groups", [1])

# Test fusing when silu is decomposed as x * sigmoid(x)
def test_conv_silu_decomposed_fusing(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def conv2d_silu_decomposed(
            input_tensor: Operand,
            conv_weight: Operand,
            conv_bias: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Create input tensor with random data
            input_tensor_data = torch.randn(shapes[0], dtype=dtypes[0])

            # Create conv2d weights and bias
            conv_weight_data = torch.randn(shapes[1], dtype=dtypes[1])
            conv_bias_data = torch.randn(shapes[2], dtype=dtypes[2])

            # Calculate golden output using torch operations
            input_tensor_data_rs = input_tensor_data.transpose(-2, -1).transpose(-3, -2)
            conv_result = torch.nn.functional.conv2d(
                input_tensor_data_rs,
                conv_weight_data,
                conv_bias_data.squeeze(),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            conv_result = conv_result.transpose(-3, -2).transpose(-2, -1)
            golden_output = conv_result * torch.sigmoid(conv_result)

            # Create conv2d builder op
            conv = builder.conv2d(
                input_tensor,
                conv_weight,
                conv_bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                unit_attrs=unit_attrs,
            )

            # Add builder ops for x * sigmoid(x)
            sigmoid_op = builder.sigmoid(conv, unit_attrs=unit_attrs)
            silu_decomposed = builder.multiply(conv, sigmoid_op)

            builder.set_goldens(
                {
                    input_tensor: input_tensor_data,
                    conv_weight: conv_weight_data,
                    conv_bias: conv_bias_data,
                },
                {conv: golden_output},
            )
            return silu_decomposed

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    assert (
        check_op(output_path, "conv2d")
        and not check_op(output_path, "sigmoid")
        and not check_op(output_path, "multiply")
    )


@pytest.mark.parametrize(
    "shapes",
    [
        # Direct matmul + 1D bias.
        [(68, 1024), (1024, 1024), (1024,)],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3])
def test_matmul_with_bias_fusing(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def matmul_add_bias(
            input_tensor: Operand,
            weight: Operand,
            bias: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            matmul_result = builder.matmul(input_tensor, weight)
            return builder.add(matmul_result, bias)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    assert check_op(output_path, "linear") and not check_op(output_path, "matmul")


@pytest.mark.parametrize(
    "shape_a,shape_b,transpose_a,transpose_b,reshape_shape,begins,ends",
    [
        # Row (M) slice -> pushed into A. The last row of the matmul output is
        # kept (greedy-decode lm_head after the rank-3 logits tensor has been
        # collapsed to 2D).
        ((256, 512), (512, 1024), False, False, None, [255, 0], [256, 1024]),
        # Column (N) slice -> pushed into B. A middle block of output columns.
        ((256, 512), (512, 1024), False, False, None, [0, 5], [256, 17]),
        # transpose_b=true: B is [N, K], so the column slice is pushed into B's
        # rank-2 dim and transpose_b is preserved.
        ((256, 512), (1024, 512), False, True, None, [0, 0], [256, 64]),
        # transpose_a=true: A is [K, M], so the row slice is pushed into A's
        # trailing dim and transpose_a is preserved.
        ((512, 256), (512, 1024), True, False, None, [255, 0], [256, 1024]),
        # Row slice through a leading-unit-dim reshape (the rank-3 logits tensor
        # [1, seq, vocab] produced by HF causal-LM heads).
        (
            (256, 512),
            (512, 1024),
            False,
            False,
            [1, 256, 1024],
            [0, 255, 0],
            [1, 256, 1024],
        ),
        # Middle block of output rows ([5:17]) -> pushed into A just like a
        # trailing row slice; only the selected rows feed the matmul.
        ((256, 512), (512, 1024), False, False, None, [5, 0], [17, 1024]),
        # Negative begin/end row indices ([-4:-1]) are normalized against the
        # dim size before being pushed into A (rows [252:255]).
        ((256, 512), (512, 1024), False, False, None, [-4, 0], [-1, 1024]),
        # Batched 3D x 3D matmul: the leading batch dim is kept full while the
        # output row dim is sliced and pushed into A's matching dim.
        (
            (2, 256, 512),
            (2, 512, 1024),
            False,
            False,
            None,
            [0, 255, 0],
            [2, 256, 1024],
        ),
        # Column (N) slice through a leading-unit-dim reshape -> pushed into B;
        # the reshape is looked through and re-applied to the narrowed result.
        (
            (256, 512),
            (512, 1024),
            False,
            False,
            [1, 256, 1024],
            [0, 0, 5],
            [1, 256, 17],
        ),
        # transpose_a=true and transpose_b=true together: A is [K, M], B is
        # [N, K]. A row slice is pushed into A's trailing (M) dim and both
        # transpose flags are preserved.
        ((512, 256), (1024, 512), True, True, None, [255, 0], [256, 1024]),
        # transpose_a=true row slice *through* a leading-unit-dim reshape: the
        # reshape's dim offset and the transpose flag are honored together; the
        # slice is pushed into A's trailing (M) dim.
        (
            (512, 256),
            (512, 1024),
            True,
            False,
            [1, 256, 1024],
            [0, 255, 0],
            [1, 256, 1024],
        ),
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 2])
def test_permute_slice_after_matmul_fusing(
    shape_a: Shape,
    shape_b: Shape,
    transpose_a: bool,
    transpose_b: bool,
    reshape_shape: Optional[List[int]],
    begins: List[int],
    ends: List[int],
    dtypes: List[torch.dtype],
    request,
    device,
):
    shapes = [shape_a, shape_b]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def matmul_slice(
            input_tensor: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            matmul_result = builder.matmul(
                input_tensor, weight, transpose_a=transpose_a, transpose_b=transpose_b
            )
            if reshape_shape is not None:
                matmul_result = builder.reshape(matmul_result, reshape_shape)
            return builder.slice(matmul_result, begins, ends)

    # Numerical correctness: compile_and_execute_ttir runs the matmul->slice
    # graph on device with check_pcc=True (default pcc=0.99) and compares the
    # result against the torch golden the builder accumulates. If the fusion
    # changed the numerics, PCC drops below threshold and this call raises.
    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
        check_pcc=True,
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    # The fusion fired: a slice now feeds the matmul instead of consuming it.
    assert check_op(output_path, "matmul") and slice_precedes_matmul(output_path)


@pytest.mark.parametrize(
    "shape_a,shape_b,bias_shape,transpose_a,transpose_b,begins,ends",
    [
        # Row (M) slice -> pushed into A; the [N] bias is indexed by N, so it
        # passes through to the narrowed linear untouched.
        ((256, 512), (512, 1024), (1024,), False, False, [255, 0], [256, 1024]),
        # Column (N) slice -> pushed into B; the [N] bias is sliced along N too.
        ((256, 512), (512, 1024), (1024,), False, False, [0, 5], [256, 17]),
        # transpose_b=true column slice: B is [N, K] and the [N] bias is still
        # sliced along the N range.
        ((256, 512), (1024, 512), (1024,), False, True, [0, 0], [256, 64]),
        # transpose_a=true row slice: A is [K, M], so the slice is pushed into
        # A's trailing dim and the [N] bias passes through untouched.
        ((512, 256), (512, 1024), (1024,), True, False, [255, 0], [256, 1024]),
        # Batched 3D linear with a full 3D bias [B, M, N]; a column slice narrows
        # B and the bias's trailing N dim (batch + M kept full).
        (
            (2, 256, 512),
            (2, 512, 1024),
            (2, 256, 1024),
            False,
            False,
            [0, 0, 5],
            [2, 256, 17],
        ),
        # Batched 3D linear with a full 3D bias, row (M) slice: pushed into A and
        # the bias is sliced along its M dim (batch + N kept full).
        (
            (2, 256, 512),
            (2, 512, 1024),
            (2, 256, 1024),
            False,
            False,
            [0, 255, 0],
            [2, 256, 1024],
        ),
        # Batched 3D linear with a [1, M, N] bias (broadcast over the batch dim):
        # a column slice narrows the bias's N dim; the size-1 batch dim stays full
        # and keeps broadcasting onto the 2-batch output.
        (
            (2, 256, 512),
            (2, 512, 1024),
            (1, 256, 1024),
            False,
            False,
            [0, 0, 5],
            [2, 256, 17],
        ),
        # Row slice with a full 2D bias [M, N]: bias is sliced along the M range.
        ((256, 512), (512, 1024), (256, 1024), False, False, [255, 0], [256, 1024]),
        # Column slice with a full 2D bias [M, N]: bias is sliced along the N
        # range, its M dim kept full.
        ((256, 512), (512, 1024), (256, 1024), False, False, [0, 5], [256, 17]),
        # Row slice with a [1, N] bias (broadcast over M): the M-aligned bias dim
        # is size-1, so the bias is left untouched and must still broadcast.
        ((256, 512), (512, 1024), (1, 1024), False, False, [255, 0], [256, 1024]),
        # Column slice with a [M, 1] bias (broadcast over N): the N-aligned bias
        # dim is size-1, so the bias is left untouched and must still broadcast.
        ((256, 512), (512, 1024), (256, 1), False, False, [0, 5], [256, 17]),
        # Column slice with a scalar bias [1] (broadcast over everything): the
        # aligned dim is size-1, so the bias is left untouched while B is narrowed.
        ((256, 512), (512, 1024), (1,), False, False, [0, 5], [256, 17]),
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3])
def test_permute_slice_after_linear_fusing(
    shape_a: Shape,
    shape_b: Shape,
    bias_shape: Shape,
    transpose_a: bool,
    transpose_b: bool,
    begins: List[int],
    ends: List[int],
    dtypes: List[torch.dtype],
    request,
    device,
):
    shapes = [shape_a, shape_b, bias_shape]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def linear_slice(
            input_tensor: Operand,
            weight: Operand,
            bias: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            linear_result = builder.linear(
                input_tensor,
                weight,
                bias=bias,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
            )
            return builder.slice(linear_result, begins, ends)

    # Numerical correctness: the on-device golden PCC comparison (check_pcc,
    # default 0.99) catches any mismatch from slicing the operand/bias.
    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
        check_pcc=True,
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    # The fusion fired: a slice now feeds the narrowed op instead of consuming
    # it. A 2D linear stays ttnn.linear; a batched linear lowers to ttnn.matmul
    # + add, so accept either.
    assert (
        check_op(output_path, "linear") or check_op(output_path, "matmul")
    ) and slice_precedes_matmul(output_path)


@pytest.mark.parametrize("dtypes", [[torch.float32] * 4])
def test_permute_slice_after_matmul_cascade_fusing(
    dtypes: List[torch.dtype],
    request,
    device,
):
    # Chain of three matmuls, each result feeding the next as its single-use
    # LHS, with a row (M) slice at the bottom. PermuteSliceAfterMatmul pushes the row
    # slice up one level into the LHS, which is itself a matmul -- re-matching
    # the pattern -- so the greedy driver cascades it to the top: the slice ends
    # up on the original input and every matmul is narrowed to a single row.
    shapes = [(256, 512), (512, 384), (384, 256), (256, 128)]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def matmul_chain_slice(
            a: Operand,
            b: Operand,
            c: Operand,
            d: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            m1 = builder.matmul(a, b)
            m2 = builder.matmul(m1, c)
            m3 = builder.matmul(m2, d)
            return builder.slice(m3, [255, 0], [256, 128])

    # PCC (default 0.99) verifies the cascade preserved numerics end to end.
    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
        check_pcc=True,
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    # The cascade fired: a slice now feeds the (first) matmul instead of a slice
    # consuming the (last) matmul's output.
    assert check_op(output_path, "matmul") and slice_precedes_matmul(output_path)


@pytest.mark.parametrize("dtypes", [[torch.float32] * 4])
def test_shared_lhs_fusion_not_undone(
    dtypes: List[torch.dtype],
    request,
    device,
):
    # Three matmuls sharing the same LHS are fused by SharedLHSMatmulFusion into
    # one matmul over the concatenated weights, split back out with per-output
    # slices. Those slices feed a matmul result with multiple uses, so
    # PermuteSliceAfterMatmul's single-use guard must leave them alone (pushing them
    # into the concatenated RHS would undo the concatenation). This verifies the
    # two fusions coexist and the result stays numerically correct.
    shapes = [(32, 512), (512, 384), (512, 384), (512, 384)]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def shared_lhs(
            a: Operand,
            b0: Operand,
            b1: Operand,
            b2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return (
                builder.matmul(a, b0),
                builder.matmul(a, b1),
                builder.matmul(a, b2),
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
        check_pcc=True,
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    # SharedLHSMatmulFusion fired and was preserved: the weights are concatenated
    # and fed to a single matmul (not pulled back apart by PermuteSliceAfterMatmul).
    assert check_op(output_path, "matmul") and check_op(output_path, "concat")


@pytest.mark.parametrize("dtypes", [[torch.float32] * 5])
def test_shared_lhs_fusion_with_final_slice(
    dtypes: List[torch.dtype],
    request,
    device,
):
    # Same shared-LHS group, but one output feeds a downstream "final" matmul
    # whose result is row-sliced. Both fusions apply: the shared-LHS group still
    # fuses to one matmul over the concatenated weights (left untouched by the
    # single-use guard), while the trailing row slice IS pushed up into the
    # final matmul, narrowing it to a single row.
    shapes = [(32, 512), (512, 384), (512, 384), (512, 384), (384, 256)]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def shared_lhs_final(
            a: Operand,
            b0: Operand,
            b1: Operand,
            b2: Operand,
            d: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            o0 = builder.matmul(a, b0)
            o1 = builder.matmul(a, b1)
            o2 = builder.matmul(a, b2)
            final = builder.matmul(o2, d)
            return (o0, o1, builder.slice(final, [31, 0], [32, 256]))

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
        check_pcc=True,
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    # The shared-LHS concatenation survives and a matmul remains for the
    # (now narrowed) final projection.
    assert check_op(output_path, "matmul") and check_op(output_path, "concat")


@pytest.mark.parametrize("dtypes", [[torch.float32] * 4])
def test_shared_lhs_vs_permute_slice_after_matmul_collision(
    dtypes: List[torch.dtype],
    request,
    device,
):
    # Direct collision: all three matmuls share the same LHS (so
    # SharedLHSMatmulFusion wants to fuse them) AND the last one's output is
    # sliced (so PermuteSliceAfterMatmul wants to push the slice into the LHS). With
    # the pass's top-down traversal the matmuls are visited before the slice, so
    # SharedLHSMatmulFusion fires first; the fused result then has multiple uses
    # and PermuteSliceAfterMatmul's single-use guard blocks it. SharedLHS wins -- the
    # row slice just composes with the fused output column slice. Either outcome
    # is numerically correct; this pins the winner and the numerics.
    shapes = [(32, 512), (512, 384), (512, 384), (512, 384)]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def shared_lhs_collision(
            a: Operand,
            b0: Operand,
            b1: Operand,
            b2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            o0 = builder.matmul(a, b0)
            o1 = builder.matmul(a, b1)
            o2 = builder.matmul(a, b2)
            return (o0, o1, builder.slice(o2, [31, 0], [32, 384]))

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
        check_pcc=True,
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    # SharedLHS won: the weights are concatenated into a single fused matmul, and
    # the slices land on its output (so a slice does NOT precede the matmul, as
    # it would have if PermuteSliceAfterMatmul had fired instead).
    assert (
        check_op(output_path, "matmul")
        and check_op(output_path, "concat")
        and not slice_precedes_matmul(output_path)
    )


@pytest.mark.parametrize(
    "matmul_shapes,bias_shape,bias_reshape,bias_broadcast",
    [
        # ViT / BERT pattern: matmul [1576, 768] + bias [768] reshaped to
        # [1, 768] and broadcast to [1576, 768].
        (
            [(1576, 768), (768, 768)],
            (768,),
            [1, 768],
            [1576, 1],
        ),
        # Phi decode pattern: matmul [32, 2048] + bias [2048] reshaped to
        # [1, 2048] and broadcast to [32, 2048].
        (
            [(32, 2048), (2048, 2048)],
            (2048,),
            [1, 2048],
            [32, 1],
        ),
        # Qwen decode pattern: matmul [32, 896] + bias [896] reshaped to
        # [1, 896] and broadcast to [32, 896].
        (
            [(32, 896), (896, 896)],
            (896,),
            [1, 896],
            [32, 1],
        ),
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3])
def test_matmul_with_bias_reshape_broadcast_fusing(
    matmul_shapes: List[Shape],
    bias_shape: Shape,
    bias_reshape: List[int],
    bias_broadcast: List[int],
    dtypes: List[torch.dtype],
    request,
    device,
):
    shapes = [matmul_shapes[0], matmul_shapes[1], bias_shape]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def matmul_bias_tm(
            input_tensor: Operand,
            weight: Operand,
            bias: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            matmul_result = builder.matmul(input_tensor, weight)
            reshaped_bias = builder.reshape(bias, bias_reshape)
            broadcast_bias = builder.broadcast(reshaped_bias, bias_broadcast)
            return builder.add(matmul_result, broadcast_bias)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        save_artifacts=True,
    )
    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )
    assert check_op(output_path, "linear") and not check_op(output_path, "matmul")
