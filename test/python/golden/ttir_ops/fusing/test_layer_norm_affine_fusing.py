# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
from typing import List, Optional

from builder.base.builder_utils import Operand, Shape, get_artifact_dir
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")

EPSILON = 1e-5


def check_op(mlir_file: str, op_name: str, dialect: str = "ttnn") -> bool:
    """Check if an op exists in the MLIR file."""
    op_pattern = f"{dialect}.{op_name}"
    with open(mlir_file, "r") as f:
        return any(op_pattern in line for line in f)


def compiled_mlir_path(request) -> str:
    return os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )


def build_torch_layer_norm_affine(
    input_data: torch.Tensor,
    weight_data: torch.Tensor,
    bias_data: torch.Tensor,
    normalized_shape: List[int],
) -> torch.Tensor:
    """Reference for layer_norm(x) * weight + bias with per-channel weight/bias."""
    normalized = torch.nn.functional.layer_norm(
        input_data.float(), normalized_shape, eps=EPSILON
    )
    return (normalized * weight_data.float() + bias_data.float()).to(input_data.dtype)


# =============================================================================
# Per-channel affine trailing an unaffine layer_norm fuses into weight/bias.
# =============================================================================
@pytest.mark.parametrize(
    "shape,normalized_shape",
    [
        ((1, 256, 512), [512]),
        ((2, 4, 64), [64]),
        ((32, 128), [128]),
    ],
    ids=["3D", "3D_small", "2D"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_layer_norm_affine_fusion(
    shape: Shape,
    normalized_shape: List[int],
    target: str,
    request,
    device,
):
    """layer_norm(x) * w + b should collapse into a single layer_norm."""
    param_shape = tuple([1] * (len(shape) - 1) + [normalized_shape[-1]])
    shapes = [shape, param_shape, param_shape]
    dtypes = [torch.float32] * 3

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def layer_norm_affine_fusion(
            input: Operand,
            weight: Operand,
            bias: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            input_data = torch.randn(shape, dtype=torch.float32)
            weight_data = torch.randn(param_shape, dtype=torch.float32)
            bias_data = torch.randn(param_shape, dtype=torch.float32)
            golden_output = build_torch_layer_norm_affine(
                input_data, weight_data, bias_data, normalized_shape
            )

            normed = builder.layer_norm(
                input, normalized_shape=normalized_shape, epsilon=EPSILON
            )
            broadcast_dims = list(shape[:-1]) + [1]
            weight_bcast = builder.broadcast(
                weight, broadcast_dimensions=broadcast_dims
            )
            scaled = builder.multiply(normed, weight_bcast)
            bias_bcast = builder.broadcast(bias, broadcast_dimensions=broadcast_dims)
            result = builder.add(scaled, bias_bcast)

            builder.set_goldens(
                {input: input_data, weight: weight_data, bias: bias_data},
                {result: golden_output},
            )
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        save_artifacts=True,
    )

    output_path = compiled_mlir_path(request)
    assert check_op(output_path, "layer_norm"), "layer_norm should survive the fusion"
    assert not check_op(
        output_path, "multiply"
    ), "The activation-sized multiply should be folded into the norm's weight"


# =============================================================================
# The adaLN shape this pattern was written for: norm(x) * (1 + scale) + shift.
# The `1 + scale` add survives, but on the per-channel tensor.
# =============================================================================
@pytest.mark.parametrize(
    "shape,normalized_shape",
    [
        ((1, 256, 512), [512]),
        ((1, 64, 128), [128]),
    ],
    ids=["3D", "3D_small"],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_layer_norm_adaln_modulation_fusion(
    shape: Shape,
    normalized_shape: List[int],
    target: str,
    request,
    device,
):
    """adaLN modulation folds into the norm; only the per-channel add remains."""
    param_shape = tuple([1] * (len(shape) - 1) + [normalized_shape[-1]])
    shapes = [shape, param_shape, param_shape]
    dtypes = [torch.float32] * 3

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def layer_norm_adaln_fusion(
            input: Operand,
            scale: Operand,
            shift: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            input_data = torch.randn(shape, dtype=torch.float32)
            scale_data = torch.randn(param_shape, dtype=torch.float32)
            shift_data = torch.randn(param_shape, dtype=torch.float32)
            golden_output = build_torch_layer_norm_affine(
                input_data, 1.0 + scale_data, shift_data, normalized_shape
            )

            normed = builder.layer_norm(
                input, normalized_shape=normalized_shape, epsilon=EPSILON
            )
            one = builder.constant(torch.ones(param_shape, dtype=torch.float32))
            one_plus_scale = builder.add(one, scale)
            broadcast_dims = list(shape[:-1]) + [1]
            scale_bcast = builder.broadcast(
                one_plus_scale, broadcast_dimensions=broadcast_dims
            )
            modulated = builder.multiply(normed, scale_bcast)
            shift_bcast = builder.broadcast(shift, broadcast_dimensions=broadcast_dims)
            result = builder.add(modulated, shift_bcast)

            builder.set_goldens(
                {input: input_data, scale: scale_data, shift: shift_data},
                {result: golden_output},
            )
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        save_artifacts=True,
    )

    output_path = compiled_mlir_path(request)
    assert check_op(output_path, "layer_norm"), "layer_norm should survive the fusion"
    assert not check_op(
        output_path, "multiply"
    ), "The activation-sized multiply should be folded into the norm's weight"


# =============================================================================
# Norm in fp32, modulation in bf16 - the shape a frontend produces when it
# casts down right after the norm. The affine moves inside the norm and is
# evaluated in fp32, so this checks the promotion holds accuracy.
# =============================================================================
@pytest.mark.parametrize("shape,normalized_shape", [((1, 256, 512), [512])], ids=["3D"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_layer_norm_affine_fusion_mixed_dtype(
    shape: Shape,
    normalized_shape: List[int],
    target: str,
    request,
    device,
):
    """layer_norm(x_f32) -> bf16 -> * w_bf16 + b_bf16 should still fuse."""
    param_shape = tuple([1] * (len(shape) - 1) + [normalized_shape[-1]])
    shapes = [shape, param_shape, param_shape]
    dtypes = [torch.float32, torch.bfloat16, torch.bfloat16]

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def layer_norm_affine_mixed_dtype(
            input: Operand,
            weight: Operand,
            bias: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            input_data = torch.randn(shape, dtype=torch.float32)
            weight_data = torch.randn(param_shape, dtype=torch.bfloat16)
            bias_data = torch.randn(param_shape, dtype=torch.bfloat16)
            # The fused form evaluates the affine in the norm's fp32, so the
            # reference does too; the bf16 parameters are widened, not the
            # other way round.
            normalized = torch.nn.functional.layer_norm(
                input_data, normalized_shape, eps=EPSILON
            )
            golden_output = (normalized * weight_data.float() + bias_data.float()).to(
                torch.bfloat16
            )

            normed = builder.layer_norm(
                input, normalized_shape=normalized_shape, epsilon=EPSILON
            )
            normed_bf16 = builder.typecast(normed, output_type=torch.bfloat16)
            broadcast_dims = list(shape[:-1]) + [1]
            weight_bcast = builder.broadcast(
                weight, broadcast_dimensions=broadcast_dims
            )
            scaled = builder.multiply(normed_bf16, weight_bcast)
            bias_bcast = builder.broadcast(bias, broadcast_dimensions=broadcast_dims)
            result = builder.add(scaled, bias_bcast)

            builder.set_goldens(
                {input: input_data, weight: weight_data, bias: bias_data},
                {result: golden_output},
            )
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        save_artifacts=True,
    )

    output_path = compiled_mlir_path(request)
    assert check_op(output_path, "layer_norm"), "layer_norm should survive the fusion"
    assert not check_op(
        output_path, "multiply"
    ), "The activation-sized multiply should be folded into the norm's weight"


# =============================================================================
# Negative: a residual addend is not a per-channel bias. The pattern must
# decline, and the unfused graph must still be numerically correct.
# =============================================================================
@pytest.mark.parametrize("shape,normalized_shape", [((1, 256, 512), [512])], ids=["3D"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_layer_norm_residual_addend_not_fused(
    shape: Shape,
    normalized_shape: List[int],
    target: str,
    request,
    device,
):
    """layer_norm(x) * w + residual must not fold the residual into the bias."""
    param_shape = tuple([1] * (len(shape) - 1) + [normalized_shape[-1]])
    shapes = [shape, param_shape, shape]
    dtypes = [torch.float32] * 3

    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def layer_norm_residual_addend(
            input: Operand,
            weight: Operand,
            residual: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            input_data = torch.randn(shape, dtype=torch.float32)
            weight_data = torch.randn(param_shape, dtype=torch.float32)
            residual_data = torch.randn(shape, dtype=torch.float32)
            normalized = torch.nn.functional.layer_norm(
                input_data.float(), normalized_shape, eps=EPSILON
            )
            golden_output = normalized * weight_data.float() + residual_data.float()

            normed = builder.layer_norm(
                input, normalized_shape=normalized_shape, epsilon=EPSILON
            )
            broadcast_dims = list(shape[:-1]) + [1]
            weight_bcast = builder.broadcast(
                weight, broadcast_dimensions=broadcast_dims
            )
            scaled = builder.multiply(normed, weight_bcast)
            result = builder.add(scaled, residual)

            builder.set_goldens(
                {input: input_data, weight: weight_data, residual: residual_data},
                {result: golden_output},
            )
            return result

    compile_and_execute_ttir(
        module,
        target=target,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        save_artifacts=True,
    )

    output_path = compiled_mlir_path(request)
    assert check_op(
        output_path, "multiply"
    ), "A residual addend must not be folded into the norm"
