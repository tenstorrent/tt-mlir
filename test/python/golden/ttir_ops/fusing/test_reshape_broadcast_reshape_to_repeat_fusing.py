# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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


def check_op(mlir_file: str, op_name: str) -> bool:
    """Check if a ttnn operation exists in the MLIR file."""
    op_name = "ttnn." + op_name
    with open(mlir_file, "r") as f:
        for line in f:
            if op_name in line:
                return True
    return False


def build_torch_golden_repeat(
    input_data: torch.Tensor,
    repeat_dim: int,
    repeat_count: int,
) -> torch.Tensor:
    """
    Build golden output for repeat operation.
    Equivalent to: unsqueeze at repeat_dim -> expand by repeat_count -> reshape to merge with repeat_dim
    """
    repeat_dims = [1] * len(input_data.shape)
    repeat_dims[repeat_dim] = repeat_count
    return input_data.repeat(repeat_dims)


def build_torch_golden_repeat_interleave(
    input_data: torch.Tensor,
    repeat_dim: int,
    repeat_count: int,
) -> torch.Tensor:
    """
    Build golden output for repeat_interleave operation.
    Equivalent to: unsqueeze at repeat_dim+1 -> expand by repeat_count -> reshape to merge with repeat_dim
    """
    return torch.repeat_interleave(input_data, repeat_count, dim=repeat_dim)


def build_ttir_reshape_broadcast_reshape_repeat(
    input: Operand,
    builder: TTIRBuilder,
    repeat_dim: int,
    repeat_count: int,
    unit_attrs: Optional[List[str]] = None,
):
    """
    Build TTIR representation of reshape -> broadcast -> reshape pattern
    that should fuse to ttir.repeat.

    Pattern (right merge - changedDim == insertedDim):
    - Input: [d0, d1, d2, d3]
    - Unsqueeze at repeat_dim: inserts dim of size 1 at repeat_dim
    - Broadcast: expands the inserted dim by repeat_count
    - Reshape: merges the expanded dim with the dimension at repeat_dim

    Example with repeat_dim=1, repeat_count=4, input=[1, 8, 128, 64]:
    - reshape: [1, 8, 128, 64] -> [1, 1, 8, 128, 64]
    - broadcast [1, 4, 1, 1, 1]: [1, 1, 8, 128, 64] -> [1, 4, 8, 128, 64]
    - reshape: [1, 4, 8, 128, 64] -> [1, 32, 128, 64]
    """
    input_shape = builder.get_shape(input)
    rank = len(input_shape)

    # Step 1: Unsqueeze at repeat_dim (insert dim of size 1)
    unsqueeze_shape = list(input_shape)
    unsqueeze_shape.insert(repeat_dim, 1)
    unsqueezed = builder.reshape(input, unsqueeze_shape, unit_attrs=unit_attrs)

    # Step 2: Broadcast the inserted dimension
    broadcast_factors = [1] * (rank + 1)
    broadcast_factors[repeat_dim] = repeat_count
    broadcasted = builder.broadcast(
        unsqueezed, broadcast_factors, unit_attrs=unit_attrs
    )

    # Step 3: Reshape to merge the broadcasted dim with the next dim (right merge)
    output_shape = list(input_shape)
    output_shape[repeat_dim] = input_shape[repeat_dim] * repeat_count
    output = builder.reshape(broadcasted, output_shape, unit_attrs=unit_attrs)

    return output


def build_ttir_reshape_broadcast_reshape_repeat_interleave(
    input: Operand,
    builder: TTIRBuilder,
    repeat_dim: int,
    repeat_count: int,
    unit_attrs: Optional[List[str]] = None,
):
    """
    Build TTIR representation of reshape -> broadcast -> reshape pattern
    that should fuse to ttir.repeat_interleave.

    Pattern (left merge - changedDim == insertedDim - 1):
    - Input: [d0, d1, d2, d3]
    - Unsqueeze at repeat_dim+1: inserts dim of size 1 after repeat_dim
    - Broadcast: expands the inserted dim by repeat_count
    - Reshape: merges the expanded dim with the dimension at repeat_dim (left merge)

    Example with repeat_dim=1, repeat_count=4, input=[1, 8, 128, 64]:
    - reshape: [1, 8, 128, 64] -> [1, 8, 1, 128, 64]
    - broadcast [1, 1, 4, 1, 1]: [1, 8, 1, 128, 64] -> [1, 8, 4, 128, 64]
    - reshape: [1, 8, 4, 128, 64] -> [1, 32, 128, 64]
    """
    input_shape = builder.get_shape(input)
    rank = len(input_shape)

    # Step 1: Unsqueeze at repeat_dim + 1 (insert dim of size 1 after repeat_dim)
    unsqueeze_shape = list(input_shape)
    unsqueeze_shape.insert(repeat_dim + 1, 1)
    unsqueezed = builder.reshape(input, unsqueeze_shape, unit_attrs=unit_attrs)

    # Step 2: Broadcast the inserted dimension
    broadcast_factors = [1] * (rank + 1)
    broadcast_factors[repeat_dim + 1] = repeat_count
    broadcasted = builder.broadcast(
        unsqueezed, broadcast_factors, unit_attrs=unit_attrs
    )

    # Step 3: Reshape to merge the broadcasted dim with the previous dim (left merge)
    output_shape = list(input_shape)
    output_shape[repeat_dim] = input_shape[repeat_dim] * repeat_count
    output = builder.reshape(broadcasted, output_shape, unit_attrs=unit_attrs)

    return output


@pytest.mark.parametrize(
    "shape,repeat_dim,repeat_count",
    [
        # GQA-like patterns: expanding KV heads to match Q heads
        ((1, 8, 128, 64), 1, 4),  # [1, 8, 128, 64] -> [1, 32, 128, 64]
        ((32, 8, 128, 64), 1, 4),  # [32, 8, 128, 64] -> [32, 32, 128, 64]
        # Repeat along different dimensions
        ((1, 32, 1, 64), 2, 128),  # [1, 32, 1, 64] -> [1, 32, 128, 64]
        ((4, 16, 64), 0, 2),  # [4, 16, 64] -> [8, 16, 64]
        ((8, 32, 64), 1, 4),  # [8, 32, 64] -> [8, 128, 64]
        # MQA pattern: single KV head expanded to all Q heads
        ((8, 1, 128, 64), 1, 32),  # [8, 1, 128, 64] -> [8, 32, 128, 64]
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_reshape_broadcast_reshape_to_repeat(
    shape: Shape, repeat_dim: int, repeat_count: int, target: str, request, device
):
    """
    Test that reshape -> broadcast -> reshape pattern fuses to ttir.repeat.

    This pattern occurs in GQA (Grouped Query Attention) when expanding
    KV heads to match Q heads using right-merge:
    - Unsqueeze at repeat_dim
    - Broadcast the inserted dim
    - Merge the broadcasted dim with the next dim
    """
    input_shapes = [shape]
    dtypes = [torch.bfloat16]

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def reshape_broadcast_reshape_repeat(
            input: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            input_data = torch.randn(shape, dtype=torch.bfloat16)
            golden_output = build_torch_golden_repeat(
                input_data, repeat_dim, repeat_count
            )

            result = build_ttir_reshape_broadcast_reshape_repeat(
                input, builder, repeat_dim, repeat_count, unit_attrs=unit_attrs
            )

            builder.set_goldens(
                {input: input_data},
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

    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )

    assert check_op(output_path, "repeat")


@pytest.mark.parametrize(
    "shape,repeat_dim,repeat_count",
    [
        # GQA-like patterns: expanding KV heads to match Q heads with interleaving
        ((1, 8, 128, 64), 1, 4),  # [1, 8, 128, 64] -> [1, 32, 128, 64]
        ((32, 8, 128, 64), 1, 4),  # [32, 8, 128, 64] -> [32, 32, 128, 64]
        # Repeat interleave along different dimensions
        ((1, 32, 1, 64), 2, 128),  # [1, 32, 1, 64] -> [1, 32, 128, 64]
        ((4, 16, 64), 0, 2),  # [4, 16, 64] -> [8, 16, 64]
        ((8, 32, 64), 1, 4),  # [8, 32, 64] -> [8, 128, 64]
        # MQA pattern
        ((8, 1, 128, 64), 1, 32),  # [8, 1, 128, 64] -> [8, 32, 128, 64]
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_reshape_broadcast_reshape_to_repeat_interleave(
    shape: Shape, repeat_dim: int, repeat_count: int, target: str, request, device
):
    """
    Test that reshape -> broadcast -> reshape pattern fuses to ttir.repeat_interleave.

    This pattern occurs when using left-merge to repeat elements:
    - Unsqueeze at repeat_dim + 1
    - Broadcast the inserted dim
    - Merge the broadcasted dim with the previous dim (repeat_dim)
    """
    input_shapes = [shape]
    dtypes = [torch.bfloat16]

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, dtypes)
        def reshape_broadcast_reshape_repeat_interleave(
            input: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            input_data = torch.randn(shape, dtype=torch.bfloat16)
            golden_output = build_torch_golden_repeat_interleave(
                input_data, repeat_dim, repeat_count
            )

            result = build_ttir_reshape_broadcast_reshape_repeat_interleave(
                input, builder, repeat_dim, repeat_count, unit_attrs=unit_attrs
            )

            builder.set_goldens(
                {input: input_data},
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

    output_path = os.path.join(
        get_artifact_dir(
            request.config.getoption("--path"), "TTIRBuilder", request.node.name
        ),
        "ttnn_compiled.mlir",
    )

    assert check_op(output_path, "repeat_interleave")
