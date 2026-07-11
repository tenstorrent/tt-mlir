# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import pytest
import torch

from ttmlir.dialects import ttcore
from ttmlir.ir import *

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


def get_dma_pipeline(
    use_tensor_accessor_dma: bool,
    force_compile_time_args: bool = False,
) -> str:
    # Back to back tolayout ops are normally folded during canonicalization into
    # a single ToLayoutOp representing the final result. The option
    # 'disable-tolayout-folding' prevents this
    pipeline_options = (
        "{disable-tolayout-folding=1 "
        f"use-tensor-accessor-dma={int(use_tensor_accessor_dma)}"
        f" force-compile-time-args={int(force_compile_time_args)}"
        "}"
    )
    return f"ttir-to-ttmetal-pipeline{pipeline_options}"


def compile_dma_test(
    test_func,
    target,
    request,
    device,
    use_tensor_accessor_dma: bool = False,
    force_compile_time_args: bool = False,
):
    compile_and_execute_ttir(
        test_func,
        target=target,
        device=device,
        custom_pipeline=get_dma_pipeline(
            use_tensor_accessor_dma, force_compile_time_args
        ),
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("shape", [(256, 256)])
@pytest.mark.parametrize("memory_space", [ttcore.MemorySpace.DeviceDRAM])
def test_host_interop_single_bank_dram_dma(
    shape: Shape,
    memory_space: ttcore.MemorySpace,
    target: str,
    request,
    device,
):
    """tests that host enqueue_read|write_buffer works for single-shard DRAM
    buffers"""

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def tilize(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):

            to_device = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, tiled=False, memorySpace=memory_space
                ),
                unit_attrs=unit_attrs,
            )

            system_out = builder.to_layout(
                to_device,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )

            return system_out

    compile_dma_test(module, target, request, device=device)


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("shape", [(256, 256)])
@pytest.mark.parametrize("start_grid", [(1, 1), (1, 2), (2, 1), (4, 4)])
@pytest.mark.parametrize("end_grid", [(1, 1), (2, 2)])
@pytest.mark.parametrize(
    "memory_space", [ttcore.MemorySpace.DeviceL1, ttcore.MemorySpace.DeviceDRAM]
)
def test_roundtrip_dma_tiled(
    target: str,
    shape: Shape,
    start_grid: tuple[int, int],
    end_grid: tuple[int, int],
    memory_space: ttcore.MemorySpace,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def tilize(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # derive sharded shapes
            assert (shape[0] % start_grid[0] == 0) and (
                shape[1] % start_grid[1] == 0
            ), "shape must be divisible by start_grid"
            assert (shape[0] % end_grid[0] == 0) and (
                shape[1] % end_grid[1] == 0
            ), "shard_shape must be divisible by end_grid"

            # tilize the tensor on a single worker
            to_device = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=True,
                    memorySpace=ttcore.MemorySpace.DeviceL1,
                ),
                unit_attrs=unit_attrs,
            )

            # WRITE tensor from L1 to initial shard layout
            tensor_layoutA = builder.to_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=True,
                    memorySpace=memory_space,
                    grid=start_grid,
                ),
                unit_attrs=unit_attrs,
            )

            # READ sharded layout to final sharded layout
            tensor_layoutB = builder.to_layout(
                tensor_layoutA,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=True,
                    memorySpace=ttcore.MemorySpace.DeviceL1,
                    grid=end_grid,
                ),
                unit_attrs=unit_attrs,
            )

            untilize_out = builder.untilize(
                tensor_layoutB,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )

            return untilize_out

    compile_dma_test(module, target, request, device=device)


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "shape,remote_kind,start_grid,end_grid,force_compile_time_args",
    [
        ((192, 384), "l1_sharded", (1, 3), (2, 2), False),
        ((192, 384), "dram_sharded", (1, 3), (2, 2), False),
        ((192, 384), "dram_interleaved", (1, 1), (2, 2), False),
        ((192, 384), "dram_sharded_force_cta", (1, 3), (2, 2), True),
    ],
)
@pytest.mark.parametrize("use_tensor_accessor_dma", [False, True])
def test_tensor_accessor_dma_tiled(
    target: str,
    shape: Shape,
    remote_kind: str,
    start_grid: tuple[int, int],
    end_grid: tuple[int, int],
    force_compile_time_args: bool,
    use_tensor_accessor_dma: bool,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        golden = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)

        @builder.func([shape], [torch.float32])
        def tensor_accessor_roundtrip(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            tiled_input = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=True,
                    memorySpace=ttcore.MemorySpace.DeviceL1,
                ),
                unit_attrs=unit_attrs,
            )

            is_dram = remote_kind.startswith("dram")
            is_interleaved = remote_kind == "dram_interleaved"
            remote = builder.to_layout(
                tiled_input,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=True,
                    memorySpace=(
                        ttcore.MemorySpace.DeviceDRAM
                        if is_dram
                        else ttcore.MemorySpace.DeviceL1
                    ),
                    grid=start_grid,
                    memory_layout=(
                        ttcore.TensorMemoryLayout.Interleaved
                        if is_interleaved
                        else ttcore.TensorMemoryLayout.Sharded
                    ),
                ),
                unit_attrs=unit_attrs,
            )

            local = builder.to_layout(
                remote,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=True,
                    memorySpace=ttcore.MemorySpace.DeviceL1,
                    grid=end_grid,
                ),
                unit_attrs=unit_attrs,
            )
            output = builder.untilize(
                local, output_type=in0.type, unit_attrs=unit_attrs
            )
            builder.set_goldens({in0: golden}, {output: golden})
            return output

    compile_dma_test(
        module,
        target,
        request,
        device=device,
        use_tensor_accessor_dma=use_tensor_accessor_dma,
        force_compile_time_args=force_compile_time_args,
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("use_tensor_accessor_dma", [False, True])
def test_tensor_accessor_dma_tiled_vgm(
    target: str,
    use_tensor_accessor_dma: bool,
    request,
    device,
):
    shape = (128, 384)

    def module(builder: TTIRBuilder):
        golden = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)

        @builder.func([shape], [torch.float32])
        def tensor_accessor_vgm(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            tiled_input = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=True,
                    memorySpace=ttcore.MemorySpace.DeviceL1,
                ),
                unit_attrs=unit_attrs,
            )
            virtual_grid = builder.to_layout(
                tiled_input,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=True,
                    memorySpace=ttcore.MemorySpace.DeviceL1,
                    grid=(1, 12),
                ),
                unit_attrs=unit_attrs,
            )
            local = builder.to_layout(
                virtual_grid,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=True,
                    memorySpace=ttcore.MemorySpace.DeviceL1,
                    grid=(2, 6),
                ),
                unit_attrs=unit_attrs,
            )
            output = builder.untilize(
                local, output_type=in0.type, unit_attrs=unit_attrs
            )
            builder.set_goldens({in0: golden}, {output: golden})
            return output

    compile_dma_test(
        module,
        target,
        request,
        device=device,
        use_tensor_accessor_dma=use_tensor_accessor_dma,
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.skip_exec(
    ("n150", "sim"), reason="TEN-3868 LLK f32 tilize undefined behavior"
)
def test_tensor_accessor_binary_add(
    target: str,
    request,
    device,
):
    """Exercise BFP8 page sizing with two TA reads and one TA write."""
    shape = (256, 256)

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [torch.float32, torch.float32])
        def tensor_accessor_add(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            input0 = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(
                shape
            ) / (shape[0] * shape[1])
            input1 = torch.flip(input0, dims=[0, 1]) * 0.5
            builder.set_goldens(inputs={in0: input0, in1: input1})
            return builder.add(
                in0, in1, loc="tensor_accessor_add", unit_attrs=unit_attrs
            )

    pipeline = (
        "ttir-to-ttmetal-pipeline{default-input-memspace=dram "
        "default-output-memspace=dram global-data-format-target=bfp_bf8 "
        "num-stream-buffers=1 test-buffer-size-policy=max "
        "use-tensor-accessor-dma=1}"
    )
    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=pipeline,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_tensor_accessor_outer_permute(
    target: str,
    request,
    device,
):
    """Exercise a non-identity page map while preserving whole tiles."""
    shape = (2, 3, 64, 128)

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def tensor_accessor_outer_permute(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            input0 = torch.arange(
                shape[0] * shape[1] * shape[2] * shape[3],
                dtype=torch.float32,
            ).reshape(shape)
            builder.set_goldens(inputs={in0: input0})
            return builder.permute(
                in0,
                permutation=[1, 0, 2, 3],
                loc="tensor_accessor_outer_permute",
                unit_attrs=unit_attrs,
            )

    pipeline = (
        "ttir-to-ttmetal-pipeline{collapse-tensors-2d=0 "
        "num-stream-buffers=1 test-buffer-size-policy=max "
        "use-tensor-accessor-dma=1}"
    )
    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=pipeline,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_tensor_accessor_matmul(
    target: str,
    request,
    device,
):
    """Exercise TensorAccessor reads through matmul reblocking and multicast."""
    lhs_shape = (192, 128)
    rhs_shape = (128, 192)
    dtype = torch.bfloat16

    def module(builder: TTIRBuilder):
        @builder.func([lhs_shape, rhs_shape], [dtype, dtype])
        def tensor_accessor_matmul(
            lhs: Operand,
            rhs: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            lhs_input = torch.linspace(
                0.001, 0.999, lhs_shape[0] * lhs_shape[1]
            ).reshape(lhs_shape)
            rhs_input = torch.linspace(
                0.999, 0.001, rhs_shape[0] * rhs_shape[1]
            ).reshape(rhs_shape)
            builder.set_goldens(
                inputs={
                    lhs: lhs_input.to(dtype),
                    rhs: rhs_input.to(dtype),
                }
            )
            return builder.matmul(
                lhs,
                rhs,
                loc="tensor_accessor_matmul",
                unit_attrs=unit_attrs,
            )

    pipeline = (
        "ttir-to-ttmetal-pipeline{default-input-memspace=dram "
        "default-output-memspace=dram matmul-interchange=2,0,1 "
        "num-stream-buffers=1 test-buffer-size-policy=max "
        "use-tensor-accessor-dma=1 use-tile-matmul=false}"
    )
    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=pipeline,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("start_grid", [(1, 1), (1, 2), (2, 1), (4, 4)])
@pytest.mark.parametrize("end_grid", [(1, 1), (2, 2)])
@pytest.mark.parametrize(
    "memory_space", [ttcore.MemorySpace.DeviceL1, ttcore.MemorySpace.DeviceDRAM]
)
def test_roundtrip_dma_rowmajor(
    shape: Shape,
    start_grid: tuple[int, int],
    end_grid: tuple[int, int],
    memory_space: ttcore.MemorySpace,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def dram_write(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):

            to_device = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, tiled=False, memorySpace=ttcore.MemorySpace.DeviceL1
                ),
                unit_attrs=unit_attrs,
            )

            # derive sharded shapes
            assert (shape[0] % start_grid[0] == 0) and (
                shape[1] % start_grid[1] == 0
            ), "shape must be divisible by grid"
            start_shard_shape = (shape[0] // start_grid[0], shape[1] // start_grid[1])
            assert (start_shard_shape[0] % end_grid[0] == 0) and (
                start_shard_shape[1] % end_grid[1] == 0
            ), "start_shard_shape must be divisible by end_grid"

            # WRITE L1 to initial shard layout
            tensor_layoutA = builder.to_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=False,
                    memorySpace=memory_space,
                    grid=start_grid,
                ),
                unit_attrs=unit_attrs,
            )

            # READ sharded layout to final sharded layout
            tensor_layoutB = builder.to_layout(
                tensor_layoutA,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=False,
                    memorySpace=ttcore.MemorySpace.DeviceL1,
                    grid=end_grid,
                ),
                unit_attrs=unit_attrs,
            )

            system_out = builder.to_layout(
                tensor_layoutB,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )

            return system_out

    compile_dma_test(module, target, request, device=device)


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("shape", [(256, 256)])
@pytest.mark.parametrize("dram_grid", [(1, 2), (2, 1), (2, 2), (4, 4)])
def test_host_sharded_dram_roundtrip(
    shape: Shape,
    dram_grid: tuple[int, int],
    target: str,
    request,
    device,
):
    """Tests direct host -> sharded DRAM -> host transfers."""

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def dram_roundtrip(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            to_dram = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=False,
                    memorySpace=ttcore.MemorySpace.DeviceDRAM,
                    grid=dram_grid,
                ),
                unit_attrs=unit_attrs,
            )

            return builder.to_layout(
                to_dram,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )

    compile_dma_test(module, target, request, device=device)


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("shape", [(2560, 256)])
def test_host_dram_roundtrip_exceeds_l1(
    shape: Shape,
    target: str,
    request,
    device,
):
    """Tests host -> DRAM -> host for a tensor too large to stage in L1."""

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def dram_roundtrip(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            to_dram = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=False,
                    memorySpace=ttcore.MemorySpace.DeviceDRAM,
                ),
                unit_attrs=unit_attrs,
            )

            return builder.to_layout(
                to_dram,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )

    compile_dma_test(module, target, request, device=device)


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("shape", [(64, 64), (64, 128), (128, 64), (128, 128)])
@pytest.mark.parametrize("end_grid", [(1, 1), (2, 2), (1, 2), (2, 1)])
def test_interleaved_dma(
    shape: Shape, end_grid: tuple[int, int], request, target, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def interleaved_dma(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # derive sharded shapes
            assert (
                (shape[0] % end_grid[0] == 0) and (shape[1] % end_grid[1] == 0),
                "shard_shape must be divisible by end_grid",
            )

            # tilize the tensor on a single worker
            to_device = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=True,
                    memorySpace=ttcore.MemorySpace.DeviceL1,
                ),
                unit_attrs=unit_attrs,
            )

            # WRITE tensor from L1 to initial shard layout
            tensor_layoutA = builder.to_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=True,
                    memorySpace=ttcore.MemorySpace.DeviceDRAM,
                    grid=(1, 1),  # interleaved grid must be 1x1
                    memory_layout=ttcore.TensorMemoryLayout.Interleaved,
                ),
                unit_attrs=unit_attrs,
            )

            # READ sharded layout to final sharded layout
            tensor_layoutB = builder.to_layout(
                tensor_layoutA,
                output_type=builder.get_metal_tensor_layout(
                    shape,
                    tiled=True,
                    memorySpace=ttcore.MemorySpace.DeviceL1,
                    grid=end_grid,
                ),
                unit_attrs=unit_attrs,
            )

            untilize_out = builder.untilize(
                tensor_layoutB,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )

            return untilize_out

    compile_dma_test(module, target, request, device=device)
