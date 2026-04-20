# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List
from conftest import get_request_kwargs

from ttmlir.dialects import ttcore
from ttmlir.ir import *

from builder.base.builder_utils import Operand, Shape
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_apis import compile_and_execute_d2m


pytestmark = pytest.mark.frontend("ttir")


def tilize_golden(input_tensor):
    shape = input_tensor.shape
    TILE_SIZE = 32
    FACE_SIZE = 16
    Y_TILES = shape[0] // TILE_SIZE
    X_TILES = shape[1] // TILE_SIZE
    FACES_PER_TILE = TILE_SIZE // FACE_SIZE

    tilized = torch.zeros_like(input_tensor)
    tilized = tilized.flatten()

    idx = 0
    for tile_y in range(Y_TILES):
        for tile_x in range(X_TILES):
            for face_y in range(FACES_PER_TILE):
                for face_x in range(FACES_PER_TILE):
                    for datum_y in range(FACE_SIZE):
                        for datum_x in range(FACE_SIZE):
                            tilized[idx] = input_tensor[
                                datum_y + tile_y * TILE_SIZE + face_y * FACE_SIZE,
                                datum_x + tile_x * TILE_SIZE + face_x * FACE_SIZE,
                            ]
                            idx += 1

    tilized = tilized.reshape(shape)
    return tilized


def untilize_golden(input_tensor):
    shape = input_tensor.shape
    TILE_SIZE = 32
    FACE_SIZE = 16
    Y_TILES = shape[0] // TILE_SIZE
    X_TILES = shape[1] // TILE_SIZE
    FACES_PER_TILE = TILE_SIZE // FACE_SIZE

    untilized = torch.zeros_like(input_tensor)
    flattened = input_tensor.clone()
    flattened = flattened.flatten()

    idx = 0
    for tile_y in range(Y_TILES):
        for tile_x in range(X_TILES):
            for face_y in range(FACES_PER_TILE):
                for face_x in range(FACES_PER_TILE):
                    for datum_y in range(FACE_SIZE):
                        for datum_x in range(FACE_SIZE):
                            # Calculate the original position
                            orig_y = datum_y + tile_y * TILE_SIZE + face_y * FACE_SIZE
                            orig_x = datum_x + tile_x * TILE_SIZE + face_x * FACE_SIZE

                            # Place the value from the tilized tensor back to its original position
                            untilized[orig_y, orig_x] = flattened[idx]
                            idx += 1

    return untilized


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64), (64, 128)])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.int32, torch.bfloat16, torch.uint16],
    ids=["f32", "i32", "bf16", "u16"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_tilize(shape: Shape, target: str, dtype: torch.dtype, request, device):
    def module(builder: D2MBuilder):
        if dtype.is_floating_point:
            in_golden = torch.randn(shape, dtype=dtype)
        else:
            in_golden = torch.randint(100, shape, dtype=dtype)
        out_golden = tilize_golden(in_golden)

        @builder.func([shape], [dtype])
        def tilize(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):

            to_device = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, tiled=True, element_dtype=dtype
                ),
                unit_attrs=unit_attrs,
            )

            # Provide an explicit identity remapping for the view op.
            id_map = AffineMap.get_identity(2 * len(shape), builder._ctx)
            view_as_rm = builder.view_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    shape, tiled=False, element_dtype=dtype
                ),
                remapping=id_map,
                reinterpret_layout=True,
                unit_attrs=unit_attrs,
            )

            from_device = builder.to_layout(
                view_as_rm,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )

            builder.set_goldens({in0: in_golden}, {from_device: out_golden})

            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64), (64, 128)])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.int32, torch.bfloat16, torch.uint16],
    ids=["f32", "i32", "bf16", "u16"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_untilize(shape: Shape, target: str, dtype: torch.dtype, request, device):
    def module(builder: D2MBuilder):
        if dtype.is_floating_point:
            in_golden = torch.randn(shape, dtype=dtype)
        else:
            in_golden = torch.randint(100, shape, dtype=dtype)
        out_golden = untilize_golden(in_golden)

        @builder.func([shape], [dtype])
        def untilize(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            to_device = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, tiled=False, grid=(1, 1), element_dtype=dtype
                ),
                unit_attrs=unit_attrs,
            )

            # Provide an explicit identity remapping for the view op.
            id_map = AffineMap.get_identity(2 * len(shape), builder._ctx)
            view_as_tiled = builder.view_layout(
                to_device,
                output_type=builder.get_metal_tensor_layout(
                    shape, grid=(1, 1), tiled=True, element_dtype=dtype
                ),
                remapping=id_map,
                reinterpret_layout=True,
                unit_attrs=unit_attrs,
            )

            from_device = builder.untilize(
                view_as_tiled,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )

            builder.set_goldens({in0: in_golden}, {from_device: out_golden})

            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("shape", [(32, 64), (64, 32), (64, 64)])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.int32, torch.bfloat16, torch.uint16],
    ids=["f32", "i32", "bf16", "u16"],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_tilize_untilize(
    shape: Shape, target: str, dtype: torch.dtype, request, device
):
    def module(builder: D2MBuilder):
        if dtype.is_floating_point:
            golden = torch.randn(shape, dtype=dtype)
        else:
            golden = torch.randint(100, shape, dtype=dtype)

        @builder.func([shape], [dtype])
        def tilize_untilize(
            in0: Operand,
            builder: D2MBuilder,
            unit_attrs: List[str] = None,
        ):
            to_device = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, tiled=True, grid=(1, 1), element_dtype=dtype
                ),
                unit_attrs=unit_attrs,
            )
            from_device = builder.untilize(
                to_device,
                output_type=in0.type,
                unit_attrs=unit_attrs,
            )

            builder.set_goldens({in0: golden}, {from_device: golden})

            return from_device

    compile_and_execute_d2m(
        module,
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        **get_request_kwargs(request),
    )
