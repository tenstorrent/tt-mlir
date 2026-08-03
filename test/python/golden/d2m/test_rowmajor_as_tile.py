# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstration: relabeling a row-major buffer as tile-typed without moving data.

The compiler's only record of "is this tilized?" is the tensor element type
(ttcore::isTiled() == isa<TileType>(elementType)).  d2m.view_layout with
reinterpret_layout=true changes that label and emits no data movement, so a
buffer holding row-major bytes can be handed to tile-consuming ops.

Consequence, demonstrated by the two tests below:
  - elementwise ops are permutation-invariant, so the relabel is harmless
  - positional ops (reduce / untilize / transpose) read the wrong data
"""

import pytest
import torch
from pathlib import Path
from typing import List

from ttmlir.ir import *

from builder.base.builder_utils import Operand, Shape
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_apis import compile_and_execute_d2m
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


PIPELINE = (
    "ttcore-mark-functions-as-forward,"
    "d2m-lower-to-layout,canonicalize,ttir-bufferization-pipeline,"
    "d2m-insert-scratch-buffers,d2m-generic-apply-interchange,"
    "d2m-generate-outer-loops,d2m-allocate,d2m-lower-multicast-loads,"
    "d2m-generic-lower-to-explicit-form,canonicalize,d2m-be-pipeline,"
    "d2m-to-ttkernel-pipeline,d2m-to-ttmetal-pipeline"
)

TILE_SIZE = 32
FACE_SIZE = 16


def face_scramble(t: torch.Tensor) -> torch.Tensor:
    """Reference model of reading row-major bytes as if they were tile-ordered.

    Mirrors untilize_golden() in test_tilize.py: walks the buffer in face order
    and writes each datum to the row-major position the tile kernel believes it
    occupies.
    """
    shape = t.shape
    out = torch.zeros_like(t)
    flat = t.clone().flatten()
    idx = 0
    for tile_y in range(shape[0] // TILE_SIZE):
        for tile_x in range(shape[1] // TILE_SIZE):
            for face_y in range(TILE_SIZE // FACE_SIZE):
                for face_x in range(TILE_SIZE // FACE_SIZE):
                    for dy in range(FACE_SIZE):
                        for dx in range(FACE_SIZE):
                            oy = dy + tile_y * TILE_SIZE + face_y * FACE_SIZE
                            ox = dx + tile_x * TILE_SIZE + face_x * FACE_SIZE
                            out[oy, ox] = flat[idx]
                            idx += 1
    return out


def face_gather(t: torch.Tensor) -> torch.Tensor:
    """Inverse of face_scramble: read row-major, emit in face order.

    Mirrors tilize_golden() in test_tilize.py.  This is what a linear reader
    sees when the bytes underneath it are actually tile/face-ordered.
    """
    shape = t.shape
    out = torch.zeros_like(t).flatten()
    idx = 0
    for tile_y in range(shape[0] // TILE_SIZE):
        for tile_x in range(shape[1] // TILE_SIZE):
            for face_y in range(TILE_SIZE // FACE_SIZE):
                for face_x in range(TILE_SIZE // FACE_SIZE):
                    for dy in range(FACE_SIZE):
                        for dx in range(FACE_SIZE):
                            out[idx] = t[
                                dy + tile_y * TILE_SIZE + face_y * FACE_SIZE,
                                dx + tile_x * TILE_SIZE + face_x * FACE_SIZE,
                            ]
                            idx += 1
    return out.reshape(shape)


def _relabel_as_tile(builder: D2MBuilder, rowmajor, shape, dtype, unit_attrs):
    """The trick: metadata-only relabel of a row-major buffer to tile type."""
    id_map = AffineMap.get_identity(2 * len(shape), builder._ctx)
    return builder.view_layout(
        rowmajor,
        output_type=builder.get_metal_tensor_layout(
            shape, tiled=True, grid=(1, 1), element_dtype=dtype
        ),
        remapping=id_map,
        reinterpret_layout=True,
        unit_attrs=unit_attrs,
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_relabel_then_positional_op_scrambles(
    target: str, request, device, tmp_path: Path
):
    """A tile-consuming positional op reads relabeled row-major bytes wrong.

    Golden is the *scrambled* tensor, so a PASS here proves the relabel took
    effect and that no tilize was inserted to fix it up.
    """
    shape = (64, 64)
    dtype = torch.float32
    ir_dump_dir = tmp_path / "ir_dumps"

    def module(builder: D2MBuilder):
        in_golden = torch.randn(shape, dtype=dtype)
        out_golden = face_scramble(in_golden)

        @builder.func([shape], [dtype])
        def relabel_positional(
            in0: Operand, builder: D2MBuilder, unit_attrs: List[str] = None
        ):
            # 1. genuine untilize: bytes really become row-major
            rowmajor = builder.to_layout(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, tiled=False, grid=(1, 1), element_dtype=dtype
                ),
                unit_attrs=unit_attrs,
            )
            # 2. relabel as tile, no data movement
            as_tile = _relabel_as_tile(builder, rowmajor, shape, dtype, unit_attrs)
            # 3. hand to a positional tile consumer
            out = builder.untilize(as_tile, output_type=in0.type, unit_attrs=unit_attrs)

            builder.set_goldens({in0: in_golden}, {out: out_golden})
            return out

    kwargs = get_request_kwargs(request)
    kwargs.update(
        test_base=request.node.name,
        output_root=str(tmp_path),
        save_artifacts=True,
        print_ir=str(ir_dump_dir),
    )
    compile_and_execute_d2m(
        module, target=target, custom_pipeline=PIPELINE, device=device, **kwargs
    )
    print(f"\nIR dumps: {ir_dump_dir}")


@pytest.mark.parametrize("target", ["ttmetal"])
def test_relabel_tile_as_rowmajor(target: str, request, device, tmp_path: Path):
    """The trick in the other direction: tile bytes relabeled as row-major.

    Genuinely tilize, relabel the tile buffer as row-major (no data movement),
    then let a row-major-consuming to_layout read it.  The reader walks the
    buffer linearly while the bytes are in face order, so the output is the
    face-order permutation of the input -- i.e. tilize_golden, not the input.
    """
    shape = (64, 64)
    dtype = torch.float32
    ir_dump_dir = tmp_path / "ir_dumps"

    def module(builder: D2MBuilder):
        in_golden = torch.randn(shape, dtype=dtype)

        @builder.func([shape], [dtype])
        def relabel_roundtrip(
            in0: Operand, builder: D2MBuilder, unit_attrs: List[str] = None
        ):
            tiled = builder.tilize(
                in0,
                output_type=builder.get_metal_tensor_layout(
                    shape, tiled=True, grid=(1, 1), element_dtype=dtype
                ),
                unit_attrs=unit_attrs,
            )
            # relabel tile -> row-major, no data movement
            id_map = AffineMap.get_identity(2 * len(shape), builder._ctx)
            as_rm = builder.view_layout(
                tiled,
                output_type=builder.get_metal_tensor_layout(
                    shape, tiled=False, grid=(1, 1), element_dtype=dtype
                ),
                remapping=id_map,
                reinterpret_layout=True,
                unit_attrs=unit_attrs,
            )
            out = builder.to_layout(as_rm, output_type=in0.type, unit_attrs=unit_attrs)

            builder.set_goldens({in0: in_golden}, {out: face_gather(in_golden)})
            return out

    kwargs = get_request_kwargs(request)
    kwargs.update(
        test_base=request.node.name,
        output_root=str(tmp_path),
        save_artifacts=True,
        print_ir=str(ir_dump_dir),
    )
    compile_and_execute_d2m(
        module, target=target, custom_pipeline=PIPELINE, device=device, **kwargs
    )
    print(f"\nIR dumps: {ir_dump_dir}")
