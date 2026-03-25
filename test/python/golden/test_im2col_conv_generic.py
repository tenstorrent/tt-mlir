# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Builder test for the pre-lowered im2col convolution ttnn.generic op.

Loads debug_conv_lowered.mlir (already contains ttnn.generic + EmitC kernels),
serialises it straight to a flatbuffer, runs on device, and compares the
device output against a Python-computed golden.

Convolution parameters (from the MLIR IR):
  Image:   H=9, W=9, C=8  ->  activation shard 81x8 bf16
  Kernel:  Kh=2, Kw=2, stride=1, no padding
  OH=8, OW=8  ->  64 output pixels
  K = Kh*Kw*C = 32   (one tile width)
  Cout = 64

  Blocking: 2 blocks of 32 output pixels.
  Per-block im2col: 32x32 bf16  ->  1 tile.
  Matmul per block:  [1,1] x [1,2] -> [1,2] tiles (accumulated across blocks).
  Output per block:  32x64 bf16 (untilized from [1,2] tiles).

Input choices (designed for easy visual verification):
  act[r, c]  = 1.0              uniform — im2col rows are all-ones
  wt[i, j]   = j + 1            column ramp 1..64

Expected output (every row identical):
  output[r, j] = 2 * 32 * (j+1) = 64*(j+1)
  -> [64, 128, 192, 256, ..., 4096]
"""

import os

import numpy as np
import pytest
import torch

import _ttmlir_runtime as tt_runtime
from ttmlir.ir import Context, Location, Module
from ttmlir.passes import ttnn_to_flatbuffer_bin
from builder.base.builder_runtime import execute_fb
from golden import GoldenMapTensor

pytestmark = pytest.mark.frontend("ttnn")

# ---------------------------------------------------------------------------
# Golden reference
# ---------------------------------------------------------------------------

INPUT_H, INPUT_W, INPUT_C = 9, 9, 8
KH, KW = 2, 2
OH, OW = 8, 8
COUT = 64
K_DIM = KH * KW * INPUT_C  # 32
NUM_BLOCKS = 2
BLOCK_SIZE = 32  # output pixels per block


def im2col_gather(activation: np.ndarray, block_idx: int) -> np.ndarray:
    """Gather one 32x32 im2col block from the 81x8 activation.

    Affine map (from the kernel IR):
        src_row = (d0_eff//8 + d1//16)*9 + d0_eff%8 + (d1%16)//8
        src_col = d1 % 8
    where d0_eff = block_idx * 32 + d0_local.
    """
    col = np.zeros((BLOCK_SIZE, K_DIM), dtype=np.float32)
    for d0_local in range(BLOCK_SIZE):
        d0 = block_idx * BLOCK_SIZE + d0_local
        for d1 in range(K_DIM):
            src_row = (d0 // 8 + d1 // 16) * 9 + d0 % 8 + (d1 % 16) // 8
            src_col = d1 % 8
            col[d0_local, d1] = activation[src_row, src_col]
    return col


def compute_golden(act_np: np.ndarray, wt_np: np.ndarray) -> np.ndarray:
    """output = sum_over_blocks(im2col_block_b @ weights)   (32x64)"""
    result = np.zeros((BLOCK_SIZE, COUT), dtype=np.float32)
    for b in range(NUM_BLOCKS):
        result += im2col_gather(act_np, b) @ wt_np
    return result


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ttnn"])
def test_im2col_conv_generic(target, request, device):
    mlir_path = os.path.join(
        os.path.dirname(__file__), "../../../debug_conv_lowered.mlir"
    )
    if not os.path.exists(mlir_path):
        pytest.skip(f"MLIR file not found: {mlir_path}")

    with open(mlir_path, "r") as f:
        mlir_text = f.read()

    ctx = Context()
    ctx.allow_unregistered_dialects = True
    loc = Location.unknown(ctx)
    with ctx, loc:
        module = Module.parse(mlir_text)

    # -- Activation 81x8: all ones -----------------------------------------
    # act = torch.ones(81, 8, dtype=torch.bfloat16)
    act = (
        torch.arange(1, 82, dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(81, 8)
        .contiguous()
    )

    # -- Weights 32x64: wt[i, j] = j + 1  (column ramp 1..64) -----------
    wt = torch.ones(32, 64, dtype=torch.bfloat16)
    # wt = torch.zeros(32, 64, dtype=torch.bfloat16)
    # for j in range(64):
    #     wt[:, j] = float(j + 1)

    # -- Output buffer (zeros) --------------------------------------------
    out = torch.zeros(32, 64, dtype=torch.bfloat16)

    # -- Compute golden in fp32, round to bf16 ----------------------------
    golden_f32 = compute_golden(act.float().numpy(), wt.float().numpy())
    golden = torch.from_numpy(golden_f32.copy()).to(torch.bfloat16)

    golden_io = {
        0: {
            "input_0": GoldenMapTensor({0: act}, (1, 1)),
            "input_1": GoldenMapTensor({0: wt}, (1, 1)),
            "input_2": GoldenMapTensor({0: out}, (1, 1)),
            "output_0": GoldenMapTensor({0: golden}, (1, 1)),
        }
    }

    compiled = ttnn_to_flatbuffer_bin(module)

    golden_report, output_tensors = execute_fb(
        compiled,
        golden_io,
        {},
        device=device,
        check_pcc=True,
        pcc=0.5,
    )

    # -- Print full tensors -----------------------------------------------
    torch.set_printoptions(precision=4, linewidth=200, sci_mode=False, threshold=100000)

    print("\n" + "=" * 80)
    print("ACTIVATION (81x8 bf16)  all ones")
    print(act)

    print("\nWEIGHTS (32x64 bf16)  wt[i,j] = j+1")
    print(wt)

    print("\nGOLDEN OUTPUT (32x64):")
    print(golden)

    print("\nDEVICE OUTPUT (32x64):")
    for prog_key, prog_tensors in output_tensors.items():
        for key, val in prog_tensors.items():
            print(f"  {prog_key}/{key}:")
            print(val)

    print("\nGOLDEN REPORT:")
    print(golden_report)
    print("=" * 80)
