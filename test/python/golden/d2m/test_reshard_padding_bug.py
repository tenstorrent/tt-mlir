# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Reproducer tests for the K-padding reshard bug.

Root cause
----------
When a producer d2m.generic runs on an N-wide core-column grid and a consumer
d2m.generic is restricted (via d2m.spatial) to an M-wide grid where M does
not divide K_tiles, the grid-aware layout alignment pads K to
ceil(K_tiles / M) * M.  The reshard DMA issues NoC reads for phantom padding
tiles that map to source column N -- out of bounds for columns 0..N-1.

Concrete setup
  - act [256, 4096]:  K = 128 tiles (4096 / 32)
  - Producer grid:    8x8  -> 16 K-tiles per column
  - Consumer grid:    8x7  via d2m.spatial core_range<(0,0),(7,6)>
  - 128 % 7 != 0  ->  K padded to ceil(128/7)*7 = 133 tiles, 19 tiles/col
  - Reshard DMA:      reads tiles 128..132 from source column 8 (OOB)

Without d2m.spatial both ops use the same 8x8 grid; 128 % 8 = 0, no phantom
tiles, bug does not trigger.

Failure modes
-------------
Case 1 - FABRIC_DISABLED, standalone n150 (single chip)
    ETH core is dormant; phantom read returns garbage immediately.
    Program completes with WRONG OUTPUT (PCC << 0.99).

Case 1 alt - FABRIC_DISABLED, LLMBox / n300 (multi-chip host)
    UMD opens all chips; ETH cores remain active even with FABRIC_DISABLED.
    submit() hangs (same outcome as Case 2).

Case 2 - FABRIC_1D / FABRIC_1D_RING (any multi-chip host)
    ETH core in forwarding mode.  Phantom read forwarded around ring, never
    acknowledged -> async_read_barrier never returns -> INFINITE HANG.

MLIR reproducer
---------------
Both tests compile the same D2M-level MLIR file:
    test/ttmlir/Silicon/TTMetal/n150/spatial/reshard_padding_hang_repro.mlir
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
import _ttmlir_runtime as tt_runtime

pytestmark = pytest.mark.frontend("ttmetal")

# Path to the D2M-level MLIR reproducer.
#   this file  : test/python/golden/d2m/test_reshard_padding_bug.py
#   MLIR source: test/ttmlir/Silicon/TTMetal/n150/spatial/
_MLIR_SRC = (
    Path(__file__).parents[3]
    / "ttmlir/Silicon/TTMetal/n150/spatial/reshard_padding_hang_repro.mlir"
)

# Tensor shapes and reference computation.
_ACT_SHAPE = (256, 4096)
_WT_SHAPE = (4096, 256)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile_to_flatbuffer(sys_desc_path: str) -> str:
    """Compile the D2M reproducer MLIR to a TTMetal flatbuffer.

    Runs ttmlir-opt then ttmlir-translate in a temporary directory.
    Returns the path to the .ttm flatbuffer file.
    Raises RuntimeError with captured stderr if either tool fails.
    """
    tmp_dir = tempfile.mkdtemp(prefix="reshard_padding_bug_")
    mlir_out = os.path.join(tmp_dir, "repro.mlir")
    fbb_out = os.path.join(tmp_dir, "repro.ttm")

    opt_cmd = [
        "ttmlir-opt",
        f"--ttir-to-ttmetal-pipeline=system-desc-path={sys_desc_path} ttnn-mode=false",
        "-o",
        mlir_out,
        str(_MLIR_SRC),
    ]
    r = subprocess.run(opt_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ttmlir-opt failed:\n{r.stderr}")

    translate_cmd = [
        "ttmlir-translate",
        "--ttmetal-to-flatbuffer",
        "-o",
        fbb_out,
        mlir_out,
    ]
    r = subprocess.run(translate_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ttmlir-translate failed:\n{r.stderr}")

    return fbb_out


def _to_runtime_tensor(t: torch.Tensor) -> tt_runtime.runtime.Tensor:
    """Wrap a contiguous CPU torch.Tensor as a borrowed runtime Tensor."""
    _dtype_map = {
        torch.float32: tt_runtime.runtime.DataType.Float32,
        torch.bfloat16: tt_runtime.runtime.DataType.BFloat16,
        torch.int32: tt_runtime.runtime.DataType.Int32,
        torch.uint16: tt_runtime.runtime.DataType.UInt16,
        torch.uint32: tt_runtime.runtime.DataType.UInt32,
    }
    return tt_runtime.runtime.create_borrowed_host_tensor(
        t.data_ptr(),
        list(t.shape),
        list(t.stride()),
        t.element_size(),
        _dtype_map[t.dtype],
    )


def _from_runtime_tensor(rt: tt_runtime.runtime.Tensor) -> torch.Tensor:
    """Copy a runtime Tensor back to a CPU torch.Tensor."""
    _dtype_map = {
        tt_runtime.runtime.DataType.Float32: torch.float32,
        tt_runtime.runtime.DataType.BFloat16: torch.bfloat16,
        tt_runtime.runtime.DataType.Int32: torch.int32,
        tt_runtime.runtime.DataType.UInt16: torch.int16,
        tt_runtime.runtime.DataType.UInt32: torch.int32,
    }
    data = bytearray(rt.get_data_buffer())
    return torch.frombuffer(data, dtype=_dtype_map[rt.get_dtype()]).reshape(
        rt.get_shape()
    )


def _run_and_get_output(device, fbb_path: str) -> torch.Tensor:
    """Load and execute the compiled flatbuffer on *device*.

    Returns the device output as a float32 CPU tensor.
    Uses the same random seed as _reference_output() so inputs match.
    """
    torch.manual_seed(42)
    act = torch.randn(*_ACT_SHAPE, dtype=torch.bfloat16).contiguous()
    wt = torch.randn(*_WT_SHAPE, dtype=torch.bfloat16).contiguous()

    fbb = tt_runtime.binary.load_binary_from_path(fbb_path)
    program_index = 0

    act_rt = _to_runtime_tensor(act)
    wt_rt = _to_runtime_tensor(wt)

    act_layout = tt_runtime.runtime.get_layout(fbb, program_index, 0)
    wt_layout = tt_runtime.runtime.get_layout(fbb, program_index, 1)
    act_conv = tt_runtime.runtime.to_layout(act_rt, device, act_layout, True)
    wt_conv = tt_runtime.runtime.to_layout(wt_rt, device, wt_layout, True)

    outputs = tt_runtime.runtime.submit(device, fbb, program_index, [act_conv, wt_conv])
    return _from_runtime_tensor(outputs[0]).to(torch.float32)


def _reference_output() -> torch.Tensor:
    """Compute exp(act) @ wt on CPU (float32). Same seed as _run_and_get_output."""
    torch.manual_seed(42)
    act = torch.randn(*_ACT_SHAPE, dtype=torch.bfloat16)
    wt = torch.randn(*_WT_SHAPE, dtype=torch.bfloat16)
    return torch.exp(act).to(torch.float32) @ wt.to(torch.float32)


# ---------------------------------------------------------------------------
# Case 1: single device, FABRIC_DISABLED -> wrong output (PCC < 0.99).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("fabric_config", [tt_runtime.runtime.FabricConfig.DISABLED])
def test_reshard_padding_wrong_output(target, fabric_config, request, device):
    """Case 1: FABRIC_DISABLED, standalone n150 -> wrong output (PCC < 0.99).

    The reshard DMA from 8x8 (16 K-tiles/col) to 8x7 (19 K-tiles/col, padded)
    issues async_read for source column 8 which is OOB.  With DISABLED fabric
    on a standalone n150 the ETH core is dormant; the phantom read returns
    garbage immediately, so the matmul reduction accumulates phantom values.

    On a multi-chip host (LLMBox, n300) submit() hangs instead (ETH cores stay
    active even with FABRIC_DISABLED).  See the skip reason above.

    Bug branch  : assertion fails (PCC << 0.99) -- confirms the bug.
    After fix   : assertion passes (PCC >= 0.99) -- regression guard.
    """
    sys_desc = request.config.getoption("--sys-desc")
    fbb_path = _compile_to_flatbuffer(sys_desc)

    dev_out = _run_and_get_output(device, fbb_path)
    ref_out = _reference_output()

    ref_flat = ref_out.flatten()
    dev_flat = dev_out.flatten()
    pcc = torch.corrcoef(torch.stack([ref_flat, dev_flat]))[0, 1].item()
    print(f"\nreshard_padding PCC = {pcc:.6f}")
    assert pcc > 0.99, (
        f"PCC {pcc:.6f} < 0.99: K-padding reshard bug produces garbage output "
        f"(phantom tiles from OOB NoC read accumulate in the matmul reduction)"
    )


# ---------------------------------------------------------------------------
# Case 2: FABRIC_1D_RING -> infinite device hang (SKIPPED in CI).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "fabric_config", [tt_runtime.runtime.FabricConfig.FABRIC_1D_RING]
)
def test_reshard_padding_hang(target, fabric_config, request, device):
    """Case 2: FABRIC_1D_RING enabled -> infinite device hang.

    With ring fabric active the ETH cores are in forwarding mode.  The phantom
    NoC read packet (for OOB source column 8) is forwarded around the ring and
    no endpoint can serve it, so async_read_barrier never completes.

    Same compiled flatbuffer as Case 1; only the fabric configuration differs.
    The test body itself hangs at ttrt.runtime.submit(); kill with Ctrl-C, then
    reset the device with tt-smi reset.
    """
    sys_desc = request.config.getoption("--sys-desc")
    fbb_path = _compile_to_flatbuffer(sys_desc)

    # This call hangs on fabric-enabled hardware:
    dev_out = _run_and_get_output(device, fbb_path)
    ref_out = _reference_output()

    ref_flat = ref_out.flatten()
    dev_flat = dev_out.flatten()
    pcc = torch.corrcoef(torch.stack([ref_flat, dev_flat]))[0, 1].item()
    print(f"\nreshard_padding PCC = {pcc:.6f}")
    assert pcc > 0.99, f"PCC {pcc:.6f} < 0.99"
