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
ceil(K_tiles / M) * M.  The DMA that reshards the N-wide source tensor into
the M-wide padded destination issues NoC reads for phantom padding tiles.
Those phantom tiles map to source column N, which is out of bounds (the source
only occupies columns 0..N-1).

Concrete numbers used here
  - Producer exp: full device grid (8x8), K = 128 tiles (4096 / 32)
  - Consumer matmul: restricted to 8x7 via d2m.spatial core_range<(0,0),(7,6)>
  - 128 % 7 != 0  ->  K padded to ceil(128 / 7) * 7 = 133 tiles
  - Reshard DMA: source col has 128 / 8 = 16 tiles; dest col expects 133 / 7 = 19
  - Phantom tiles 128..132 map to source column 8 (out of bounds)

Without d2m.spatial both ops use the same 8x8 grid; 128 % 8 = 0, no phantom
tiles, bug does not trigger.

Observed failure modes
----------------------
Case 1 - FABRIC DISABLED (single device only)
    The OOB NoC address translates to a dormant ETH core (not in forwarding
    mode).  The read returns garbage data immediately, so the program
    completes but the matmul reduction accumulates phantom values.
    Result: wrong output (PCC << 0.99).

Case 2 - FABRIC_1D / FABRIC_1D_RING (any fabric-enabled configuration)
    ETH cores are in forwarding mode.  The OOB NoC read packet is forwarded
    through the fabric ring and no endpoint can serve it, so
    async_read_barrier never completes.
    Result: infinite device hang.

MLIR reproducer
---------------
Both cases use the same compiled MLIR:
    test/ttmlir/Silicon/TTMetal/n150/spatial/reshard_padding_hang_repro.mlir

The test compiles it with ttmlir-opt + ttmlir-translate on the fly, then
executes the resulting flatbuffer through the ttrt Python runtime API.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
import _ttmlir_runtime as tt_runtime
from conftest import get_request_kwargs
from ttrt.common.util import create_tensor, convert_runtime_to_torch_tensor

pytestmark = pytest.mark.frontend("ttmetal")

# Path to the D2M-level MLIR reproducer (relative to this file).
#   this file  : test/python/golden/d2m/test_reshard_padding_bug.py
#   MLIR source: test/ttmlir/Silicon/TTMetal/n150/spatial/
_MLIR_SRC = (
    Path(__file__)
    .resolve()
    .parent.parent.parent.parent  # d2m/  # golden/  # python/  # test/
    / "ttmlir"
    / "Silicon"
    / "TTMetal"
    / "n150"
    / "spatial"
    / "reshard_padding_hang_repro.mlir"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile_to_flatbuffer(sys_desc_path: str) -> str:
    """Compile the D2M reproducer MLIR to a TTMetal flatbuffer.

    Runs ttmlir-opt then ttmlir-translate in a temporary directory.
    Returns the path of the .ttm flatbuffer file.
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


def _run_and_get_output(device, fbb_path: str) -> torch.Tensor:
    """Load and execute the compiled flatbuffer on *device*.

    Returns the device output as a float32 CPU tensor.
    Uses the same random seed as _reference_output() so inputs match.
    """
    import ttrt.runtime
    import ttrt.binary

    torch.manual_seed(42)
    act = torch.randn(256, 4096, dtype=torch.bfloat16).contiguous()
    wt = torch.randn(4096, 256, dtype=torch.bfloat16).contiguous()

    fbb = ttrt.binary.load_binary_from_path(fbb_path)
    program_index = 0

    act_tensor = create_tensor([act], (1, 1))
    wt_tensor = create_tensor([wt], (1, 1))

    act_layout = ttrt.runtime.get_layout(fbb, program_index, 0)
    wt_layout = ttrt.runtime.get_layout(fbb, program_index, 1)
    act_conv = ttrt.runtime.to_layout(act_tensor, device, act_layout, True)
    wt_conv = ttrt.runtime.to_layout(wt_tensor, device, wt_layout, True)

    outputs = ttrt.runtime.submit(device, fbb, program_index, [act_conv, wt_conv])
    return convert_runtime_to_torch_tensor(outputs[0]).to(torch.float32)


def _reference_output() -> torch.Tensor:
    """Compute exp(act) @ weight on CPU (float32). Same seed as _run_and_get_output."""
    torch.manual_seed(42)
    act = torch.randn(256, 4096, dtype=torch.bfloat16)
    wt = torch.randn(4096, 256, dtype=torch.bfloat16)
    return torch.exp(act).to(torch.float32) @ wt.to(torch.float32)


# ---------------------------------------------------------------------------
# Case 1: single device, fabric DISABLED -> wrong output (PCC < 0.99)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("mesh_shape", [(1, 1)])
def test_reshard_padding_wrong_output(target, request, device):
    """Case 1: fabric disabled, single device.

    With no fabric the OOB NoC read to the dormant ETH core returns garbage
    immediately, so the matmul reduction accumulates phantom values and the
    output PCC is significantly below 0.99.

    On the BUGGY branch  : assertion fails (PCC << 0.99) -- demonstrates bug.
    After the fix is applied: assertion passes (PCC >= 0.99) -- regression guard.
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
        f"PCC {pcc:.6f} < 0.99: K-padding reshard bug causes garbage "
        f"matmul output (phantom tiles from OOB NoC read)"
    )


# ---------------------------------------------------------------------------
# Case 2: fabric enabled -> infinite device hang (SKIPPED in CI)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "KNOWN DEVICE HANG -- do not run in CI.\n"
        "\n"
        "With FABRIC_1D_RING active the OOB NoC read to an ETH core in\n"
        "forwarding mode is never acknowledged, so async_read_barrier hangs\n"
        "indefinitely and the device must be reset after the test.\n"
        "\n"
        "To observe the hang manually:\n"
        "  1. Remove this @pytest.mark.skip decorator.\n"
        "  2. Run on a Wormhole host with >= 2 chips (N300 or TG).\n"
        "  3. The test will hang at ttrt.runtime.submit(); kill it with\n"
        "     Ctrl-C, then reset the device with tt-smi reset."
    )
)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("mesh_shape", [(1, 1)])
@pytest.mark.parametrize(
    "fabric_config", [tt_runtime.runtime.FabricConfig.FABRIC_1D_RING]
)
def test_reshard_padding_hang(target, fabric_config, request, device):
    """Case 2: FABRIC_1D_RING enabled.

    When ring fabric is active on the host the ETH cores on the local chip are
    in forwarding mode.  The phantom NoC read packet is forwarded around the
    fabric ring; no endpoint can serve it, so async_read_barrier never
    returns.

    The test body is identical to test_reshard_padding_wrong_output; the only
    difference is the fabric configuration injected through the *device*
    fixture via the fabric_config parametrize.

    Root cause is the same OOB NoC read -- only the observable symptom differs
    (silent wrong output vs. infinite hang) depending on whether ETH cores are
    in forwarding mode.
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
