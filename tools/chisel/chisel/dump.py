# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Dump device/golden input+output torch tensors for a failing numerics check.

A debug aid: on a PCC (or atol/rtol) failure for an op output, we torch.save the
host torch tensors already retrieved for the comparison - no extra device readback.
Failures here are logged, never raised: a debug dump must not break the run.

Layout under <debug_dir>:
    tensors/binary_{id}/prog{index}/{sanitized_ssa}/{mode}/
        input_{ssa}_device.pt
        input_{ssa}_golden.pt      # accumulation only (== device in isolation)
        output_{ssa}_device.pt
        output_{ssa}_golden.pt
        meta.json
"""
import json
import logging
import os
from typing import Dict, Optional

import torch

from golden import GoldenMapTensor

from .report import NumericsMode

logger = logging.getLogger("chisel")


def _sanitize(ssa: str) -> str:
    """Make an SSA name (e.g. "%0", "%arg1:1") safe as a path component."""
    return "".join(c if c.isalnum() else "_" for c in ssa).strip("_") or "anon"


def _save(tensor: GoldenMapTensor, path: str) -> None:
    # shard_map is Dict[int, torch.Tensor] - picklable as-is, keeps multi-shard
    # tensors in one file keyed by device id.
    torch.save(tensor.shard_map, path)


def dump_failing_op_tensors(
    debug_dir: str,
    *,
    binary_id: int,
    program_index: int,
    op_name: str,
    ssa: str,
    mode: NumericsMode,
    golden_out: GoldenMapTensor,
    device_out: GoldenMapTensor,
    device_inputs: Dict[str, GoldenMapTensor],
    golden_inputs: Optional[Dict[str, GoldenMapTensor]],
    pcc: Optional[float] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    min_pcc: Optional[float] = None,
) -> Optional[str]:
    """Dump one failing op's tensors. Returns the dir written, or None on error.

    `golden_inputs` is None in isolation mode (golden inputs == device inputs);
    pass the pool entries in accumulation mode to capture the divergent chain.
    """
    out_dir = os.path.join(
        debug_dir,
        "tensors",
        f"binary_{binary_id}",
        f"prog{program_index}",
        _sanitize(ssa),
        mode.value,
    )
    os.makedirs(out_dir, exist_ok=True)

    _save(device_out, os.path.join(out_dir, f"output_{_sanitize(ssa)}_device.pt"))
    _save(golden_out, os.path.join(out_dir, f"output_{_sanitize(ssa)}_golden.pt"))

    for in_ssa, dev_t in device_inputs.items():
        _save(dev_t, os.path.join(out_dir, f"input_{_sanitize(in_ssa)}_device.pt"))
    if golden_inputs is not None:
        for in_ssa, gold_t in golden_inputs.items():
            _save(
                gold_t,
                os.path.join(out_dir, f"input_{_sanitize(in_ssa)}_golden.pt"),
            )

    meta = {
        "op": op_name,
        "ssa": ssa,
        "mode": mode.value,
        "binary_id": binary_id,
        "program_index": program_index,
        "pcc": pcc,
        "atol": atol,
        "rtol": rtol,
        "min_pcc": min_pcc,
        "device_ids": sorted(device_out.shard_map.keys()),
        "input_ssas": sorted(device_inputs.keys()),
        "golden_inputs_dumped": golden_inputs is not None,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir
