# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Log PCC / atol / rtol for a single op output comparison."""
import logging

import torch

from golden.metrics import compute_pcc, compute_atol, compute_rtol

logger = logging.getLogger("chisel")

PCC_WARN_THRESHOLD = 0.99


def compute_metrics(
    op_name: str,
    out_name: str,
    golden: torch.Tensor,
    device: torch.Tensor,
) -> None:
    pcc = compute_pcc(golden, device)
    atol = compute_atol(golden, device)
    rtol = compute_rtol(golden, device)
    msg = f"{op_name} {out_name}: PCC={pcc:.4f} atol={atol:.4e} rtol={rtol:.4e}"
    if pcc < PCC_WARN_THRESHOLD:
        logger.warning(msg)
    else:
        logger.info(msg)
