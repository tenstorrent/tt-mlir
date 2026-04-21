# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unified tensor comparison metrics — PCC, absolute error, relative error.

Ported from tools/builder/base/builder_runtime.py::get_atol_rtol_pcc.
"""
import numpy as np
import torch


def _mask_inf_nan(tensor: torch.Tensor) -> torch.Tensor:
    tensor[
        torch.logical_or(
            torch.isnan(tensor),
            torch.logical_or(torch.isinf(tensor), torch.isneginf(tensor)),
        )
    ] = 0
    return tensor


def _to_float(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(t):
        return t.to(torch.float64)
    return t


def compute_atol(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Max absolute difference: max(|golden - calculated|)."""
    golden = _to_float(golden.clone())
    calculated = _to_float(calculated.clone())

    if golden.numel() == 0 or calculated.numel() == 0:
        return 0.0

    return torch.max(torch.abs(golden - calculated)).item()


def compute_rtol(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Max relative difference: max(|(golden - calculated) / calculated|)."""
    golden = _to_float(golden.clone())
    calculated = _to_float(calculated.clone())

    if golden.numel() == 0 or calculated.numel() == 0:
        return 0.0

    return torch.max(torch.abs((golden - calculated) / calculated)).item()


def compute_pcc(
    golden: torch.Tensor,
    calculated: torch.Tensor,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> float:
    """Pearson correlation coefficient between two tensors.

    Returns 1.0 for identical tensors, 0.0 for completely different,
    -1 if shapes cannot be reconciled.
    """
    golden = _to_float(golden.clone())
    calculated = _to_float(calculated.clone())

    # Empty tensor handling
    if golden.numel() == 0 and calculated.numel() == 0:
        return 1.0 if golden.shape == calculated.shape else 0.0
    if golden.numel() == 0 or calculated.numel() == 0:
        return 0.0

    # Single non-zero scalar: use cosine similarity
    if golden.numel() == 1 and golden.item() != 0:
        return (
            1.0
            if torch.nn.functional.cosine_similarity(
                golden.float().unsqueeze(0),
                calculated.float().unsqueeze(0),
                dim=0,
            ).item()
            else 0.0
        )

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        return 1.0
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        return 0.0
    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        return 0.0

    golden = _mask_inf_nan(golden)
    calculated = _mask_inf_nan(calculated)

    if torch.equal(golden, calculated):
        return 1.0

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
    if calculated.dtype == torch.bfloat16:
        calculated = calculated.type(torch.float32)

    if golden.numel() == 1:
        return float(torch.isclose(golden, calculated, atol=atol, rtol=rtol))

    # Constant tensors: Pearson undefined, fall back to isclose on the constant value
    if torch.max(golden) == torch.min(golden) and torch.max(calculated) == torch.min(
        calculated
    ):
        return float(
            torch.isclose(
                torch.max(golden), torch.max(calculated), atol=atol, rtol=rtol
            ).item()
        )

    cal_pcc = np.ma.corrcoef(
        np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
        np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
    )
    mask = np.ones(cal_pcc.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    cal_pcc = np.min(cal_pcc[mask])

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return 1.0

    return float(cal_pcc)
