# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np


def mask_torch_inf_nan(tensor):
    tensor[
        torch.logical_or(
            torch.isnan(tensor),
            torch.logical_or(torch.isinf(tensor), torch.isneginf(tensor)),
        )
    ] = 0
    return tensor


def get_atol_rtol(golden, calculated):
    """Calculate absolute and relative tolerance between golden and calculated tensors."""
    # Clone tensors to avoid modifying the originals
    golden = golden.clone()
    calculated = calculated.clone()
    if not torch.is_floating_point(golden):
        golden = golden.to(torch.float64)
    if not torch.is_floating_point(calculated):
        calculated = calculated.to(torch.float64)

    if golden.numel() == 0 or calculated.numel() == 0:
        cal_atol = 0.0
        cal_rtol = 0.0
    else:
        cal_atol = torch.max(torch.abs(golden - calculated)).item()
        cal_rtol = torch.max(torch.abs((golden - calculated) / calculated)).item()

    return cal_atol, cal_rtol


def get_pcc(golden, calculated, atol, rtol):
    """Calculate Pearson Correlation Coefficient (PCC) between golden and calculated tensors."""
    # Clone tensors to avoid modifying the originals
    golden = golden.clone()
    calculated = calculated.clone()
    if not torch.is_floating_point(golden):
        golden = golden.to(torch.float64)
    if not torch.is_floating_point(calculated):
        calculated = calculated.to(torch.float64)

    def _calculate_pcc(golden, calculated):
        if golden.numel() == 0 and calculated.numel() == 0:
            if golden.shape == calculated.shape:
                return 1.0
            else:
                return 0.0
        elif golden.numel() == 0 or calculated.numel() == 0:
            return 0.0
        if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
            return 1.0
        elif torch.any(golden.bool()) != torch.any(calculated.bool()):
            return 0.0
        elif torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
            return 0.0
        else:
            golden = mask_torch_inf_nan(golden)
            calculated = mask_torch_inf_nan(calculated)

            if torch.equal(golden, calculated):
                return 1.0

            if golden.dtype == torch.bfloat16:
                golden = golden.type(torch.float32)
            if calculated.dtype == torch.bfloat16:
                calculated = calculated.type(torch.float32)

            if golden.numel() == 1:
                return float(torch.isclose(golden, calculated, atol=atol, rtol=rtol))

            if torch.max(golden) == torch.min(golden) and torch.max(
                calculated
            ) == torch.min(calculated):
                return float(
                    torch.isclose(
                        torch.max(golden), torch.max(calculated), atol=atol, rtol=rtol
                    ).item()
                )

            cal_pcc = np.ma.corrcoef(
                np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
                np.ma.masked_invalid(
                    torch.squeeze(calculated).detach().numpy()
                ).flatten(),
            )
            mask = np.ones(cal_pcc.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            cal_pcc = np.min(cal_pcc[mask])

            if isinstance(cal_pcc, np.ma.core.MaskedConstant):
                return 1.0

            return cal_pcc

    if golden.numel() == 1 and golden.item() != 0:
        cal_pcc = (
            1.0
            if torch.nn.functional.cosine_similarity(
                golden.float().unsqueeze(0),
                calculated.float().unsqueeze(0),
                dim=0,
            ).item()
            else 0.0
        )
    else:
        cal_pcc = _calculate_pcc(golden, calculated)

    return cal_pcc


def get_relative_l2(golden, calculated):
    """Calculate relative L2 error between golden and calculated tensors.

    rel_l2 = ||calculated - golden||_2 / ||golden||_2

    Norms are computed in float64 to avoid bf16/fp32 underflow. Complements PCC:
    PCC is scale-blind, max-atol is dominated by a single outlier, max-rtol blows
    up near zero. rel_l2 is scale-sensitive, stable near zero globally (denominator
    is the golden norm, not per-element |y|), and captures distributed degradation
    rather than one bad element. Larger = worse.

    Returns a non-negative float: 0.0 if both tensors are empty or the golden norm
    is zero and the tensors match exactly, inf if the golden norm is zero but they
    differ, NaN propagated through if either tensor contains NaN.
    """
    golden = golden.to(torch.float64).flatten()
    calculated = calculated.to(torch.float64).flatten()

    if golden.numel() == 0 or calculated.numel() == 0:
        return 0.0

    diff_norm = torch.linalg.vector_norm(calculated - golden)
    golden_norm = torch.linalg.vector_norm(golden)

    if torch.isnan(diff_norm) or torch.isnan(golden_norm):
        return float("nan")

    if golden_norm.item() == 0.0:
        # Both exactly zero -> perfect match; otherwise undefined (inf).
        return 0.0 if diff_norm.item() == 0.0 else float("inf")

    return (diff_norm / golden_norm).item()


def get_atol_rtol_pcc(golden, calculated, atol, rtol):
    """Wrapper function that calculates atol, rtol, and pcc."""
    cal_atol, cal_rtol = get_atol_rtol(golden, calculated)
    cal_pcc = get_pcc(golden, calculated, atol, rtol)

    return (
        cal_atol,
        cal_rtol,
        cal_pcc,
    )
