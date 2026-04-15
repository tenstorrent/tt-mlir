# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unified tensor comparison metrics — PCC, absolute error, relative error.

Pure torch, no numpy dependency. Ported from the original chisel metrics
module at runtime/tools/chisel/chisel/utils/metrics.py.
"""
import torch


def _to_common_shape(x: torch.Tensor, y: torch.Tensor):
    if x.shape == y.shape:
        return x, y

    x = x.squeeze()
    y = y.squeeze()
    if x.shape == y.shape:
        return x, y
    try:
        x = torch.broadcast_to(x, y.shape)
        try:
            y = torch.broadcast_to(y, x.shape)
        except Exception:
            pass
        return x, y
    except Exception:
        pass
    try:
        y = torch.broadcast_to(y, x.shape)
        try:
            x = torch.broadcast_to(x, y.shape)
        except Exception:
            pass
        return x, y
    except Exception:
        pass

    # Last ditch attempt, we try to permute the dimensions
    # to match the other tensor
    x_shapes = list(x.shape)
    y_shapes = list(y.shape)
    if x.ndim == y.ndim and not len(set(x_shapes)) < len(x_shapes):
        try:
            permutation = [x_shapes.index(i) for i in y_shapes]
            y = y.permute(permutation)
            return x, y
        except Exception:
            pass

    # Last Last ditch attempt, try to merge the last two dimensions to match
    # the other tensor
    if len(x_shapes) - len(y_shapes) == 1:
        sz = x_shapes[-1] * x_shapes[-2]
        if sz == y_shapes[-1]:
            y = y.view(x.shape)
            return x, y

    # Last Last Last ditch attempt, flatten both
    if x.numel() == y.numel():
        x = x.flatten()
        y = y.flatten()
        return x, y

    return None, None


def compute_atol(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Max absolute difference: max(|golden - calculated|)."""
    golden, calculated = _to_common_shape(golden, calculated)
    if golden is None or calculated is None:
        return -1

    golden = golden.to(torch.float32)
    calculated = calculated.to(torch.float32)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        return 0.0

    if torch.all(torch.isinf(golden)) and torch.all(torch.isinf(calculated)):
        if torch.all(golden == calculated):
            return 0.0
        return torch.inf

    abs_err = torch.max(torch.abs(golden - calculated))
    return abs_err.item()


def compute_rtol(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Max relative difference: max(|(golden - calculated) / (|golden| + eps)|)."""
    golden, calculated = _to_common_shape(golden, calculated)
    if golden is None or calculated is None:
        return -1

    golden = golden.to(torch.float32)
    calculated = calculated.to(torch.float32)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        return 0.0

    if torch.all(torch.isinf(golden)) and torch.all(torch.isinf(calculated)):
        if torch.all(golden == calculated):
            return 0.0
        return torch.inf

    rel_err = torch.max(torch.abs((golden - calculated) / (torch.abs(golden) + 1e-8)))
    return rel_err.item()


def compute_pcc(golden: torch.Tensor, calculated: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors.

    Returns 1.0 for identical tensors, 0.0 for completely different,
    -1 if shapes cannot be reconciled.
    """
    golden, calculated = _to_common_shape(golden, calculated)
    if golden is None or calculated is None:
        return -1

    x = golden.to(torch.float32).flatten()
    y = calculated.to(torch.float32).flatten()

    if torch.all(torch.isnan(x)) and torch.all(torch.isnan(y)):
        return 1.0

    if torch.all(torch.isinf(x)) and torch.all(torch.isinf(y)):
        if torch.all(x == y):
            return 1.0
        return 0.0

    mask = ~(torch.isnan(x) | torch.isinf(x) | torch.isnan(y) | torch.isinf(y))

    try:
        x = x[mask]
        y = y[mask]
    except RuntimeError:
        pass

    def equal(a, b, rtol=1e-2, atol=1e-2) -> float:
        """Normally NaN; for debugging we use equality-style fallback."""
        return 1.0 if torch.allclose(a, b, rtol=rtol, atol=atol) else 0.0

    # Not enough valid samples for Pearson:
    # NOTE: This would normally be NaN, but for result verification we return
    # 1.0 if tensors are (numerically) equal, otherwise 0.0.
    if min(x.numel(), y.numel()) < 2:
        return equal(x, y)

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    # Zero variance (constant vector) -> Pearson undefined -> fallback
    # NOTE: This would normally be NaN, but for result verification we return
    # 1.0 if tensors are (numerically) equal, otherwise 0.0.
    sx2 = torch.sum(x_centered**2)
    sy2 = torch.sum(y_centered**2)
    if sx2.item() == 0.0 or sy2.item() == 0.0:
        return equal(x, y)

    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(sx2 * sy2)
    pcc = numerator / denominator

    return float(pcc)
