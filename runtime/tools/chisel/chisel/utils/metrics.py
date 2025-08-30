# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


def _to_common_shape(x, y):
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
    ttir_shapes = list(x.shape)
    ttnn_shapes = list(y.shape)
    if x.ndim == y.ndim and not len(set(ttir_shapes)) < len(ttir_shapes):
        try:
            permutation = [ttir_shapes.index(i) for i in ttnn_shapes]
            y = y.permute(permutation)
            return x, y
        except Exception:
            pass

    # Last Last ditch attempt, try to merge the last two dimensions to match
    # the other tensor
    if len(ttir_shapes) - len(ttnn_shapes) == 1:
        sz = ttir_shapes[-1] * ttir_shapes[-2]
        if sz == ttnn_shapes[-1]:
            y = y.view(x.shape)
            return x, y

    # Last Last Last ditch attempt, flatten both
    if x.numel() == y.numel():
        x = x.flatten()
        y = y.flatten()
        return x, y

    return None, None


def compute_abs_err(ttir_result, ttnn_result):
    ttir_result, ttnn_result = _to_common_shape(ttir_result, ttnn_result)
    if ttir_result is None or ttnn_result is None:
        return -1

    ttir_result = ttir_result.to(torch.float32)
    ttnn_result = ttnn_result.to(torch.float32)

    if torch.all(torch.isnan(ttir_result)) and torch.all(torch.isnan(ttnn_result)):
        return 0.0

    # compare if inf
    if torch.all(torch.isinf(ttir_result)) and torch.all(torch.isinf(ttnn_result)):
        if torch.all(ttir_result == ttnn_result):
            return 0.0
        return torch.inf

    abs_err = torch.max(torch.abs(ttir_result - ttnn_result))
    return abs_err.item()


def compute_rel_err(ttir_result, ttnn_result):
    ttir_result, ttnn_result = _to_common_shape(ttir_result, ttnn_result)
    if ttir_result is None or ttnn_result is None:
        return -1

    ttir_result = ttir_result.to(torch.float32)
    ttnn_result = ttnn_result.to(torch.float32)

    if torch.all(torch.isnan(ttir_result)) and torch.all(torch.isnan(ttnn_result)):
        return 0.0

    # compare if inf
    if torch.all(torch.isinf(ttir_result)) and torch.all(torch.isinf(ttnn_result)):
        if torch.all(ttir_result == ttnn_result):
            return 0.0
        return torch.inf

    rel_err = torch.max(torch.abs((ttir_result - ttnn_result) / (ttir_result + 1e-8)))
    return rel_err.item()


def compute_pcc(ttir_result, ttnn_result):
    ttir_result, ttnn_result = _to_common_shape(ttir_result, ttnn_result)
    if ttir_result is None or ttnn_result is None:
        return -1

    x = ttir_result.to(torch.float32).flatten()
    y = ttnn_result.to(torch.float32).flatten()

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
    except RuntimeError as e:
        print(f"Warning: Masking failed with error: {e}")
        pass

    # We can adjust tolerances here if needed
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
