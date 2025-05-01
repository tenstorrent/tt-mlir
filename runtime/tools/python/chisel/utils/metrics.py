# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


def compute_abs_err(ttir_result, ttnn_result):
    if ttir_result.shape != ttnn_result.shape:
        # if it's just unsqueezing, unsqueeze the result
        try:
            ttir_result = torch.broadcast_to(ttir_result, ttnn_result.shape)
        except Exception:
            return None
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


def compute_pcc(ttir_result, ttnn_result):
    if ttir_result.shape != ttnn_result.shape:
        try:
            ttir_result = torch.broadcast_to(ttir_result, ttnn_result.shape)
        except Exception:
            return None

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

    if x.numel() == 0 or y.numel() == 0:
        return 1.0

    x_centered = x - x.mean()
    y_centered = y - y.mean()

    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2))

    if denominator == 0:
        pcc = 1.0
    else:
        pcc = numerator / denominator

    return float(pcc)
